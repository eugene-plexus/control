"""The trust root's auth surface: status, initialize, login, logout.

Moved here from the agent, which held it only because it used to be the
single process in the install. Two states, as before:

  * **Uninitialized** — no passphrase. Only `GET /v1/auth/status`,
    `POST /v1/auth/initialize` and `/healthz` work. Note what that is
    *not*: a mode that lets everything through. A trust root with an
    open window until it is configured is a trust root with an open
    window.
  * **Initialized** — login is open, everything else needs a token.

What initialization does that the agent's version did not: it mints the
install's **service-token signing key**, the control root's **identity
keypair** and the **recovery recipient**, seals all three under the
passphrase-derived key, and snapshots them. That bundle is the
replication set, and it is why a standby can be promoted from a
passphrase rather than from a key somebody copied between hosts.

Login is rate-limited per source IP — five failures in sixty seconds
locks that source out for sixty. Per process rather than replicated: a
failed login on one host is not a fact about the install, and a standby
that inherited a lockout would be a denial of service with extra steps.
"""

from __future__ import annotations

import base64
import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .. import sealing, security
from .._generated.models import (
    AuthInitializeRequest,
    AuthStatus,
    SessionToken,
)
from ..auth_state import AuthState
from ..dependencies import problem, require_operator
from ..state_machine import AlreadyActive, NotActive, StateMachine

log = logging.getLogger(__name__)

router = APIRouter(tags=["auth"])

_bearer_scheme = HTTPBearer(auto_error=False)

_RATE_LIMIT_WINDOW_SECONDS = 60
_RATE_LIMIT_MAX_FAILURES = 5

# Names the current signing-key generation. Not a secret: it is what a
# node reports back so a rotation can tell which hosts are stale.
_INITIAL_KEY_ID = "1"


@router.get("/v1/auth/status", response_model=AuthStatus)
async def auth_status(request: Request) -> AuthStatus:
    """Has this install been through first-run setup?

    Unauthenticated by necessity — it is what the UI asks *before* it
    has a token, on every page load, to tell a fresh install from a
    logged-out one. One boolean, no secrets, no rate limit.
    """
    machine: StateMachine = request.app.state.machine
    return AuthStatus(initialized=machine.state.identity.salt is not None)


@router.post("/v1/auth/initialize", status_code=204)
async def initialize(request: Request, body: AuthInitializeRequest) -> None:
    """First-run only. Establishes the whole trust root.

    The passphrase is unrecoverable by design: losing it means losing
    every sealed secret in the install, **including the recovery
    recipient** that would otherwise reach a dead node's credentials.
    There is no reset path, and adding one would make the passphrase
    decorative.
    """
    machine: StateMachine = request.app.state.machine
    auth: AuthState = request.app.state.auth_state

    passphrase = body.passphrase.get_secret_value()
    if not passphrase:
        raise problem(status.HTTP_400_BAD_REQUEST, "Empty passphrase", "A passphrase is required.")

    salt = security.generate_master_key_salt()
    master_key = security.derive_master_key(passphrase, salt)
    verifier = security.hash_passphrase(passphrase)

    signing_key = security.generate_signing_key()
    control_identity = sealing.generate_control_identity()
    recovery = sealing.generate_sealing_keypair()

    try:
        machine.seed_identity(
            {
                "salt": base64.b64encode(salt).decode("ascii"),
                "passphraseVerifier": verifier,
                "sealedSigningKey": security.seal_b64(signing_key, master_key),
                "sealedControlKey": security.seal_b64(
                    base64.b64decode(control_identity.private), master_key
                ),
                "controlPublicKey": control_identity.public,
                "sealedRecoveryKey": security.seal_b64(
                    base64.b64decode(recovery.private), master_key
                ),
                "signingKeyId": _INITIAL_KEY_ID,
            }
        )
    except AlreadyActive as exc:
        raise problem(
            status.HTTP_409_CONFLICT,
            "Already initialized",
            "This install already has a passphrase. There is no reset endpoint by "
            "design — the passphrase is what the sealed secrets are sealed under.",
        ) from exc
    except NotActive as exc:
        raise problem(
            status.HTTP_409_CONFLICT,
            "Not the active root",
            "Only the active control root can initialize an install. A standby joins "
            "one that already exists.",
        ) from exc

    auth.set_master_key(master_key)
    auth.set_signing_key(signing_key)
    auth.control_private_key = control_identity.private
    auth.recovery_private_key = recovery.private
    log.info(
        "install initialized; epoch %d, signing key generation %s",
        machine.state.epoch,
        _INITIAL_KEY_ID,
    )


@router.post("/v1/auth/login", response_model=SessionToken)
async def login(request: Request, body: AuthInitializeRequest) -> SessionToken:
    """Verify the passphrase, unseal the keys, issue a session token.

    Unsealing on login is what makes a restart not a re-key: the signing
    key is in applied state sealed under the master key, so the process
    recovers the install's existing key rather than minting a new one.
    Through M4 a restart re-keyed everything, which was harmless with
    one process and an outage with N nodes.
    """
    machine: StateMachine = request.app.state.machine
    auth: AuthState = request.app.state.auth_state
    identity = machine.state.identity
    remote = request.client.host if request.client else "unknown"

    if identity.salt is None or identity.passphraseVerifier is None:
        raise problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Setup required",
            "No passphrase set yet. Call POST /v1/auth/initialize first.",
        )

    if auth.is_login_rate_limited(
        remote,
        window_seconds=_RATE_LIMIT_WINDOW_SECONDS,
        max_in_window=_RATE_LIMIT_MAX_FAILURES,
    ):
        raise problem(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Rate limited",
            f"Too many failed logins from {remote}. Wait {_RATE_LIMIT_WINDOW_SECONDS} "
            "seconds and try again.",
        )

    passphrase = body.passphrase.get_secret_value()
    if not security.verify_passphrase(passphrase, identity.passphraseVerifier):
        auth.record_login_failure(
            remote,
            window_seconds=_RATE_LIMIT_WINDOW_SECONDS,
            max_in_window=_RATE_LIMIT_MAX_FAILURES,
        )
        log.warning("failed login attempt from %s", remote)
        raise problem(
            status.HTTP_401_UNAUTHORIZED,
            "Wrong passphrase",
            "Passphrase did not match. Repeated failures from one source are rate-limited.",
        )

    auth.clear_login_failures(remote)
    unseal_into(auth, machine, passphrase)

    token, expires = security.issue_operator_token(signing_key=_require_signing_key(auth))
    log.info("operator login from %s", remote)
    return SessionToken(token=token, expiresAt=datetime.fromtimestamp(expires, tz=UTC))


@router.delete("/v1/auth/sessions/current", status_code=204)
async def logout(request: Request) -> None:
    """Revoke the presented session token.

    Validated through the same dependency as any protected route, so a
    caller can only revoke the token it is holding rather than an
    arbitrary one.
    """
    creds: HTTPAuthorizationCredentials | None = await _bearer_scheme(request)
    if creds is None or not creds.credentials:
        raise problem(
            status.HTTP_401_UNAUTHORIZED,
            "Missing token",
            "Provide the session token to revoke via Authorization: Bearer.",
        )
    _ = require_operator(request, creds)
    auth: AuthState = request.app.state.auth_state
    auth.revoke(creds.credentials)
    log.info("session revoked")


def unseal_into(auth: AuthState, machine: StateMachine, passphrase: str) -> None:
    """Derive the master key and open the three sealed values.

    Shared by login and promotion, because they need exactly the same
    thing: a standby being promoted is a host that has the salt and the
    verifier and is being handed the passphrase for the first time.

    A sealed value that will not open is reported loudly rather than
    skipped. On a promotion in particular, a control root that came up
    without the install's signing key would hand out tokens nothing
    accepts, and the operator needs to hear that now rather than from
    every component at once.
    """
    import base64 as _b64

    identity = machine.state.identity
    if identity.salt is None:
        raise problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Setup required",
            "This install has no salt recorded, so no key can be derived.",
        )

    master_key = security.derive_master_key(passphrase, _b64.b64decode(identity.salt))
    auth.set_master_key(master_key)

    if identity.sealedSigningKey is not None:
        try:
            auth.set_signing_key(security.open_b64(identity.sealedSigningKey, master_key))
        except ValueError as exc:
            raise problem(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "Sealed signing key unreadable",
                f"The install's signing key did not open with this passphrase ({exc}). "
                "Every service token in the install is signed with it, so this host "
                "cannot serve until it is recovered.",
            ) from exc

    if identity.sealedControlKey is not None:
        try:
            auth.control_private_key = _b64.b64encode(
                security.open_b64(identity.sealedControlKey, master_key)
            ).decode("ascii")
        except ValueError as exc:
            log.error("control identity key did not open: %s", exc)

    if identity.sealedRecoveryKey is not None:
        # Opened but not used in normal operation. Held so an explicit
        # recovery — reaching a dead node's secrets — does not need a
        # second passphrase prompt in the middle of an incident.
        try:
            auth.recovery_private_key = _b64.b64encode(
                security.open_b64(identity.sealedRecoveryKey, master_key)
            ).decode("ascii")
        except ValueError as exc:
            log.error("recovery key did not open: %s", exc)


def _require_signing_key(auth: AuthState) -> bytes:
    if auth.signing_key is None:
        raise problem(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "No signing key",
            "The install's signing key is not available in this process.",
        )
    return auth.signing_key
