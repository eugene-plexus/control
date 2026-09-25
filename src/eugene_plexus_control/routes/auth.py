"""The trust root's auth surface: status, initialize, login, logout.

Moved here from the agent, which held it only because it used to be the
single process in the install. Two states, as before:

  * **Uninitialized** — no passphrase. Only `GET /v1/auth/status`,
    `POST /v1/auth/initialize` and `/healthz` work. Note what that is
    *not*: a mode that lets everything through. A trust root with an
    open window until it is configured is a trust root with an open
    window.
  * **Initialized** — login is open, everything else needs a token.

What initialization does that the agent's version did not: it mints
this root's **token key**, its **identity keypair** and the **recovery
recipient**, seals all three under the passphrase-derived key, and
snapshots them. That bundle is the replication set, and it is why a
standby can be promoted from a passphrase rather than from a key
somebody copied between hosts.

**This is the only place an operator session is minted** in an enrolled
install (2026-09-25). An agent's sign-in forwards here with its own
service token, and the session comes back addressed to that machine and
to this root. The console reaches every other machine by exchanging it
(`POST /v1/auth/token`), never by sending it on.

Login is rate-limited per source IP — five failures in sixty seconds
locks that source out for sixty. Per process rather than replicated: a
failed login on one host is not a fact about the install, and a standby
that inherited a lockout would be a denial of service with extra steps.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Request, status
from fastapi.security import HTTPBearer

from .. import config as config_module
from .. import keyring_store, peer, sealing, security, tokens
from .._generated.models import (
    AuthInitializeRequest,
    AuthLoginResponse,
    AuthStatus,
    TokenExchangeRequest,
    TokenExchangeResponse,
)
from ..applied import OP_REVOKE_SESSION
from ..auth_state import AuthState
from ..dependencies import (
    optional_node_actor,
    problem,
    require_node_actor,
    require_operator,
    verify_bearer,
)
from ..state_machine import AlreadyActive, NotActive, StateMachine
from ..trusted_host import require_trusted_host

log = logging.getLogger(__name__)

router = APIRouter(tags=["auth"])

_bearer_scheme = HTTPBearer(auto_error=False)

_RATE_LIMIT_WINDOW_SECONDS = 60
_RATE_LIMIT_MAX_FAILURES = 5

# Names the current signing-key generation. Not a secret: it is what a
# node reports back so a rotation can tell which hosts are stale.
_INITIAL_KEY_ID = "1"


MIN_PASSPHRASE_LENGTH = 12
"""The shortest passphrase first-run setup accepts, in characters.

The same number as the agent's and the UI's, deliberately: the wizard
posts one passphrase to both the agent and this root, and three
different minimums would be a form that passes on one screen and is
refused by the next. Applied where a passphrase is **chosen** and never
where one is presented -- an install that already has a shorter one
still signs in and still unlocks, because there is no way to change it
and refusing it would be a lockout."""


# How long `GET /v1/auth/status` waits for the keyring probe before
# answering without it. A locked Secret Service can block on a prompt
# nobody will answer; a status endpoint must not hang with it.
KEYRING_PROBE_BUDGET_SECONDS = 3.0


def install_id_of(machine: StateMachine) -> str:
    """The keyring scope for this install — a fingerprint of its master-key
    salt. Callers run past the `salt is None` checks; the empty fallback
    only keeps a corrupt state from raising inside a keyring call."""
    salt = machine.state.identity.salt
    return keyring_store.install_id_for(salt) if salt else ""


@router.get("/v1/auth/status", response_model=AuthStatus)
async def auth_status(request: Request) -> AuthStatus:
    """Has this install been through first-run setup, is it unlocked, and
    can this host's keyring keep it that way?

    Unauthenticated by necessity — it is what the UI asks *before* it
    has a token, on every page load, to tell a fresh install from a
    logged-out one. `unlocked` is the sealed root said in a word instead
    of a 503, for the Issues list; `keyringAvailable` is what the wizard
    reads to default `securityMode`. No secrets, no rate limit. The probe
    runs once per process, in a thread, with a deadline — past it the
    field is absent, not False.
    """
    machine: StateMachine = request.app.state.machine
    auth: AuthState = request.app.state.auth_state
    try:
        available: bool | None = await asyncio.wait_for(
            asyncio.to_thread(keyring_store.probe_sync), timeout=KEYRING_PROBE_BUDGET_SECONDS
        )
    except TimeoutError:
        log.warning(
            "the OS keyring probe did not finish within %.0fs; reporting keyringAvailable "
            "as unknown",
            KEYRING_PROBE_BUDGET_SECONDS,
        )
        available = None
    return AuthStatus(
        initialized=machine.state.identity.salt is not None,
        unlocked=auth.master_key is not None,
        keyringAvailable=available,
    )


# The two unauthenticated doors a browser uses, and the only routes here
# that check `Host`: first-run is first-come, so a page that rebinds its
# own name to this machine could otherwise claim the install. See
# `trusted_host` for the rule and for why the machine routes are exempt.
_BROWSER_DOOR = [Depends(require_trusted_host)]


@router.post("/v1/auth/initialize", status_code=204, dependencies=_BROWSER_DOOR)
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
    # Before anything is minted: a refusal here leaves a fresh install,
    # so the next try is first-run again rather than "already initialized".
    if len(passphrase) < MIN_PASSPHRASE_LENGTH:
        raise problem(
            status.HTTP_400_BAD_REQUEST,
            "Passphrase too short",
            f"Choose a passphrase of at least {MIN_PASSPHRASE_LENGTH} characters. It locks "
            "every secret this install keeps and it can never be reset, so a short one is "
            "the easiest part of the install to guess. A few ordinary words in a row works.",
        )

    salt = security.generate_master_key_salt()
    master_key = security.derive_master_key(passphrase, salt)
    verifier = security.hash_passphrase(passphrase)

    signing_key = security.generate_signing_key()
    root_public = tokens.public_b64(tokens.load_private(signing_key))
    control_identity = sealing.generate_control_identity()
    recovery = sealing.generate_sealing_keypair()

    try:
        machine.seed_identity(
            {
                "salt": base64.b64encode(salt).decode("ascii"),
                "passphraseVerifier": verifier,
                "sealedSigningKey": security.seal_b64(signing_key, master_key),
                "rootTokenPublicKey": root_public,
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
    request.app.state.trust.publish(request.app)
    log.info(
        "install initialized; epoch %d, token key generation %s",
        machine.state.epoch,
        _INITIAL_KEY_ID,
    )


@router.post("/v1/auth/login", response_model=AuthLoginResponse, dependencies=_BROWSER_DOOR)
async def login(request: Request, body: AuthInitializeRequest) -> AuthLoginResponse:
    """Verify the passphrase, unseal the keys, issue a session token.

    Unsealing on login is what makes a restart not a sign-out: the token
    key is in applied state sealed under the master key, so the process
    recovers the existing key rather than minting a new one.

    **Who asked decides the session's `aud`.** An agent forwarding a
    sign-in presents its own `agent` service token as `Authorization`,
    and the session is addressed to that machine and to this root. A
    login with no such token gets this root alone. The header is read
    after unsealing, because a sealed root cannot verify anything, and a
    header that does not verify only narrows the session.
    """
    machine: StateMachine = request.app.state.machine
    auth: AuthState = request.app.state.auth_state
    identity = machine.state.identity
    # **The browser's address, not the proxy's.** Every login here
    # arrives through a node agent's `/api/proxy/control/...`, so the
    # peer is loopback for every caller there has ever been: one bucket
    # for the whole install, and until R1.2 a bucket the caller chose.
    # `peer.py` says why this header can be believed and
    # `X-Forwarded-For` could not.
    remote = (
        peer.peer_of(
            request.client.host if request.client else None,
            request.headers.get(peer.PEER_HEADER),
        )
        or "unknown"
    )

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
    was_locked = auth.signing_key is None or auth.control_private_key is None
    unseal_into(auth, machine, passphrase)
    if was_locked:
        request.app.state.trust.publish(request.app)

    actor = optional_node_actor(request, request.headers.get("authorization"))
    audience = [tokens.RECIPIENT_CONTROL]
    if actor is not None and actor.issuer_node is not None:
        audience = [tokens.node_recipient(actor.issuer_node), tokens.RECIPIENT_CONTROL]
    token, expires = _signer(auth).mint(
        typ=tokens.TYP_SESSION,
        sub=tokens.SUB_OPERATOR,
        aud=audience,
        ttl_seconds=tokens.SESSION_TTL_SECONDS,
    )

    # Persist for auto-unlock if the operator asked for it. Here rather
    # than only on the config flip, because the flip can happen while
    # the root is locked (there is nothing to store yet) - both paths
    # converge on "the next restart auto-recovers".
    if config_module.effective(machine.state.config)["securityMode"] == "os_keyring":
        if auth.master_key is not None and keyring_store.set_master_key(
            auth.master_key, install_id_of(machine)
        ):
            log.info("master key persisted to the OS keyring for auto-unlock")
        else:
            log.warning(
                "securityMode is os_keyring but the master key could not be stored; "
                "this root will ask for the passphrase again after a restart"
            )

    log.info(
        "operator login from %s%s",
        remote,
        f" through node {actor.issuer_node}" if actor is not None else "",
    )
    return AuthLoginResponse(
        sessionToken=token,
        expiresAt=datetime.fromtimestamp(expires, tz=UTC),
        operatorName="operator",
    )


@router.delete("/v1/auth/sessions/current", status_code=204)
async def logout(request: Request) -> None:
    """Sign the presented session out, everywhere in the install.

    Verified through the same dependency as any protected route, so a
    caller can only sign out the session it holds. A replicated
    `revokeSession` entry, then a new trust bundle pushed to every
    machine: the session and everything exchanged from it (by `sid`)
    stop verifying wherever they were. An exchanged token presented here
    signs out the session it came from.
    """
    claims = require_operator(request, await _bearer_scheme(request))
    machine: StateMachine = request.app.state.machine
    jti = claims.sid or claims.jti
    try:
        machine.append(
            OP_REVOKE_SESSION,
            {
                "jti": jti,
                "exp": claims.exp,
                # Stamped by this writer, once, so replay prunes the same entries.
                "prunedBefore": int(time.time()) - tokens.LEEWAY_SECONDS,
            },
        )
    except NotActive as exc:
        raise problem(
            status.HTTP_409_CONFLICT,
            "Not the active root",
            "Sign-out is a write, and this control root is a standby.",
        ) from exc
    request.app.state.trust.publish(request.app)
    log.info("session signed out")


@router.post("/v1/auth/token", response_model=TokenExchangeResponse)
async def exchange_token(request: Request, body: TokenExchangeRequest) -> TokenExchangeResponse:
    """RFC 8693: a session for a 5-minute token addressed to one other machine.

    How the console acts on another machine without handing it the
    session. The actor is the console's own agent, signing for itself;
    the subject is the operator's session, which must be addressed to
    that same machine: a session cannot be exchanged by a machine it was
    not issued to, and an exchanged token cannot be exchanged again,
    since it is not addressed to this root.
    """
    actor = require_node_actor(request, await _bearer_scheme(request))
    machine: StateMachine = request.app.state.machine
    auth: AuthState = request.app.state.auth_state
    subject = verify_bearer(request, body.subjectToken, classes=(tokens.TYP_SESSION,))
    console = tokens.node_recipient(actor.issuer_node or "")
    if subject.act is not None or console not in subject.aud:
        raise problem(
            status.HTTP_401_UNAUTHORIZED,
            "Not this machine's session",
            f"The session is addressed to {list(subject.aud)}; {console} cannot exchange it.",
        )
    target = tokens.node_of(body.audience)
    if target is None or target not in machine.state.nodes:
        raise problem(
            status.HTTP_400_BAD_REQUEST,
            "No such node",
            f"{body.audience!r} names no enrolled node, so there is nothing to address.",
        )
    now = int(time.time())
    ttl = min(tokens.EXCHANGED_TTL_SECONDS, subject.exp - now)
    if ttl <= 0:
        raise problem(status.HTTP_401_UNAUTHORIZED, "Session expired", "Sign in again.")
    token, expires = _signer(auth).mint(
        typ=tokens.TYP_SESSION,
        sub=tokens.SUB_OPERATOR,
        aud=[body.audience],
        ttl_seconds=ttl,
        now=now,
        extra={"act": {"sub": console}, "sid": subject.jti},
    )
    return TokenExchangeResponse(
        accessToken=token, expiresAt=datetime.fromtimestamp(expires, tz=UTC)
    )


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
    unseal_with_master_key(auth, machine, master_key)


def unseal_with_master_key(auth: AuthState, machine: StateMachine, master_key: bytes) -> None:
    """The half of `unseal_into` after the key is derived.

    Split out for OS keyring auto-unlock, which holds the derived master
    key and never sees the passphrase. One place opens the sealed values
    either way, so the "a sealed value that will not open is reported
    loudly" rule cannot drift between the two entry points.
    """
    import base64 as _b64

    identity = machine.state.identity
    auth.set_master_key(master_key)

    if identity.sealedSigningKey is not None:
        try:
            auth.set_signing_key(security.open_b64(identity.sealedSigningKey, master_key))
        except ValueError as exc:
            raise problem(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "Sealed token key unreadable",
                f"This root's token key did not open with this passphrase ({exc}). Every "
                "session and client key in the install is signed with it, so this host "
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


def _signer(auth: AuthState) -> tokens.Signer:
    signer = auth.signer()
    if signer is None:
        raise problem(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "No token key",
            "This root's token key is not available in this process.",
        )
    return signer
