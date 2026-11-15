"""Named directory sign-in, separate from root unlock and app sign-in."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Request, Response

from .. import peer, security, tokens
from .._generated.models import (
    AccessAudit,
    AdministrationGrants,
    AdministrationLoginRequest,
    AdministrationLoginResponse,
    AdministrationSession,
    OperationsPower,
    OperationsToken,
    OperationsTokenRequest,
)
from ..administration import (
    DirectoryAdministrator,
    active,
    has_grants,
    operations_powers,
    operations_reach,
)
from ..applied import OP_REVOKE_SESSION
from ..dependencies import problem, require_operator
from ..trusted_host import require_trusted_host

router = APIRouter(tags=["people"])


@router.get("/v1/auth/audit", response_model=AccessAudit, dependencies=[Depends(require_operator)])
async def audit(request: Request, response: Response) -> AccessAudit:
    response.headers["Cache-Control"] = "no-store"
    return AccessAudit.model_validate(
        {"events": list(reversed(active(request).state.access_audit))}
    )


@router.post(
    "/v1/auth/administration/login",
    response_model=AdministrationLoginResponse,
    dependencies=[Depends(require_trusted_host)],
)
async def login(
    request: Request, body: AdministrationLoginRequest, response: Response
) -> AdministrationLoginResponse:
    machine = active(request)
    auth = request.app.state.auth_state
    if auth.signing_key is None:
        raise problem(
            503, "Locked", "The owner must unlock Eugene before named administrators can sign in."
        )
    remote = (
        peer.peer_of(
            request.client.host if request.client else None, request.headers.get(peer.PEER_HEADER)
        )
        or "unknown"
    )
    if auth.is_login_rate_limited(remote, window_seconds=60, max_in_window=5):
        raise problem(429, "Rate limited", "Wait 60 seconds before trying to sign in again.")
    person = next(
        (
            p
            for p in machine.state.people.values()
            if p["name"].casefold() == body.name.strip().casefold()
        ),
        None,
    )
    # Always perform one password verification, including unknown names.
    verifier = person["passwordVerifier"] if person else machine.state.identity.passphraseVerifier
    assert verifier is not None  # an unlocked root is initialized
    valid_password = security.verify_passphrase(body.password, verifier)
    if not valid_password or person is None or person.get("disabled") or not has_grants(person):
        auth.record_login_failure(remote, window_seconds=60, max_in_window=5)
        raise problem(
            401,
            "Sign-in refused",
            "Check your name and password, and that the owner has granted administration access.",
        )
    auth.clear_login_failures(remote)
    response.headers["Cache-Control"] = "no-store"
    return issue_session(request, person)


def issue_session(
    request: Request, person: dict[str, Any], *, browser_session: str | None = None
) -> AdministrationLoginResponse:
    token, expires = request.app.state.auth_state.signer().mint(
        typ=tokens.TYP_ADMINISTRATION,
        sub=person["id"],
        aud=[tokens.RECIPIENT_CONTROL],
        ttl_seconds=tokens.ADMINISTRATION_TTL_SECONDS,
        extra={
            "administration_version": person.get("administrationVersion", 0),
            **({"eugene_browser_session": browser_session} if browser_session else {}),
        },
    )
    return AdministrationLoginResponse(
        sessionToken=token,
        expiresAt=datetime.fromtimestamp(expires, UTC),
        session=AdministrationSession(
            personId=person["id"],
            name=person["name"],
            owner=False,
            grants=AdministrationGrants.model_validate(person["administration"]),
        ),
    )


@router.get("/v1/auth/administration/session", response_model=AdministrationSession)
async def current(admin: DirectoryAdministrator, response: Response) -> AdministrationSession:
    response.headers["Cache-Control"] = "no-store"
    return admin.session


@router.delete("/v1/auth/administration/session", status_code=204)
async def logout(request: Request, admin: DirectoryAdministrator) -> Response:
    if admin.owner:
        raise problem(403, "Owner session", "Use the owner's console sign-out for this session.")
    active(request).append(
        OP_REVOKE_SESSION,
        {
            "jti": admin.claims.sid or admin.claims.jti,
            "exp": admin.claims.exp,
            "prunedBefore": int(time.time()) - tokens.LEEWAY_SECONDS,
        },
        actor=admin.claims.sub,
    )
    request.app.state.trust.publish(request.app)
    admin.audit("sign-out", admin.claims.sub)
    return Response(status_code=204)


@router.post("/v1/auth/operations-token", response_model=OperationsToken)
async def operations_token(
    request: Request,
    body: OperationsTokenRequest,
    admin: DirectoryAdministrator,
    response: Response,
) -> OperationsToken:
    """Row 39: a person's operations powers on one machine, for 10 minutes.

    Checked at every exchange, from the root's current state: the session
    (version, enabled, not signed out: `require_administrator`), the
    powers, and the machine scope. The token is addressed to that one
    machine and carries the session it came from, so signing out ends it
    on every machine that can fetch the trust bundle."""
    if admin.owner:
        raise problem(
            403,
            "Owner session",
            "The owner's console reaches every machine already. This exchange is for people "
            "the owner gave an operations power.",
        )
    machine = active(request)
    person = machine.state.people.get(admin.claims.sub)
    powers = operations_powers(person) if person is not None else []
    if person is None or not powers:
        raise problem(
            403,
            "No operations power",
            "The owner has not given you an operations power (Models, or Updates and "
            "restarts). The owner can, on Eugene's People page.",
        )
    name = body.machine
    if not operations_reach(person, name):
        raise problem(
            403,
            "Outside your machines",
            f"Your operations powers do not reach the machine {name!r}. The owner chooses your "
            "machines on Eugene's People page.",
        )
    if name not in machine.state.nodes:
        raise problem(400, "No such machine", f"{name!r} is not a machine in this install.")
    auth = request.app.state.auth_state
    if auth.signing_key is None:
        raise problem(503, "Locked", "The owner must unlock Eugene first.")
    now = int(time.time())
    ttl = min(tokens.OPERATIONS_TTL_SECONDS, admin.claims.exp - now)
    if ttl <= 0:
        raise problem(401, "Administration session ended", "Sign in again.")
    token, expires = auth.signer().mint(
        typ=tokens.TYP_OPERATIONS,
        sub=person["id"],
        aud=[tokens.node_recipient(name)],
        ttl_seconds=ttl,
        now=now,
        extra={
            "ops": powers,
            "name": person.get("displayName") or person["name"],
            "sid": admin.claims.jti,
            "administration_version": person.get("administrationVersion", 0),
        },
    )
    admin.audit("operations-token", name)
    response.headers["Cache-Control"] = "no-store"
    return OperationsToken(
        token=token,
        expiresAt=datetime.fromtimestamp(expires, UTC),
        machine=name,
        powers=[OperationsPower(p) for p in powers],
    )


def revoke_operations(request: Request, person: dict[str, Any], *, actor: str) -> None:
    """End every operations token `person` holds now (row 39).

    Called before a change that bumps their administration version or
    removes them: their grants, enabled state, password, or the account.
    One sign-out entry names the person at their current version; it
    outlives the longest token minted before it."""
    if not operations_powers(person):
        return
    now = int(time.time())
    active(request).append(
        OP_REVOKE_SESSION,
        {
            "jti": tokens.operations_revocation(
                person["id"], person.get("administrationVersion", 0)
            ),
            "exp": now + tokens.OPERATIONS_TTL_SECONDS + tokens.LEEWAY_SECONDS,
            "prunedBefore": now - tokens.LEEWAY_SECONDS,
        },
        actor=actor,
    )
    request.app.state.trust.publish(request.app)
