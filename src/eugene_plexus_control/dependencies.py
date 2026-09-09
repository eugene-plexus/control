"""FastAPI dependencies for bearer auth.

Same shape as every other component: `require_authorized` accepts
operator OR any `service:*` audience; `require_operator` accepts
operator only. All rejection paths produce RFC 7807 Problem JSON so the
UI renders one error template across components.

Two differences from the other components, both because this one is the
trust root:

**There is no auth-disabled dev path.** Elsewhere a missing signing key
means "running standalone, let everything through". Here a missing
signing key means the install has not been initialized, and the answer
is 503 with a pointer at `POST /v1/auth/initialize` — not open access. A
trust root that waved requests through until it was configured would be
a trust root with a window in it.

**Reads accept a service token; mutations do not.** An agent needs
`GET /v1/control/status` to learn the current epoch, and a standby needs
`GET /v1/control/log` to replicate, so reads have to accept a service
audience. Everything that changes control state is operator-only,
declared on the route rather than on the router so the level is visible
next to the handler.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import security
from ._generated.models import Problem
from .auth_state import AuthState

_bearer_scheme = HTTPBearer(auto_error=False)


def problem(status_code: int, title: str, detail: str) -> HTTPException:
    slug = title.replace(" ", "-").lower()
    return HTTPException(
        status_code=status_code,
        detail=Problem(
            type=f"https://github.com/eugene-plexus/control#{slug}",
            title=title,
            status=status_code,
            detail=detail,
            component="control",
        ).model_dump(exclude_none=True),
    )


def _validate(
    request: Request,
    creds: HTTPAuthorizationCredentials | None,
    *,
    accept_operator: bool,
    accept_any_service: bool,
) -> security.TokenPayload:
    auth: AuthState = request.app.state.auth_state

    if auth.signing_key is None:
        # Two different situations, and telling them apart matters most
        # on the day it matters at all. A **standby** is normally in the
        # second one: it holds the signing key sealed and cannot open it
        # until someone supplies the passphrase, so an operator arriving
        # to promote it needs to be told "log in", not "run first-run
        # setup" — which would be advice to wipe the install.
        initialized = request.app.state.machine.state.identity.salt is not None
        if initialized:
            raise problem(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "Locked",
                "This control root is initialized but locked: it holds the install's "
                "signing key sealed and has not been given the passphrase. Call "
                "POST /v1/auth/login. (On a standby this is the normal state until "
                "someone logs in — it is not a reason to re-initialize anything.)",
            )
        raise problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Setup required",
            "This install has no passphrase yet. Call POST /v1/auth/initialize first. "
            "Until then the trust root has no key to verify anything against — it does "
            "not fall back to accepting everything.",
        )

    if creds is None or not creds.credentials:
        raise problem(
            status.HTTP_401_UNAUTHORIZED,
            "Missing token",
            "Provide a bearer token via the Authorization: Bearer header.",
        )

    if auth.is_revoked(creds.credentials):
        raise problem(
            status.HTTP_401_UNAUTHORIZED,
            "Session revoked",
            "This session was logged out. Log in again.",
        )

    try:
        return security.decode_token(
            token=creds.credentials,
            signing_key=auth.signing_key,
            accept_operator=accept_operator,
            accept_any_service=accept_any_service,
        )
    except Exception as exc:
        raise problem(
            status.HTTP_401_UNAUTHORIZED,
            "Invalid token",
            f"Bearer token rejected: {exc}",
        ) from exc


def require_authorized(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> security.TokenPayload:
    """Operator OR any service-audience token.

    The level for reads. An agent polling for the current epoch and a
    standby pulling the log both arrive with a service token, and
    refusing them would make replication an operator-supervised
    activity."""
    return _validate(request, creds, accept_operator=True, accept_any_service=True)


def require_operator(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> security.TokenPayload:
    """Operator-audience tokens only.

    The level for anything that mutates control state, mints a join
    token, revokes a node, or rotates a key. A compromised peer holding
    a service token must not be able to enroll a host or re-key the
    install."""
    return _validate(request, creds, accept_operator=True, accept_any_service=False)
