"""FastAPI dependencies for bearer auth, against the trust bundle.

Every bearer is checked by `tokens.verify` against the keys this root's
applied state names (`trust.view`), addressed to `control`. The route
levels, per `specs/docs/design/per-node-token-keys.md` D5:

* `require_operator` — an operator session. Everything that changes
  control state, and the replication surface.
* `require_authorized` — a session, or a service token with `sub`
  `agent` or `gateway` from a member node's key. The reads a node or the
  gateway legitimately needs: the node list, topology, the epoch.
* `require_key_policy` — the same two service kinds, for the client-key
  policy the gateway and agents enforce.
* `require_node_actor` — a service token with `sub: agent` from a member
  node, and nothing else: the credential an agent presents when it asks
  on its own behalf (token exchange, a forwarded sign-in).

**There is no auth-disabled dev path.** A missing token key means the
install has not been initialized, or is sealed, and the answer is 503.
A trust root that waved requests through until it was configured would
be a trust root with a window in it.

**Until 2026-09-25 "any `service:*`" opened every read here**, and this
root handed each node a year-long `service:control` token on every poll,
which opened the replication snapshot and its passphrase verifier. A
service token is addressed to one machine now, and none is addressed
here except by a node speaking for itself.
"""

from __future__ import annotations

from collections.abc import Collection

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import tokens
from ._generated.models import Problem
from .auth_state import AuthState

_bearer_scheme = HTTPBearer(auto_error=False)

READ_SERVICE_SUBS = frozenset({tokens.SUB_AGENT, tokens.SUB_GATEWAY})


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


def _locked_or_uninitialized(request: Request) -> HTTPException:
    # Two different situations, and telling them apart matters most on
    # the day it matters at all. A **standby** is normally in the second
    # one: it holds the token key sealed and cannot open it until someone
    # supplies the passphrase, so an operator arriving to promote it
    # needs to be told "log in", not "run first-run setup" — which would
    # be advice to wipe the install.
    initialized = request.app.state.machine.state.identity.salt is not None
    if initialized:
        return problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Locked",
            "This control root is initialized but locked: it holds its token key sealed "
            "and has not been given the passphrase. Call POST /v1/auth/login. (On a standby "
            "this is the normal state until someone logs in — it is not a reason to "
            "re-initialize anything.)",
        )
    return problem(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "Setup required",
        "This install has no passphrase yet. Call POST /v1/auth/initialize first. Until "
        "then the trust root has no key to verify anything against — it does not fall "
        "back to accepting everything.",
    )


def verify_bearer(
    request: Request,
    token: str,
    *,
    classes: Collection[str],
) -> tokens.Claims:
    """Verify one bearer addressed to this root, or raise the 401 that says why."""
    auth: AuthState = request.app.state.auth_state
    if auth.signing_key is None:
        raise _locked_or_uninitialized(request)
    bundle = request.app.state.trust.view_for(request.app.state.machine.state)
    try:
        return tokens.verify(
            token, bundle=bundle, recipient=tokens.RECIPIENT_CONTROL, classes=classes
        )
    except tokens.TokenError as exc:
        raise problem(
            status.HTTP_401_UNAUTHORIZED, "Invalid token", f"Bearer token rejected: {exc}"
        ) from exc


def _bearer(request: Request, creds: HTTPAuthorizationCredentials | None) -> str:
    # Locked or uninitialized first: "log in" and "run setup" are the
    # answers, and a missing token would otherwise hide them behind a 401.
    if request.app.state.auth_state.signing_key is None:
        raise _locked_or_uninitialized(request)
    if creds is None or not creds.credentials:
        raise problem(
            status.HTTP_401_UNAUTHORIZED,
            "Missing token",
            "Provide a bearer token via the Authorization: Bearer header.",
        )
    return creds.credentials


def _session_or_services(
    request: Request, creds: HTTPAuthorizationCredentials | None, subs: frozenset[str]
) -> tokens.Claims:
    """The route's own list of service kinds, on top of the bundle's grants.

    **Redundant today, and kept on purpose** (sabotage pass, 2026-09-25):
    `tokens.verify` already refuses every `sub` but `agent` and a granted
    `gateway` on a token that leaves its machine, so no token that exists
    can tell this check from its absence. It stays as the route's policy,
    stated where the route is, so that widening a grant later cannot
    silently widen what this root accepts.
    """
    claims = verify_bearer(
        request, _bearer(request, creds), classes=(tokens.TYP_SESSION, tokens.TYP_SERVICE)
    )
    if claims.is_service and (claims.issuer_node is None or claims.sub not in subs):
        raise problem(
            status.HTTP_401_UNAUTHORIZED,
            "Invalid token",
            f"A {claims.sub!r} service token from {claims.iss!r} does not open this route.",
        )
    return claims


def require_authorized(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> tokens.Claims:
    """A session, or an `agent` or `gateway` service token from a member node.

    The level for reads an agent or the gateway needs: the node list,
    topology, the current epoch."""
    return _session_or_services(request, creds, READ_SERVICE_SUBS)


def require_operator(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> tokens.Claims:
    """An operator session only.

    The level for anything that mutates control state, mints a join
    token, revokes a node or rotates a key, and for the replication
    surface, which carries the salt and the passphrase verifier."""
    return verify_bearer(request, _bearer(request, creds), classes=(tokens.TYP_SESSION,))


require_replica = require_operator
"""The replication surface takes a session and nothing else (2026-09-25).

A standby that follows unattended needs a credential of its own, and no
production path has ever given it one; see the design's §5."""


def require_key_policy(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> tokens.Claims:
    return _session_or_services(request, creds, READ_SERVICE_SUBS)


def require_node_actor(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> tokens.Claims:
    """A member node speaking for itself: `sub: agent`, signed by its own key."""
    claims = verify_bearer(request, _bearer(request, creds), classes=(tokens.TYP_SERVICE,))
    if claims.sub != tokens.SUB_AGENT or claims.issuer_node is None:
        raise problem(
            status.HTTP_401_UNAUTHORIZED,
            "Invalid token",
            "Only an agent's own service token, signed by its node's key, is accepted here.",
        )
    return claims


def optional_node_actor(request: Request, authorization: str | None) -> tokens.Claims | None:
    """The asking node, when an `Authorization` header says so and verifies.

    For the one route that works with or without it: a sign-in forwarded
    by an agent is addressed to that machine, one made directly is not.
    A header that does not verify is ignored rather than refused, so the
    worst a caller can do is get a session addressed to this root alone.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    try:
        claims = verify_bearer(request, authorization[7:].strip(), classes=(tokens.TYP_SERVICE,))
    except HTTPException:
        return None
    if claims.sub != tokens.SUB_AGENT or claims.issuer_node is None:
        return None
    return claims
