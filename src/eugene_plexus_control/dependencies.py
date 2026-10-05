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
* `require_operator_or_acting_node` — an operator session, or a member
  node acting for an operator who is acting on it. Only the routes an app
  install spends (its key and its sign-in registration), and each such
  route then confines the node to things named for itself.
* `require_member_actor` — `require_node_actor`, or a **Job Site** speaking
  for itself. The file helper's three routes take it, and nothing else:
  a site's token is refused by every other route here
  (`verify_bearer(..., sites=False)`, the default), so a route added later
  cannot open to one by forgetting (remote-nodes.md §3.2).

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

import re
from collections.abc import Collection
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import tokens
from ._generated.models import Problem
from .auth_state import AuthState

_bearer_scheme = HTTPBearer(auto_error=False)

READ_SERVICE_SUBS = frozenset({tokens.SUB_AGENT, tokens.SUB_GATEWAY})

ENTRY_HEADER = "X-Eugene-Plexus-Entry"
PUBLIC_NODES = "public-nodes"
"""Set by the container's entry point on a request that arrived through the
public node route (`public_nodes`, J3), and stripped by it from everything
else. It can only narrow what a request may do: a caller who adds it by hand
on the LAN gets the public route's rules, never more."""


def via_public_nodes(request: Request) -> bool:
    return request.headers.get(ENTRY_HEADER, "").strip().lower() == PUBLIC_NODES


def is_job_site(request: Request, node: str | None) -> bool:
    record = request.app.state.machine.state.nodes.get(node) if node else None
    return record is not None and tokens.GRANT_FILES in record.grants


def _refuse_job_site(request: Request, claims: tokens.Claims) -> None:
    if claims.is_service and is_job_site(request, claims.issuer_node):
        raise problem(
            status.HTTP_401_UNAUTHORIZED,
            "Invalid token",
            f"{claims.issuer_node} is a job site: its token opens only the file helper's "
            "routes and the trust bundle here.",
        )


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
    sites: bool = False,
) -> tokens.Claims:
    """Verify one bearer addressed to this root, or raise the 401 that says why.

    A Job Site's service token is refused unless the route says `sites=True`.
    """
    auth: AuthState = request.app.state.auth_state
    if auth.signing_key is None:
        raise _locked_or_uninitialized(request)
    claims = verify_with_view(request, token, classes=classes)
    if not sites:
        _refuse_job_site(request, claims)
    return claims


def verify_with_view(request: Request, token: str, *, classes: Collection[str]) -> tokens.Claims:
    """`tokens.verify` against the keys applied state names, needing no
    private key: a sealed root still knows who its members are."""
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
    return _actor(request, creds, sites=False)


def require_member_actor(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> tokens.Claims:
    """`require_node_actor`, or a Job Site speaking for itself (the file helper)."""
    return _actor(request, creds, sites=True)


def _actor(
    request: Request, creds: HTTPAuthorizationCredentials | None, *, sites: bool
) -> tokens.Claims:
    claims = verify_bearer(
        request, _bearer(request, creds), classes=(tokens.TYP_SERVICE,), sites=sites
    )
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


SUBJECT_TOKEN_HEADER = "X-Eugene-Plexus-Subject-Token"
"""Where a node acting for an operator puts the operator's token (RFC 8693's
"subject"); its own service token, in `Authorization`, is the actor."""


@dataclass(frozen=True)
class Operator:
    """Who an operator-level request speaks for, and through which machine.

    `node` is None for an operator session presented here directly. It names
    a member node when that node presented the operator's token for them, and
    then the route confines it to what `named_for_node` allows.
    """

    claims: tokens.Claims
    node: str | None = None

    def may_name(self, name: str | None) -> bool:
        return self.node is None or named_for_node(name, self.node)


def named_for_node(name: str | None, node: str) -> bool:
    """An app's key or sign-in owner on that node: `app:<id>@<node>`."""
    return bool(name) and re.fullmatch(rf"app:[^@\s]+@{re.escape(node)}", str(name)) is not None


def require_operator_or_acting_node(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> Operator:
    """An operator session; or a member node acting for an operator acting on it.

    **Why the second form exists (2026-10-01).** The console reaches another
    machine with a five-minute token addressed to that machine alone (D7), so
    an install the operator starts there cannot send that token on here --
    it is not addressed here, and forwarding a bearer past its audience is
    what D7 forbids. Installing an app needs two things only this root can
    do: its key, and its sign-in registration. So the machine presents its
    own `sub: agent` token as the actor, and the operator's token addressed
    to that same machine as the subject.

    The pair proves that this node is asking and that an operator is acting
    on it right now: the subject lives five minutes, and a sign-out revokes
    it by `sid`. A node's key alone opens nothing here, and the routes that
    take this let the node touch only keys and sign-in registrations named
    for itself.
    """
    subject = request.headers.get(SUBJECT_TOKEN_HEADER)
    if not subject:
        return Operator(require_operator(request, creds))
    actor = require_node_actor(request, creds)
    node = actor.issuer_node
    assert node is not None  # require_node_actor said so
    auth: AuthState = request.app.state.auth_state
    if auth.signing_key is None:
        raise _locked_or_uninitialized(request)
    bundle = request.app.state.trust.view_for(request.app.state.machine.state)
    try:
        claims = tokens.verify(
            subject.strip(),
            bundle=bundle,
            recipient=tokens.node_recipient(node),
            classes=(tokens.TYP_SESSION,),
        )
    except tokens.TokenError as exc:
        raise problem(
            status.HTTP_401_UNAUTHORIZED,
            "Not an operator acting on this machine",
            f"{node} sent an operator token this root will not take for it: {exc}",
        ) from exc
    return Operator(claims, node=node)


ActingOperator = Annotated[Operator, Depends(require_operator_or_acting_node)]
"""A route parameter that takes `require_operator_or_acting_node`."""


def require_operator_or_node(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> Operator:
    """`require_operator_or_acting_node`, or a member node on its own.

    For the one route a node may call with no operator present: replacing
    the return addresses of its own apps' sign-in clients (one HTTPS port,
    2026-10-05). A container moved onto one HTTPS port changes where its
    Workbench is opened, and the agent updates that at boot, when nobody is
    signed in. The node gets nothing new by it: it runs those apps and holds
    their client secrets, so every sign-in to them already passes through it.
    The route confines it to clients named `app:<id>@<node>`.

    The acting form (`SubjectToken`) needs no branch of its own here: a node
    acting for an operator is confined to the same clients as the node on its
    own, so the subject would add nothing (found by sabotage). A subject sent
    anyway is ignored.
    """
    try:
        return Operator(require_operator(request, creds))
    except HTTPException as refused:
        if refused.status_code != status.HTTP_401_UNAUTHORIZED:
            raise
        try:
            actor = require_node_actor(request, creds)
        except HTTPException:
            raise refused from None
    assert actor.issuer_node is not None  # require_node_actor said so
    return Operator(actor, node=actor.issuer_node)


OperatorOrNode = Annotated[Operator, Depends(require_operator_or_node)]
"""A route parameter that takes `require_operator_or_node`."""


def refuse_for_node(op: Operator, what: str) -> HTTPException:
    """The 403 for a node that reached past its own apps."""
    return problem(
        status.HTTP_403_FORBIDDEN,
        "Not this machine's to change",
        f"{op.node} is acting for you, so it may {what} only for its own apps, "
        f"named app:<id>@{op.node}. Do this from a console signed in to the control root.",
    )
