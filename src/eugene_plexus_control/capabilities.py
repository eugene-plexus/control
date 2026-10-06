"""Named capabilities, not "the operator" (J15, remote-nodes.md §3.3).

Troy, 2026-10-05: *"I wouldn't count on the owner getting to see everything
in the future."* An MSP will want its own tools to join and remove customer
machines, and a paid service automated user tools, without either one reading
anybody's files. So a surface added for Job Sites asks for the capability it
needs, never whether the caller is the operator, even while one party holds
them all.

**The novice default:** the passphrase session holds every administrative
capability and never data access. Nothing assigns capabilities yet; when
something does, it changes `held_by` and no route.

Data access is not in this list on purpose. Using a Job Site's folders is the
site owner's grant, which the site keeps and enforces itself (J8), and no
administrative capability reaches it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, Request, status
from fastapi.security import HTTPAuthorizationCredentials

from . import tokens
from .dependencies import _bearer, _bearer_scheme, problem, verify_bearer

MEMBERSHIP = "membership"
"""Invite machines (Job Sites included), remove them, and see that they exist
and whether they are online. Not their folders, their contents or who holds
grants (remote-nodes.md §3.3)."""

INSTALL_MODE = "install-mode"
"""Switch the install between production and dev mode (J13, J18)."""

NODE_FILES = "node-files"
"""Administer ordinary nodes' file support from the console: turn it on,
register folders, and give people folders there, where the root's grant is
final (J6d). On a Job Site it reaches nothing but, in dev mode, the owner's
own grant (J13b). It grants no file access by itself (J15)."""

ADMINISTRATIVE = frozenset({MEMBERSHIP, INSTALL_MODE, NODE_FILES})


def held_by(claims: tokens.Claims) -> frozenset[str]:
    """What a verified caller may do. Today: the operator session, everything
    administrative; anything else, nothing."""
    if claims.typ == tokens.TYP_SESSION and claims.sub == tokens.SUB_OPERATOR:
        return ADMINISTRATIVE
    return frozenset()


def require(capability: str) -> Callable[..., tokens.Claims]:
    """A dependency: a session that holds `capability`, or 401/403."""

    def dependency(
        request: Request,
        creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
    ) -> tokens.Claims:
        claims = verify_bearer(request, _bearer(request, creds), classes=(tokens.TYP_SESSION,))
        if capability not in held_by(claims):
            raise problem(
                status.HTTP_403_FORBIDDEN,
                "Not permitted",
                f"This needs the {capability!r} capability, which this sign-in does not hold.",
            )
        return claims

    dependency.__name__ = f"require_{capability.replace('-', '_')}"
    return dependency
