"""Current directory authority; never an owner or app credential.

Only the owner delegates these grants. They are independent of app/site
permission groups, so editing a group cannot manufacture an administrator.
Every request reads active root state, including session invalidation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import tokens
from ._generated.models import AdministrationGrants, AdministrationSession
from .dependencies import problem, verify_bearer
from .state_machine import ROLE_ACTIVE, StateMachine

log = logging.getLogger(__name__)
_bearer = HTTPBearer(auto_error=False)


def has_grants(person: dict[str, Any]) -> bool:
    grants = person.get("administration", {})
    return bool(
        grants.get("managePeople") or grants.get("manageGroups") or grants.get("membershipGroups")
    )


def active(request: Request) -> StateMachine:
    machine: StateMachine = request.app.state.machine
    if machine.role != ROLE_ACTIVE:
        raise problem(503, "Not the active root", "Directory administration needs the active root.")
    return machine


@dataclass(frozen=True)
class Administrator:
    claims: tokens.Claims
    session: AdministrationSession

    @property
    def owner(self) -> bool:
        return self.session.owner

    def require(self, permission: str) -> None:
        if not self.owner and not getattr(self.session.grants, permission, False):
            raise problem(403, "Permission required", "You are not allowed to make this change.")

    def ordinary_account(self, person: dict[str, Any]) -> None:
        if not self.owner and (person["id"] == self.claims.sub or has_grants(person)):
            raise problem(
                403,
                "Protected administrator",
                "Only the owner can change an administrator's account.",
            )

    def memberships(self, before: list[str] | None, after: list[str] | None) -> None:
        if self.owner:
            return
        if before is None or after is None:
            raise problem(
                403,
                "Owner required",
                "Only the owner can switch between individual and group access.",
            )
        scope = set(self.session.grants.model_dump()["membershipGroups"])
        if not scope or not (set(before) ^ set(after)).issubset(scope):
            raise problem(
                403,
                "Outside your groups",
                "Change membership only in your assigned groups; keep all other memberships.",
            )

    def audit(self, action: str, target: str) -> None:
        log.info("directory actor=%s action=%s target=%s", self.claims.sub, action, target)


async def require_administrator(
    request: Request,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Administrator:
    machine = active(request)
    if creds is None:
        raise problem(401, "Missing token", "Sign in to manage people and groups.")
    claims = verify_bearer(
        request, creds.credentials, classes=(tokens.TYP_SESSION, tokens.TYP_ADMINISTRATION)
    )
    if claims.typ == tokens.TYP_SESSION:
        if claims.sub != tokens.SUB_OPERATOR:
            raise problem(403, "Owner required", "This is not the owner's session.")
        return Administrator(
            claims,
            AdministrationSession(
                personId=tokens.SUB_OPERATOR,
                name=tokens.SUB_OPERATOR,
                owner=True,
                grants=AdministrationGrants.model_validate(
                    {
                        "managePeople": True,
                        "manageGroups": True,
                        # The owner is not restricted to a delegate's 64-group scope.
                        "membershipGroups": [],
                    }
                ),
            ),
        )
    from .external_sign_in import valid_session

    if claims.browser_session and not valid_session(
        machine.state, claims.browser_session, claims.sub
    ):
        raise problem(401, "Sign-in ended", "Your shared sign-in ended. Sign in again.")
    person = machine.state.people.get(claims.sub)
    if (
        person is None
        or person.get("disabled")
        or not has_grants(person)
        or claims.administration_version != person.get("administrationVersion", 0)
    ):
        raise problem(
            401,
            "Administration session ended",
            "Your account or administration access changed. Sign in again.",
        )
    return Administrator(
        claims,
        AdministrationSession(
            personId=person["id"],
            name=person["name"],
            owner=False,
            grants=AdministrationGrants.model_validate(person["administration"]),
        ),
    )


DirectoryAdministrator = Annotated[Administrator, Depends(require_administrator)]
