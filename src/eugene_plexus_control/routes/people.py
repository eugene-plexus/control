"""People, and the apps that sign in with Eugene (C2). Operator only.

Design: `specs/docs/design/sign-in-with-eugene.md` §2-§4. A person is an
account the owner adds so someone can sign in to the apps they are given;
an app that signs in with Eugene is an OpenID Connect client. Both live in
the replicated log, so a standby keeps them.

Nothing here answers with a password, a verifier or a client secret, except
a new client's secret, once, in the answer that made it.

A password's length is the contract's rule (`minLength: 12` on both
requests), so a short one is a 422 before any handler here runs.
"""

from __future__ import annotations

import secrets
import uuid
from typing import Any
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from .. import node_helpers, oidc, security
from .._generated.models import (
    OidcClient,
    OidcClientCreated,
    OidcClientCreateRequest,
    OidcClientList,
    Person,
    PersonCreateRequest,
    PersonList,
    PersonPasswordRequest,
    PersonUpdateRequest,
)
from ..applied import (
    OP_DELETE_OIDC_CLIENT,
    OP_DELETE_PERSON,
    OP_PUT_OIDC_CLIENT,
    OP_PUT_PERSON,
    OP_SET_PERSON_PASSWORD,
    OPERATOR_NAME,
)
from ..dependencies import (
    ActingOperator,
    problem,
    refuse_for_node,
    require_operator,
)
from ..state_machine import StateMachine

router = APIRouter(tags=["people"], dependencies=[Depends(require_operator)])
# Registering and removing an app's sign-in is also what installing an app on
# another machine does, so these two take a node acting for the operator as
# well, confined to that node's own apps (`require_operator_or_acting_node`).
app_clients = APIRouter(tags=["people"])


def _active(request: Request) -> StateMachine:
    machine: StateMachine = request.app.state.machine
    if not machine.is_active:
        raise problem(503, "Not the active root", "People are managed at the active control root.")
    return machine


def _view(record: dict[str, Any]) -> Person:
    return Person.model_validate({k: v for k, v in record.items() if k != "passwordVerifier"})


def _person(machine: StateMachine, person_id: str) -> dict[str, Any]:
    record = machine.state.people.get(person_id)
    if record is None:
        raise problem(404, "No such person", "Nobody with that id is on this install.")
    return record


def _check_name(machine: StateMachine, name: str, *, except_id: str | None = None) -> str:
    name = name.strip()
    if not name or "@" in name:
        raise problem(
            422, "Invalid name", "A name is what a person signs in with: no @, not blank."
        )
    if name.casefold() == OPERATOR_NAME:
        raise problem(
            409, "That name is the owner's", f"{OPERATOR_NAME!r} signs in with the passphrase."
        )
    if any(
        p["name"].casefold() == name.casefold() and p["id"] != except_id
        for p in machine.state.people.values()
    ):
        raise problem(409, "That name is taken", f"Someone on this install is already {name!r}.")
    return name


def _check_email(
    machine: StateMachine, email: str | None, *, except_id: str | None = None
) -> str | None:
    """An address, or None. Unique compared case-folded: an app that knows
    people by it (Open WebUI) refuses a second person with the same one."""
    email = (email or "").strip()
    if not email:
        return None
    if any(
        (p.get("email") or "").casefold() == email.casefold() and p["id"] != except_id
        for p in machine.state.people.values()
    ):
        raise problem(409, "That email is taken", f"Someone on this install already has {email!r}.")
    return email


def _check_apps(machine: StateMachine, apps: list[str] | None) -> list[str] | None:
    if apps is None:
        return None
    unknown = [a for a in apps if a not in machine.state.oidc_clients]
    if unknown:
        raise problem(
            422, "Unknown apps", f"No app that signs in with Eugene has the id {unknown}."
        )
    return list(dict.fromkeys(apps))


@router.get("/v1/people", response_model=PersonList, response_model_exclude_none=True)
async def list_people(request: Request) -> PersonList:
    machine = _active(request)
    people = sorted(machine.state.people.values(), key=lambda p: p["name"].casefold())
    return PersonList(people=[_view(p) for p in people], operatorName=OPERATOR_NAME)


@router.post("/v1/people", response_model=Person, status_code=201, response_model_exclude_none=True)
async def create_person(request: Request, body: PersonCreateRequest) -> Person:
    machine = _active(request)
    name = _check_name(machine, body.name)
    when = oidc.utcnow_iso()
    record: dict[str, Any] = {
        "id": uuid.uuid4().hex,
        "name": name,
        "passwordVerifier": security.hash_passphrase(body.password),
        "apps": _check_apps(machine, body.apps),
        "helperGrants": node_helpers.check_grants(
            machine.state, [g.model_dump(mode="json") for g in body.helperGrants or []]
        ),
        "disabled": False,
        "createdAt": when,
        "passwordChangedAt": when,
    }
    if body.displayName and body.displayName.strip():
        record["displayName"] = body.displayName.strip()
    email = _check_email(machine, body.email)
    if email:
        record["email"] = email
    machine.append(OP_PUT_PERSON, {"person": record})
    return _view(machine.state.people[record["id"]])


@router.patch("/v1/people/{person_id}", response_model=Person, response_model_exclude_none=True)
async def update_person(request: Request, person_id: str, body: PersonUpdateRequest) -> Person:
    machine = _active(request)
    record = dict(_person(machine, person_id))
    changes = body.model_dump(exclude_unset=True)
    if "displayName" in changes:
        display = (changes["displayName"] or "").strip()
        if display:
            record["displayName"] = display
        else:
            record.pop("displayName", None)
    if "email" in changes:
        email = _check_email(machine, changes["email"], except_id=person_id)
        if email:
            record["email"] = email
        else:
            record.pop("email", None)
    if "apps" in changes:
        record["apps"] = _check_apps(machine, changes["apps"])
    if "helperGrants" in changes:
        record["helperGrants"] = node_helpers.check_grants(
            machine.state, changes["helperGrants"] or []
        )
    if "disabled" in changes and changes["disabled"] is not None:
        record["disabled"] = bool(changes["disabled"])
    machine.append(OP_PUT_PERSON, {"person": record})
    return _view(machine.state.people[person_id])


@router.delete("/v1/people/{person_id}", status_code=204)
async def delete_person(request: Request, person_id: str) -> Response:
    machine = _active(request)
    _person(machine, person_id)
    machine.append(OP_DELETE_PERSON, {"id": person_id})
    return Response(status_code=204)


@router.put("/v1/people/{person_id}/password", status_code=204)
async def set_password(request: Request, person_id: str, body: PersonPasswordRequest) -> Response:
    machine = _active(request)
    _person(machine, person_id)
    machine.append(
        OP_SET_PERSON_PASSWORD,
        {
            "id": person_id,
            "passwordVerifier": security.hash_passphrase(body.password),
            "passwordChangedAt": oidc.utcnow_iso(),
        },
    )
    return Response(status_code=204)


# --------------------------------------------------------------------------- #
# the apps that sign in with Eugene
# --------------------------------------------------------------------------- #


def _client_view(record: dict[str, Any]) -> OidcClient:
    return OidcClient.model_validate({k: v for k, v in record.items() if k != "secretVerifier"})


def _check_redirects(uris: list[str]) -> list[str]:
    """Absolute, no fragment (RFC 6749 §3.1.2), matched exactly later.
    Plain http is allowed: a home network has no certificates."""
    checked = []
    for uri in uris:
        parts = urlsplit(uri.strip())
        if parts.scheme not in ("http", "https") or not parts.netloc or parts.fragment:
            raise problem(
                422,
                "Invalid redirect URI",
                f"{uri!r} must be an absolute http or https address with no #fragment.",
            )
        checked.append(uri.strip())
    return list(dict.fromkeys(checked))


@router.get("/v1/oidc/clients", response_model=OidcClientList, response_model_exclude_none=True)
async def list_clients(request: Request) -> OidcClientList:
    machine = _active(request)
    clients = sorted(machine.state.oidc_clients.values(), key=lambda c: c["name"].casefold())
    issuer = machine.state.config.get("oidcIssuer")
    return OidcClientList(
        clients=[_client_view(c) for c in clients],
        issuer=issuer if isinstance(issuer, str) and issuer.strip() else None,
    )


@app_clients.post(
    "/v1/oidc/clients",
    response_model=OidcClientCreated,
    status_code=201,
    response_model_exclude_none=True,
)
async def create_client(
    request: Request,
    body: OidcClientCreateRequest,
    op: ActingOperator,
) -> OidcClientCreated:
    machine = _active(request)
    name = body.name.strip()
    if not name:
        raise problem(422, "Name required", "Give the app a name people will recognise.")
    if not op.may_name(body.owner):
        raise refuse_for_node(op, "register sign-in")
    secret = secrets.token_urlsafe(32)
    record: dict[str, Any] = {
        "clientId": "c-" + secrets.token_hex(12),
        "name": name,
        "secretVerifier": oidc.secret_verifier(secret),
        "redirectUris": _check_redirects([uri.root for uri in body.redirectUris]),
        "createdAt": oidc.utcnow_iso(),
    }
    if body.owner:
        record["owner"] = body.owner
    machine.append(OP_PUT_OIDC_CLIENT, {"client": record})
    return OidcClientCreated(client=_client_view(record), clientSecret=secret)


@app_clients.delete("/v1/oidc/clients/{client_id}", status_code=204)
async def delete_client(request: Request, client_id: str, op: ActingOperator) -> Response:
    machine = _active(request)
    record = machine.state.oidc_clients.get(client_id)
    if record is None:
        raise problem(404, "No such app", "No app that signs in with Eugene has that id.")
    if not op.may_name(record.get("owner")):
        raise refuse_for_node(op, "remove sign-in")
    machine.append(OP_DELETE_OIDC_CLIENT, {"clientId": client_id})
    return Response(status_code=204)
