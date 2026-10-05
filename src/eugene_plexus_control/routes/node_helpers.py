"""Node file capability management, outbound delivery, and app-scoped use.

**Job Sites** (`docs/design/remote-nodes.md` §3.2-§3.3). A machine joined
with the `files` grant belongs to the person who confirmed its join. Only that
person turns its helper on, registers its folders and says who may use them,
through `/oidc/job-sites`, which Workbench calls with their sign-in. Eugene's
owner manages membership and sees that a site exists and whether it is online.
In production mode that is all; in dev mode the owner also sees its folders
and who may use them, and may give themselves access (J13b). Every
administrative change to a site here is refused with a sentence saying whose
it is.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, StrictBool

from .. import node_helpers as helpers
from .. import oidc, tokens
from .._generated.models import (
    HelperCall,
    HelperCancel,
    HelperDiscovery,
    HelperFolderCreate,
    HelperReport,
    HelperResult,
    JobSiteEnable,
    JobSiteFolderCreate,
    JobSiteInviteRequest,
    JobSitePeopleRequest,
)
from ..applied import OP_PUT_NODE_HELPER
from ..dependencies import problem, require_member_actor, require_operator, verify_bearer
from ..tokens import TYP_SESSION, Claims
from .oidc import _client_auth
from .people import _active


def _private_response(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"


log = logging.getLogger(__name__)
router = APIRouter(tags=["node-helpers"], dependencies=[Depends(_private_response)])
admin = APIRouter(
    tags=["node-helpers"], dependencies=[Depends(require_operator), Depends(_private_response)]
)
NodeActor = Annotated[Claims, Depends(require_member_actor)]


class Enabled(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: StrictBool


class OwnerAccess(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ownerAccess: Literal["none", "read", "write"]


@admin.patch("/v1/node-helpers/{node}/folders/{folder_id}")
async def owner_access(
    request: Request, node: str, folder_id: str, body: OwnerAccess
) -> dict[str, Any]:
    machine = _active(request)
    config = helpers.configuration(machine.state, node)
    if helpers.is_site(machine.state, node) and helpers.install_mode(machine.state) != helpers.DEV:
        raise problem(
            403,
            "Production mode",
            f"{node} is a job site. In production mode Eugene's owner cannot give "
            "themselves its folders; its owner gives access, from Workbench.",
        )
    folder = next((f for f in config["folders"] if f["id"] == folder_id), None)
    if folder is None:
        raise problem(404, "Folder unavailable", "This folder is no longer registered.")
    if body.ownerAccess == "write" and not folder["writable"]:
        raise problem(422, "Read-only folder", "This folder permits reads only.")
    updated = {**folder, "ownerAccess": body.ownerAccess}
    _save(
        machine,
        {**config, "folders": [updated if f["id"] == folder_id else f for f in config["folders"]]},
    )
    return updated


def _node(request: Request, actor: Any) -> str:
    _active(request)
    node = str(actor.issuer_node)
    # A Job Site's status is its last contact: it has no address to probe.
    request.app.state.node_contacts[node] = datetime.now(UTC)
    return node


def _not_a_site(request: Request, node: str) -> None:
    state = request.app.state.machine.state
    if helpers.is_site(state, node):
        record = state.nodes[node]
        owner = state.people.get(record.owner or "", {}).get("name") or "its owner"
        raise problem(
            403,
            "A job site's files are its owner's",
            f"{node} is {owner}'s job site. Only {owner} turns its file helper on, adds "
            "its folders and says who may use them, from Workbench (Job sites).",
        )


def _save(machine: Any, config: dict[str, Any]) -> None:
    machine.append(OP_PUT_NODE_HELPER, {"helper": config})


def _same(machine: Any, original: dict[str, Any]) -> dict[str, Any]:
    current = helpers.configuration(machine.state, original["node"])
    if (current["nodeKey"], current["enrolledAt"]) != (original["nodeKey"], original["enrolledAt"]):
        raise problem(
            409, "Machine changed", "This machine re-enrolled. Configure its folders again."
        )
    if not current["enabled"]:
        raise problem(403, "File helper disabled", "File access is disabled on this machine.")
    return current


@admin.get("/v1/node-helpers")
async def listing(request: Request) -> dict[str, Any]:
    machine = _active(request)
    delivery = helpers.broker(request)
    state = machine.state
    dev = helpers.install_mode(state) == helpers.DEV
    listed = []
    for node in sorted(state.nodes):
        if not state.nodes[node].signingPublicKey:
            continue
        config = helpers.configuration(state, node)
        entry = dict(config, **delivery.status(config), jobSite=False, hidden=False)
        if helpers.is_site(state, node):
            owner = state.people.get(state.nodes[node].owner or "", {})
            entry.update(jobSite=True, owner=owner.get("id"), ownerName=owner.get("name"))
            if not dev:
                # Membership only (J11): that it exists, whose it is and
                # whether it is online. Not its folders, nor who uses them.
                for key in ("enabled", "ready", "supported", "reason", "account"):
                    entry.pop(key, None)
                entry.update(folders=[], hidden=True)
        listed.append(entry)
    return {"helpers": listed, "installMode": helpers.mode_view(state)}


@admin.put("/v1/node-helpers/{node}")
async def configure(request: Request, node: str, body: Enabled) -> dict[str, Any]:
    machine = _active(request)
    _not_a_site(request, node)
    config = {**helpers.configuration(machine.state, node), "enabled": body.enabled}
    _save(machine, config)
    helpers.broker(request).changed.set()
    return config


@admin.post("/v1/node-helpers/{node}/folders", status_code=201)
async def add_folder(request: Request, node: str, body: HelperFolderCreate) -> dict[str, Any]:
    machine = _active(request)
    _not_a_site(request, node)
    config = helpers.configuration(machine.state, node)
    # Code-generated enum defaults are strings until validated explicitly.
    values = body.model_dump(mode="json", exclude_unset=True)
    values.setdefault("writable", False)
    values.setdefault("ownerAccess", "none")
    if (
        not values["name"].strip()
        or not values["path"].strip()
        or any(ord(c) < 32 for c in values["name"] + values["path"])
    ):
        raise problem(
            422, "Invalid folder", "Use a nonempty folder name and path without control characters."
        )
    if values["ownerAccess"] == "write" and not values["writable"]:
        raise problem(
            422, "Read-only folder", "Enable text writes before giving the owner write access."
        )
    bearer = request.headers.get("authorization", "").partition(" ")[2]

    def check() -> dict[str, Any]:
        verify_bearer(request, bearer, classes=(TYP_SESSION,))
        current = _same(machine, config)
        if len(current["folders"]) >= 64:
            raise problem(
                409, "Folder limit", "Remove a folder before adding another; this node has 64."
            )
        return {
            "node": node,
            "nodeKey": config["nodeKey"],
            "enrolledAt": config["enrolledAt"],
            "subject": "operator",
            "tool": "inspect",
            "arguments": {"path": values["path"]},
        }

    outcome = await helpers.broker(request).submit(config, check)
    if outcome["status"] != "done":
        raise problem(
            400,
            "Folder could not be opened",
            outcome.get("message")
            or "Check the helper's OS account has permission to this folder.",
        )
    result = outcome.get("result") or {}
    if (
        not isinstance(result.get("identity"), str)
        or not 0 < len(result["identity"]) <= 256
        or not isinstance(result.get("path"), str)
        or not 0 < len(result["path"]) <= 4096
    ):
        raise problem(502, "Invalid helper answer", "The helper did not return a folder identity.")
    folder = {
        **values,
        "name": values["name"].strip(),
        "id": secrets.token_hex(16),
        "path": result["path"],
        "identity": result["identity"],
    }
    current = _same(machine, config)
    if len(current["folders"]) >= 64:
        raise problem(409, "Folder limit", "This machine already has 64 registered folders.")
    _save(machine, {**current, "folders": [*current["folders"], folder]})
    return folder


@admin.delete("/v1/node-helpers/{node}/folders/{folder_id}", status_code=204)
async def remove_folder(request: Request, node: str, folder_id: str) -> None:
    machine = _active(request)
    _not_a_site(request, node)
    config = helpers.configuration(machine.state, node)
    _save(machine, {**config, "folders": [f for f in config["folders"] if f["id"] != folder_id]})


@router.post("/v1/node-helpers/poll")
async def poll(request: Request, body: HelperReport, actor: NodeActor) -> dict[str, Any]:
    node = _node(request, actor)
    config = helpers.configuration(request.app.state.machine.state, node)
    ident = await helpers.broker(request).poll(config, body.model_dump(mode="json"))
    # A setting may have changed during the long poll.
    state = request.app.state.machine.state
    current = helpers.configuration(state, node)
    # A job site is told its owner: the one person whose commands may
    # register its folders, which its own relay checks again (J11).
    owner = state.nodes[node].owner if helpers.is_site(state, node) else None
    return {"configuration": current, "operation": ident, "siteOwner": owner}


@router.post("/v1/node-helpers/operations/{ident}/claim")
async def claim(request: Request, ident: str, actor: NodeActor) -> dict[str, Any]:
    return helpers.broker(request).claim(_node(request, actor), ident)


@router.post("/v1/node-helpers/operations/{ident}/result", status_code=204)
async def result(request: Request, ident: str, body: HelperResult, actor: NodeActor) -> None:
    value = body.model_dump(mode="json", exclude_none=True)
    if len(json.dumps(value, ensure_ascii=False).encode()) > 70_000:
        raise problem(413, "Result too large", "A helper result may contain at most 70,000 bytes.")
    helpers.broker(request).finish(_node(request, actor), ident, value)


def _app_subject(request: Request, token: str) -> str:
    machine = _active(request)
    if request.app.state.auth_state.signing_key is None:
        raise problem(503, "Eugene is locked", "Unlock Eugene before using node folders.")
    client = _client_auth(request)
    if client is None:
        raise problem(401, "App sign-in required", "Workbench's client authentication failed.")
    if not str(client.get("owner") or "").startswith("app:workbench@"):
        raise problem(403, "Workbench required", "This app is not an enrolled Workbench.")
    try:
        claims = oidc.Provider.decode(
            token, machine.state, typ=oidc.TYP_REFRESH, audience=client["clientId"]
        )
    except jwt.InvalidTokenError:
        raise problem(
            401, "Sign-in ended", "Sign into Workbench again to use node folders."
        ) from None
    subject = oidc.still_valid(machine.state, claims, client["clientId"])
    if subject is None:
        raise problem(
            403, "Access withdrawn", "Your sign-in or Workbench access has been withdrawn."
        )
    return subject.sub


@router.post("/oidc/node-helpers/folders")
async def discover(request: Request, body: HelperDiscovery) -> dict[str, Any]:
    subject = _app_subject(request, body.refreshToken)
    state = request.app.state.machine.state
    grants = []
    for grant in helpers.grants_for(state, subject):
        try:
            config, folder = helpers.find_folder(state, grant["folderId"])
        except HTTPException:
            continue
        status = helpers.broker(request).status(config)
        grants.append(
            {
                "id": folder["id"],
                "node": config["node"],
                "name": folder["name"],
                "subject": subject,
                "writable": bool(grant["writable"] and folder["writable"]),
                "usable": True,
                "available": bool(config["enabled"] and status["ready"]),
                "reason": status.get("reason") if config["enabled"] else "File helper is disabled.",
                "jobSite": helpers.is_site(state, config["node"]),
            }
        )
    return {"grants": grants, "installMode": helpers.mode_view(state)}


@router.post("/oidc/node-helpers/execute")
async def execute(request: Request, body: HelperCall) -> dict[str, Any]:
    _app_subject(request, body.refreshToken)
    if len(json.dumps(body.arguments, ensure_ascii=False).encode()) > 40_000:
        raise problem(413, "Arguments too large", "Use a smaller file operation.")
    machine = _active(request)
    tool = body.model_dump(mode="json")["tool"]
    config, _ = helpers.find_folder(machine.state, body.folderId)

    def check() -> dict[str, Any]:
        subject = _app_subject(request, body.refreshToken)
        _same(machine, config)
        _, folder = helpers.find_folder(machine.state, body.folderId)
        grant = next(
            (
                g
                for g in helpers.grants_for(machine.state, subject)
                if g["folderId"] == body.folderId
            ),
            None,
        )
        if grant is None:
            raise problem(403, "Folder unavailable", "This folder is not granted to you.")
        writable = bool(folder["writable"] and grant["writable"])
        if tool == "write_text" and not writable:
            raise problem(403, "Read-only folder", "You may only read this folder. No write ran.")
        return {
            "node": config["node"],
            "nodeKey": config["nodeKey"],
            "enrolledAt": config["enrolledAt"],
            "subject": subject,
            "folder": {**folder, "writable": writable, "subject": subject},
            "tool": tool,
            "arguments": body.arguments,
        }

    operation_id = _operation_id(body.refreshToken, body.operationId) if body.operationId else None
    outcome = await helpers.broker(request).submit(
        config, check, write=tool == "write_text", operation_id=operation_id
    )
    # Workbench keeps these with the result: what production mode hides
    # from the owner's chat reading is decided by them (J13a, J18).
    state = machine.state
    return {
        **outcome,
        "jobSite": helpers.is_site(state, config["node"]),
        "installMode": helpers.install_mode(state),
    }


def _operation_id(token: str, ident: str) -> str:
    return hashlib.sha256((token + "\0" + ident).encode()).hexdigest()


@router.post("/oidc/node-helpers/cancel", status_code=204)
async def cancel(request: Request, body: HelperCancel) -> None:
    _app_subject(request, body.refreshToken)
    helpers.broker(request).cancel(_operation_id(body.refreshToken, body.operationId))


# --------------------------------------------------------------------------- #
# A Job Site's owner (J9, J11), through Workbench with their own sign-in
# --------------------------------------------------------------------------- #

INVITES_PER_PERSON = 3
INVITE_SECONDS = 900


@router.post("/oidc/install-mode")
async def install_mode(request: Request) -> dict[str, Any]:
    """The mode, for an app to show every person it serves (J18). Any enrolled
    app client: the mode is the same for everyone and is not a secret."""
    if _client_auth(request) is None:
        raise problem(401, "App sign-in required", "The app's client authentication failed.")
    return helpers.mode_view(request.app.state.machine.state)


def _own_site(request: Request, token: str, node: str) -> tuple[Any, str, dict[str, Any]]:
    """The signed-in person, and their site's helper configuration, or 404.

    One answer for "no such site" and "not yours", so a person cannot learn
    which machines other people own by asking."""
    subject = _app_subject(request, token)
    machine = _active(request)
    record = machine.state.nodes.get(node)
    if record is None or not helpers.is_site(machine.state, node) or record.owner != subject:
        raise problem(404, "No such job site", f"You have no job site named {node!r}.")
    return machine, subject, helpers.configuration(machine.state, node)


def _person_name(state: Any, person: str) -> str:
    return str(state.people.get(person, {}).get("name") or "someone who has left")


def _site_view(request: Request, config: dict[str, Any]) -> dict[str, Any]:
    state = request.app.state.machine.state
    status = helpers.broker(request).status(config)
    contact = request.app.state.node_contacts.get(config["node"])
    return {
        "node": config["node"],
        "enabled": config["enabled"],
        "online": bool(status.get("online")),
        "ready": bool(status.get("ready")),
        "supported": status.get("supported"),
        "reason": status.get("reason"),
        "account": status.get("account"),
        "lastContactAt": contact.isoformat() if contact else None,
        "folders": [
            {
                "id": folder["id"],
                "name": folder["name"],
                "path": folder["path"],
                "writable": folder["writable"],
                "people": [
                    {
                        "person": entry["person"],
                        "name": _person_name(state, entry["person"]),
                        "writable": entry["writable"],
                    }
                    for entry in folder.get("people") or []
                ],
            }
            for folder in config["folders"]
        ],
    }


def _can_invite(request: Request) -> bool:
    settings = getattr(request.app.state, "settings", None)
    return bool(
        getattr(settings, "nodes_origin", None) and getattr(settings, "nodes_public", False)
    )


@router.post("/oidc/job-sites")
async def my_sites(request: Request, body: HelperDiscovery) -> dict[str, Any]:
    subject = _app_subject(request, body.refreshToken)
    state = request.app.state.machine.state
    sites = [
        _site_view(request, helpers.configuration(state, node))
        for node in sorted(state.nodes)
        if helpers.is_site(state, node) and state.nodes[node].owner == subject
    ]
    return {
        "sites": sites,
        "installMode": helpers.mode_view(state),
        "canInvite": subject != "operator" and _can_invite(request),
    }


@router.post("/oidc/job-sites/invite")
async def invite(request: Request, body: JobSiteInviteRequest) -> dict[str, Any]:
    """J9: any signed-in person may add a Job Site for a machine of their own."""
    subject = _app_subject(request, body.refreshToken)
    machine = _active(request)
    if subject == "operator":
        raise problem(
            403,
            "Job sites belong to people",
            "Eugene's owner signs in with the passphrase, which owns no job sites. Add "
            "yourself as a person on Eugene's People page, sign in to Workbench as them, "
            "and add the machine from there.",
        )
    if not _can_invite(request):
        raise problem(
            409,
            "No route for machines yet",
            "Eugene's owner has not opened a route for machines outside the network "
            "(Settings, Container access setup: let machines join from any network).",
        )
    store = request.app.state.join_tokens
    if store.outstanding_for(subject) >= INVITES_PER_PERSON:
        raise problem(
            429,
            "Too many open invitations",
            f"You have {INVITES_PER_PERSON} unused job site invitations. Use one, or wait "
            f"{INVITE_SECONDS // 60} minutes for them to expire.",
        )
    if body.nodeName and body.nodeName in machine.state.nodes:
        raise problem(
            409, "That name is taken", f"A machine named {body.nodeName!r} is already joined."
        )
    minted = store.mint(
        ttl_seconds=INVITE_SECONDS,
        node_name=body.nodeName,
        grants=(tokens.GRANT_FILES,),
        owner=subject,
    )
    settings = request.app.state.settings
    log.info("%s minted a job site invitation", _person_name(machine.state, subject))
    return {
        "token": minted.token,
        "expiresAt": datetime.fromtimestamp(minted.expires_at, tz=UTC).isoformat(),
        "nodeName": minted.node_name,
        "nodesUrl": str(settings.nodes_origin).rstrip("/"),
        "rootKey": machine.state.identity.controlPublicKey or "",
        "owner": _person_name(machine.state, subject),
    }


@router.post("/oidc/job-sites/{node}/enabled")
async def site_enabled(request: Request, node: str, body: JobSiteEnable) -> dict[str, Any]:
    machine, _, config = _own_site(request, body.refreshToken, node)
    config = {**config, "enabled": body.enabled}
    _save(machine, config)
    helpers.broker(request).changed.set()
    return _site_view(request, config)


@router.post("/oidc/job-sites/{node}/folders", status_code=201)
async def site_add_folder(request: Request, node: str, body: JobSiteFolderCreate) -> dict[str, Any]:
    machine, subject, config = _own_site(request, body.refreshToken, node)
    name, path = body.name.strip(), body.path.strip()
    if not name or not path or any(ord(c) < 32 for c in name + path):
        raise problem(
            422,
            "Invalid folder",
            "Use a nonempty folder name and path without control characters.",
        )

    def check() -> dict[str, Any]:
        _own_site(request, body.refreshToken, node)
        current = _same(machine, config)
        if len(current["folders"]) >= 64:
            raise problem(409, "Folder limit", "Remove a folder before adding another; 64 at most.")
        return {
            "node": node,
            "nodeKey": config["nodeKey"],
            "enrolledAt": config["enrolledAt"],
            "subject": subject,
            "tool": "inspect",
            "arguments": {"path": path},
        }

    outcome = await helpers.broker(request).submit(config, check)
    if outcome["status"] != "done":
        raise problem(
            400,
            "Folder could not be opened",
            outcome.get("message")
            or "Check the helper's OS account has permission to this folder.",
        )
    result = outcome.get("result") or {}
    if (
        not isinstance(result.get("identity"), str)
        or not 0 < len(result["identity"]) <= 256
        or not isinstance(result.get("path"), str)
        or not 0 < len(result["path"]) <= 4096
    ):
        raise problem(502, "Invalid helper answer", "The helper did not return a folder identity.")
    folder = {
        "id": secrets.token_hex(16),
        "name": name,
        "path": result["path"],
        "identity": result["identity"],
        "writable": body.writable,
        "ownerAccess": "none",
        "people": [],
    }
    current = _same(machine, config)
    if len(current["folders"]) >= 64:
        raise problem(409, "Folder limit", "This machine already has 64 registered folders.")
    _save(machine, {**current, "folders": [*current["folders"], folder]})
    return folder


@router.post("/oidc/job-sites/{node}/folders/{folder_id}/remove", status_code=204)
async def site_remove_folder(
    request: Request, node: str, folder_id: str, body: HelperDiscovery
) -> None:
    machine, _, config = _own_site(request, body.refreshToken, node)
    if not any(f["id"] == folder_id for f in config["folders"]):
        raise problem(404, "Folder unavailable", "This folder is no longer registered.")
    _save(machine, {**config, "folders": [f for f in config["folders"] if f["id"] != folder_id]})


@router.post("/oidc/job-sites/{node}/folders/{folder_id}/people")
async def site_folder_people(
    request: Request, node: str, folder_id: str, body: JobSitePeopleRequest
) -> dict[str, Any]:
    """J11: the site's owner says who may use a folder, themselves included."""
    machine, _, config = _own_site(request, body.refreshToken, node)
    folder = next((f for f in config["folders"] if f["id"] == folder_id), None)
    if folder is None:
        raise problem(404, "Folder unavailable", "This folder is no longer registered.")
    by_name = {p["name"].casefold(): p for p in machine.state.people.values()}
    people: list[dict[str, Any]] = []
    for entry in body.people:
        person = by_name.get(entry.name.strip().casefold())
        if person is None:
            raise problem(
                404, "No such person", f"Nobody on this install signs in as {entry.name!r}."
            )
        if any(p["person"] == person["id"] for p in people):
            raise problem(422, "Named twice", f"{person['name']} is listed twice.")
        if entry.writable and not folder["writable"]:
            raise problem(422, "Read-only folder", "This folder permits reads only.")
        people.append({"person": person["id"], "writable": bool(entry.writable)})
    updated = {**folder, "people": people}
    _save(
        machine,
        {**config, "folders": [updated if f["id"] == folder_id else f for f in config["folders"]]},
    )
    return updated


@router.post("/oidc/job-sites/{node}/leave", status_code=204)
async def site_leave(request: Request, node: str, body: HelperDiscovery) -> None:
    """Leaving needs no one's permission (§3.3)."""
    from .nodes import revoke

    _own_site(request, body.refreshToken, node)
    revoke(request, node)
