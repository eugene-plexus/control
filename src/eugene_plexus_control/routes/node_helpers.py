"""Machines' site hosts: their file support, MCP between them and the root,
and a Job Site's owner managing it from Workbench.

**MCP between site and root** (`docs/design/remote-nodes.md` §3.4, J6). Each
machine runs a site host, a local MCP host whose policy is final on that
machine (J8). Workbench asks `/oidc/sites/servers` what it may use, and sends
one MCP request of the 2026-07-28 revision to `/oidc/sites/mcp`; the root
queues it, the machine claims it on its next poll, and its host decides. On
an ordinary node the root's grants are final (J6d) and travel with the call
(`grants`). On a Job Site the site's own list is final, and what the root
holds of it is a cache of the site's last report (rule 2 of §3.3).

**Job Sites** (§3.2-§3.3). A machine joined with the `files` grant belongs to
the person who confirmed its join. Only that person turns its file support
on, registers its folders, says who may use them and its local servers, and
reads its audit log, through `/oidc/job-sites`, which Workbench calls with
their sign-in; each change is relayed to the site as a management action,
which the site accepts only from the owner it pinned (J6b). Eugene's owner
manages membership and sees that a site exists and whether it is online. In
production mode that is all; in dev mode the owner also sees its folders and
may give themselves access (J13b), which works only if the site's owner let
them in there (J6e). Every administrative change to a site here is refused
with a sentence saying whose it is.

**Capabilities, not the operator** (J15): the console's routes need
`node-files`, which the passphrase session holds.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

import jwt
from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, ConfigDict, StrictBool

from .. import capabilities, oidc, tokens
from .. import node_helpers as helpers
from .._generated.models import (
    HelperDiscovery,
    HelperFolderCreate,
    HelperReport,
    HelperResult,
    JobSiteAccessRequest,
    JobSiteAuditRequest,
    JobSiteEnable,
    JobSiteFolderCreate,
    JobSiteInviteRequest,
    JobSitePeopleRequest,
    JobSiteServerEnable,
    JobSiteSettings,
    SiteCancel,
    SiteMcpCall,
)
from ..applied import OP_PUT_NODE_HELPER
from ..dependencies import problem, require_member_actor
from ..tokens import Claims
from .oidc import _client_auth
from .people import _active


def _private_response(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"


log = logging.getLogger(__name__)
router = APIRouter(tags=["node-helpers"], dependencies=[Depends(_private_response)])
admin = APIRouter(
    tags=["node-helpers"],
    dependencies=[
        Depends(capabilities.require(capabilities.NODE_FILES)),
        Depends(_private_response),
    ],
)
NodeActor = Annotated[Claims, Depends(require_member_actor)]
MAX_REQUEST = 40_000
NEEDS_UPDATE = (
    "This machine runs an older Eugene, whose file support takes no MCP. Update Eugene on it "
    "(Nodes, Versions), then try again."
)


class Enabled(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: StrictBool


class OwnerAccess(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ownerAccess: Literal["none", "read", "write"]


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
            f"{node} is {owner}'s job site. Only {owner} turns its file support on, adds "
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
        raise problem(403, "File support disabled", "File access is disabled on this machine.")
    return current


def _speaks_mcp(request: Request, config: dict[str, Any]) -> None:
    """An agent older than slice 2 reports no `protocol`: say it needs an
    update, rather than send it MCP it cannot take."""
    delivery = helpers.broker(request)
    status = delivery.status(config)
    report = delivery.report(config["node"]) or {}
    if status.get("online") and report.get("ready") and report.get("protocol") != helpers.PROTOCOL:
        raise problem(503, "This machine needs an update", NEEDS_UPDATE)


def _availability(request: Request, config: dict[str, Any]) -> tuple[bool, str | None]:
    if not config["enabled"]:
        return False, "File support is off on this machine."
    status = helpers.broker(request).status(config)
    if not status.get("ready"):
        return False, status.get("reason") or "This machine's file support is not ready."
    report = helpers.broker(request).report(config["node"]) or {}
    if report.get("protocol") != helpers.PROTOCOL:
        return False, NEEDS_UPDATE
    return True, None


# --------------------------------------------------------------------------- #
# The console: ordinary nodes' file support (`node-files`, J15)
# --------------------------------------------------------------------------- #


def _site_folders_for_console(request: Request, node: str) -> list[dict[str, Any]]:
    """A Job Site's folders as it last reported them, with Eugene's owner's
    own dev-mode access on each (J13b). Shown in dev mode only."""
    state = request.app.state.machine.state
    summary = helpers.broker(request).summary(node) or {}
    owned = {g["folderId"]: g for g in helpers.configuration(state, node).get("devGrants") or []}
    return [
        {
            "id": folder["id"],
            "name": folder["name"],
            "path": folder["path"],
            "identity": folder["identity"],
            "writable": folder["writable"],
            "ownerAccess": (
                "none"
                if folder["id"] not in owned
                else ("write" if owned[folder["id"]]["writable"] else "read")
            ),
            "people": [
                {"person": p["subject"], "writable": p["writable"]}
                for p in folder.get("people") or []
            ],
        }
        for folder in summary.get("folders") or []
    ]


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
        entry.pop("devGrants", None)
        if helpers.is_site(state, node):
            owner = state.people.get(state.nodes[node].owner or "", {})
            entry.update(jobSite=True, owner=owner.get("id"), ownerName=owner.get("name"))
            if dev:
                entry["folders"] = _site_folders_for_console(request, node)
                summary = delivery.summary(node)
                entry["ownerInDevMode"] = summary.get("ownerInDevMode") if summary else None
            else:
                # Membership only (J11): that it exists, whose it is and
                # whether it is online. Not its folders, nor who uses them.
                for key in ("enabled", "ready", "supported", "reason", "account"):
                    entry.pop(key, None)
                entry.update(folders=[], hidden=True)
        listed.append(entry)
    return {"helpers": listed, "installMode": helpers.mode_view(state)}


@admin.patch("/v1/node-helpers/{node}/folders/{folder_id}")
async def owner_access(
    request: Request, node: str, folder_id: str, body: OwnerAccess
) -> dict[str, Any]:
    machine = _active(request)
    config = helpers.configuration(machine.state, node)
    if helpers.is_site(machine.state, node):
        if helpers.install_mode(machine.state) != helpers.DEV:
            raise problem(
                403,
                "Production mode",
                f"{node} is a job site. In production mode Eugene's owner cannot give "
                "themselves its folders; its owner gives access, from Workbench.",
            )
        folder = next(
            (f for f in _site_folders_for_console(request, node) if f["id"] == folder_id), None
        )
        if folder is None:
            raise problem(404, "Folder unavailable", "This site has not reported that folder.")
        if body.ownerAccess == "write" and not folder["writable"]:
            raise problem(422, "Read-only folder", "This folder permits reads only.")
        kept = [g for g in config.get("devGrants") or [] if g["folderId"] != folder_id]
        if body.ownerAccess != "none":
            kept.append({"folderId": folder_id, "writable": body.ownerAccess == "write"})
        _save(machine, {**config, "devGrants": kept})
        return {**folder, "ownerAccess": body.ownerAccess}
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
    """An ordinary node's folder, registered by Eugene's owner: the node's
    host opens it and reports its identity (`folder.inspect`)."""
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
    name = values["name"].strip()

    def unused(current: dict[str, Any]) -> None:
        # The file server's `folder` argument names it (J6g).
        if name.casefold() in {n.casefold() for n in helpers.folder_names(current).values()}:
            raise problem(
                409, "Name in use", f"{node} already has a folder named {name}. Choose another."
            )

    unused(config)
    _speaks_mcp(request, config)

    def check() -> dict[str, Any]:
        current = _same(machine, config)
        if len(current["folders"]) >= 64:
            raise problem(
                409, "Folder limit", "Remove a folder before adding another; this node has 64."
            )
        return helpers.envelope(
            machine.state,
            config,
            helpers.OPERATOR,
            kind="manage",
            action="folder.inspect",
            arguments={"path": values["path"]},
        )

    outcome = await helpers.broker(request).submit(config, check)
    if outcome["status"] != "done":
        raise problem(
            400,
            "Folder could not be opened",
            outcome.get("message")
            or "Check the file support's OS account has permission to this folder.",
        )
    result = outcome.get("result") or {}
    if (
        not isinstance(result.get("identity"), str)
        or not 0 < len(result["identity"]) <= 256
        or not isinstance(result.get("path"), str)
        or not 0 < len(result["path"]) <= 4096
    ):
        raise problem(502, "Invalid answer", "The machine did not return a folder identity.")
    folder = {
        **values,
        "name": name,
        "id": secrets.token_hex(16),
        "path": result["path"],
        "identity": result["identity"],
    }
    current = _same(machine, config)
    unused(current)
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


# --------------------------------------------------------------------------- #
# The machine: poll, claim, result (a member node's own token)
# --------------------------------------------------------------------------- #


@router.post("/v1/node-helpers/poll")
async def poll(request: Request, body: HelperReport, actor: NodeActor) -> dict[str, Any]:
    node = _node(request, actor)
    config = helpers.configuration(request.app.state.machine.state, node)
    report = body.model_dump(mode="json", exclude_none=True)
    ident = await helpers.broker(request).poll(config, report)
    # A setting may have changed during the long poll.
    state = request.app.state.machine.state
    current = helpers.configuration(state, node)
    # A job site is told its owner. It pins the one it was told at its join
    # and refuses a different one (J6b).
    owner = state.nodes[node].owner if helpers.is_site(state, node) else None
    return {"configuration": current, "operation": ident, "siteOwner": owner}


@router.post("/v1/node-helpers/operations/{ident}/claim")
async def claim(request: Request, ident: str, actor: NodeActor) -> dict[str, Any]:
    return helpers.broker(request).claim(_node(request, actor), ident)


@router.post("/v1/node-helpers/operations/{ident}/result", status_code=204)
async def result(request: Request, ident: str, body: HelperResult, actor: NodeActor) -> None:
    value = body.model_dump(mode="json", exclude_none=True)
    if len(json.dumps(value, ensure_ascii=False).encode()) > 70_000:
        raise problem(413, "Result too large", "A machine's answer may be at most 70,000 bytes.")
    helpers.broker(request).finish(_node(request, actor), ident, value)


# --------------------------------------------------------------------------- #
# Workbench: the servers a person may use, and one MCP request to one
# --------------------------------------------------------------------------- #


def _app_subject(request: Request, token: str) -> str:
    machine = _active(request)
    if request.app.state.auth_state.signing_key is None:
        raise problem(503, "Eugene is locked", "Unlock Eugene before using machines' tools.")
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
            401, "Sign-in ended", "Sign into Workbench again to use machines' tools."
        ) from None
    subject = oidc.still_valid(machine.state, claims, client["clientId"])
    if subject is None:
        raise problem(
            403, "Access withdrawn", "Your sign-in or Workbench access has been withdrawn."
        )
    return subject.sub


def _members(state: Any) -> list[str]:
    return [n for n in sorted(state.nodes) if state.nodes[n].signingPublicKey]


@router.post("/oidc/sites/servers")
async def servers(request: Request, body: HelperDiscovery) -> dict[str, Any]:
    subject = _app_subject(request, body.refreshToken)
    state = request.app.state.machine.state
    delivery = helpers.broker(request)
    listed: list[dict[str, Any]] = []
    for node in _members(state):
        config = helpers.configuration(state, node)
        available, reason = _availability(request, config)
        site = helpers.is_site(state, node)
        common = {"node": node, "jobSite": site, "available": available, "reason": reason}
        if not site:
            folders = [
                {"id": g["folderId"], "name": g["name"], "writable": g["writable"]}
                for g in helpers.node_grants(state, subject, node)
            ]
            if folders:
                listed.append(
                    {
                        **common,
                        "server": helpers.FILES,
                        "name": "Files",
                        "kind": "files",
                        "folders": folders,
                    }
                )
            continue
        summary = delivery.summary(node)
        if subject == helpers.OPERATOR:
            if not (summary or {}).get("ownerInDevMode"):
                continue
            folders = [
                {"id": g["folderId"], "name": g["name"], "writable": g["writable"]}
                for g in helpers.dev_grants(state, node, summary)
            ]
        else:
            folders = helpers.site_folders_for(summary, subject)
        if folders:
            listed.append(
                {
                    **common,
                    "server": helpers.FILES,
                    "name": "Files",
                    "kind": "files",
                    "folders": folders,
                }
            )
        if subject == helpers.OPERATOR:
            continue
        for server in helpers.site_local_servers_for(summary, subject):
            listed.append(
                {
                    **common,
                    "server": server["id"],
                    "name": server["name"],
                    "kind": "local",
                    "available": available and bool(server.get("available")),
                    "reason": reason or server.get("reason"),
                    "folders": [],
                }
            )
    return {"servers": listed, "installMode": helpers.mode_view(state)}


def _named_folder(request_body: dict[str, Any]) -> tuple[str | None, Any]:
    if request_body.get("method") != "tools/call":
        return None, None
    params = request_body.get("params") or {}
    arguments = params.get("arguments") if isinstance(params, dict) else None
    tool = params.get("name") if isinstance(params, dict) else None
    folder = arguments.get("folder") if isinstance(arguments, dict) else None
    return (tool if isinstance(tool, str) else None), folder


@router.post("/oidc/sites/mcp")
async def mcp(request: Request, body: SiteMcpCall) -> dict[str, Any]:
    """One MCP request to one server on one machine. The root checks the
    Workbench client, the person's sign-in and, on an ordinary node, its own
    grants; on a Job Site the site's own policy decides (J8)."""
    _app_subject(request, body.refreshToken)
    request_body = body.request.model_dump(mode="json", exclude_none=True, by_alias=True)
    if len(json.dumps(request_body, ensure_ascii=False).encode()) > MAX_REQUEST:
        raise problem(413, "Request too large", "Send a smaller request (40,000 bytes at most).")
    machine = _active(request)
    state = machine.state
    node, server = body.node, str(body.server)
    if node not in _members(state):
        raise problem(403, "Not granted", "You have nothing on that machine.")
    config = helpers.configuration(state, node)
    site = helpers.is_site(state, node)
    tool, folder = _named_folder(request_body)

    def authorized() -> tuple[str, list[dict[str, Any]]]:
        """Who asks, and the grants their call carries; checked at submit,
        at the claim and again before an answer is released."""
        subject = _app_subject(request, body.refreshToken)
        current = machine.state
        _same(machine, config)
        if site:
            summary = helpers.broker(request).summary(node)
            if subject == helpers.OPERATOR:
                if helpers.install_mode(current) != helpers.DEV:
                    raise problem(
                        403,
                        "Production mode",
                        "Eugene is in production mode: its owner's own access to a job site "
                        "does not work.",
                    )
                grants = helpers.dev_grants(current, node, summary)
                if server != helpers.FILES or not grants:
                    raise problem(403, "Not granted", "You have nothing on that machine.")
                return subject, grants
            listed = (
                bool(helpers.site_folders_for(summary, subject))
                if server == helpers.FILES
                else any(
                    s["id"] == server for s in helpers.site_local_servers_for(summary, subject)
                )
            )
            if not listed:
                raise problem(403, "Not granted", "You have nothing on that machine.")
            return subject, []
        if server != helpers.FILES:
            raise problem(403, "Not granted", "An ordinary node offers its folders only.")
        grants = helpers.node_grants(current, subject, node)
        if not grants:
            raise problem(403, "Not granted", "You have no folders on that machine.")
        if tool is not None:
            # The root's grant is final here (J6d): a folder it did not give
            # never reaches the machine.
            named = next((g for g in grants if g["name"] == folder), None)
            if named is None:
                raise problem(403, "Folder unavailable", "This folder is not granted to you.")
            if tool == "write_text" and not named["writable"]:
                raise problem(
                    403, "Read-only folder", "You may only read this folder. No write ran."
                )
        return subject, grants

    def check() -> dict[str, Any]:
        subject, grants = authorized()
        return helpers.envelope(
            machine.state, config, subject, server=server, request=request_body, grants=grants
        )

    check()
    _speaks_mcp(request, config)
    operation_id = _operation_id(body.refreshToken, body.operationId) if body.operationId else None
    outcome = await helpers.broker(request).submit(
        config, check, write=tool is not None, operation_id=operation_id
    )
    # Workbench keeps these with the result: what production mode hides
    # from the owner's chat reading is decided by them (J13a, J18).
    answer: dict[str, Any] = {
        "status": outcome.get("status", "failed"),
        "jobSite": site,
        "installMode": helpers.install_mode(machine.state),
    }
    for key in ("message", "response"):
        if outcome.get(key) is not None:
            answer[key] = outcome[key]
    return answer


def _operation_id(token: str, ident: str) -> str:
    return hashlib.sha256((token + "\0" + ident).encode()).hexdigest()


@router.post("/oidc/sites/cancel", status_code=204)
async def cancel(request: Request, body: SiteCancel) -> None:
    _app_subject(request, body.refreshToken)
    helpers.broker(request).cancel(_operation_id(body.refreshToken, body.operationId))


# --------------------------------------------------------------------------- #
# A Job Site's owner (J9, J11, J6b), through Workbench with their own sign-in
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
    """The signed-in person, and their site's configuration, or 404.

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


def _person_id(state: Any, name: str) -> str:
    by_name = {p["name"].casefold(): p for p in state.people.values()}
    person = by_name.get(name.strip().casefold())
    if person is None:
        raise problem(404, "No such person", f"Nobody on this install signs in as {name!r}.")
    return str(person["id"])


def _folder_view(state: Any, folder: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": folder["id"],
        "name": folder["name"],
        "path": folder["path"],
        "writable": folder["writable"],
        "people": [
            {
                "person": p["subject"],
                "name": _person_name(state, p["subject"]),
                "writable": p["writable"],
            }
            for p in folder.get("people") or []
        ],
    }


def _server_view(
    state: Any, server: dict[str, Any], people: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "server": server,
        "people": [
            {
                "person": p["subject"],
                "name": _person_name(state, p["subject"]),
                "tools": p["tools"],
            }
            for p in people
        ],
    }


def _site_view(request: Request, config: dict[str, Any]) -> dict[str, Any]:
    state = request.app.state.machine.state
    delivery = helpers.broker(request)
    status = delivery.status(config)
    contact = request.app.state.node_contacts.get(config["node"])
    summary = delivery.summary(config["node"]) or {}
    access = summary.get("access") or []
    return {
        "node": config["node"],
        "enabled": config["enabled"],
        "online": bool(status.get("online")),
        "ready": bool(status.get("ready")),
        "supported": status.get("supported"),
        "reason": status.get("reason"),
        "account": status.get("account"),
        "lastContactAt": contact.isoformat() if contact else None,
        "folders": [_folder_view(state, f) for f in summary.get("folders") or []],
        "servers": [
            _server_view(
                state,
                server,
                [
                    {"subject": e["subject"], "tools": e["tools"]}
                    for e in access
                    if e["server"] == server["id"]
                ],
            )
            for server in summary.get("servers") or []
            if server["kind"] == "local"
        ],
        "ownerInDevMode": summary.get("ownerInDevMode") if summary else None,
    }


async def _manage(
    request: Request, token: str, node: str, action: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """One management action, relayed to the site, which takes it from the
    owner it pinned and nobody else (J6b). Its refusal comes back as 422 in
    the site's own words."""
    machine, subject, config = _own_site(request, token, node)
    _speaks_mcp(request, config)

    def check() -> dict[str, Any]:
        _own_site(request, token, node)
        _same(machine, config)
        return helpers.envelope(
            machine.state, config, subject, kind="manage", action=action, arguments=arguments
        )

    outcome = await helpers.broker(request).submit(config, check)
    if outcome.get("status") != "done":
        raise problem(
            422,
            "The machine refused it",
            outcome.get("message") or "The machine did not take this change.",
        )
    value = outcome.get("result")
    return value if isinstance(value, dict) else {}


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
        "canInvite": subject != helpers.OPERATOR and _can_invite(request),
    }


@router.post("/oidc/job-sites/invite")
async def invite(request: Request, body: JobSiteInviteRequest) -> dict[str, Any]:
    """J9: any signed-in person may add a Job Site for a machine of their own."""
    subject = _app_subject(request, body.refreshToken)
    machine = _active(request)
    if subject == helpers.OPERATOR:
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
    name, path = body.name.strip(), body.path.strip()
    if not name or not path or any(ord(c) < 32 for c in name + path):
        raise problem(
            422,
            "Invalid folder",
            "Use a nonempty folder name and path without control characters.",
        )
    folder = await _manage(
        request,
        body.refreshToken,
        node,
        "folder.add",
        {"name": name, "path": path, "writable": bool(body.writable)},
    )
    return _folder_view(request.app.state.machine.state, folder)


@router.post("/oidc/job-sites/{node}/folders/{folder_id}/remove", status_code=204)
async def site_remove_folder(
    request: Request, node: str, folder_id: str, body: HelperDiscovery
) -> None:
    await _manage(request, body.refreshToken, node, "folder.remove", {"id": folder_id})


@router.post("/oidc/job-sites/{node}/folders/{folder_id}/people")
async def site_folder_people(
    request: Request, node: str, folder_id: str, body: JobSitePeopleRequest
) -> dict[str, Any]:
    """J11: the site's owner says who may use a folder, themselves included,
    and who may change files in it; the site keeps the list (J6b, J6g)."""
    _own_site(request, body.refreshToken, node)
    state = request.app.state.machine.state
    people = [
        {"subject": _person_id(state, entry.name), "writable": bool(entry.writable)}
        for entry in body.people
    ]
    folder = await _manage(
        request, body.refreshToken, node, "folder.people", {"id": folder_id, "people": people}
    )
    return _folder_view(request.app.state.machine.state, folder)


@router.post("/oidc/job-sites/{node}/servers/{server}/access")
async def site_server_access(
    request: Request, node: str, server: str, body: JobSiteAccessRequest
) -> dict[str, Any]:
    """Who may use which tools of one local server (J6b)."""
    _own_site(request, body.refreshToken, node)
    state = request.app.state.machine.state
    people = [
        {
            "subject": _person_id(state, entry.name),
            "tools": [t.model_dump(mode="json", exclude_none=True) for t in entry.tools],
        }
        for entry in body.people
    ]
    value = await _manage(
        request, body.refreshToken, node, "access.set", {"server": server, "people": people}
    )
    return _server_view(request.app.state.machine.state, value["server"], value["people"])


@router.post("/oidc/job-sites/{node}/servers/{server}/enabled")
async def site_server_enabled(
    request: Request, node: str, server: str, body: JobSiteServerEnable
) -> dict[str, Any]:
    value = await _manage(
        request,
        body.refreshToken,
        node,
        "server.enable",
        {"server": server, "enabled": body.enabled},
    )
    return _server_view(request.app.state.machine.state, value["server"], value["people"])


@router.post("/oidc/job-sites/{node}/settings")
async def site_settings(request: Request, node: str, body: JobSiteSettings) -> dict[str, Any]:
    """J6e: let Eugene's owner in while Eugene is in dev mode. Off by
    default; dev mode alone opens nothing on a site."""
    value = await _manage(
        request,
        body.refreshToken,
        node,
        "settings.set",
        {"ownerInDevMode": body.ownerInDevMode},
    )
    _, _, config = _own_site(request, body.refreshToken, node)
    return {**_site_view(request, config), "ownerInDevMode": value.get("ownerInDevMode")}


@router.post("/oidc/job-sites/{node}/audit")
async def site_audit(request: Request, node: str, body: JobSiteAuditRequest) -> dict[str, Any]:
    """The site's own audit log, read from the machine for its owner alone."""
    value = await _manage(
        request, body.refreshToken, node, "audit.read", {"limit": body.limit or 50}
    )
    return {"entries": value.get("entries") or []}


@router.post("/oidc/job-sites/{node}/leave", status_code=204)
async def site_leave(request: Request, node: str, body: HelperDiscovery) -> None:
    """Leaving needs no one's permission (§3.3)."""
    from .nodes import revoke

    _own_site(request, body.refreshToken, node)
    revoke(request, node)
