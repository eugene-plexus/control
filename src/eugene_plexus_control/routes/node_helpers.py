"""Node file capability management, outbound delivery, and app-scoped use."""

from __future__ import annotations

import hashlib
import json
import secrets
from typing import Annotated, Any, Literal

import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, StrictBool

from .. import node_helpers as helpers
from .. import oidc
from .._generated.models import (
    HelperCall,
    HelperCancel,
    HelperDiscovery,
    HelperFolderCreate,
    HelperReport,
    HelperResult,
)
from ..applied import OP_PUT_NODE_HELPER
from ..dependencies import problem, require_node_actor, require_operator, verify_bearer
from ..tokens import TYP_SESSION, Claims
from .oidc import _client_auth
from .people import _active


def _private_response(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"


router = APIRouter(tags=["node-helpers"], dependencies=[Depends(_private_response)])
admin = APIRouter(
    tags=["node-helpers"], dependencies=[Depends(require_operator), Depends(_private_response)]
)
NodeActor = Annotated[Claims, Depends(require_node_actor)]


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
    return str(actor.issuer_node)


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
    return {
        "helpers": [
            dict(config, **delivery.status(config))
            for node in sorted(machine.state.nodes)
            if machine.state.nodes[node].signingPublicKey
            for config in [helpers.configuration(machine.state, node)]
        ]
    }


@admin.put("/v1/node-helpers/{node}")
async def configure(request: Request, node: str, body: Enabled) -> dict[str, Any]:
    machine = _active(request)
    config = {**helpers.configuration(machine.state, node), "enabled": body.enabled}
    _save(machine, config)
    helpers.broker(request).changed.set()
    return config


@admin.post("/v1/node-helpers/{node}/folders", status_code=201)
async def add_folder(request: Request, node: str, body: HelperFolderCreate) -> dict[str, Any]:
    machine = _active(request)
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
    config = helpers.configuration(machine.state, node)
    _save(machine, {**config, "folders": [f for f in config["folders"] if f["id"] != folder_id]})


@router.post("/v1/node-helpers/poll")
async def poll(request: Request, body: HelperReport, actor: NodeActor) -> dict[str, Any]:
    node = _node(request, actor)
    config = helpers.configuration(request.app.state.machine.state, node)
    ident = await helpers.broker(request).poll(config, body.model_dump(mode="json"))
    # A setting may have changed during the long poll.
    current = helpers.configuration(request.app.state.machine.state, node)
    return {"configuration": current, "operation": ident}


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
            }
        )
    return {"grants": grants}


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
    return await helpers.broker(request).submit(
        config, check, write=tool == "write_text", operation_id=operation_id
    )


def _operation_id(token: str, ident: str) -> str:
    return hashlib.sha256((token + "\0" + ident).encode()).hexdigest()


@router.post("/oidc/node-helpers/cancel", status_code=204)
async def cancel(request: Request, body: HelperCancel) -> None:
    _app_subject(request, body.refreshToken)
    helpers.broker(request).cancel(_operation_id(body.refreshToken, body.operationId))
