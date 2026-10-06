"""Real enrollment, sign-in, replication and at-most-once delivery of MCP to an
ordinary node, where the root's grants are final (Job Sites J6, J6d, J6g)."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_control import applied, node_helpers

from .conftest import PASSPHRASE, NodeKeys, enroll
from .test_oidc import CALLBACK, _tokens

REPORT = {
    "supported": True,
    "ready": True,
    "account": "isolated-file-helper",
    "protocol": "mcp-2026-07-28",
}


@pytest.fixture(autouse=True)
def quick_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(node_helpers, "POLL_SECONDS", 0.03)


PRODUCTION = {"mode": "production", "changedAt": None}


def agent(keys: NodeKeys) -> dict[str, str]:
    return {"Authorization": "Bearer " + keys.service_token()}


def rpc(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": 7,
        "method": method,
        "params": {
            **(params or {}),
            "_meta": {"io.modelcontextprotocol/protocolVersion": "2026-07-28"},
        },
    }


def call(tool: str, folder: str = "Notes", **arguments: Any) -> dict[str, Any]:
    return rpc("tools/call", {"name": tool, "arguments": {"folder": folder, **arguments}})


def deliver(
    c: TestClient, keys: NodeKeys, report: dict[str, Any] | None = None
) -> tuple[str, dict[str, Any]]:
    for _ in range(100):
        answer = c.post("/v1/node-helpers/poll", headers=agent(keys), json=report or REPORT)
        assert answer.status_code == 200, answer.text
        ident = answer.json()["operation"]
        if ident:
            claim = c.post(f"/v1/node-helpers/operations/{ident}/claim", headers=agent(keys))
            assert claim.status_code == 200, claim.text
            return ident, claim.json()
    pytest.fail("no operation was delivered")


def register(
    c: TestClient, keys: NodeKeys, name: str, path: str, identity: str, **extra: Any
) -> Any:
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(
            c.post,
            "/v1/node-helpers/desktop/folders",
            json={"name": name, "path": path, **extra},
        )
        ident, command = deliver(c, keys)
        assert command["kind"] == "manage" and command["action"] == "folder.inspect"
        assert command["subject"] == "operator" and command["arguments"] == {"path": path}
        assert (
            c.post(
                f"/v1/node-helpers/operations/{ident}/result",
                headers=agent(keys),
                json={"status": "done", "result": {"path": path, "identity": identity}},
            ).status_code
            == 204
        )
        return pending.result(timeout=5)


def setup(c: TestClient) -> tuple[NodeKeys, dict[str, Any], dict[str, Any]]:
    keys, enrolled = enroll(c, "desktop")
    assert enrolled.status_code == 201
    assert c.put("/v1/node-helpers/desktop", json={"enabled": True}).status_code == 200
    assert c.post("/v1/node-helpers/poll", headers=agent(keys), json=REPORT).status_code == 200
    created = register(c, keys, "Notes", "/srv/notes", "device:inode:birth", writable=True)
    assert created.status_code == 201, created.text
    app = c.post(
        "/v1/oidc/clients",
        json={"name": "Workbench", "owner": "app:workbench@desktop", "redirectUris": [CALLBACK]},
    ).json()
    return keys, created.json(), app


def person(
    c: TestClient,
    app: dict[str, Any],
    folder: dict[str, Any],
    *,
    name: str = "ada",
    writable: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    made = c.post(
        "/v1/people",
        json={
            "name": name,
            "password": PASSPHRASE,
            "apps": [app["client"]["clientId"]],
            "helperGrants": [{"folderId": folder["id"], "writable": writable}],
        },
    )
    assert made.status_code == 201, made.text
    return made.json(), _tokens(c, app, name=name)


def auth(app: dict[str, Any]) -> tuple[str, str]:
    return app["client"]["clientId"], app["clientSecret"]


def servers(c: TestClient, app: dict[str, Any], token: str) -> dict[str, Any]:
    answer = c.post("/oidc/sites/servers", auth=auth(app), json={"refreshToken": token})
    assert answer.status_code == 200, answer.text
    return dict(answer.json())


def mcp(token: str, request: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "refreshToken": token,
        "node": "desktop",
        "server": "files",
        "request": request,
        **extra,
    }


def test_configuration_needs_the_capability_and_survives_snapshot(
    active_client: TestClient,
) -> None:
    c = active_client
    keys, folder, app = setup(c)
    member, issued = person(c, app, folder)
    assert c.get("/v1/node-helpers", headers=agent(keys)).status_code == 401
    assert (
        c.put("/v1/node-helpers/desktop", headers=agent(keys), json={"enabled": False}).status_code
        == 401
    )
    assert (
        c.get(
            "/v1/node-helpers", headers={"Authorization": "Bearer " + issued["access_token"]}
        ).status_code
        == 401
    )
    raw = applied.to_canonical(c.app.state.machine.state)
    assert applied.to_canonical(applied.from_canonical(raw)) == raw
    assert raw["people"][0]["helperGrants"] == member["helperGrants"]
    assert raw["nodeHelpers"][0]["folders"][0] == folder
    assert "devGrants" not in raw["nodeHelpers"][0]


def test_an_old_log_entry_with_folder_people_still_replays(active_client: TestClient) -> None:
    """Slice 1 kept a Job Site's people on the root's folder record; slice 2
    reads them nowhere, and a log holding them must still replay."""
    c = active_client
    _, _folder, _ = setup(c)
    raw = applied.to_canonical(c.app.state.machine.state)
    raw["nodeHelpers"][0]["folders"][0]["people"] = [{"person": "p-old", "writable": False}]
    again = applied.from_canonical(raw)
    assert again.node_helpers["desktop"]["folders"][0]["people"][0]["person"] == "p-old"
    assert node_helpers.node_grants(again, "p-old", "desktop") == []


def test_the_servers_a_person_may_use_and_never_a_host_path(active_client: TestClient) -> None:
    c = active_client
    _, folder, app = setup(c)
    owner = _tokens(c, app)
    assert servers(c, app, owner["refresh_token"]) == {"servers": [], "installMode": PRODUCTION}
    _, issued = person(c, app, folder)
    listed = servers(c, app, issued["refresh_token"])["servers"]
    assert listed == [
        {
            "node": "desktop",
            "jobSite": False,
            "available": True,
            "reason": None,
            "server": "files",
            "name": "Files",
            "kind": "files",
            "folders": [{"id": folder["id"], "name": "Notes", "writable": False}],
        }
    ]
    other = c.post("/v1/people", json={"name": "bo", "password": PASSPHRASE}).json()
    bo = _tokens(c, app, name=other["name"])
    text = json.dumps(servers(c, app, bo["refresh_token"]))
    assert text == json.dumps({"servers": [], "installMode": PRODUCTION})
    assert "/srv/notes" not in json.dumps(listed) and "device:inode" not in json.dumps(listed)


@pytest.mark.parametrize("kind", ["id_token", "access_token", "wrong-client", "non-workbench"])
def test_only_the_apps_own_person_credential_can_use_machines(
    active_client: TestClient, kind: str
) -> None:
    c = active_client
    _, folder, app = setup(c)
    _, issued = person(c, app, folder)
    credential = issued["refresh_token"]
    if kind in {"id_token", "access_token"}:
        credential = issued[kind]
    else:
        app = c.post(
            "/v1/oidc/clients",
            json={
                "name": "Other",
                "redirectUris": [CALLBACK],
                **({"owner": "app:workbench@other"} if kind == "wrong-client" else {}),
            },
        ).json()
    assert c.post(
        "/oidc/sites/servers", auth=auth(app), json={"refreshToken": credential}
    ).status_code in {401, 403}


def test_a_folder_the_root_did_not_give_never_reaches_the_node(active_client: TestClient) -> None:
    c = active_client
    _keys, folder, app = setup(c)
    member, issued = person(c, app, folder)
    token = issued["refresh_token"]
    write = call("write_text", path="a.txt", text="no", expectedSha256="")
    assert c.post("/oidc/sites/mcp", auth=auth(app), json=mcp(token, write)).status_code == 403
    elsewhere = call("read_text", folder="Elsewhere", path="a.txt")
    assert c.post("/oidc/sites/mcp", auth=auth(app), json=mcp(token, elsewhere)).status_code == 403
    local = mcp(token, rpc("tools/list"), server="fixture")
    assert c.post("/oidc/sites/mcp", auth=auth(app), json=local).status_code == 403
    c.patch(f"/v1/people/{member['id']}", json={"helperGrants": []})
    read = call("read_text", path="a.txt")
    assert c.post("/oidc/sites/mcp", auth=auth(app), json=mcp(token, read)).status_code == 403
    assert c.app.state.node_helper_broker.jobs == {}


def test_a_call_carries_the_persons_grants_once_and_contents_never_persist(
    active_client: TestClient,
) -> None:
    c = active_client
    keys, folder, app = setup(c)
    _, issued = person(c, app, folder)
    other, _ = enroll(c, "other")
    body = mcp(issued["refresh_token"], call("read_text", path="a.txt"))
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(c.post, "/oidc/sites/mcp", auth=auth(app), json=body)
        ident, command = deliver(c, keys)
        assert command["kind"] == "mcp" and command["server"] == "files"
        assert command["request"]["method"] == "tools/call"
        assert command["grants"] == [
            {
                "folderId": folder["id"],
                "name": "Notes",
                "path": "/srv/notes",
                "identity": "device:inode:birth",
                "writable": False,
            }
        ]
        assert command["installMode"] == "production"
        assert issued["refresh_token"] not in json.dumps(command)
        assert (
            c.post(f"/v1/node-helpers/operations/{ident}/claim", headers=agent(keys)).status_code
            == 409
        )
        answer = {
            "status": "done",
            "response": {"jsonrpc": "2.0", "id": 7, "result": {"text": "PRIVATE-FILE-CONTENT"}},
        }
        assert (
            c.post(
                f"/v1/node-helpers/operations/{ident}/result", headers=agent(other), json=answer
            ).status_code
            == 409
        )
        assert (
            c.post(
                f"/v1/node-helpers/operations/{ident}/result", headers=agent(keys), json=answer
            ).status_code
            == 204
        )
        assert pending.result(timeout=5).json() == {
            **answer,
            "jobSite": False,
            "installMode": "production",
        }
        assert (
            c.post(
                f"/v1/node-helpers/operations/{ident}/result", headers=agent(keys), json=answer
            ).status_code
            == 409
        )
    assert b"PRIVATE-FILE-CONTENT" not in applied.canonical_bytes(c.app.state.machine.state)
    assert not c.app.state.node_helper_broker.jobs


def test_a_machine_that_speaks_no_mcp_is_told_to_update(active_client: TestClient) -> None:
    c = active_client
    keys, folder, app = setup(c)
    _, issued = person(c, app, folder)
    old = {k: v for k, v in REPORT.items() if k != "protocol"}
    assert c.post("/v1/node-helpers/poll", headers=agent(keys), json=old).status_code == 200
    listed = servers(c, app, issued["refresh_token"])["servers"][0]
    assert listed["available"] is False and "older Eugene" in listed["reason"]
    body = mcp(issued["refresh_token"], call("read_text", path="a.txt"))
    refused = c.post("/oidc/sites/mcp", auth=auth(app), json=body)
    assert refused.status_code == 503 and "Update Eugene" in refused.text
    added = c.post("/v1/node-helpers/desktop/folders", json={"name": "More", "path": "/srv/more"})
    assert added.status_code == 503
    assert not c.app.state.node_helper_broker.jobs


def test_a_folder_name_is_unique_on_its_node_and_an_old_duplicate_reads_name_2(
    active_client: TestClient,
) -> None:
    c = active_client
    keys, folder, app = setup(c)
    taken = c.post("/v1/node-helpers/desktop/folders", json={"name": "notes", "path": "/x"})
    assert taken.status_code == 409 and not c.app.state.node_helper_broker.jobs
    second = register(c, keys, "Drafts", "/srv/drafts", "device:inode:2")
    assert second.status_code == 201, second.text
    machine = c.app.state.machine
    config = node_helpers.configuration(machine.state, "desktop")
    renamed = [
        {**f, "name": "Notes"} if f["id"] == second.json()["id"] else f for f in config["folders"]
    ]
    machine.append(applied.OP_PUT_NODE_HELPER, {"helper": {**config, "folders": renamed}})
    made = c.post(
        "/v1/people",
        json={
            "name": "ada",
            "password": PASSPHRASE,
            "apps": [app["client"]["clientId"]],
            "helperGrants": [
                {"folderId": folder["id"], "writable": False},
                {"folderId": second.json()["id"], "writable": False},
            ],
        },
    )
    assert made.status_code == 201, made.text
    token = _tokens(c, app, name="ada")["refresh_token"]
    names = [f["name"] for f in servers(c, app, token)["servers"][0]["folders"]]
    assert names == ["Notes", "Notes (2)"]


@pytest.mark.parametrize(
    "revoke", ["person", "grant", "helper", "folder", "app", "password", "sign-in"]
)
def test_queued_calls_recheck_current_access_before_execution(
    active_client: TestClient, revoke: str
) -> None:
    c = active_client
    keys, folder, app = setup(c)
    member, issued = person(c, app, folder, writable=True)
    body = mcp(
        issued["refresh_token"], call("write_text", path="a.txt", text="no", expectedSha256="")
    )
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(c.post, "/oidc/sites/mcp", auth=auth(app), json=body)
        ident = None
        for _ in range(100):
            ident = c.post("/v1/node-helpers/poll", headers=agent(keys), json=REPORT).json()[
                "operation"
            ]
            if ident:
                break
        assert ident
        if revoke == "person":
            c.patch(f"/v1/people/{member['id']}", json={"disabled": True})
        if revoke == "grant":
            c.patch(f"/v1/people/{member['id']}", json={"helperGrants": []})
        if revoke == "helper":
            c.put("/v1/node-helpers/desktop", json={"enabled": False})
        if revoke == "folder":
            c.delete(f"/v1/node-helpers/desktop/folders/{folder['id']}")
        if revoke == "app":
            c.patch(f"/v1/people/{member['id']}", json={"apps": []})
        if revoke == "password":
            c.put(f"/v1/people/{member['id']}/password", json={"password": "a-different-password"})
        if revoke == "sign-in":
            c.post("/oidc/revoke", auth=auth(app), data={"token": issued["refresh_token"]})
        assert (
            c.post(f"/v1/node-helpers/operations/{ident}/claim", headers=agent(keys)).status_code
            == 403
        )
        assert pending.result(timeout=5).status_code == 403


def test_reusing_a_node_name_cannot_inherit_file_access(active_client: TestClient) -> None:
    c = active_client
    keys, folder, app = setup(c)
    _, issued = person(c, app, folder)
    assert c.delete("/v1/nodes/desktop").status_code == 202
    _, replacement = enroll(c, "desktop")
    assert replacement.status_code == 201
    helper = c.get("/v1/node-helpers").json()["helpers"][0]
    assert helper["enabled"] is False and helper["folders"] == []
    assert c.post("/v1/node-helpers/poll", headers=agent(keys), json=REPORT).status_code == 401
    assert servers(c, app, issued["refresh_token"]) == {"servers": [], "installMode": PRODUCTION}


@pytest.mark.parametrize("stage", ["before-submit", "queued", "running"])
def test_cancellation_is_signin_bound_and_prevents_replay(
    active_client: TestClient, stage: str
) -> None:
    c = active_client
    keys, folder, app = setup(c)
    _, issued = person(c, app, folder, writable=True)
    _, other = person(c, app, folder, name="bo", writable=True)
    body = mcp(
        issued["refresh_token"],
        call("write_text", path="new.txt", text="Hello", expectedSha256=""),
        operationId="one-cancellable-operation",
    )
    cancellation = {k: body[k] for k in ("refreshToken", "operationId")}
    if stage == "before-submit":
        assert c.post("/oidc/sites/cancel", auth=auth(app), json=cancellation).status_code == 204
    else:
        with ThreadPoolExecutor() as pool:
            pending = pool.submit(c.post, "/oidc/sites/mcp", auth=auth(app), json=body)
            for _ in range(100):
                ident = c.post("/v1/node-helpers/poll", headers=agent(keys), json=REPORT).json()[
                    "operation"
                ]
                if ident:
                    break
            assert ident
            assert (
                c.post(
                    "/oidc/sites/cancel",
                    auth=auth(app),
                    json={**cancellation, "refreshToken": other["refresh_token"]},
                ).status_code
                == 204
            )
            assert not pending.done()  # Another person's matching id cannot cancel ours.
            if stage == "running":
                assert (
                    c.post(
                        f"/v1/node-helpers/operations/{ident}/claim", headers=agent(keys)
                    ).status_code
                    == 200
                )
            assert (
                c.post("/oidc/sites/cancel", auth=auth(app), json=cancellation).status_code == 204
            )
            result = pending.result(timeout=5)
            assert result.status_code == 200
            assert result.json()["status"] == ("uncertain" if stage == "running" else "failed")
            assert (
                c.post(
                    f"/v1/node-helpers/operations/{ident}/claim", headers=agent(keys)
                ).status_code
                == 409
            )
    assert c.post("/oidc/sites/mcp", auth=auth(app), json=body).status_code == 409


@pytest.mark.parametrize("acting", [False, True])
async def test_timeout_never_requeues_claimed_work(
    monkeypatch: pytest.MonkeyPatch, acting: bool
) -> None:
    from fastapi import HTTPException

    monkeypatch.setattr(node_helpers, "JOB_SECONDS", 0.05)
    broker = node_helpers.Broker()
    config = {"node": "desktop", "nodeKey": "key", "enrolledAt": "now", "enabled": True}
    await broker.poll(config, REPORT)
    pending = asyncio.create_task(broker.submit(config, lambda: {"kind": "mcp"}, write=acting))
    ident = await broker.poll(config, REPORT)
    assert ident
    broker.claim("desktop", ident)
    if acting:
        assert (await pending)["status"] == "uncertain"
    else:
        with pytest.raises(HTTPException) as error:
            await pending
        assert error.value.status_code == 504
    assert not broker.jobs
    with pytest.raises(HTTPException):
        broker.claim("desktop", ident)


def test_owner_access_is_explicit_and_offline_is_inline_metadata(active_client: TestClient) -> None:
    c = active_client
    _, folder, app = setup(c)
    issued = _tokens(c, app)

    def listed() -> list[dict[str, Any]]:
        return list(servers(c, app, issued["refresh_token"])["servers"])

    assert listed() == []
    assert (
        c.patch(
            f"/v1/node-helpers/desktop/folders/{folder['id']}", json={"ownerAccess": "read"}
        ).status_code
        == 200
    )
    assert listed()[0]["folders"][0]["writable"] is False
    c.app.state.node_helper_broker.reports.clear()
    assert listed()[0]["available"] is False
    assert "offline" in listed()[0]["reason"]


@pytest.mark.parametrize("acting", [False, True])
def test_withdrawn_access_does_not_return_inflight_results(
    active_client: TestClient, acting: bool
) -> None:
    c = active_client
    keys, folder, app = setup(c)
    member, issued = person(c, app, folder, writable=True)
    request = (
        call("write_text", path="note.txt", text="x", expectedSha256="")
        if acting
        else rpc("tools/list")
    )
    body = mcp(issued["refresh_token"], request)
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(c.post, "/oidc/sites/mcp", auth=auth(app), json=body)
        ident, _ = deliver(c, keys)
        c.patch(f"/v1/people/{member['id']}", json={"helperGrants": []})
        assert (
            c.post(
                f"/v1/node-helpers/operations/{ident}/result",
                headers=agent(keys),
                json={
                    "status": "done",
                    "response": {"jsonrpc": "2.0", "id": 7, "result": {"x": "PRIVATE"}},
                },
            ).status_code
            == 204
        )
        response = pending.result(timeout=5)
        assert "PRIVATE" not in response.text
        assert response.status_code == (200 if acting else 403)
        if acting:
            assert response.json()["status"] == "uncertain"
