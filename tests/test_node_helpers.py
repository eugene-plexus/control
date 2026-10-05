"""Real enrollment, sign-in, replication and at-most-once helper delivery."""

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

REPORT = {"supported": True, "ready": True, "account": "isolated-file-helper"}


@pytest.fixture(autouse=True)
def quick_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(node_helpers, "POLL_SECONDS", 0.03)


PRODUCTION = {"mode": "production", "changedAt": None}


def agent(keys: NodeKeys) -> dict[str, str]:
    return {"Authorization": "Bearer " + keys.service_token()}


def deliver(c: TestClient, keys: NodeKeys) -> tuple[str, dict[str, Any]]:
    for _ in range(100):
        answer = c.post("/v1/node-helpers/poll", headers=agent(keys), json=REPORT)
        assert answer.status_code == 200, answer.text
        ident = answer.json()["operation"]
        if ident:
            claim = c.post(f"/v1/node-helpers/operations/{ident}/claim", headers=agent(keys))
            assert claim.status_code == 200, claim.text
            return ident, claim.json()
    pytest.fail("no file operation was delivered")


def setup(c: TestClient) -> tuple[NodeKeys, dict[str, Any], dict[str, Any]]:
    keys, enrolled = enroll(c, "desktop")
    assert enrolled.status_code == 201
    assert c.put("/v1/node-helpers/desktop", json={"enabled": True}).status_code == 200
    assert c.post("/v1/node-helpers/poll", headers=agent(keys), json=REPORT).status_code == 200
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(
            c.post,
            "/v1/node-helpers/desktop/folders",
            json={
                "name": "Notes",
                "path": "/srv/notes",
                "writable": True,
            },
        )
        ident, command = deliver(c, keys)
        assert command["tool"] == "inspect" and command["subject"] == "operator"
        assert (
            c.post(
                f"/v1/node-helpers/operations/{ident}/result",
                headers=agent(keys),
                json={
                    "status": "done",
                    "result": {"path": "/srv/notes", "identity": "device:inode:birth"},
                },
            ).status_code
            == 204
        )
        created = pending.result(timeout=5)
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


def test_configuration_requires_operator_and_survives_snapshot(active_client: TestClient) -> None:
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


def test_discovery_is_person_specific_and_never_returns_host_paths(
    active_client: TestClient,
) -> None:
    c = active_client
    _, folder, app = setup(c)
    owner = _tokens(c, app)
    assert c.post(
        "/oidc/node-helpers/folders", auth=auth(app), json={"refreshToken": owner["refresh_token"]}
    ).json() == {"grants": [], "installMode": PRODUCTION}
    _, issued = person(c, app, folder)
    data = c.post(
        "/oidc/node-helpers/folders", auth=auth(app), json={"refreshToken": issued["refresh_token"]}
    )
    assert data.status_code == 200, data.text
    assert data.json()["grants"][0]["node"] == "desktop"
    assert data.json()["grants"][0]["writable"] is False
    assert "path" not in data.text and "identity" not in data.text
    other = c.post("/v1/people", json={"name": "bo", "password": PASSPHRASE}).json()
    bo = _tokens(c, app, name=other["name"])
    assert c.post(
        "/oidc/node-helpers/folders", auth=auth(app), json={"refreshToken": bo["refresh_token"]}
    ).json() == {"grants": [], "installMode": PRODUCTION}


@pytest.mark.parametrize("kind", ["id_token", "access_token", "wrong-client", "non-workbench"])
def test_only_the_apps_own_person_credential_can_use_helpers(
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
        "/oidc/node-helpers/folders", auth=auth(app), json={"refreshToken": credential}
    ).status_code in {401, 403}


def test_read_only_write_and_cross_person_calls_are_refused(active_client: TestClient) -> None:
    c = active_client
    _, folder, app = setup(c)
    member, issued = person(c, app, folder)
    body = {
        "refreshToken": issued["refresh_token"],
        "folderId": folder["id"],
        "tool": "write_text",
        "arguments": {"path": "a.txt", "text": "no", "expectedSha256": ""},
    }
    assert c.post("/oidc/node-helpers/execute", auth=auth(app), json=body).status_code == 403
    c.patch(f"/v1/people/{member['id']}", json={"helperGrants": []})
    body["tool"] = "read_text"
    body["arguments"] = {"path": "a.txt"}
    assert c.post("/oidc/node-helpers/execute", auth=auth(app), json=body).status_code == 403
    assert c.app.state.node_helper_broker.jobs == {}


def test_delivered_read_is_once_and_contents_never_enter_snapshot(
    active_client: TestClient,
) -> None:
    c = active_client
    keys, folder, app = setup(c)
    _, issued = person(c, app, folder)
    other, _ = enroll(c, "other")
    body = {
        "refreshToken": issued["refresh_token"],
        "folderId": folder["id"],
        "tool": "read_text",
        "arguments": {"path": "a.txt"},
    }
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(c.post, "/oidc/node-helpers/execute", auth=auth(app), json=body)
        ident, command = deliver(c, keys)
        assert command["folder"]["subject"] == command["subject"]
        assert issued["refresh_token"] not in json.dumps(command)
        assert (
            c.post(f"/v1/node-helpers/operations/{ident}/claim", headers=agent(keys)).status_code
            == 409
        )
        result = {"status": "done", "result": {"text": "PRIVATE-FILE-CONTENT"}}
        assert (
            c.post(
                f"/v1/node-helpers/operations/{ident}/result", headers=agent(other), json=result
            ).status_code
            == 409
        )
        assert (
            c.post(
                f"/v1/node-helpers/operations/{ident}/result", headers=agent(keys), json=result
            ).status_code
            == 204
        )
        assert pending.result(timeout=5).json() == {
            **result,
            "jobSite": False,
            "installMode": "production",
        }
        assert (
            c.post(
                f"/v1/node-helpers/operations/{ident}/result", headers=agent(keys), json=result
            ).status_code
            == 409
        )
    assert b"PRIVATE-FILE-CONTENT" not in applied.canonical_bytes(c.app.state.machine.state)
    assert not c.app.state.node_helper_broker.jobs


@pytest.mark.parametrize(
    "revoke", ["person", "grant", "helper", "folder", "app", "password", "sign-in"]
)
def test_queued_calls_recheck_current_access_before_execution(
    active_client: TestClient, revoke: str
) -> None:
    c = active_client
    keys, folder, app = setup(c)
    member, issued = person(c, app, folder, writable=True)
    body = {
        "refreshToken": issued["refresh_token"],
        "folderId": folder["id"],
        "tool": "write_text",
        "arguments": {"path": "a.txt", "text": "no", "expectedSha256": ""},
    }
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(c.post, "/oidc/node-helpers/execute", auth=auth(app), json=body)
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
    assert c.post(
        "/oidc/node-helpers/folders", auth=auth(app), json={"refreshToken": issued["refresh_token"]}
    ).json() == {"grants": [], "installMode": PRODUCTION}


@pytest.mark.parametrize("stage", ["before-submit", "queued", "running"])
def test_cancellation_is_signin_bound_and_prevents_replay(
    active_client: TestClient, stage: str
) -> None:
    c = active_client
    keys, folder, app = setup(c)
    _, issued = person(c, app, folder, writable=True)
    _, other = person(c, app, folder, name="bo", writable=True)
    body = {
        "refreshToken": issued["refresh_token"],
        "operationId": "one-cancellable-operation",
        "folderId": folder["id"],
        "tool": "write_text",
        "arguments": {"path": "new.txt", "text": "Hello", "expectedSha256": ""},
    }
    cancellation = {k: body[k] for k in ("refreshToken", "operationId")}
    if stage == "before-submit":
        assert (
            c.post("/oidc/node-helpers/cancel", auth=auth(app), json=cancellation).status_code
            == 204
        )
    else:
        with ThreadPoolExecutor() as pool:
            pending = pool.submit(c.post, "/oidc/node-helpers/execute", auth=auth(app), json=body)
            for _ in range(100):
                ident = c.post("/v1/node-helpers/poll", headers=agent(keys), json=REPORT).json()[
                    "operation"
                ]
                if ident:
                    break
            assert ident
            assert (
                c.post(
                    "/oidc/node-helpers/cancel",
                    auth=auth(app),
                    json={**cancellation, "refreshToken": other["refresh_token"]},
                ).status_code
                == 204
            )
            assert not pending.done()  # Another user's matching client ID cannot cancel ours.
            if stage == "running":
                assert (
                    c.post(
                        f"/v1/node-helpers/operations/{ident}/claim", headers=agent(keys)
                    ).status_code
                    == 200
                )
            assert (
                c.post("/oidc/node-helpers/cancel", auth=auth(app), json=cancellation).status_code
                == 204
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
    assert c.post("/oidc/node-helpers/execute", auth=auth(app), json=body).status_code == 409


@pytest.mark.parametrize("write", [False, True])
async def test_timeout_never_requeues_claimed_work(
    monkeypatch: pytest.MonkeyPatch, write: bool
) -> None:
    from fastapi import HTTPException

    monkeypatch.setattr(node_helpers, "JOB_SECONDS", 0.05)
    broker = node_helpers.Broker()
    config = {"node": "desktop", "nodeKey": "key", "enrolledAt": "now", "enabled": True}
    await broker.poll(config, REPORT)
    pending = asyncio.create_task(
        broker.submit(config, lambda: {"tool": "write_text" if write else "read_text"}, write=write)
    )
    ident = await broker.poll(config, REPORT)
    assert ident
    broker.claim("desktop", ident)
    if write:
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

    def discover() -> list[dict[str, Any]]:
        return c.post(
            "/oidc/node-helpers/folders",
            auth=auth(app),
            json={"refreshToken": issued["refresh_token"]},
        ).json()["grants"]

    assert discover() == []
    assert (
        c.patch(
            f"/v1/node-helpers/desktop/folders/{folder['id']}", json={"ownerAccess": "read"}
        ).status_code
        == 200
    )
    assert discover()[0]["writable"] is False
    c.app.state.node_helper_broker.reports.clear()
    assert discover()[0]["available"] is False
    assert "offline" in discover()[0]["reason"]


@pytest.mark.parametrize("write", [False, True])
def test_withdrawn_access_does_not_return_inflight_read_data(
    active_client: TestClient, write: bool
) -> None:
    c = active_client
    keys, folder, app = setup(c)
    member, issued = person(c, app, folder, writable=True)
    body = {
        "refreshToken": issued["refresh_token"],
        "folderId": folder["id"],
        "tool": "write_text" if write else "read_text",
        "arguments": {"path": "note.txt"},
    }
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(c.post, "/oidc/node-helpers/execute", auth=auth(app), json=body)
        ident, _ = deliver(c, keys)
        c.patch(f"/v1/people/{member['id']}", json={"helperGrants": []})
        assert (
            c.post(
                f"/v1/node-helpers/operations/{ident}/result",
                headers=agent(keys),
                json={"status": "done", "result": {"text": "PRIVATE"}},
            ).status_code
            == 204
        )
        response = pending.result(timeout=5)
        assert "PRIVATE" not in response.text
        assert response.status_code == (200 if write else 403)
        if write:
            assert response.json()["status"] == "uncertain"
