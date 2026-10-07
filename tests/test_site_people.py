"""2b.3b at the root: per-person permissions (J77), linked people's own
workspaces and keys (J67, J68), names that match the site's (J76), and
`asked` carried as is (J72). `job-sites-own-enrollment.md` §3.3."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_control.settings import Settings

from .conftest import PASSPHRASE
from .test_oidc import _request_id
from .test_site_links import _link_authorize
from .test_sites import (
    NODES_ORIGIN,
    Ada,
    _root,
    ada_with_a_folder,
    add_person,
    answer,
    auth,
    deliver,
    manage_through,
    quick_poll,  # noqa: F401  (autouse fixture)
    rpc,
)


@pytest.fixture
def root(tmp_path: Path) -> Iterator[TestClient]:
    yield from _root(
        Settings(
            config_file=tmp_path / "control.yaml",
            state_dir=tmp_path / "state",
            nodes_origin=NODES_ORIGIN,
        )
    )


SIGNING = {"state": "signed", "held": 0, "passkeys": True, "people": True}
NOTES = "a" * 32
MINE = "b" * 32


def people_site(c: TestClient) -> tuple[Ada, dict[str, Any], str]:
    """Ada's site reporting as a 2b.3b site: her `Notes` workspace shared with
    Bo to read, and Bo linked with a workspace of his own, `Mine`."""
    a = ada_with_a_folder(c)
    bo, bo_token = add_person(c, a.app, "bo")
    a.held.summary.pop("folders")
    a.held.summary["workspaces"] = [
        {
            "id": NOTES,
            "name": "Notes",
            "holder": a.person["id"],
            "writable": True,
            "rules": {"read": "allow", "change": "ask"},
            "people": [{"subject": bo["id"], "read": "allow", "change": "deny"}],
        },
        {
            "id": MINE,
            "name": "Mine",
            "holder": bo["id"],
            "writable": True,
            "rules": {"read": "allow", "change": "ask"},
            "people": [],
        },
    ]
    a.held.summary["links"] = [
        {
            "subject": a.person["id"],
            "accountName": "HOST/ada",
            "available": True,
            "keys": 1,
            "signing": "signed",
            "held": 0,
        },
        {
            "subject": bo["id"],
            "accountName": "HOST/bo",
            "available": True,
            "keys": 1,
            "signing": "unconfirmed",
            "held": 2,
        },
    ]
    a.held.summary["signing"] = dict(SIGNING)
    c.post("/v1/sites/poll", headers=a.keys.bearer(a.site), json=a.held.report())
    return a, bo, bo_token


def files_folders(c: TestClient, a: Ada, token: str) -> list[dict[str, Any]]:
    listed = c.post("/oidc/sites/servers", auth=auth(a.app), json={"refreshToken": token})
    assert listed.status_code == 200, listed.text
    files = [s for s in listed.json()["servers"] if s["server"] == "files"]
    return files[0]["folders"] if files else []


def call(c: TestClient, a: Ada, token: str, folder: str, **extra: Any) -> Any:
    return c.post(
        "/oidc/sites/mcp",
        auth=auth(a.app),
        json={
            "refreshToken": token,
            "site": a.site,
            "server": "files",
            "request": rpc(
                "tools/call",
                {"name": "read_text", "arguments": {"folder": folder, "path": "a.txt"}},
            ),
            **extra,
        },
    )


def call_through(c: TestClient, a: Ada, token: str, folder: str, **extra: Any) -> dict[str, Any]:
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(call, c, a, token, folder, **extra)
        ident, command = deliver(c, a.keys, a.site, a.held.report())
        answer(
            c,
            a.keys,
            a.site,
            ident,
            {"status": "done", "response": {"jsonrpc": "2.0", "id": 3, "result": {}}},
        )
        assert pending.result(timeout=5).status_code == 200
    return command


# --- permissions (J77) ------------------------------------------------------------------


def test_permissions_follow_the_defaults_until_a_person_has_their_own(root: TestClient) -> None:
    c = root
    made = c.post("/v1/people", json={"name": "bo", "password": PASSPHRASE})
    assert made.status_code == 201, made.text
    bo = made.json()
    assert "permissions" not in bo and bo["permissionsInEffect"] == [
        "add-job-sites",
        "use-job-sites",
    ]
    # On the defaults the record carries no key: an older standby applies it.
    assert "permissions" not in c.app.state.machine.state.people[bo["id"]]
    assert c.patch("/v1/config", json={"peopleMayAddJobSites": False}).status_code == 200
    listed = {p["name"]: p for p in c.get("/v1/people").json()["people"]}
    assert listed["bo"]["permissionsInEffect"] == ["use-job-sites"]
    own = c.patch(f"/v1/people/{bo['id']}", json={"permissions": ["add-job-sites"]})
    assert own.status_code == 200 and own.json()["permissionsInEffect"] == ["add-job-sites"]
    assert own.json()["permissions"] == ["add-job-sites"]
    back = c.patch(f"/v1/people/{bo['id']}", json={"permissions": None})
    assert back.json()["permissionsInEffect"] == ["use-job-sites"]
    assert "permissions" not in c.app.state.machine.state.people[bo["id"]]
    unknown = c.patch(f"/v1/people/{bo['id']}", json={"permissions": ["run-everything"]})
    assert unknown.status_code == 422


def test_without_add_job_sites_a_person_adds_no_machine(root: TestClient) -> None:
    c = root
    a = ada_with_a_folder(c)
    assert c.patch("/v1/config", json={"siteJoinUrl": "http://192.168.1.5:8083"}).status_code == 200
    c.patch(f"/v1/people/{a.person['id']}", json={"permissions": ["use-job-sites"]})
    refused = c.post(
        "/oidc/job-sites/invite",
        auth=auth(a.app),
        json={"refreshToken": a.token, "label": "laptop"},
    )
    assert refused.status_code == 403 and "add job sites" in refused.text
    mine = c.post("/oidc/job-sites", auth=auth(a.app), json={"refreshToken": a.token})
    assert mine.json()["canInvite"] is False


def test_without_use_job_sites_a_person_links_nothing(root: TestClient) -> None:
    c = root
    a = ada_with_a_folder(c)
    bo, _ = add_person(c, a.app, "bo")
    c.patch(f"/v1/people/{bo['id']}", json={"permissions": []})
    page = _link_authorize(c)
    refused = c.post(
        "/oidc/authorize",
        data={"request": _request_id(page.text), "name": "bo", "password": PASSPHRASE},
        follow_redirects=False,
    )
    assert refused.status_code == 403 and "use job sites" in refused.text
    assert "location" not in refused.headers
    check = c.post(
        "/v1/sites/links/check",
        headers=a.keys.bearer(a.site),
        json={"name": "bo", "password": PASSPHRASE},
    )
    assert check.status_code == 403 and "use job sites" in check.text


def test_without_use_job_sites_ones_own_workspaces_are_left_out_and_refused(
    root: TestClient,
) -> None:
    c = root
    a, bo, bo_token = people_site(c)
    assert [(f["name"], f["mine"]) for f in files_folders(c, a, bo_token)] == [
        ("Mine", True),
        ("Notes", False),
    ]
    c.patch(f"/v1/people/{bo['id']}", json={"permissions": []})
    assert [(f["name"], f["mine"]) for f in files_folders(c, a, bo_token)] == [("Notes", False)]
    refused = call(c, a, bo_token, "Mine")
    assert refused.status_code == 403 and "use job sites" in refused.text
    # What the owner shared is the owner's grant (J77).
    assert call_through(c, a, bo_token, "Notes")["subject"] == bo["id"]
    added = c.post(
        f"/oidc/job-sites/{a.site}/workspaces",
        auth=auth(a.app),
        json={"refreshToken": bo_token, "name": "X", "path": "C:/X"},
    )
    assert added.status_code == 403
    # Taking access away needs no permission: the site removes it.
    _, removed = manage_through(
        c,
        a.keys,
        a.site,
        a.held,
        lambda: c.post(
            f"/oidc/job-sites/{a.site}/workspaces/{MINE}/remove",
            auth=auth(a.app),
            json={"refreshToken": bo_token},
        ),
        lambda _: {"status": "done", "result": {"id": MINE}},
        action="workspace.remove",
    )
    assert removed.status_code == 204, removed.text


# --- a linked person's own (J67, J68, J76) -----------------------------------------------


def test_a_linked_person_adds_a_workspace_held_for_their_own_key(root: TestClient) -> None:
    c = root
    a, bo, bo_token = people_site(c)
    command, held = manage_through(
        c,
        a.keys,
        a.site,
        a.held,
        lambda: c.post(
            f"/oidc/job-sites/{a.site}/workspaces",
            auth=auth(a.app),
            json={
                "refreshToken": bo_token,
                "name": "Code",
                "path": "D:/code",
                "rules": {"read": "allow", "change": "ask"},
                "deny": [".env"],
            },
        ),
        lambda _: {"status": "held", "message": "Waiting for your own key."},
        action="workspace.add",
    )
    assert command["subject"] == bo["id"]
    assert command["arguments"] == {
        "name": "Code",
        "path": "D:/code",
        "writable": True,
        "rules": {"read": "allow", "change": "ask"},
        "deny": [".env"],
    }
    assert held.status_code == 202 and held.json()["held"] is True


def test_a_linked_person_sees_only_their_own_and_the_owner_sees_none_of_it(
    root: TestClient,
) -> None:
    c = root
    a, bo, bo_token = people_site(c)
    bos = c.post("/oidc/job-sites", auth=auth(a.app), json={"refreshToken": bo_token}).json()
    [site] = bos["sites"]
    assert site["role"] == "linked" and site["folders"] == [] and site["servers"] == []
    assert [w["name"] for w in site["workspaces"]] == ["Mine"]
    assert [link["subject"] for link in site["links"]] == [bo["id"]]
    assert site["signing"]["state"] == "unconfirmed" and site["signing"]["held"] == 2
    adas = c.post("/oidc/job-sites", auth=auth(a.app), json={"refreshToken": a.token}).json()
    [own] = adas["sites"]
    assert own["role"] == "owner" and [w["name"] for w in own["workspaces"]] == ["Notes"]
    assert own["workspaces"][0]["people"] == [
        {"person": bo["id"], "name": "bo", "read": "allow", "change": "deny"}
    ]
    assert [f["name"] for f in own["folders"]] == ["Notes"] and own["folders"][0]["path"] is None
    # Bo's workspace reaches nobody else, and Ada's list never names it.
    assert "Mine" not in str(adas)


def test_workspaces_are_listed_live_and_their_paths_kept_nowhere(root: TestClient) -> None:
    c = root
    a, bo, bo_token = people_site(c)
    detail = {
        "id": MINE,
        "name": "Mine",
        "path": "D:/bo/mine",
        "holder": bo["id"],
        "writable": True,
        "rules": {"read": "allow", "change": "ask"},
        "deny": [".env"],
        "people": [],
    }
    _, listed = manage_through(
        c,
        a.keys,
        a.site,
        a.held,
        lambda: c.post(
            f"/oidc/job-sites/{a.site}/workspaces/list",
            auth=auth(a.app),
            json={"refreshToken": bo_token},
        ),
        lambda _: {"status": "done", "result": {"workspaces": [detail]}},
        action="workspace.list",
    )
    assert listed.status_code == 200, listed.text
    assert listed.json()["workspaces"][0]["path"] == "D:/bo/mine"
    assert "D:/bo/mine" not in str(c.app.state.machine.state)


def test_only_the_owner_shares_and_nobody_unlinked_reaches_a_site(root: TestClient) -> None:
    c = root
    a, _bo, bo_token = people_site(c)
    share = {
        "refreshToken": bo_token,
        "people": [{"name": "ada", "read": "allow", "change": "deny"}],
    }
    assert (
        c.post(
            f"/oidc/job-sites/{a.site}/workspaces/{MINE}/people", auth=auth(a.app), json=share
        ).status_code
        == 404
    )
    _, cy_token = add_person(c, a.app, "cy")
    for path in ("workspaces/list", "held", "audit", f"workspaces/{MINE}/remove"):
        refused = c.post(
            f"/oidc/job-sites/{a.site}/{path}", auth=auth(a.app), json={"refreshToken": cy_token}
        )
        assert refused.status_code == 404, (path, refused.text)


def test_a_site_from_before_keeps_no_ones_workspaces_and_says_so(root: TestClient) -> None:
    c = root
    a, _bo, bo_token = people_site(c)
    a.held.summary["signing"] = {"state": "signed", "held": 0, "passkeys": True}
    c.post("/v1/sites/poll", headers=a.keys.bearer(a.site), json=a.held.report())
    refused = c.post(
        f"/oidc/job-sites/{a.site}/workspaces/list",
        auth=auth(a.app),
        json={"refreshToken": bo_token},
    )
    assert refused.status_code == 503 and "Update Eugene" in refused.text


# --- names, asked, dev mode (J72, J76) ----------------------------------------------------


def test_a_persons_view_is_named_as_the_site_names_it(root: TestClient) -> None:
    c = root
    a, _bo, bo_token = people_site(c)
    a.held.summary["workspaces"][1]["name"] = "notes"
    c.post("/v1/sites/poll", headers=a.keys.bearer(a.site), json=a.held.report())
    assert [(f["name"], f["mine"]) for f in files_folders(c, a, bo_token)] == [
        ("notes", True),
        ("Notes (2)", False),
    ]
    assert [f["writable"] for f in files_folders(c, a, bo_token)] == [True, False]


def test_asked_is_carried_to_the_site_as_is(root: TestClient) -> None:
    c = root
    a, _bo, bo_token = people_site(c)
    assert call_through(c, a, bo_token, "Mine", asked=True)["asked"] is True
    assert call_through(c, a, bo_token, "Mine")["asked"] is False


def test_a_dev_grant_names_a_workspace_by_id_and_carries_no_path(root: TestClient) -> None:
    c = root
    a, _bo, _bo_token = people_site(c)
    a.held.summary["ownerInDevMode"] = True
    c.post("/v1/sites/poll", headers=a.keys.bearer(a.site), json=a.held.report())
    assert c.patch("/v1/config", json={"installMode": "dev"}).status_code == 200
    granted = c.patch(f"/v1/sites/{a.site}/folders/{NOTES}", json={"ownerAccess": "read"})
    assert granted.status_code == 200, granted.text
    folders = granted.json()["dev"]["folders"]
    assert folders == [
        {"id": NOTES, "name": "Notes", "path": None, "writable": True, "ownerAccess": "read"}
    ]
    record = c.app.state.machine.state.sites[a.site]
    summary = c.app.state.site_broker.summary({"site": a.site, "enrolledAt": record.enrolledAt})
    from eugene_plexus_control import sites

    assert sites.dev_grants(c.app.state.machine.state, record, summary) == [
        {"folderId": NOTES, "name": "Notes", "writable": False}
    ]
    assert (
        c.patch(f"/v1/sites/{a.site}/folders/{MINE}", json={"ownerAccess": "read"}).status_code
        == 404
    )
