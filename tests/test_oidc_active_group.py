"""Row 37 (R37-FIRST): a person chooses which of their groups they work from."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from eugene_plexus_control.routes import oidc as oidc_routes
from eugene_plexus_control.routes.oidc import set_active_group

from .test_oidc import PASSPHRASE, _tokens
from .test_permission_groups import group
from .test_person_access import call, setup


@pytest.fixture
def no_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oidc_routes, "ACTIVE_GROUP_INTERVAL_SECONDS", 0.0)


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_a_person_chooses_their_active_group(active_client: TestClient, no_interval) -> None:
    c = active_client
    _, g, p, signed, key = setup(c)
    proof = signed["access_token"]
    sales = group(c, name="Sales", apps=None, policies={"roleRouting": {"chat": ["m1"]}})
    ops = group(c, name="Ops", apps=None, policies={"roleRouting": {"chat": ["m3"]}})
    c.patch(f"/v1/people/{p['id']}", json={"groups": [sales["id"], ops["id"], g["id"]]})
    read = c.get("/oidc/active-group", headers=_bearer(proof))
    assert read.status_code == 200, read.text
    assert read.json() == {
        "groups": [
            {"id": sales["id"], "name": "Sales"},
            {"id": ops["id"], "name": "Ops"},
            {"id": g["id"], "name": "Builders"},
        ],
        "activeGroup": None,
        "inEffect": sales["id"],
    }
    chose = c.post("/oidc/active-group", headers=_bearer(proof), json={"group": ops["id"]})
    assert chose.status_code == 200, chose.text
    assert chose.json()["activeGroup"] == chose.json()["inEffect"] == ops["id"]
    # The gateway sees it at the person's next request.
    assert call(c, key, proof).json()["roleRouting"] == {"chat": ["m3"]}
    listed = {x["id"]: x for x in c.get("/v1/people").json()["people"]}
    assert listed[p["id"]]["activeGroup"] == ops["id"]
    assert listed[p["id"]]["policiesInEffect"]["activeGroup"] == ops["id"]
    back = c.post("/oidc/active-group", headers=_bearer(proof), json={"group": None})
    assert back.json()["activeGroup"] is None and back.json()["inEffect"] == sales["id"]
    assert call(c, key, proof).json()["roleRouting"] == {"chat": ["m1"]}


def test_a_person_cannot_choose_a_group_they_are_not_in(active_client: TestClient) -> None:
    c = active_client
    _, _g, p, signed, _key = setup(c)
    proof = signed["access_token"]
    elsewhere = group(c, name="Elsewhere", apps=None, policies={"allowedModels": []})
    refused = c.post("/oidc/active-group", headers=_bearer(proof), json={"group": elsewhere["id"]})
    assert refused.status_code == 403, refused.text
    assert refused.json()["error"] == "not_a_member"
    assert "activeGroup" not in c.app.state.machine.state.people[p["id"]]
    for body in ({}, {"group": 5}, {"group": "x", "more": 1}, []):
        bad = c.post("/oidc/active-group", headers=_bearer(proof), json=body)
        assert bad.status_code == 400, body
    for token in ("not-a-token", signed["id_token"], signed["refresh_token"]):
        assert c.get("/oidc/active-group", headers=_bearer(token)).status_code == 401
        assert (
            c.post("/oidc/active-group", headers=_bearer(token), json={"group": None}).status_code
            == 401
        )
    assert c.get("/oidc/active-group").status_code == 401


def test_a_second_change_within_a_minute_is_told_when(active_client: TestClient) -> None:
    """Row 37 review S5: each change is a record in the replicated log."""
    c = active_client
    _, g, p, signed, _key = setup(c)
    proof = signed["access_token"]
    sales = group(c, name="Sales", apps=None)
    c.patch(f"/v1/people/{p['id']}", json={"groups": [sales["id"], g["id"]]})
    first = c.post("/oidc/active-group", headers=_bearer(proof), json={"group": g["id"]})
    assert first.status_code == 200, first.text
    index = c.app.state.machine.state.index
    again = c.post("/oidc/active-group", headers=_bearer(proof), json={"group": None})
    assert again.status_code == 429, again.text
    assert again.json()["error"] == "slow_down"
    wait = int(again.headers["Retry-After"])
    assert 1 <= wait <= 60
    assert f"Try again in {wait} seconds" in again.json()["error_description"]
    assert c.app.state.machine.state.index == index, "nothing was written"
    # The same choice again is no change, so no write and no wait.
    same = c.post("/oidc/active-group", headers=_bearer(proof), json={"group": g["id"]})
    assert same.status_code == 200


def test_the_owner_is_told_they_are_in_no_group(active_client: TestClient) -> None:
    c = active_client
    app, *_ = setup(c)
    owner = _tokens(c, app, name="operator", password=PASSPHRASE)["access_token"]
    for answer in (
        c.get("/oidc/active-group", headers=_bearer(owner)),
        c.post("/oidc/active-group", headers=_bearer(owner), json={"group": None}),
    ):
        assert answer.status_code == 403, answer.text
        assert answer.json()["error"] == "not_a_person"
        assert "People page" in answer.json()["error_description"]


def _slow_post(c: TestClient, proof: str, body: dict[str, Any], meanwhile: Any) -> Any:
    """POST /oidc/active-group whose body arrives only after `meanwhile` ran
    (the reviewer's probe: the person holds the body back)."""

    async def receive() -> dict[str, Any]:
        meanwhile()
        return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/oidc/active-group",
        "raw_path": b"/oidc/active-group",
        "query_string": b"",
        "headers": [
            (b"authorization", f"Bearer {proof}".encode()),
            (b"content-type", b"application/json"),
        ],
        "app": c.app,
        "client": ("127.0.0.1", 1),
        "server": ("testserver", 80),
        "scheme": "http",
    }
    return asyncio.run(set_active_group(Request(scope, receive)))


def test_a_slow_body_cannot_undo_a_disable_or_a_removed_group(active_client: TestClient) -> None:
    """Row 37 review B1: the route read the person, awaited the body, then
    wrote the whole record it read before, enabling a disabled account and
    restoring a removed membership."""
    c = active_client
    _, g, p, signed, key = setup(c)
    proof = signed["access_token"]
    sales = group(c, name="Sales", apps=None, policies={"allowedModels": ["m1"]})
    wide = group(c, name="Wide", apps=None, policies={"allowedModels": ["*"]})
    c.patch(f"/v1/people/{p['id']}", json={"groups": [sales["id"], wide["id"], g["id"]]})
    machine = c.app.state.machine

    def owner_acts() -> None:
        changed = c.patch(
            f"/v1/people/{p['id']}", json={"groups": [sales["id"], g["id"]], "disabled": True}
        )
        assert changed.status_code == 200, changed.text

    response = _slow_post(c, proof, {"group": wide["id"]}, owner_acts)
    after = machine.state.people[p["id"]]
    assert wide["id"] not in after["groups"], "the removed membership came back"
    assert after["disabled"] is True, "the disabled account is enabled again"
    assert response.status_code == 401
    assert call(c, key, proof).status_code in (401, 403)


def test_a_slow_body_cannot_undo_a_password_reset_or_revoked_grants(
    active_client: TestClient,
) -> None:
    c = active_client
    _, g, p, signed, key = setup(c)
    proof = signed["access_token"]
    sales = group(c, name="Sales", apps=None, policies={"allowedModels": ["m1"]})
    c.patch(f"/v1/people/{p['id']}", json={"groups": [g["id"], sales["id"]]})
    grants = {"managePeople": True, "manageGroups": True, "membershipGroups": []}
    assert c.put(f"/v1/people/{p['id']}/administration", json=grants).status_code == 200
    machine = c.app.state.machine
    before = dict(machine.state.people[p["id"]])
    seen: dict[str, Any] = {}

    def owner_acts() -> None:
        revoked = {"managePeople": False, "manageGroups": False, "membershipGroups": []}
        assert c.put(f"/v1/people/{p['id']}/administration", json=revoked).status_code == 200
        reset = c.put(f"/v1/people/{p['id']}/password", json={"password": "a-new-password-123"})
        assert reset.status_code == 204
        seen.update(machine.state.people[p["id"]])

    _slow_post(c, proof, {"group": sales["id"]}, owner_acts)
    after = machine.state.people[p["id"]]
    assert after.get("administration") == seen["administration"], "revoked grants came back"
    assert after["passwordVerifier"] == seen["passwordVerifier"] != before["passwordVerifier"]
    assert after["administrationVersion"] == seen["administrationVersion"]
    assert call(c, key, proof).status_code in (401, 403), "the old token is admitted again"


def test_row_37_writes_wait_for_a_standby_that_can_copy_them(
    active_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Log version 3: a standby that reports less would stop its copy at a
    group with routing or models, a person's chosen group, or a choice entry,
    so the root refuses to write them and names the standby."""
    from eugene_plexus_control.routes import control as control_routes

    c = active_client
    _, g, p, signed, _key = setup(c)
    proof = signed["access_token"]
    sales = group(c, name="Sales", apps=None)
    c.patch(f"/v1/people/{p['id']}", json={"groups": [g["id"], sales["id"]]})
    monkeypatch.setattr(
        control_routes, "standby_short_of", lambda _app, needed: "spare" if needed >= 3 else None
    )
    index = c.app.state.machine.state.index
    routed = c.put(
        f"/v1/permission-groups/{sales['id']}",
        json={"name": "Sales", "apps": None, "policies": {"allowedModels": ["m1"]}},
    )
    assert routed.status_code == 409, routed.text
    assert "The standby on spare is older than this root" in routed.text
    chosen = c.patch(f"/v1/people/{p['id']}", json={"activeGroup": sales["id"]})
    assert chosen.status_code == 409, chosen.text
    own = c.post("/oidc/active-group", headers=_bearer(proof), json={"group": sales["id"]})
    assert own.status_code == 409, own.text
    assert "update it" in own.json()["error_description"]
    assert c.app.state.machine.state.index == index, "nothing was written"
    # A row 70 policy still saves: the standby can copy it.
    banded = c.put(
        f"/v1/permission-groups/{sales['id']}",
        json={"name": "Sales", "apps": None, "policies": {"queueBand": "background"}},
    )
    assert banded.status_code == 200, banded.text
