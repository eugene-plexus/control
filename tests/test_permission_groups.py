"""Group policy must survive replication and govern real authorization paths."""

from copy import deepcopy
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_control import permissions
from eugene_plexus_control.applied import ApplyError, canonical_bytes, from_canonical, to_canonical

from .test_oidc import CALLBACK, _client, _sign_in, _tokens

PASSWORD = "group member password"


def group(c: TestClient, **extra: Any) -> dict[str, Any]:
    r = c.post("/v1/permission-groups", json={"name": "Builders", **extra})
    assert r.status_code == 201, r.text
    return r.json()


def person(c: TestClient, **extra: Any) -> dict[str, Any]:
    r = c.post("/v1/people", json={"name": "Ada", "password": PASSWORD, **extra})
    assert r.status_code == 201, r.text
    return r.json()


def test_explicit_group_mode_never_inherits_individual_defaults(active_client: TestClient) -> None:
    c = active_client
    app = _client(c)["client"]["clientId"]
    g = group(c, apps=[app], permissions=["use-job-sites"])
    p = person(c)
    assert p["permissionsInEffect"] == ["add-job-sites", "use-job-sites"]
    path = f"/v1/people/{p['id']}"
    changed = c.patch(path, json={"groups": [g["id"]]}).json()
    assert changed["appsInEffect"] == [app]
    assert changed["permissionsInEffect"] == ["use-job-sites"]
    assert changed["id"] == p["id"]
    empty = c.patch(path, json={"groups": []}).json()
    assert empty["appsInEffect"] == empty["permissionsInEffect"] == []
    restored = c.patch(path, json={"groups": None}).json()
    assert restored.get("appsInEffect") is None
    assert restored["permissionsInEffect"] == p["permissionsInEffect"]


def test_union_and_deleting_last_group_never_restore_access(active_client: TestClient) -> None:
    c = active_client
    a, b = _client(c, "A")["client"]["clientId"], _client(c, "B")["client"]["clientId"]
    first = group(c, apps=[a], permissions=["use-job-sites"])
    second = group(c, name="All apps", apps=None, permissions=["add-job-sites"])
    p = person(c, groups=[first["id"], second["id"]], apps=[b], permissions=[])
    assert p.get("appsInEffect") is None
    assert p["permissionsInEffect"] == ["add-job-sites", "use-job-sites"]
    assert c.delete(f"/v1/permission-groups/{second['id']}").status_code == 204
    remaining = c.get("/v1/people").json()["people"][0]
    assert remaining["appsInEffect"] == [a] and remaining["groups"] == [first["id"]]
    assert c.delete(f"/v1/oidc/clients/{a}").status_code == 204
    assert c.get("/v1/permission-groups").json()["groups"][0]["apps"] == []
    assert c.delete(f"/v1/permission-groups/{first['id']}").status_code == 204
    empty = c.get("/v1/people").json()["people"][0]
    assert empty["groups"] == empty["appsInEffect"] == empty["permissionsInEffect"] == []
    assert empty["apps"] == [b]  # retained, inactive individual policy


def test_group_edit_revokes_refresh_and_new_sign_in(active_client: TestClient) -> None:
    c = active_client
    app = _client(c)
    g = group(c, apps=[app["client"]["clientId"]])
    person(c, groups=[g["id"]])
    signed = _tokens(c, app, name="Ada", password=PASSWORD, scope="openid profile offline_access")
    # A valid member identity is never a console/group administrator.
    member = {"Authorization": "Bearer " + signed["access_token"]}
    assert c.get("/v1/permission-groups", headers=member).status_code == 401
    assert (
        c.post(
            "/v1/permission-groups", headers=member, json={"name": "Escalate", "apps": None}
        ).status_code
        == 401
    )
    assert c.put(f"/v1/permission-groups/{g['id']}", json={"name": "Builders"}).status_code == 200
    denied = c.post(
        "/oidc/token",
        auth=(app["client"]["clientId"], app["clientSecret"]),
        data={
            "grant_type": "refresh_token",
            "refresh_token": signed["refresh_token"],
        },
    )
    assert denied.status_code == 400 and denied.json()["error"] == "invalid_grant"
    assert _sign_in(c, app["client"]["clientId"], PASSWORD, "Ada").status_code == 403


def test_group_and_membership_validation_and_protected_owner(active_client: TestClient) -> None:
    c = active_client
    g = group(c)
    assert g["apps"] == g["permissions"] == []
    assert c.post("/v1/permission-groups", json={"name": " builders "}).status_code == 409
    for bad in (
        {"name": " "},
        {"name": "X", "apps": ["unknown"]},
        {"name": "X", "permissions": ["admin"]},
    ):
        assert c.post("/v1/permission-groups", json=bad).status_code == 422
    p = person(c)
    for groups in (["missing"], [g["id"], g["id"]]):
        assert c.patch(f"/v1/people/{p['id']}", json={"groups": groups}).status_code == 422
    listing = c.get("/v1/people").json()
    assert listing["owner"]["id"] == "operator"
    assert c.patch("/v1/people/operator", json={"groups": []}).status_code == 404
    assert c.delete("/v1/people/operator").status_code == 404


def test_group_state_survives_snapshot_and_rejects_dangling_members(
    active_client: TestClient,
) -> None:
    c = active_client
    g = group(c, permissions=["use-job-sites"])
    p = person(c, groups=[g["id"]])
    state = c.app.state.machine.state
    snapshot = to_canonical(state)
    restored = from_canonical(snapshot)
    assert canonical_bytes(restored) == canonical_bytes(state)
    assert permissions.has(restored, p["id"], "use-job-sites")
    assert not permissions.has(restored, p["id"], "add-job-sites")
    broken = deepcopy(snapshot)
    broken["permissionGroups"] = []
    with pytest.raises(ApplyError, match="unknown permission group"):
        from_canonical(broken)
    legacy = deepcopy(snapshot)
    legacy.pop("permissionGroups")
    legacy["people"][0].pop("groups")
    assert permissions.has(from_canonical(legacy), p["id"], "add-job-sites")
    broken = deepcopy(snapshot)
    broken["permissionGroups"][0]["permissions"] = [{}]
    with pytest.raises(ApplyError):
        from_canonical(broken)


def test_disabled_group_member_has_no_job_site_grants(active_client: TestClient) -> None:
    c = active_client
    g = group(c, permissions=["use-job-sites"])
    p = person(c, groups=[g["id"]])
    c.patch(f"/v1/people/{p['id']}", json={"disabled": True})
    assert not permissions.has(c.app.state.machine.state, p["id"], "use-job-sites")


def test_group_revocation_between_approval_and_code_exchange(active_client: TestClient) -> None:
    c = active_client
    app = _client(c)
    g = group(c, apps=[app["client"]["clientId"]])
    person(c, groups=[g["id"]])
    approved = _sign_in(c, app["client"]["clientId"], PASSWORD, "Ada")
    code = parse_qs(urlsplit(approved.headers["location"]).query)["code"][0]
    c.delete(f"/v1/permission-groups/{g['id']}")
    response = c.post(
        "/oidc/token",
        auth=(app["client"]["clientId"], app["clientSecret"]),
        data={
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": "v" * 50,
            "redirect_uri": CALLBACK,
        },
    )
    assert response.status_code == 400 and response.json()["error"] == "invalid_grant"
