"""Real signed tokens and durable policy; test boundaries, not just visibility."""

from copy import deepcopy
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_control import tokens
from eugene_plexus_control.applied import ApplyError, canonical_bytes, from_canonical, to_canonical
from eugene_plexus_control.state_machine import StateMachine

from .conftest import enroll
from .test_permission_groups import PASSWORD, group, person


def delegate(c: TestClient, **grants: Any) -> tuple[dict[str, Any], dict[str, str]]:
    p = person(c, name="Manager")
    r = c.put(f"/v1/people/{p['id']}/administration", json=grants)
    assert r.status_code == 200, r.text
    return r.json(), sign_in(c)


def sign_in(c: TestClient) -> dict[str, str]:
    r = c.post("/v1/auth/administration/login", json={"name": "Manager", "password": PASSWORD})
    assert r.status_code == 200, r.text
    assert r.headers["cache-control"] == "no-store"
    return {"Authorization": "Bearer " + r.json()["sessionToken"]}


def test_account_management_never_grants_identity_or_owner_access(
    active_client: TestClient,
) -> None:
    c = active_client
    manager, auth = delegate(c, managePeople=True)
    other_admin = person(c, name="Other admin")
    c.put(f"/v1/people/{other_admin['id']}/administration", json={"manageGroups": True})
    made = c.post("/v1/people", headers=auth, json={"name": "New member", "password": PASSWORD})
    assert made.status_code == 201, made.text
    member = made.json()
    assert member["groups"] == member["appsInEffect"] == member["permissionsInEffect"] == []
    path = f"/v1/people/{member['id']}"
    assert c.patch(path, headers=auth, json={"displayName": "Changed"}).status_code == 200
    assert c.patch(path, headers=auth, json={"disabled": True}).status_code == 200
    assert c.patch(path, headers=auth, json={"disabled": False}).status_code == 200
    for body in (
        {"apps": None},
        {"permissions": None},
        {"email": "victim@example.test"},
        {"groups": None},
    ):
        assert c.patch(path, headers=auth, json=body).status_code == 403
    assert (
        c.put(path + "/password", headers=auth, json={"password": "new password here"}).status_code
        == 401
    )
    assert (
        c.put(path + "/administration", headers=auth, json={"managePeople": True}).status_code
        == 401
    )
    for protected in (manager, other_admin):
        assert (
            c.patch(
                f"/v1/people/{protected['id']}", headers=auth, json={"disabled": True}
            ).status_code
            == 403
        )
        assert c.delete(f"/v1/people/{protected['id']}", headers=auth).status_code == 403
    for body in ({"apps": None}, {"email": "victim@example.test"}, {"groups": []}):
        assert (
            c.post(
                "/v1/people", headers=auth, json={"name": "Bypass", "password": PASSWORD, **body}
            ).status_code
            == 403
        )
    for route in (
        "/v1/config",
        "/v1/nodes",
        "/v1/topology",
        "/v1/replication/snapshot",
        "/v1/client-keys",
        "/v1/sites",
    ):
        r = c.get(route, headers=auth)
        assert r.status_code in (401, 404), (route, r.text)
    assert (
        c.post("/v1/permission-groups", headers=auth, json={"name": "Escalate"}).status_code == 403
    )
    assert c.delete(path, headers=auth).status_code == 204


def test_membership_delegate_preserves_out_of_scope_and_cannot_manage_accounts(
    active_client: TestClient,
) -> None:
    c = active_client
    allowed, other = group(c), group(c, name="Other")
    manager, auth = delegate(c, membershipGroups=[allowed["id"]])
    p = person(c, groups=[other["id"]])
    path = f"/v1/people/{p['id']}"
    assert (
        c.patch(path, headers=auth, json={"groups": [other["id"], allowed["id"]]}).status_code
        == 200
    )
    assert c.patch(path, headers=auth, json={"groups": [allowed["id"]]}).status_code == 403
    assert c.patch(path, headers=auth, json={"groups": [other["id"]]}).status_code == 200
    assert c.patch(path, headers=auth, json={"groups": None}).status_code == 403
    assert c.patch(path, headers=auth, json={"disabled": True}).status_code == 403
    assert (
        c.post("/v1/people", headers=auth, json={"name": "X", "password": PASSWORD}).status_code
        == 403
    )
    assert c.delete(path, headers=auth).status_code == 403
    assert (
        c.patch(
            f"/v1/people/{manager['id']}", headers=auth, json={"groups": [allowed["id"]]}
        ).status_code
        == 403
    )
    individual = person(c, name="Individual")
    assert (
        c.patch(
            f"/v1/people/{individual['id']}", headers=auth, json={"groups": [allowed["id"]]}
        ).status_code
        == 403
    )


def test_group_designer_has_no_membership_or_account_authority(active_client: TestClient) -> None:
    c = active_client
    _, auth = delegate(c, manageGroups=True)
    r = c.post("/v1/permission-groups", headers=auth, json={"name": "Designed"})
    assert r.status_code == 201, r.text
    g = r.json()["id"]
    assert (
        c.put(
            f"/v1/permission-groups/{g}", headers=auth, json={"name": "Designed", "apps": None}
        ).status_code
        == 200
    )
    p = person(c, groups=[])
    assert c.patch(f"/v1/people/{p['id']}", headers=auth, json={"groups": [g]}).status_code == 403
    assert c.delete(f"/v1/permission-groups/{g}", headers=auth).status_code == 204


def test_owner_is_not_limited_to_a_delegates_scope_size(active_client: TestClient) -> None:
    c = active_client
    for i in range(65):
        group(c, name=f"Group {i}")
    assert c.get("/v1/people").status_code == 200
    session = c.get("/v1/auth/administration/session")
    assert session.status_code == 200 and session.json()["owner"] is True
    assert c.delete("/v1/auth/administration/session").status_code == 403


@pytest.mark.parametrize(
    "change", ["grants", "disable", "password", "delete", "logout", "scope-delete"]
)
def test_current_sessions_revoke_without_waiting_for_expiry(
    active_client: TestClient, change: str
) -> None:
    c = active_client
    g = group(c)
    p, auth = delegate(c, managePeople=True, membershipGroups=[g["id"]])
    assert c.get("/v1/people", headers=auth).status_code == 200
    path = f"/v1/people/{p['id']}"
    if change == "grants":
        c.put(path + "/administration", json={"manageGroups": True})
    elif change == "disable":
        c.patch(path, json={"disabled": True})
        c.patch(path, json={"disabled": False})
    elif change == "password":
        c.put(path + "/password", json={"password": PASSWORD})
    elif change == "delete":
        c.delete(path)
    elif change == "scope-delete":
        c.delete(f"/v1/permission-groups/{g['id']}")
    else:
        assert c.delete("/v1/auth/administration/session", headers=auth).status_code == 204
    assert c.get("/v1/people", headers=auth).status_code == 401


def test_grants_replay_and_snapshot_and_legacy_default(active_client: TestClient) -> None:
    c = active_client
    g = group(c)
    p, auth = delegate(c, membershipGroups=[g["id"]])
    machine = c.app.state.machine
    snapshot = to_canonical(machine.state)
    assert canonical_bytes(from_canonical(snapshot)) == canonical_bytes(machine.state)
    replay = StateMachine(machine._store)
    replay.load()
    assert replay.state.people[p["id"]]["administration"] == p["administration"]
    for bad in (
        {"managePeople": "true"},
        {"membershipGroups": ["missing"]},
        {"manageEverything": True},
    ):
        broken = deepcopy(snapshot)
        broken["people"][0]["administration"] = bad
        with pytest.raises(ApplyError):
            from_canonical(broken)
    legacy = deepcopy(snapshot)
    legacy["people"][0].pop("administration")
    legacy["people"][0].pop("administrationVersion")
    assert "administration" not in from_canonical(legacy).people[p["id"]]
    assert c.get("/v1/auth/administration/session", headers=auth).json()["personId"] == p["id"]


def test_login_refusals_rate_limit_and_sealed_root(active_client: TestClient) -> None:
    c = active_client
    delegate(c, managePeople=True)
    for name in ("Unknown", "operator", "Manager", "Manager", "Unknown"):
        r = c.post("/v1/auth/administration/login", json={"name": name, "password": "wrong"})
        assert r.status_code == 401
    assert (
        c.post(
            "/v1/auth/administration/login", json={"name": "Manager", "password": PASSWORD}
        ).status_code
        == 429
    )
    c.app.state.auth_state.signing_key = None
    assert (
        c.post(
            "/v1/auth/administration/login", json={"name": "Manager", "password": PASSWORD}
        ).status_code
        == 503
    )


def test_no_grants_no_login_and_worker_cannot_forge_or_exchange(active_client: TestClient) -> None:
    c = active_client
    person(c, name="Ordinary")
    assert (
        c.post(
            "/v1/auth/administration/login", json={"name": "Ordinary", "password": PASSWORD}
        ).status_code
        == 401
    )
    p, auth = delegate(c, managePeople=True)
    node, joined = enroll(c, "worker")
    assert joined.status_code == 201
    forged, _ = node.token.mint(
        typ=tokens.TYP_ADMINISTRATION,
        sub=p["id"],
        aud=["control"],
        ttl_seconds=3600,
        extra={"administration_version": 1},
    )
    assert c.get("/v1/people", headers={"Authorization": "Bearer " + forged}).status_code == 401
    assert (
        c.post(
            "/v1/auth/token",
            headers={"Authorization": "Bearer " + node.service_token()},
            json={"subjectToken": auth["Authorization"][7:], "audience": "node:worker"},
        ).status_code
        == 401
    )
    assert (
        c.put(
            f"/v1/people/{p['id']}/administration", json={"membershipGroups": ["missing"]}
        ).status_code
        == 422
    )

    assert (
        c.put(f"/v1/people/{p['id']}/administration", json={"manageEverything": True}).status_code
        == 422
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"sub": "operator"},
        {"aud": ["control", "node:worker"]},
        {"ttl_seconds": 3601},
        {"extra": {}},
        {"extra": {"administration_version": True}},
        {"extra": {"administration_version": -1}},
        {"extra": {"administration_version": 1, "sid": "other-session"}},
    ],
)
def test_administration_token_profile_is_narrow(
    active_client: TestClient, overrides: dict[str, Any]
) -> None:
    c = active_client
    p, _ = delegate(c, managePeople=True)
    args = {
        "typ": tokens.TYP_ADMINISTRATION,
        "sub": p["id"],
        "aud": ["control"],
        "ttl_seconds": 3600,
        "extra": {"administration_version": 1},
        **overrides,
    }
    token, _ = c.app.state.auth_state.signer().mint(**args)
    assert c.get("/v1/people", headers={"Authorization": "Bearer " + token}).status_code == 401
