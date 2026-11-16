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


# --------------------------------------------------------------------------- #
# row 70: policies per group
# --------------------------------------------------------------------------- #

#: Row 40 added the usage allowances, answered as null when not set.
UNSET = {
    "queueBand": None,
    "summaryContextPercent": None,
    "tokensPerDay": None,
    "tokensPerMonth": None,
    "costPerMonthMicros": None,
    "roleRouting": None,
    "allowedModels": None,
}


def test_group_policies_are_set_kept_by_an_older_console_and_cleared(
    active_client: TestClient,
) -> None:
    c = active_client
    plain = group(c)
    assert plain["policies"] == UNSET, "a root that knows policies always answers them"
    assert "policies" not in c.app.state.machine.state.permission_groups[plain["id"]]
    set_ = group(
        c, name="Night shift", policies={"queueBand": "background", "summaryContextPercent": 70}
    )
    assert set_["policies"] == {**UNSET, "queueBand": "background", "summaryContextPercent": 70}
    path = f"/v1/permission-groups/{set_['id']}"
    # A console that predates policies sends none: the group keeps them.
    kept = c.put(path, json={"name": "Night shift", "description": "Late"}).json()
    assert kept["policies"] == set_["policies"] and kept["description"] == "Late"
    # Row 40 (review S2): a policy left out keeps its value; null clears it.
    replaced = c.put(path, json={"name": "Night shift", "policies": {"summaryContextPercent": 90}})
    assert replaced.json()["policies"] == {
        **UNSET,
        "queueBand": "background",
        "summaryContextPercent": 90,
    }
    cleared = c.put(path, json={"name": "Night shift", "policies": UNSET}).json()
    assert cleared["policies"] == UNSET
    assert "policies" not in c.app.state.machine.state.permission_groups[set_["id"]]
    listed = {g["id"]: g for g in c.get("/v1/permission-groups").json()["groups"]}
    assert listed[set_["id"]]["policies"] == UNSET
    for bad in (
        {"queueBand": "urgent"},
        {"summaryContextPercent": 49},
        {"summaryContextPercent": 96},
        {"routing": "fast"},
    ):
        refused = c.put(path, json={"name": "Night shift", "policies": bad})
        assert refused.status_code == 422, bad


def test_a_persons_policies_in_effect_name_the_groups_that_set_them(
    active_client: TestClient,
) -> None:
    c = active_client
    night = group(
        c, name="Archive", policies={"queueBand": "background", "summaryContextPercent": 90}
    )
    desk = group(c, name="Desk", policies={"queueBand": "interactive", "summaryContextPercent": 70})
    p = person(c, groups=[night["id"], desk["id"]])
    assert p["policiesInEffect"] == {
        "queueBand": "interactive",
        "summaryContextPercent": 90,
        "setBy": {"queueBand": [desk["id"]], "summaryContextPercent": [night["id"]]},
    }
    individual = c.patch(f"/v1/people/{p['id']}", json={"groups": None}).json()
    assert individual["policiesInEffect"] == {}
    c.patch(f"/v1/people/{p['id']}", json={"groups": [night["id"]]})
    listed = c.get("/v1/people").json()["people"][0]
    assert listed["policiesInEffect"]["queueBand"] == "background"
    # Deleting the group takes its policies with it.
    c.delete(f"/v1/permission-groups/{night['id']}")
    assert c.get("/v1/people").json()["people"][0]["policiesInEffect"] == {}


def test_a_group_administrator_sets_policies_and_a_member_cannot(
    active_client: TestClient,
) -> None:
    from .test_administration import delegate

    c = active_client
    _, auth = delegate(c, manageGroups=True)
    made = c.post(
        "/v1/permission-groups",
        headers=auth,
        json={"name": "Batch", "policies": {"queueBand": "sheddable"}},
    )
    assert made.status_code == 201, made.text
    assert made.json()["policies"]["queueBand"] == "sheddable"
    _, people_only = delegate_people(c)
    refused = c.put(
        f"/v1/permission-groups/{made.json()['id']}",
        headers=people_only,
        json={"name": "Batch", "policies": {"queueBand": "interactive"}},
    )
    assert refused.status_code == 403


def delegate_people(c: TestClient) -> tuple[dict[str, Any], dict[str, str]]:
    p = person(c, name="Clerk")
    c.put(f"/v1/people/{p['id']}/administration", json={"managePeople": True})
    r = c.post("/v1/auth/administration/login", json={"name": "Clerk", "password": PASSWORD})
    assert r.status_code == 200, r.text
    return p, {"Authorization": "Bearer " + r.json()["sessionToken"]}


# --------------------------------------------------------------------------- #
# row 37: models and routing per group, and the group a person works from
# --------------------------------------------------------------------------- #


def test_routing_and_models_are_kept_by_a_row_70_console(active_client: TestClient) -> None:
    """A console from row 70 sends only the two policies it knows; the
    group keeps its routing and model list."""
    c = active_client
    made = group(
        c,
        name="Sales",
        policies={"roleRouting": {"chat": ["m2", "m1"]}, "allowedModels": ["m1", "m*"]},
    )
    assert made["policies"]["roleRouting"] == {"chat": ["m2", "m1"]}
    path = f"/v1/permission-groups/{made['id']}"
    old_console = c.put(
        path,
        json={
            "name": "Sales",
            "policies": {"queueBand": "background", "summaryContextPercent": None},
        },
    ).json()
    assert old_console["policies"] == {
        **UNSET,
        "queueBand": "background",
        "roleRouting": {"chat": ["m2", "m1"]},
        "allowedModels": ["m1", "m*"],
    }
    # Every model is null; no model at all is an empty list.
    none = c.put(
        path, json={"name": "Sales", "policies": {"roleRouting": None, "allowedModels": []}}
    ).json()
    assert none["policies"]["allowedModels"] == []
    cleared = c.put(
        path, json={"name": "Sales", "policies": {"roleRouting": None, "allowedModels": None}}
    ).json()
    assert cleared["policies"] == {**UNSET, "queueBand": "background"}
    # An empty routing is no routing.
    empty = c.put(path, json={"name": "Sales", "policies": {"roleRouting": {}}}).json()
    assert empty["policies"]["roleRouting"] is None
    for bad in (
        {"roleRouting": {"typo": ["m1"]}},
        {"roleRouting": {"chat": []}},
        {"roleRouting": {"chat": ["m1", "m1"]}},
        {"allowedModels": ["m1", "m1"]},
        {"allowedModels": [""]},
    ):
        refused = c.put(path, json={"name": "Sales", "policies": bad})
        assert refused.status_code == 422, bad
    typo = c.put(path, json={"name": "Sales", "policies": {"roleRouting": {"typo": ["m1"]}}})
    assert "typo is not a model role" in typo.text


def test_a_person_cannot_pick_a_group_they_are_not_in(active_client: TestClient) -> None:
    c = active_client
    sales = group(c, name="Sales", policies={"roleRouting": {"chat": ["m1"]}})
    ops = group(c, name="Ops", policies={"roleRouting": {"chat": ["m3"]}})
    other = group(c, name="Other")
    p = person(c, groups=[sales["id"], ops["id"]])
    path = f"/v1/people/{p['id']}"
    assert p["policiesInEffect"]["activeGroup"] == sales["id"]
    assert "activeGroup" not in p
    refused = c.patch(path, json={"activeGroup": other["id"]})
    assert refused.status_code == 422, refused.text
    assert "Choose a group this person is in" in refused.text
    chosen = c.patch(path, json={"activeGroup": ops["id"]}).json()
    assert chosen["activeGroup"] == ops["id"]
    assert chosen["policiesInEffect"]["roleRouting"] == {"chat": ["m3"]}
    assert chosen["policiesInEffect"]["setBy"]["roleRouting"] == [ops["id"]]
    # Moved out of the group they worked from: back to the first of their list.
    moved = c.patch(path, json={"groups": [sales["id"]]}).json()
    assert "activeGroup" not in moved
    assert moved["policiesInEffect"]["roleRouting"] == {"chat": ["m1"]}
    # Chosen and left in one change: refused, not silently dropped.
    both = c.patch(path, json={"groups": [sales["id"]], "activeGroup": ops["id"]})
    assert both.status_code == 422
    # None goes back to the first.
    c.patch(path, json={"groups": [sales["id"], ops["id"]], "activeGroup": ops["id"]})
    back = c.patch(path, json={"activeGroup": None}).json()
    assert "activeGroup" not in back
    assert back["policiesInEffect"]["activeGroup"] == sales["id"]


def test_a_route_outside_the_groups_own_models_is_refused(active_client: TestClient) -> None:
    """Row 37 review S3 (settings never lie): the console would show a route
    the gateway drops."""
    c = active_client
    refused = c.post(
        "/v1/permission-groups",
        json={
            "name": "Sales",
            "policies": {"allowedModels": ["m1", "qwen3-*"], "roleRouting": {"chat": ["m1", "m3"]}},
        },
    )
    assert refused.status_code == 422, refused.text
    assert "The role chat lists m3, which this group's models do not include" in refused.text
    made = group(
        c,
        name="Sales",
        policies={"allowedModels": ["m1", "qwen3-*"], "roleRouting": {"chat": ["qwen3-8b", "m1"]}},
    )
    path = f"/v1/permission-groups/{made['id']}"
    # Narrowing the models under a route is refused too, naming the model.
    narrowed = c.put(path, json={"name": "Sales", "policies": {"allowedModels": ["m1"]}})
    assert narrowed.status_code == 422
    assert "qwen3-8b" in narrowed.text
    # With no model list, any route stands.
    assert (
        c.put(path, json={"name": "Sales", "policies": {"allowedModels": None}}).status_code == 200
    )


def test_a_pattern_with_too_many_stars_is_refused(active_client: TestClient) -> None:
    """Row 37 review S1: each `*` costs the matcher; at most 8 in one entry."""
    c = active_client
    eight = "*a*b*c*d*e*f*g*h"
    assert group(c, name="Eight", policies={"allowedModels": [eight]})
    nine = c.post(
        "/v1/permission-groups",
        json={"name": "Nine", "policies": {"allowedModels": ["*a*b*c*d*e*f*g*h*i"]}},
    )
    assert nine.status_code == 422, nine.text
    assert "more than 8 *" in nine.text
