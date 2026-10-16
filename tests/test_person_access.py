"""Real sign-in proofs must narrow the app key, across policy edits and replay."""

import json

import pytest

from eugene_plexus_control import oidc
from eugene_plexus_control.applied import from_canonical, to_canonical

from .conftest import machine_at
from .test_administration import delegate
from .test_oidc import CALLBACK, _tokens
from .test_permission_groups import PASSWORD, group, person


def setup(c, *, owner="app:workbench@node"):
    r = c.post(
        "/v1/oidc/clients", json={"name": "Workbench", "owner": owner, "redirectUris": [CALLBACK]}
    )
    assert r.status_code == 201, r.text
    app = r.json()
    g = group(c, apps=[app["client"]["clientId"]])
    p = person(c, groups=[g["id"]])
    signed = _tokens(c, app, name="Ada", password=PASSWORD)
    key = c.post(
        "/v1/auth/client-keys",
        json={
            "name": owner,
            "limits": {
                "allowedModels": ["allowed", "alias"],
                "allowedTools": [],
                "localOnly": True,
                "writeLogs": True,
            },
        },
    ).json()["key"]["id"]
    return app, g, p, signed, key


def call(c, key, token=None, action="check", request_id="request", **extra):
    body = {"keyId": key, "requestId": request_id, "action": action, **extra}
    if token is not None:
        body["personToken"] = token
    if action != "check":
        body.setdefault("model", "allowed")
    return c.post("/v1/auth/client-keys/admission", json=body)


def test_person_proof_and_key_limits_are_both_required(active_client):
    c = active_client
    app, _, p, signed, key = setup(c)
    assert call(c, key).status_code == 401
    assert call(c, key, "forged-person-id").status_code == 401
    for kind in ("id_token", "refresh_token"):
        assert call(c, key, signed[kind]).status_code == 401
    proof = signed["access_token"]
    good = call(c, key, proof).json()
    assert good["personId"] == p["id"] and good["appClientId"] == app["client"]["clientId"]
    assert good["limits"]["allowedTools"] == [] and good["limits"]["localOnly"] is True
    assert call(c, key, proof, "acquire", model="excluded").status_code == 403
    assert call(c, key, proof, "acquire").status_code == 200
    assert call(c, key, proof, "renew").status_code == 200
    assert call(c, key, action="release").status_code == 200
    # App logging is deliberately app-owned, never an inference reservation.
    assert call(c, key, purpose="logs").status_code == 200
    assert call(c, key, action="acquire", purpose="logs").status_code == 403


@pytest.mark.parametrize(
    "withdraw", ["disable", "delete", "membership", "group", "app", "password", "logout", "key"]
)
def test_withdrawal_stops_existing_lease_and_new_requests(active_client, withdraw):
    c = active_client
    app, g, p, signed, key = setup(c)
    proof = signed["access_token"]
    assert call(c, key, proof, "acquire").status_code == 200
    path = f"/v1/people/{p['id']}"
    if withdraw == "disable":
        c.patch(path, json={"disabled": True})
    elif withdraw == "delete":
        c.delete(path)
    elif withdraw == "membership":
        c.patch(path, json={"groups": []})
    elif withdraw == "group":
        c.delete(f"/v1/permission-groups/{g['id']}")
    elif withdraw == "app":
        c.delete(f"/v1/oidc/clients/{app['client']['clientId']}")
    elif withdraw == "password":
        c.put(path + "/password", json={"password": "a new password for Ada"})
    elif withdraw == "logout":
        c.post(
            "/oidc/revoke",
            auth=(app["client"]["clientId"], app["clientSecret"]),
            data={"token": signed["refresh_token"]},
        )
    else:
        c.delete(f"/v1/auth/client-keys/{key}")
    assert call(c, key, proof, "renew").status_code in (401, 403)
    assert call(c, key, proof, "acquire", request_id="next").status_code in (401, 403)
    assert call(c, key, action="release").status_code == 200


def test_proof_cannot_cross_app_keys_or_people_reservations(active_client):
    c = active_client
    app, _, _, signed, key = setup(c)
    other_key = c.post("/v1/auth/client-keys", json={"name": "app:workbench@other"}).json()["key"][
        "id"
    ]
    assert call(c, other_key, signed["access_token"]).status_code == 401
    person(c, name="Bob")
    bob = _tokens(c, app, name="Bob", password=PASSWORD)
    assert call(c, key, signed["access_token"], "acquire").status_code == 200
    assert call(c, key, bob["access_token"], "renew").status_code == 409
    assert call(c, key, bob["access_token"], "acquire").status_code == 409


def test_access_proof_requires_auth_time_and_current_signature(active_client):
    c = active_client
    _, _, _, signed, key = setup(c)
    state = c.app.state.machine.state
    claims = oidc.Provider.decode(signed["access_token"], state, typ=oidc.TYP_ACCESS, audience=None)
    for change in ({"exp": 1}, {"sid": ""}, {"aud": "elsewhere"}, {"eugene_auth_at": None}):
        token = c.app.state.oidc.mint(
            c.app.state.machine, c.app.state.auth_state, {**claims, **change}, oidc.TYP_ACCESS
        )
        assert call(c, key, token).status_code == 401


def test_audit_is_attributed_secret_free_owner_only_and_survives_compaction(
    active_client, tmp_path
):
    c = active_client
    _, _, _, signed, key = setup(c)
    manager, auth = delegate(c, managePeople=True)
    made = c.post("/v1/people", headers=auth, json={"name": "New person", "password": PASSWORD})
    assert made.status_code == 201
    events = c.get("/v1/auth/audit").json()["events"]
    assert events[0]["actor"] == manager["id"] and events[0]["target"] == made.json()["id"]
    assert any(e["action"] == "putClientKey" and e["target"] == key for e in events)
    assert c.get("/v1/auth/audit", headers=auth).status_code == 401
    serialized = json.dumps(events)
    assert PASSWORD not in serialized and signed["access_token"] not in serialized
    state = c.app.state.machine.state
    assert from_canonical(to_canonical(state)).access_audit == state.access_audit
    c.app.state.machine.snapshot_now()
    recovered = machine_at(tmp_path / "active" / "state")
    assert recovered.state.access_audit == state.access_audit
