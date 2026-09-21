"""A5: one durable allowance shared by every gateway using the registry."""

import pytest

from eugene_plexus_control import client_admission as admission
from eugene_plexus_control import security
from eugene_plexus_control.applied import from_canonical, to_canonical
from tests.conftest import machine_at


def mint(client, **limits):
    response = client.post("/v1/auth/client-keys", json={"name": "bounded", "limits": limits})
    assert response.status_code == 201, response.text
    return response.json()


def call(client, key, action, request_id="one", model="allowed", **kwargs):
    return client.post(
        "/v1/auth/client-keys/admission",
        json={
            "keyId": key,
            "action": action,
            "requestId": request_id,
            "model": model,
        },
        **kwargs,
    )


def test_restart_and_snapshot_preserve_rate_and_active_reservations(active_client, tmp_path):
    c = active_client
    key = mint(c, maxConcurrentRequests=1, requestsPerMinute=2)["key"]["id"]
    assert call(c, key, "acquire").status_code == 200
    machine = c.app.state.machine
    for snapshot in (False, True):
        if snapshot:
            machine.snapshot_now()
        recovered = machine_at(tmp_path / "active" / "state")
        assert recovered.state.client_admission == machine.state.client_admission
        assert (
            from_canonical(to_canonical(recovered.state)).client_admission
            == machine.state.client_admission
        )
        c.app.state.machine = recovered
        del c.app.state.admission_clock
        assert call(c, key, "acquire", "another").status_code == 429
        machine = recovered
    assert call(c, key, "release").status_code == 200
    assert call(c, key, "acquire", "another").status_code == 200
    assert call(c, key, "release", "another").status_code == 200
    assert call(c, key, "acquire", "last").status_code == 429


def test_idempotency_late_acquire_and_separate_keys(active_client):
    c = active_client
    a = mint(c, maxConcurrentRequests=1, requestsPerMinute=1)["key"]["id"]
    b = mint(c, maxConcurrentRequests=1, requestsPerMinute=1)["key"]["id"]
    assert call(c, a, "release", "cancelled").status_code == 200
    assert call(c, a, "acquire", "cancelled").status_code == 409
    assert call(c, a, "acquire").status_code == 200
    assert call(c, a, "acquire").status_code == 200
    assert call(c, b, "acquire").status_code == 200
    assert call(c, a, "release").status_code == 200
    assert call(c, a, "release").status_code == 200
    assert call(c, a, "acquire", "two").status_code == 429


def test_lease_expiration_does_not_reset_request_rate(active_client):
    c = active_client
    key = mint(c, maxConcurrentRequests=1, requestsPerMinute=2)["key"]["id"]
    assert call(c, key, "acquire").json()["leaseSeconds"] == 30
    c.app.state.admission_clock.base += 31
    assert call(c, key, "renew").status_code == 409
    assert call(c, key, "acquire", "two").status_code == 200
    assert call(c, key, "release", "two").status_code == 200
    assert call(c, key, "acquire", "three").status_code == 429
    c.app.state.admission_clock.base += 61
    assert call(c, key, "acquire", "three").status_code == 200
    assert len(c.app.state.machine.state.client_admission["buckets"][key]) == 1


def test_policy_change_and_revocation_stop_renewal_without_resetting_rate(active_client):
    c = active_client
    key = mint(c, requestsPerMinute=1)["key"]["id"]
    assert call(c, key, "acquire").status_code == 200
    response = c.put(
        f"/v1/auth/client-keys/{key}/limits",
        json={
            "limits": {
                "allowedModels": ["allowed"],
                "requestsPerMinute": 1,
            }
        },
    )
    assert response.status_code == 200
    assert call(c, key, "renew").status_code == 409
    assert call(c, key, "acquire", "two").status_code == 429
    assert c.delete(f"/v1/auth/client-keys/{key}").status_code == 204
    assert call(c, key, "check").status_code == 401
    assert call(c, key, "release").status_code == 200


def test_failed_commit_does_not_grant_admission(active_client, monkeypatch):
    c = active_client
    key = mint(c)["key"]["id"]
    before = c.app.state.machine.state.client_admission

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(c.app.state.machine._store, "append", fail)
    assert call(c, key, "acquire").status_code == 503
    assert c.app.state.machine.state.client_admission == before


def test_admission_and_edit_audiences(active_client):
    c = active_client
    made = mint(c)
    key = made["key"]["id"]
    private = c.app.state.auth_state.signing_key
    for kind, expected in [("agent", 200), ("gateway", 200), ("library", 401)]:
        h = {
            "Authorization": "Bearer "
            + security.issue_service_token(signing_key=private, kind=kind)
        }
        assert call(c, key, "check", headers=h).status_code == expected
        assert (
            c.put(f"/v1/auth/client-keys/{key}/limits", json={"limits": {}}, headers=h).status_code
            == 401
        )
    assert (
        call(c, key, "check", headers={"Authorization": "Bearer " + made["token"]}).status_code
        == 401
    )
    c.app.state.machine.force_role("standby")
    assert call(c, key, "check").status_code == 503


@pytest.mark.parametrize(
    "limits",
    [
        {"allowedModels": [" "]},
        {"allowedModels": ["a", "a"]},
        {"maxConcurrentRequests": 0},
        {"requestsPerMinute": 10001},
    ],
)
def test_bad_limits_are_client_errors(active_client, limits):
    assert (
        active_client.post(
            "/v1/auth/client-keys", json={"name": "bad", "limits": limits}
        ).status_code
        == 422
    )


def test_logical_clock_ignores_wall_time_and_restart_downtime(monkeypatch):
    tick = [100.0]
    monkeypatch.setattr(admission.time, "perf_counter", lambda: tick[0])
    clock = admission.AdmissionClock(25)
    monkeypatch.setattr(admission.time, "time", lambda: 1e12)
    tick[0] += 3
    assert clock.now(25) == 28
    restarted = admission.AdmissionClock(28)
    assert restarted.now(28) == 28


def test_model_scope_concurrency_and_rate_are_one_allowance(active_client):
    client = active_client
    made = client.post(
        "/v1/auth/client-keys",
        json={
            "name": "bounded",
            "limits": {
                "allowedModels": ["allowed"],
                "maxConcurrentRequests": 1,
                "requestsPerMinute": 2,
            },
        },
    ).json()
    key = made["key"]["id"]
    path = "/v1/auth/client-keys/admission"

    def call(action, request_id, model="allowed"):
        return client.post(
            path, json={"action": action, "keyId": key, "requestId": request_id, "model": model}
        )

    assert call("acquire", "denied", "excluded").status_code == 403
    assert call("acquire", "gateway-one").status_code == 200
    refusal = call("acquire", "gateway-two")
    assert refusal.status_code == 429
    assert int(refusal.headers["Retry-After"]) > 0
    assert call("release", "gateway-one").status_code == 200
    assert call("acquire", "gateway-two").status_code == 200
    assert call("release", "gateway-two").status_code == 200
    assert call("acquire", "third").status_code == 429
    assert client.get("/v1/auth/client-keys").status_code == 200
