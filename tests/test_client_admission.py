"""A5: one durable allowance shared by every gateway using the registry."""

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pytest

from eugene_plexus_control import client_admission as admission
from eugene_plexus_control.applied import from_canonical, to_canonical
from tests.conftest import enroll, machine_at


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


def test_local_only_survives_snapshot_and_policy_edit_invalidates_lease(active_client, tmp_path):
    c = active_client
    key = mint(c, localOnly=True)["key"]["id"]
    assert call(c, key, "acquire").json()["limits"]["localOnly"] is True
    c.app.state.machine.snapshot_now()
    c.app.state.machine = machine_at(tmp_path / "active" / "state")
    assert call(c, key, "check").json()["limits"]["localOnly"] is True
    edited = c.put(f"/v1/auth/client-keys/{key}/limits", json={"limits": {"localOnly": False}})
    assert edited.status_code == 200
    assert call(c, key, "renew").status_code == 409


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


def test_slow_admission_disk_does_not_block_health(active_client, monkeypatch):
    c = active_client
    key = mint(c)["key"]["id"]
    entered, release = threading.Event(), threading.Event()
    original = c.app.state.machine._store.append

    def slow(entry):
        entered.set()
        assert release.wait(5)
        return original(entry)

    monkeypatch.setattr(c.app.state.machine._store, "append", slow)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(call, c, key, "acquire")
        assert entered.wait(2)
        timer = threading.Timer(1, release.set)
        timer.start()
        try:
            response = c.get("/healthz")
            assert response.status_code == 200
            assert not release.is_set(), "health waited for admission fsync"
        finally:
            release.set()
            timer.cancel()
        assert pending.result(timeout=2).status_code == 200


def test_concurrent_acquires_share_one_atomic_allowance(active_client):
    c = active_client
    key = mint(c, maxConcurrentRequests=1)["key"]["id"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda i: call(c, key, "acquire", f"parallel-{i}"), range(8)))
    assert sorted(r.status_code for r in responses) == [200] + [429] * 7


def test_admission_and_edit_audiences(active_client):
    c = active_client
    made = mint(c)
    key = made["key"]["id"]
    node, _ = enroll(c, "nas", grants=["gateway"])
    for kind, expected in [("agent", 200), ("gateway", 200), ("library", 401)]:
        h = {"Authorization": "Bearer " + node.service_token(sub=kind)}
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
        {"allowedTools": [" "]},
        {"allowedTools": ["web_search", "web_search"]},
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


def test_an_account_pattern_admits_its_models_and_nothing_else():
    """P1: the control root applies the same `*` rule as the agent and the
    gateway, so a key scoped to one account cannot reach another."""
    key = {"name": "App", "expiresAt": 4_102_444_800, "limits": {"allowedModels": ["openrouter/*"]}}
    ledger = {"clock": 0.0, "buckets": {}}
    result, _ = admission.decide(
        ledger,
        key,
        key_id="k",
        action="acquire",
        request_id="r1",
        model="openrouter/anthropic/claude-opus-5.5",
        now=1.0,
    )
    assert result["leaseSeconds"] > 0
    with pytest.raises(admission.AdmissionRefusal) as caught:
        admission.decide(
            ledger,
            key,
            key_id="k",
            action="acquire",
            request_id="r2",
            model="work/anthropic/claude-opus-5.5",
            now=1.0,
        )
    assert caught.value.status == 403


def test_a_key_that_may_send_logs_says_so_to_the_ingress(active_client):
    """C1: an app's key is minted at the root with `writeLogs`, and the
    agent's log ingress asks the root about it with `check`."""
    key = mint(active_client, writeLogs=True)["key"]
    assert key["limits"]["writeLogs"] is True
    answer = call(active_client, key["id"], "check", model=None)
    assert answer.status_code == 200, answer.text
    assert answer.json()["limits"]["writeLogs"] is True
    plain = mint(active_client)["key"]
    # The response fills in the default, as it does for localOnly.
    assert plain["limits"].get("writeLogs") in (None, False)
    answer = call(active_client, plain["id"], "check", model=None)
    assert answer.json()["limits"].get("writeLogs") is not True
    with pytest.raises(ValueError):
        admission.validate_limits({"writeLogs": 1})


def search_call(client, key, action, request_id="s1", **extra):
    return client.post(
        "/v1/auth/client-keys/admission",
        json={
            "keyId": key,
            "action": action,
            "requestId": request_id,
            "purpose": "search",
            **extra,
        },
    )


def test_a_search_is_admitted_with_no_model_under_the_keys_limits(active_client):
    """Row 34: Open WebUI's web search has no model, and still counts."""
    c = active_client
    key = mint(c, maxConcurrentRequests=1, requestsPerMinute=2, allowedModels=["chat"])["key"]["id"]
    # The model list does not apply to a search; the concurrency limit does.
    assert search_call(c, key, "acquire", "s1").json()["leaseSeconds"] > 0
    assert search_call(c, key, "renew", "s1").status_code == 200
    busy = search_call(c, key, "acquire", "s2")
    assert busy.status_code == 429
    assert int(busy.headers["Retry-After"]) > 0
    assert search_call(c, key, "release", "s1").status_code == 200
    assert search_call(c, key, "acquire", "s2").status_code == 200
    assert search_call(c, key, "release", "s2").status_code == 200
    # Two searches used the rolling-minute allowance.
    assert search_call(c, key, "acquire", "s3").status_code == 429


def test_a_search_names_no_model_and_an_inference_still_does(active_client):
    c = active_client
    key = mint(c)["key"]["id"]
    assert search_call(c, key, "acquire", model="chat").status_code == 422
    assert call(c, key, "acquire", "i1", model=None).status_code == 422
    assert search_call(c, key, "acquire", "s1").status_code == 200
    # One request identifier is one kind of work.
    assert call(c, key, "acquire", "s1").status_code == 409


# --------------------------------------------------------------------------- #
# row 70: a verified person's queue band from their groups
# --------------------------------------------------------------------------- #


def _band_group(client, name, band):
    from tests.test_permission_groups import group

    made = group(client, name=name, apps=None, policies={"queueBand": band})
    assert made["policies"]["queueBand"] == band
    return made


def test_a_verified_person_carries_their_groups_band(active_client):
    from tests.test_person_access import call as person_call
    from tests.test_person_access import setup

    c = active_client
    _, g, p, signed, key = setup(c)
    proof = signed["access_token"]
    # In a group that sets no band: no band, so the key's own or the default applies.
    assert "queueBand" not in person_call(c, key, proof).json()
    # Named to sort first, so taking the first group's band would be caught.
    night = _band_group(c, "Archive", "background")
    c.patch(f"/v1/people/{p['id']}", json={"groups": [g["id"], night["id"]]})
    for action in ("check", "acquire", "renew"):
        answer = person_call(c, key, proof, action)
        assert answer.status_code == 200, answer.text
        assert answer.json()["queueBand"] == "background", action
    # The most generous band of all their groups, read now, not at sign-in.
    desk = _band_group(c, "Desk", "interactive")
    c.patch(f"/v1/people/{p['id']}", json={"groups": [g["id"], night["id"], desk["id"]]})
    assert person_call(c, key, proof).json()["queueBand"] == "interactive"
    # A band change is not a permission change: the lease renews.
    assert person_call(c, key, proof, "renew").status_code == 200
    assert "queueBand" not in person_call(c, key, action="release").json()
    # The caller cannot name its own band.
    forged = c.post(
        "/v1/auth/client-keys/admission",
        json={
            "keyId": key,
            "requestId": "forged",
            "action": "check",
            "personToken": proof,
            "queueBand": "interactive",
        },
    )
    assert forged.status_code == 422


def test_a_key_without_a_person_carries_no_band(active_client):
    from tests.test_permission_groups import person

    c = active_client
    night = _band_group(c, "Night shift", "background")
    person(c, groups=[night["id"]])
    key = mint(c)["key"]["id"]
    for action in ("check", "acquire", "renew", "release"):
        answer = call(c, key, action)
        assert answer.status_code == 200, answer.text
        assert "queueBand" not in answer.json() and "personId" not in answer.json()


def _rotate(c, key_id, overlap=600):
    response = c.post(f"/v1/auth/client-keys/{key_id}/rotate", json={"overlapSeconds": overlap})
    assert response.status_code == 201, response.text
    return response.json()["key"]["id"]


def test_acquire_records_last_used_to_the_minute(active_client, tmp_path):
    c = active_client
    key = mint(c)["key"]["id"]
    assert "lastUsedAt" not in c.get("/v1/auth/client-keys").json()["keys"][0]
    assert call(c, key, "acquire").status_code == 200
    used = c.get("/v1/auth/client-keys").json()["keys"][0]["lastUsedAt"]
    assert used.endswith("00Z") or ":00" in used  # to the minute
    assert datetime.fromisoformat(used.replace("Z", "+00:00")).second == 0
    # A check writes at most one log entry a minute, and the value survives a restart.
    machine = c.app.state.machine
    before = machine.state.index
    for n in range(5):
        assert call(c, key, "check", f"c{n}").status_code == 200
    assert machine.state.index - before <= 1
    recovered = machine_at(tmp_path / "active" / "state")
    stored = recovered.state.client_keys[key]["lastUsedAt"]
    assert datetime.fromisoformat(stored) == datetime.fromisoformat(used.replace("Z", "+00:00"))


def test_old_and_new_tokens_share_one_concurrency_limit(active_client):
    c = active_client
    old = mint(c, maxConcurrentRequests=1, requestsPerMinute=60)["key"]["id"]
    new = _rotate(c, old)
    assert call(c, old, "acquire", "stream").status_code == 200
    assert call(c, new, "acquire", "other").status_code == 429
    assert call(c, old, "release", "stream").status_code == 200
    assert call(c, new, "acquire", "other").status_code == 200


def test_a_retired_key_renews_but_cannot_acquire(active_client):
    c = active_client
    old = mint(c, maxConcurrentRequests=2)["key"]["id"]
    assert call(c, old, "acquire", "stream").status_code == 200
    new = _rotate(c, old, overlap=0)
    refused = call(c, old, "acquire", "next")
    assert refused.status_code == 401 and "replaced" in refused.text
    assert call(c, old, "check", "chk").status_code == 401
    assert call(c, old, "renew", "stream").status_code == 200
    assert call(c, old, "release", "stream").status_code == 200
    assert call(c, new, "acquire", "next").status_code == 200


# --- row 11: the gateway's queue reads these ------------------------------------


def test_concurrency_refusal_names_its_reason(active_client):
    """A 429 says which limit it is: the gateway keeps a request refused for
    `concurrency` in its queue (another gateway holds the key's leases) and
    refuses one refused for `rate`."""
    c = active_client
    key = mint(c, maxConcurrentRequests=1, requestsPerMinute=2)["key"]["id"]
    assert call(c, key, "acquire", "one").status_code == 200
    busy = call(c, key, "acquire", "two")
    assert busy.status_code == 429
    assert busy.headers["X-Eugene-Refusal-Reason"] == "concurrency"
    assert int(busy.headers["Retry-After"]) > 0
    assert call(c, key, "release", "one").status_code == 200
    assert call(c, key, "acquire", "two").status_code == 200
    assert call(c, key, "release", "two").status_code == 200
    used = call(c, key, "acquire", "three")
    assert used.status_code == 429
    assert used.headers["X-Eugene-Refusal-Reason"] == "rate"


def test_band_change_does_not_cut_a_renewal(active_client):
    """The band and the queue bound are not part of the policy a running
    request is bound to: changing them never cuts a request in flight."""
    c = active_client
    key = mint(c, maxConcurrentRequests=2, requestsPerMinute=60)["key"]["id"]
    assert call(c, key, "acquire").status_code == 200
    for limits in (
        {"band": "background"},
        {"band": "sheddable", "maxQueuedRequests": 0},
        {"maxQueuedRequests": 64},
    ):
        edited = c.put(
            f"/v1/auth/client-keys/{key}/limits",
            json={"limits": {"maxConcurrentRequests": 2, "requestsPerMinute": 60, **limits}},
        )
        assert edited.status_code == 200, edited.text
        assert call(c, key, "renew").status_code == 200, limits
    answered = call(c, key, "check", "peek").json()["limits"]
    assert answered["maxQueuedRequests"] == 64 and "band" not in answered
    # What the request is bound to still is: a new concurrency limit cuts it.
    edited = c.put(
        f"/v1/auth/client-keys/{key}/limits",
        json={"limits": {"maxConcurrentRequests": 3, "requestsPerMinute": 60}},
    )
    assert edited.status_code == 200
    assert call(c, key, "renew").status_code == 409


def test_a_key_band_and_queue_bound_are_kept_and_checked(active_client):
    c = active_client
    made = mint(c, band="background", maxQueuedRequests=0)["key"]
    assert made["limits"]["band"] == "background"
    assert made["limits"]["maxQueuedRequests"] == 0
    answered = call(c, made["id"], "check").json()["limits"]
    assert answered["band"] == "background" and answered["maxQueuedRequests"] == 0
    # A key nobody gave a band or a bound is answered without them (absent is
    # interactive and 8), so a gateway a release behind, which refuses fields it
    # does not know, still reads its admission answers.
    plain = mint(c)["key"]
    assert not {"band", "maxQueuedRequests"} & plain["limits"].keys()
    answered = call(c, plain["id"], "acquire", "plain").json()["limits"]
    assert not {"band", "maxQueuedRequests"} & answered.keys()
    for bad in ({"band": "urgent"}, {"maxQueuedRequests": 65}, {"maxQueuedRequests": -1}):
        response = c.post("/v1/auth/client-keys", json={"name": "bad", "limits": bad})
        assert response.status_code == 422, bad


def test_a_check_with_a_model_applies_allowed_models(active_client):
    """The gateway checks when a request arrives and acquires only when it
    leaves its queue, so the check is where a model outside the key's scope
    is refused, before the request waits for anything."""
    c = active_client
    key = mint(c, allowedModels=["allowed"])["key"]["id"]
    assert call(c, key, "check", model="allowed").status_code == 200
    refused = call(c, key, "check", model="excluded")
    assert refused.status_code == 403
    assert "does not permit" in refused.text
    # A check without a model (the model list, the app catalogue) still answers.
    bare = c.post(
        "/v1/auth/client-keys/admission",
        json={"keyId": key, "action": "check", "requestId": "list"},
    )
    assert bare.status_code == 200
    # Nothing was taken: the key's whole allowance is still there.
    assert call(c, key, "acquire", "after").status_code == 200
