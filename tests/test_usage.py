"""Row 40: usage counted install-wide, allowances, and whose price is used."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from eugene_plexus_control import usage as counts
from eugene_plexus_control.applied import from_canonical, to_canonical
from tests.conftest import machine_at
from tests.test_oidc import CALLBACK, _tokens
from tests.test_permission_groups import PASSWORD, group, person

LOCAL = {"locality": "local"}
CLOUD = {"locality": "external"}
#: $3 and $15 a million tokens, as a provider publishes them.
PROVIDER = {
    "locality": "external",
    "providerPrice": {"inputPerMillionMicros": 3_000_000, "outputPerMillionMicros": 15_000_000},
}


def mint(c, name="metered", **limits):
    response = c.post("/v1/auth/client-keys", json={"name": name, "limits": limits})
    assert response.status_code == 201, response.text
    return response.json()["key"]["id"]


def admit(c, key, action, request_id, *, model="m", token=None, **extra):
    body: dict[str, Any] = {"keyId": key, "action": action, "requestId": request_id, **extra}
    if action != "check" or model:
        body["model"] = model
    if token is not None and action != "release":
        body["personToken"] = token
    return c.post("/v1/auth/client-keys/admission", json=body)


def serve(c, key, request_id, *, model="m", prompt=10, completion=5, hint=None, token=None):
    """One request through the authority, as a gateway runs it: acquire, release with use."""
    hint = hint if hint is not None else LOCAL
    got = admit(c, key, "acquire", request_id, model=model, token=token, price=hint)
    if got.status_code != 200:
        return got
    used = {"model": model, "promptTokens": prompt, "completionTokens": completion, "price": hint}
    released = admit(c, key, "release", request_id, model=model, usage=used)
    assert released.status_code == 200, released.text
    return got


def subject(c, kind, ident, **query):
    report = c.get("/v1/usage", params=query)
    assert report.status_code == 200, report.text
    return next(s for s in report.json()["subjects"] if s["kind"] == kind and s["id"] == ident)


def test_two_gateways_count_into_one_total(active_client):
    """Every gateway releases to the one authority: the key's day is one total,
    a second release of one request counts nothing, and the allowance holds
    whichever gateway asks next."""
    c = active_client
    key = mint(c, tokensPerDay=1000)
    assert serve(c, key, "on-gateway-a", prompt=400, completion=200).status_code == 200
    assert serve(c, key, "on-gateway-b", prompt=300, completion=150).status_code == 200
    # The same request released again (a retried release) counts nothing.
    again = {"model": "m", "promptTokens": 400, "completionTokens": 200, "price": LOCAL}
    assert admit(c, key, "release", "on-gateway-a", usage=again).status_code == 200
    day = subject(c, "key", key)["day"]
    assert (day["requests"], day["tokens"]) == (2, 1050), day
    refused = admit(c, key, "acquire", "third", price=LOCAL)
    assert refused.status_code == 429, refused.text
    assert refused.headers["X-Eugene-Refusal-Reason"] == "allowance"
    assert "This key used 1,050 of its 1,000 tokens" in refused.json()["detail"]
    assert "It resets at 00:00 UTC on" in refused.json()["detail"]
    # A check on arrival with a model is refused the same way, before any queue.
    assert admit(c, key, "check", "fourth").status_code == 429
    # A check without a model (a model list) is not: the key still lists models.
    assert admit(c, key, "check", "list", model=None).status_code == 200


def test_a_rotated_key_shares_its_lineage_allowance(active_client):
    c = active_client
    old = mint(c, tokensPerDay=100)
    assert serve(c, old, "before", prompt=60, completion=40).status_code == 200
    rotated = c.post(f"/v1/auth/client-keys/{old}/rotate", json={"overlapSeconds": 600})
    new = rotated.json()["key"]["id"]
    assert admit(c, new, "acquire", "after", price=LOCAL).status_code == 429
    assert subject(c, "key", old)["day"]["tokens"] == 100


def _signed(c, owner="app:workbench@node"):
    app = c.post(
        "/v1/oidc/clients", json={"name": "Workbench", "owner": owner, "redirectUris": [CALLBACK]}
    ).json()
    key = mint(c, name=owner)
    return app, key


def test_a_group_allowance_pools_its_members(active_client):
    c = active_client
    app, key = _signed(c)
    sales = group(c, name="Sales", apps=[app["client"]["clientId"]], policies={"tokensPerDay": 30})
    ada = person(c, name="Ada", groups=[sales["id"]])
    bob = person(c, name="Bob", groups=[sales["id"]])
    ada_token = _tokens(c, app, name="Ada", password=PASSWORD)["access_token"]
    bob_token = _tokens(c, app, name="Bob", password=PASSWORD)["access_token"]
    assert serve(c, key, "ada-1", prompt=15, completion=5, token=ada_token).status_code == 200
    assert serve(c, key, "bob-1", prompt=8, completion=4, token=bob_token).status_code == 200
    # Together they used 32 of the group's 30: either is refused, naming the group.
    for token in (ada_token, bob_token):
        refused = admit(c, key, "acquire", f"next-{token[-6:]}", token=token, price=LOCAL)
        assert refused.status_code == 429, refused.text
        assert refused.json()["detail"].startswith("Your group Sales used 32 of its 30 tokens")
    assert subject(c, "group", sales["id"])["day"]["tokens"] == 32
    assert subject(c, "person", ada["id"])["day"]["tokens"] == 20
    assert subject(c, "person", bob["id"])["day"]["tokens"] == 12


def test_the_most_generous_group_admits_a_person(active_client):
    c = active_client
    app, key = _signed(c)
    client_id = app["client"]["clientId"]
    small = group(c, name="Small", apps=[client_id], policies={"tokensPerDay": 10})
    large = group(c, name="Large", apps=[client_id], policies={"tokensPerDay": 1000})
    person(c, name="Ada", groups=[small["id"], large["id"]])
    token = _tokens(c, app, name="Ada", password=PASSWORD)["access_token"]
    assert serve(c, key, "one", prompt=20, token=token).status_code == 200
    # Small is spent, Large is not: she is still admitted.
    assert admit(c, key, "acquire", "two", token=token, price=LOCAL).status_code == 200
    mine = c.get("/oidc/usage", headers={"Authorization": f"Bearer {token}"})
    assert mine.status_code == 200, mine.text
    kinds = {(s["kind"], s["name"]) for s in mine.json()["subjects"]}
    assert kinds == {("person", "Ada"), ("group", "Large"), ("group", "Small")}
    assert c.get("/oidc/usage", headers={"Authorization": "Bearer forged"}).status_code == 401


def test_a_spent_monthly_allowance_refuses_with_the_reset_date(active_client):
    c = active_client
    key = mint(c, costPerMonthMicros=10_000)  # $0.01
    # 1,000 tokens in at $3 a million and 500 out at $15: $0.0105.
    assert serve(c, key, "first", prompt=1000, completion=500, hint=PROVIDER).status_code == 200
    refused = admit(c, key, "acquire", "second", price=PROVIDER)
    assert refused.status_code == 429, refused.text
    now = datetime.now(UTC)
    nxt = counts._next_month(counts.month_of(now))
    month_name = now.strftime("%B")
    detail = refused.json()["detail"]
    assert detail == (
        f"This key used $0.0105 of its $0.01 for {month_name}. It resets on 1 {nxt.strftime('%B')}."
    ), detail
    retry = int(refused.headers["Retry-After"])
    reset = datetime(nxt.year, nxt.month, 1, tzinfo=UTC)
    assert abs(retry - (reset - now).total_seconds()) < 5
    # A model that costs nothing is still served under a spent money allowance.
    assert admit(c, key, "acquire", "free", price=LOCAL).status_code == 200
    month = subject(c, "key", key)["month"]
    assert month["costUsd"] == "0.010500000000"
    allowance = subject(c, "key", key)["allowances"][0]
    assert allowance == {
        "kind": "costPerMonth",
        "limit": "0.010000000000",
        "used": "0.010500000000",
        "left": "0.000000000000",
        "spent": True,
        "resetsAt": reset.isoformat().replace("+00:00", "Z"),
    }


def test_an_unknown_price_is_unknown_not_zero(active_client):
    c = active_client
    open_key = mint(c, name="open")
    assert serve(c, open_key, "cloud", prompt=100, completion=100, hint=CLOUD).status_code == 200
    month = subject(c, "key", open_key)["month"]
    assert month["unpricedRequests"] == 1 and month["costUsd"] == "0.000000000000"
    assert month["models"][0]["unpricedRequests"] == 1
    # Under a money allowance a model with no price is refused before it runs.
    metered = mint(c, name="metered", costPerMonthMicros=1_000_000)
    refused = admit(c, metered, "acquire", "cloud-2", model="cloud/x", price=CLOUD)
    assert refused.status_code == 429
    assert refused.json()["detail"] == (
        "No price is set for cloud/x, so a money limit cannot allow it. "
        "The owner sets one on the Usage page."
    )
    # An older gateway sends no hint: unknown, refused the same way.
    assert admit(c, metered, "acquire", "old-gateway", model="cloud/x").status_code == 429
    # The owner's rate prices it.
    c.put(
        "/v1/usage/prices",
        json={
            "model": "cloud/x",
            "price": {"inputPerMillionMicros": 1, "outputPerMillionMicros": 1},
        },
    ).raise_for_status()
    assert admit(c, metered, "acquire", "cloud-3", model="cloud/x", price=CLOUD).status_code == 200


def test_the_owners_rate_wins_over_the_providers_price(active_client):
    c = active_client
    key = mint(c)
    set_ = c.put(
        "/v1/usage/prices",
        json={
            "model": "m",
            "price": {"inputPerMillionMicros": 1_000_000, "outputPerMillionMicros": 2_000_000},
        },
    )
    assert set_.status_code == 200, set_.text
    assert set_.json() == {
        "prices": [
            {"model": "m", "inputPerMillionMicros": 1_000_000, "outputPerMillionMicros": 2_000_000}
        ]
    }
    assert serve(c, key, "one", prompt=1000, completion=1000, hint=PROVIDER).status_code == 200
    # $0.001 in and $0.002 out at the owner's rate, not $0.018 at the provider's.
    assert subject(c, "key", key)["day"]["costUsd"] == "0.003000000000"
    cleared = c.put("/v1/usage/prices", json={"model": "m", "price": None})
    assert cleared.json() == {"prices": []}
    assert serve(c, key, "two", prompt=1000, completion=1000, hint=PROVIDER).status_code == 200
    assert subject(c, "key", key)["day"]["costUsd"] == "0.021000000000"


def test_a_local_model_counts_once_it_has_a_rate(active_client):
    c = active_client
    key = mint(c, costPerMonthMicros=1)
    assert serve(c, key, "free", prompt=10_000, completion=10_000).status_code == 200
    assert subject(c, "key", key)["month"]["costUsd"] == "0.000000000000"
    assert serve(c, key, "free-2").status_code == 200
    c.put(
        "/v1/usage/prices",
        json={"model": "m", "price": {"inputPerMillionMicros": 100, "outputPerMillionMicros": 100}},
    ).raise_for_status()
    # Past requests keep the cost they had; this one counts at the new rate.
    assert serve(c, key, "priced", prompt=10_000, completion=10_000).status_code == 200
    assert subject(c, "key", key)["month"]["costUsd"] == "0.000002000000"
    refused = admit(c, key, "acquire", "after", price=LOCAL)
    assert refused.status_code == 429
    assert refused.json()["detail"].startswith("This key used $0.000002 of its $0.000001")


def test_a_changed_allowance_does_not_cut_a_running_request(active_client):
    c = active_client
    key = mint(c, tokensPerDay=1000)
    assert admit(c, key, "acquire", "running", price=LOCAL).status_code == 200
    edited = c.put(f"/v1/auth/client-keys/{key}/limits", json={"limits": {"tokensPerDay": 5}})
    assert edited.status_code == 200 and edited.json()["limits"]["tokensPerDay"] == 5
    assert admit(c, key, "renew", "running").status_code == 200
    # Allowances never leave the authority in an admission answer.
    assert "tokensPerDay" not in admit(c, key, "renew", "running").json()["limits"]


@pytest.mark.parametrize(
    "limits",
    [
        {"tokensPerDay": 0},
        {"tokensPerMonth": -1},
        {"costPerMonthMicros": -1},
        {"costPerMonthMicros": 1.5},
    ],
)
def test_bad_allowances_are_refused(active_client, limits):
    response = active_client.post("/v1/auth/client-keys", json={"name": "bad", "limits": limits})
    assert response.status_code == 422, response.text


def test_counters_survive_a_snapshot_and_replay(active_client, tmp_path):
    c = active_client
    key = mint(c)
    assert serve(c, key, "one", hint=PROVIDER).status_code == 200
    c.put(
        "/v1/usage/prices",
        json={"model": "x", "price": {"inputPerMillionMicros": 7, "outputPerMillionMicros": 9}},
    ).raise_for_status()
    machine = c.app.state.machine
    for snapshot in (False, True):
        if snapshot:
            machine.snapshot_now()
        recovered = machine_at(tmp_path / "active" / "state")
        assert recovered.state.client_usage == machine.state.client_usage
        assert recovered.state.model_prices == machine.state.model_prices
        assert from_canonical(to_canonical(recovered.state)).client_usage == (
            machine.state.client_usage
        )


def test_a_directory_administrator_sees_no_keys(active_client):
    from tests.test_administration import delegate

    c = active_client
    mint(c)
    sales = group(c, name="Sales", policies={"costPerMonthMicros": 5_000_000})
    _, auth = delegate(c, membershipGroups=[sales["id"]])
    seen = c.get("/v1/usage", headers=auth)
    assert seen.status_code == 200, seen.text
    kinds = {s["kind"] for s in seen.json()["subjects"]}
    assert kinds <= {"group", "person"} and "group" in kinds


def test_an_older_standby_holds_back_the_first_allowance(active_client, monkeypatch):
    from eugene_plexus_control.routes import control

    c = active_client
    monkeypatch.setattr(control, "standby_short_of", lambda app, needed: "tower")
    refused = c.post("/v1/auth/client-keys", json={"name": "x", "limits": {"tokensPerDay": 5}})
    assert refused.status_code == 409 and "Update tower first" in refused.text
    price = c.put(
        "/v1/usage/prices",
        json={"model": "m", "price": {"inputPerMillionMicros": 1, "outputPerMillionMicros": 1}},
    )
    assert price.status_code == 409
    # A key with no allowance is written as before.
    assert c.post("/v1/auth/client-keys", json={"name": "y"}).status_code == 201


def test_counters_are_bounded_and_dropped_after_keeping():
    usage = counts.empty()
    for n in range(40):
        usage = counts.counted(
            usage,
            {
                "requestId": f"r{n}",
                "subjects": ["key:k"],
                "day": "2026-01-01",
                "month": "2026-01",
                "model": f"model-{n}",
                "promptTokens": 1,
                "completionTokens": 1,
                "costPicos": None,
            },
        )
    models = usage["counters"]["key:k"]["d:2026-01-01"]["models"]
    assert len(models) == counts.MODELS_KEPT + 1 and models["*"]["requests"] == 8
    later = counts.counted(
        usage,
        {
            "requestId": "late",
            "subjects": ["key:other"],
            "day": "2027-03-01",
            "month": "2027-03",
            "model": "m",
            "promptTokens": 1,
            "completionTokens": 0,
            "costPicos": 0,
        },
    )
    assert "key:k" not in later["counters"], "14 months on, nothing of k is kept"
    assert counts.validate(later) == later
