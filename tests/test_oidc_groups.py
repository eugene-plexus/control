"""Row 70c: what an app learns of permission groups, and how big a sign-in may grow.

`GET /oidc/groups` lists the groups that give the calling app, so Workbench's owner can
attach a memory server to one before any member signs in. It needs the app's own secret,
and says nothing of who is in a group.

The ID token carries `eugene_groups` and `eugene_policies` in at most 1024 bytes: an app
may keep the token in a cookie, and a browser drops a cookie over 4 KB. Past the cap the
token lists the groups that fit and says so; `/oidc/userinfo` always has them all.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_control import oidc

from .test_oidc import FIRST, _client, _id_claims, _tokens, _userinfo

CAP = 1024


def _groups(c: TestClient, app: dict[str, Any] | None = None) -> Any:
    bare = TestClient(c.app)
    if app is None:
        return bare.get("/oidc/groups")
    return bare.get("/oidc/groups", auth=(app["client"]["clientId"], app["clientSecret"]))


def _group(c: TestClient, name: str, **extra: Any) -> dict[str, Any]:
    made = c.post("/v1/permission-groups", json={"name": name, **extra})
    assert made.status_code == 201, made.text
    return dict(made.json())


def _size(claims: dict[str, Any]) -> int:
    kept = {k: claims[k] for k in ("eugene_groups", "eugene_policies") if k in claims}
    return len(json.dumps(kept, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def test_an_app_lists_the_groups_that_give_it(active_client: TestClient) -> None:
    c = active_client
    workbench, other = _client(c, "Workbench"), _client(c, "Other")
    mine = workbench["client"]["clientId"]
    every = _group(c, "Zeta everyone", apps=None)
    given = _group(c, "Alpha", apps=[mine])
    _group(c, "Beta elsewhere", apps=[other["client"]["clientId"]])
    _group(c, "Gamma nothing", apps=[])
    answer = _groups(c, workbench)
    assert answer.status_code == 200, answer.text
    assert answer.headers["cache-control"] == "no-store"
    assert answer.json() == {
        "groups": [
            {"id": given["id"], "name": "Alpha"},
            {"id": every["id"], "name": "Zeta everyone"},
        ]
    }


def test_without_the_apps_secret_no_group_is_listed(active_client: TestClient) -> None:
    c = active_client
    app = _client(c)
    _group(c, "Alpha", apps=None)
    unsigned = _groups(c)
    wrong = TestClient(c.app).get("/oidc/groups", auth=(app["client"]["clientId"], "not-it"))
    for refused in (unsigned, wrong):
        assert refused.status_code == 401
        assert refused.json()["error"] == "invalid_client"


def test_a_standby_lists_no_groups(active_client: TestClient, monkeypatch: Any) -> None:
    app = _client(active_client)
    monkeypatch.setattr(
        type(active_client.app.state.machine), "is_active", property(lambda s: False)
    )
    answer = _groups(active_client, app)
    assert answer.status_code == 503
    assert answer.json()["error"] == "temporarily_unavailable"


def test_the_id_token_caps_group_claims_and_userinfo_has_them_all(
    active_client: TestClient,
) -> None:
    c = active_client
    made = _client(c)
    app = made["client"]["clientId"]
    # 40 groups, each with an 80-character name and a policy, far past the cap.
    groups = [
        _group(
            c,
            f"Group {n:02d} " + "x" * 71,
            apps=[app],
            policies={"queueBand": "background", "summaryContextPercent": 70},
        )
        for n in range(40)
    ]
    person = c.post(
        "/v1/people", json={"name": "Bo", "password": FIRST, "groups": [g["id"] for g in groups]}
    )
    assert person.status_code == 201, person.text
    issued = _tokens(c, made, name="Bo", password=FIRST)
    claims = _id_claims(issued)
    assert _size(claims) <= CAP
    assert claims["eugene_groups_truncated"] is True
    listed = claims["eugene_groups"]
    assert 0 < len(listed) < 40
    # The groups that fit, in name order, from the start.
    assert listed == [{"id": g["id"], "name": g["name"]} for g in groups[: len(listed)]]
    # Every policy value stays; `setBy` names only the groups the token lists.
    assert claims["eugene_policies"]["queueBand"] == "background"
    assert claims["eugene_policies"]["summaryContextPercent"] == 70
    named = {g["id"] for g in listed}
    for ids in claims["eugene_policies"]["setBy"].values():
        assert set(ids) <= named
    # The whole ID token stays well inside a 4 KB cookie.
    assert len(issued["id_token"]) < 3500
    # userinfo is never cut.
    full = _userinfo(c, issued)
    assert len(full["eugene_groups"]) == 40
    assert "eugene_groups_truncated" not in full
    assert len(full["eugene_policies"]["setBy"]["queueBand"]) == 40


@pytest.mark.parametrize("count", [0, 1, 3])
def test_under_the_cap_the_id_token_holds_every_group(
    active_client: TestClient, count: int
) -> None:
    c = active_client
    made = _client(c)
    app = made["client"]["clientId"]
    groups = [_group(c, f"Team {n}", apps=[app]) for n in range(count)]
    body: dict[str, Any] = {"name": "Cy", "password": FIRST}
    body["groups"] = [g["id"] for g in groups] if count else None
    if body["groups"] is None:
        body["apps"] = [app]
    assert c.post("/v1/people", json=body).status_code == 201
    claims = _id_claims(_tokens(c, made, name="Cy", password=FIRST))
    assert "eugene_groups_truncated" not in claims
    assert len(claims["eugene_groups"]) == count


def test_the_cap_is_measured_in_bytes_not_characters() -> None:
    """A name in another script takes more bytes than letters: the cap counts the bytes the
    signer writes (non-ASCII escaped, as PyJWT serializes), and everything the cut adds to the
    token, the flag included, fits inside it."""
    groups = [{"id": f"{n:032x}", "name": "日本語のグループ名" * 8} for n in range(10)]
    policies = {"queueBand": "interactive", "setBy": {"queueBand": [g["id"] for g in groups]}}
    added = oidc.capped_group_claims(groups, policies, CAP)
    assert added["eugene_groups_truncated"] is True
    assert added["eugene_groups"] == groups[: len(added["eugene_groups"])]
    assert added["eugene_policies"]["queueBand"] == "interactive"
    assert added["eugene_policies"]["setBy"]["queueBand"] == [
        g["id"] for g in added["eugene_groups"]
    ]
    assert len(json.dumps(added, separators=(",", ":")).encode("ascii")) <= CAP
    # Under the cap nothing is cut and no flag is added.
    small = oidc.capped_group_claims(groups[:1], {}, CAP)
    assert small == {"eugene_groups": groups[:1], "eugene_policies": {}}


def _signed_claims_bytes(token: str) -> int:
    """The group claims' bytes as the signed token carries them, in the payload's own JSON
    (PyJWT escapes non-ASCII), never re-encoded some other way."""
    import base64

    payload = token.split(".")[1]
    claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    kept = {
        k: claims[k]
        for k in ("eugene_groups", "eugene_policies", "eugene_groups_truncated")
        if k in claims
    }
    return len(json.dumps(kept, separators=(",", ":")).encode("ascii"))


def test_the_cap_holds_for_the_bytes_signed_even_with_emoji(active_client: TestClient) -> None:
    """Review blocker 1: a name outside the Basic Multilingual Plane is escaped by the signer
    (12 bytes for one emoji), so the cap must count what is signed, not UTF-8."""
    c = active_client
    made = _client(c)
    app = made["client"]["clientId"]
    emoji = chr(0x1F600)
    groups = [_group(c, f"{n:02d}" + emoji * 65, apps=[app]) for n in range(64)]
    person = c.post(
        "/v1/people", json={"name": "Em", "password": FIRST, "groups": [g["id"] for g in groups]}
    )
    assert person.status_code == 201, person.text
    issued = _tokens(c, made, name="Em", password=FIRST)
    claims = _id_claims(issued)
    assert claims["eugene_groups_truncated"] is True
    assert _signed_claims_bytes(issued["id_token"]) <= CAP
    assert len(issued["id_token"]) < 3500
    assert len(_userinfo(c, issued)["eugene_groups"]) == 64


def test_a_wrong_secret_is_tried_five_times_a_minute_at_most(active_client: TestClient) -> None:
    """Review should-fix 4: guessing an app's secret at `/oidc/groups` is bounded per caller."""
    c = active_client
    app = _client(c)
    bare = TestClient(c.app)
    for _ in range(5):
        wrong = bare.get("/oidc/groups", auth=(app["client"]["clientId"], "wrong"))
        assert wrong.status_code == 401
    limited = bare.get("/oidc/groups", auth=(app["client"]["clientId"], "wrong"))
    assert limited.status_code == 429 and limited.headers.get("retry-after")
    assert limited.json()["error"] == "slow_down"
    # The right secret from the same caller waits too, as a sign-in does.
    assert _groups(c, app).status_code == 429


def test_an_app_reads_its_groups_sixty_times_a_minute_at_most(active_client: TestClient) -> None:
    c = active_client
    app = _client(c)
    for _ in range(60):
        assert _groups(c, app).status_code == 200
    limited = _groups(c, app)
    assert limited.status_code == 429 and limited.headers.get("retry-after")
