"""Row 5 (5a): the sign-in key changes, by itself or on demand.

`specs/docs/design/rows/row-05.md` §2. A routine change keeps the old key
verifying until the last refresh token it signed has expired, so apps
stay signed in; *End every app sign-in now* retires it at once. The whole
flow through a real Authlib client is `specs/scripts/c2-sign-in-acceptance.py`.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_control import oidc_keys
from eugene_plexus_control.applied import (
    OP_PUT_OIDC_KEY,
    OP_RETIRE_OIDC_KEYS,
    ApplyError,
    apply,
    from_canonical,
    to_canonical,
)

from .conftest import enroll
from .test_oidc import _client, _refresh, _tokens


def _kid(token: str) -> str:
    return str(jwt.get_unverified_header(token)["kid"])


def _later(monkeypatch: pytest.MonkeyPatch, days: float) -> datetime:
    """Move the key schedule's clock, and nothing else: tokens still expire
    by the real one, so a refresh token signed now is live throughout."""
    moment = datetime.now(UTC) + timedelta(days=days)
    monkeypatch.setattr(oidc_keys, "now", lambda: moment)
    return moment


def _jwks_kids(c: TestClient) -> list[str]:
    return [k["kid"] for k in c.get("/oidc/jwks").json()["keys"]]


def test_change_signs_with_new_key_and_old_refresh_still_works(
    active_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = active_client
    made = _client(c)
    issued = _tokens(c, made)
    old = _kid(issued["id_token"])

    listed = c.get("/v1/oidc/keys").json()
    assert [k["kid"] for k in listed["keys"]] == [old]
    assert listed["keys"][0]["signs"] is True and "verifiesUntil" not in listed["keys"][0]
    made_at = datetime.fromisoformat(listed["keys"][0]["createdAt"])
    assert datetime.fromisoformat(listed["nextChangeAt"]) == made_at + timedelta(days=90)

    # Too soon: the key it would drop (none yet, but the rule is the age of
    # the newest) may have signed refresh tokens that are still live.
    refused = c.post("/v1/oidc/keys/change", json={})
    assert refused.status_code == 409, refused.text
    detail = refused.json()["detail"]["detail"]
    due = made_at + timedelta(days=30, minutes=10)
    assert detail == (
        f"The sign-in key changed on {made_at.day} {made_at:%b}. Changing it again before "
        f"{due.day} {due:%b} ends some app sign-ins. Choose End every app sign-in now to do "
        "it anyway."
    )

    _later(monkeypatch, 31)
    changed = c.post("/v1/oidc/keys/change", json={})
    assert changed.status_code == 200, changed.text
    keys = changed.json()["keys"]
    new = keys[0]["kid"]
    assert new != old and keys[0]["signs"] is True
    assert keys[1]["kid"] == old and keys[1]["signs"] is False
    new_at = datetime.fromisoformat(keys[0]["createdAt"])
    assert datetime.fromisoformat(keys[1]["verifiesUntil"]) == new_at + timedelta(days=30)
    assert _jwks_kids(c) == [new, old]

    # The app stays signed in: the refresh token the old key signed still
    # works, and what it gets back is signed with the new key.
    refreshed = _refresh(c, made, issued["refresh_token"])
    assert refreshed.status_code == 200, refreshed.text
    assert _kid(refreshed.json()["id_token"]) == new
    assert _kid(refreshed.json()["access_token"]) == new
    # And a new sign-in verifies against the new key.
    again = _tokens(c, made)
    assert _kid(again["id_token"]) == new


def test_end_sign_ins_refuses_old_refresh(active_client: TestClient) -> None:
    c = active_client
    made = _client(c)
    issued = _tokens(c, made)
    old = _kid(issued["id_token"])

    # Not held back by the key's age: after a leak, now is the time.
    changed = c.post("/v1/oidc/keys/change", json={"endSignIns": True})
    assert changed.status_code == 200, changed.text
    keys = changed.json()["keys"]
    assert len(keys) == 1 and keys[0]["kid"] != old and keys[0]["signs"] is True
    assert _jwks_kids(c) == [keys[0]["kid"]]

    refreshed = _refresh(c, made, issued["refresh_token"])
    assert refreshed.json().get("error") == "invalid_grant", refreshed.text
    userinfo = c.get(
        "/oidc/userinfo", headers={"Authorization": f"Bearer {issued['access_token']}"}
    )
    assert userinfo.status_code == 401

    again = _tokens(c, made)
    assert _kid(again["id_token"]) == keys[0]["kid"]
    assert _refresh(c, made, again["refresh_token"]).status_code == 200


def test_the_key_change_is_operator_only(active_client: TestClient) -> None:
    c = active_client
    for method, path, body in (
        ("get", "/v1/oidc/keys", None),
        ("post", "/v1/oidc/keys/change", {}),
    ):
        answer = c.request(method, path, json=body, headers={"Authorization": ""})
        assert answer.status_code == 401, (path, answer.text)


def test_end_sign_ins_waits_for_a_standby_that_cannot_apply_it(active_client: TestClient) -> None:
    c = active_client
    enroll(c, "b")
    assert c.put("/v1/nodes/b/standby", json={"on": True}).status_code in (200, 202, 204)
    reports: dict[str, Any] = c.app.state.standby_reports  # type: ignore[attr-defined]
    _tokens(c, _client(c))

    # A standby older than row 5 sends no log version: it would stop here.
    reports["b"] = {"appliedIndex": 0, "at": datetime.now(UTC), "seen": time.perf_counter()}
    refused = c.post("/v1/oidc/keys/change", json={"endSignIns": True})
    assert refused.status_code == 409, refused.text
    assert (
        refused.json()["detail"]["detail"] == "Update b first; its copy would stop at this change."
    )
    status = c.get("/v1/control/status").json()
    assert status["standbys"][0]["logVersion"] == 0

    reports["b"]["logVersion"] = oidc_keys.LOG_VERSION
    assert c.post("/v1/oidc/keys/change", json={"endSignIns": True}).status_code == 200


def test_the_root_changes_the_key_when_it_is_due(
    active_client: TestClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    c = active_client
    app = c.app
    _tokens(c, _client(c))
    first = c.get("/v1/oidc/keys").json()["keys"][0]["kid"]

    _later(monkeypatch, 89)
    assert oidc_keys.tick(app) is False
    _later(monkeypatch, 90.01)
    assert oidc_keys.tick(app) is True
    keys = c.get("/v1/oidc/keys").json()["keys"]
    assert len(keys) == 2 and keys[1]["kid"] == first
    assert oidc_keys.tick(app) is False

    # 0 means never.
    assert c.patch("/v1/config", json={"signInKeyDays": 0}).status_code == 200
    _later(monkeypatch, 400)
    assert oidc_keys.tick(app) is False
    assert "nextChangeAt" not in c.get("/v1/oidc/keys").json()

    # A locked root says so, once, and waits.
    assert c.patch("/v1/config", json={"signInKeyDays": 31}).status_code == 200
    app.state.auth_state.forget_master_key()  # type: ignore[attr-defined]
    caplog.set_level(logging.INFO, logger="eugene_plexus_control.oidc_keys")
    assert oidc_keys.tick(app) is False
    assert oidc_keys.tick(app) is False
    locked = [r for r in caplog.records if "locked" in r.getMessage()]
    assert len(locked) == 1


@pytest.mark.parametrize(
    ("days", "ok"), [(0, True), (31, True), (365, True), (1, False), (30, False)]
)
def test_sign_in_key_days_is_zero_or_at_least_31(
    active_client: TestClient, days: int, ok: bool
) -> None:
    answer = active_client.patch("/v1/config", json={"signInKeyDays": days})
    body = answer.json()
    rejected = [e["key"] for e in body.get("rejected") or []]
    assert ("signInKeyDays" not in rejected) is ok, body


def test_retire_keeps_only_held_keys_and_replays(active_client: TestClient) -> None:
    c = active_client
    _tokens(c, _client(c))
    machine = c.app.state.machine  # type: ignore[attr-defined]
    kid = machine.state.oidc_keys[0]["kid"]
    state = machine.state
    with pytest.raises(ApplyError):
        apply(state, {"index": state.index + 1, "epoch": state.epoch, "op": OP_RETIRE_OIDC_KEYS,
                      "payload": {"keep": ["not-a-held-key"]}})  # fmt: skip
    with pytest.raises(ApplyError):
        apply(state, {"index": state.index + 1, "epoch": state.epoch, "op": OP_RETIRE_OIDC_KEYS,
                      "payload": {"keep": []}})  # fmt: skip
    after = apply(
        state,
        {"index": state.index + 1, "epoch": state.epoch, "op": OP_RETIRE_OIDC_KEYS,
         "payload": {"keep": [kid]}},
    )  # fmt: skip
    assert [k["kid"] for k in after.oidc_keys] == [kid]
    assert to_canonical(from_canonical(to_canonical(after))) == to_canonical(after)
    assert OP_PUT_OIDC_KEY != OP_RETIRE_OIDC_KEYS


def test_two_routine_changes_at_once_change_the_key_once(
    active_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reproduction: with the newest key 31 days old, two `{}` changes at
    the same moment both passed the age check and dropped the key whose
    refresh tokens were still live. The check and the change are one step."""
    import contextlib
    import threading

    from eugene_plexus_control import oidc

    c = active_client
    _tokens(c, _client(c))
    first = c.get("/v1/oidc/keys").json()["keys"][0]["kid"]
    _later(monkeypatch, 31)
    made = oidc.new_signing_key
    both = threading.Barrier(2, timeout=2)

    def slow_key() -> Any:
        # Hold the first change here until the second has had every chance
        # to pass the age check too.
        with contextlib.suppress(threading.BrokenBarrierError):
            both.wait()
        return made()

    monkeypatch.setattr(oidc, "new_signing_key", slow_key)
    outcomes: list[str] = []

    def change() -> None:
        try:
            oidc_keys.change(c.app, end_sign_ins=False)
            outcomes.append("changed")
        except oidc_keys.TooSoon:
            outcomes.append("too soon")

    threads = [threading.Thread(target=change) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert sorted(outcomes) == ["changed", "too soon"], outcomes
    keys = [k["kid"] for k in c.get("/v1/oidc/keys").json()["keys"]]
    assert len(keys) == 2 and keys[1] == first, "the live key is kept"
