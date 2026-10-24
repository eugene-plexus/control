"""`POST /oidc/subjects` (row 7): which people Eugene still has.

An app that keeps people's data (Workbench) sends the `sub`s it has seen sign in.
A deleted person comes back `unknown`, and the app hides, then deletes, their data.
So what is pinned here is what a wrong answer would cost: a person who is turned
off is still `known`, the owner is always `known`, the root says nothing while it
is locked or a standby, and nobody without the app's own secret is answered.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from eugene_plexus_control.applied import OPERATOR_NAME

CALLBACK = "http://127.0.0.1:9/callback"
PASSWORD = "a-long-enough-password"


def _app(c: TestClient) -> dict[str, Any]:
    made = c.post("/v1/oidc/clients", json={"name": "Workbench", "redirectUris": [CALLBACK]})
    assert made.status_code == 201, made.text
    return made.json()


def _person(c: TestClient, name: str) -> str:
    made = c.post("/v1/people", json={"name": name, "password": PASSWORD})
    assert made.status_code == 201, made.text
    return str(made.json()["id"])


def _ask(c: TestClient, app: dict[str, Any], subjects: Any, **kw: Any) -> Any:
    bare = TestClient(c.app)
    return bare.post(
        "/oidc/subjects",
        auth=kw.get("auth", (app["client"]["clientId"], app["clientSecret"])),
        json={"subjects": subjects},
    )


def test_removed_person_is_unknown(active_client: TestClient) -> None:
    app = _app(active_client)
    ann, ben = _person(active_client, "Ann"), _person(active_client, "Ben")
    assert active_client.delete(f"/v1/people/{ann}").status_code == 204
    answer = _ask(active_client, app, [ann, ben, OPERATOR_NAME])
    assert answer.status_code == 200, answer.text
    assert answer.headers["cache-control"] == "no-store"
    assert answer.json() == {"known": [ben, OPERATOR_NAME], "unknown": [ann]}


def test_without_the_apps_secret_nothing_is_said(active_client: TestClient) -> None:
    app = _app(active_client)
    ben = _person(active_client, "Ben")
    bare = TestClient(active_client.app)
    unsigned = bare.post("/oidc/subjects", json={"subjects": [ben]})
    wrong = _ask(active_client, app, [ben], auth=(app["client"]["clientId"], "not-the-secret"))
    for refused in (unsigned, wrong):
        assert refused.status_code == 401
        assert refused.json()["error"] == "invalid_client"
        assert ben not in refused.text
    # The hub's own owner token is not an app's secret either.
    owner = active_client.post("/oidc/subjects", json={"subjects": [ben]})
    assert owner.status_code == 401


def test_a_turned_off_person_is_still_known(active_client: TestClient) -> None:
    app = _app(active_client)
    ann = _person(active_client, "Ann")
    turned_off = active_client.patch(f"/v1/people/{ann}", json={"disabled": True})
    assert turned_off.status_code == 200, turned_off.text
    assert _ask(active_client, app, [ann]).json() == {"known": [ann], "unknown": []}


def test_a_locked_or_standby_root_says_nothing(active_client: TestClient, monkeypatch: Any) -> None:
    app = _app(active_client)
    ann = _person(active_client, "Ann")
    active_client.delete(f"/v1/people/{ann}")
    state = active_client.app.state  # type: ignore[attr-defined]
    key = state.auth_state.master_key
    state.auth_state.forget_master_key()
    locked = _ask(active_client, app, [ann])
    state.auth_state.set_master_key(key, salt=state.machine.state.identity.salt)
    monkeypatch.setattr(type(state.machine), "is_active", property(lambda self: False))
    standby = _ask(active_client, app, [ann])
    for answer in (locked, standby):
        assert answer.status_code == 503
        assert answer.json()["error"] == "temporarily_unavailable"
        assert ann not in answer.text


def test_only_one_to_five_hundred_ids_of_text(active_client: TestClient) -> None:
    app = _app(active_client)
    for subjects in ([], ["x"] * 501, [7], [""], "operator", None):
        refused = _ask(active_client, app, subjects)
        assert refused.status_code == 400, subjects
        assert refused.json()["error"] == "invalid_request"
    assert _ask(active_client, app, ["x"] * 500).status_code == 200
