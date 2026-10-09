"""J14b and 2b.4 at the root: a held call's approval carried as Workbench
sent it, the site's `held` answer relayed, and the window and the consent to
commands taken away from Workbench (`person-held-keys.md` §13)."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_control.settings import Settings

from .test_site_people import MINE, call, people_site
from .test_sites import (
    NODES_ORIGIN,
    Ada,
    _root,
    answer,
    auth,
    deliver,
    manage_through,
    quick_poll,  # noqa: F401  (autouse fixture)
)

COMMANDS = {"allowed": True, "consentedAt": "2026-10-08T12:00:00Z"}
HELD = {
    "id": "a1b2c3d4e5f60718",
    "kind": "call",
    "words": ["Run this command on desk as HOST/bo, starting in “Mine”:", "npm test"],
    "expiresAt": "2026-10-08T12:30:00Z",
    "approved": False,
    "envelopes": {"f" * 32: '{"act":"call"}'},
    "approvePage": "http://127.0.0.1:8079/link/approve",
}


@pytest.fixture
def root(tmp_path: Path) -> Iterator[TestClient]:
    yield from _root(
        Settings(
            config_file=tmp_path / "control.yaml",
            state_dir=tmp_path / "state",
            nodes_origin=NODES_ORIGIN,
        )
    )


def calls_site(c: TestClient) -> tuple[Ada, dict[str, Any], str]:
    a, bo, bo_token = people_site(c)
    a.held.summary["commands"] = dict(COMMANDS)
    c.post("/v1/sites/poll", headers=a.keys.bearer(a.site), json=a.held.report())
    return a, bo, bo_token


def through(
    c: TestClient, a: Ada, token: str, result: dict[str, Any], **extra: Any
) -> tuple[dict[str, Any], Any]:
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(call, c, a, token, "Mine", **extra)
        ident, command = deliver(c, a.keys, a.site, a.held.report())
        answer(c, a.keys, a.site, ident, result)
        response = pending.result(timeout=5)
    return command, response


def test_a_held_answer_comes_back_with_what_to_sign(root: TestClient) -> None:
    a, _, bo_token = calls_site(root)
    _, response = through(
        root,
        a,
        bo_token,
        {"status": "held", "message": "Sign it.", "held": HELD, "windowUntil": None},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "held" and body["held"] == HELD
    assert body.get("windowUntil") is None


def test_an_approval_is_carried_to_the_site_as_is(root: TestClient) -> None:
    a, _, bo_token = calls_site(root)
    approval = {
        "held": HELD["id"],
        "envelope": '{"act":"call"}',
        "key": "f" * 32,
        "credentialId": "cred",
        "authenticatorData": "auth",
        "clientDataJSON": "client",
        "signature": "sig",
    }
    command, response = through(
        root,
        a,
        bo_token,
        {
            "status": "done",
            "response": {"jsonrpc": "2.0", "id": 3, "result": {}},
            "windowUntil": "2026-10-08T13:00:00Z",
        },
        approval=approval,
    )
    assert command["approval"] == approval
    assert response.json()["windowUntil"] == "2026-10-08T13:00:00Z"
    plain, _ = through(
        root, a, bo_token, {"status": "done", "response": {"jsonrpc": "2.0", "id": 3}}
    )
    assert plain["approval"] is None


def test_a_linked_person_closes_their_window(root: TestClient) -> None:
    a, _, bo_token = calls_site(root)
    command, response = manage_through(
        root,
        a.keys,
        a.site,
        a.held,
        lambda: root.post(
            f"/oidc/job-sites/{a.site}/window/close",
            auth=auth(a.app),
            json={"refreshToken": bo_token},
        ),
        lambda _: {"status": "done", "result": {"closed": True}},
        action="window.close",
    )
    assert response.status_code == 204, response.text
    assert command["arguments"] == {}


def test_only_the_owner_takes_back_the_consent_to_commands(root: TestClient) -> None:
    a, _, bo_token = calls_site(root)
    refused = root.post(
        f"/oidc/job-sites/{a.site}/commands/withdraw",
        auth=auth(a.app),
        json={"refreshToken": bo_token},
    )
    assert refused.status_code == 404
    _, response = manage_through(
        root,
        a.keys,
        a.site,
        a.held,
        lambda: root.post(
            f"/oidc/job-sites/{a.site}/commands/withdraw",
            auth=auth(a.app),
            json={"refreshToken": a.token},
        ),
        lambda _: {"status": "done", "result": {"allowed": False}},
        action="commands.withdraw",
    )
    assert response.status_code == 204, response.text


def test_a_site_older_than_signed_calls_says_it_needs_an_update(root: TestClient) -> None:
    a, _, bo_token = people_site(root)
    for path, token in (("window/close", bo_token), ("commands/withdraw", a.token)):
        answer_ = root.post(
            f"/oidc/job-sites/{a.site}/{path}", auth=auth(a.app), json={"refreshToken": token}
        )
        assert answer_.status_code == 503 and "signed calls" in answer_.text


def test_the_site_lists_say_whether_commands_run_there(root: TestClient) -> None:
    a, _, bo_token = calls_site(root)
    mine = root.post("/oidc/job-sites", auth=auth(a.app), json={"refreshToken": a.token})
    assert mine.status_code == 200, mine.text
    assert mine.json()["sites"][0]["commands"] == COMMANDS
    linked = root.post("/oidc/job-sites", auth=auth(a.app), json={"refreshToken": bo_token})
    site = next(s for s in linked.json()["sites"] if s["id"] == a.site)
    assert site["role"] == "linked" and site["commands"] == COMMANDS
    assert MINE in [w["id"] for w in site["workspaces"]]
