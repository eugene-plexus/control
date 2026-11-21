"""control#16: what a promotion refuses, and what a refused one leaves.

Two rules, each with the case that broke it.

* **A standby that cannot say how far behind it is needs `force`.** Its
  lag is unknown until its first pull since it started, and that is the
  ordinary failover case: the active root is dead, the standby restarts,
  and nothing it had not pulled can be counted. It used to promote
  without a word, because an unknown lag read as no lag.
* **A promotion that fails leaves a standby that still follows.** The
  follower used to be stopped first, so a passphrase change, an unreadable
  sealed key or a failed write left a standby that replicated nothing
  until it was restarted.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_control import security
from eugene_plexus_control.app import create_app
from eugene_plexus_control.applied import OP_PATCH_CONFIG
from eugene_plexus_control.replication import Follower
from eugene_plexus_control.state_machine import StateMachine

from .conftest import PASSPHRASE, settings_for, standby_at

#: Nothing listens here, so the follower's every pull fails and its lag
#: stays unknown.
DEAD_ROOT = "http://127.0.0.1:1"


def _seed(tmp_path: Path) -> int:
    """An initialized root, copied to `<tmp>/standby/state`; its epoch."""
    active_app = create_app(settings_for(tmp_path / "active"))
    with TestClient(active_app) as active:
        assert (
            active.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
        )
        machine: StateMachine = active_app.state.machine
        machine.append(OP_PATCH_CONFIG, {"values": {"uiTheme": "dark"}})
        copy = standby_at(tmp_path / "standby" / "state")
        copy.install_snapshot(machine.snapshot_document())
        for entry in machine.read_page(copy.state.index, 10_000):
            copy.accept_replicated(entry)
        return machine.state.epoch


@pytest.fixture
def standby(tmp_path: Path) -> Iterator[tuple[Any, TestClient, dict[str, str], int]]:
    """A standby whose active root is gone: it restarted, and has not
    reached the root since."""
    epoch = _seed(tmp_path)
    app = create_app(settings_for(tmp_path / "standby", role="standby", active_url=DEAD_ROOT))
    with TestClient(app, raise_server_exceptions=False) as client:
        token = client.post("/v1/auth/login", json={"passphrase": PASSPHRASE}).json()[
            "sessionToken"
        ]
        yield app, client, {"Authorization": f"Bearer {token}"}, epoch


def _promote(client: TestClient, headers: dict[str, str], **extra: Any) -> Any:
    return client.post(
        "/v1/control/promote", json={"passphrase": PASSPHRASE, **extra}, headers=headers
    )


def _problem(response: Any) -> dict[str, Any]:
    body = response.json()
    detail = body.get("detail")
    return detail if isinstance(detail, dict) else body


def _still_a_following_standby(app: Any, follower: Follower, epoch: int, index: int) -> None:
    machine: StateMachine = app.state.machine
    assert machine.role == "standby"
    assert machine.state.epoch == epoch
    assert machine.state.index == index
    assert app.state.follower is follower, "the follower was dropped"
    assert follower.running, "the standby stopped following"


def test_a_standby_that_does_not_know_its_lag_refuses_without_force(
    standby: tuple[Any, TestClient, dict[str, str], int],
) -> None:
    app, client, headers, epoch = standby
    follower: Follower = app.state.follower
    assert follower.status.lag_entries is None
    index = app.state.machine.state.index

    refused = _promote(client, headers)

    assert refused.status_code == 409, refused.text
    problem = _problem(refused)
    assert problem["title"] == "How far behind is unknown"
    assert "has not read the position of the active root" in problem["detail"]
    assert f"It last applied entry {index}." in problem["detail"]
    assert "force=true" in problem["detail"]
    assert "entries behind" not in problem["detail"]
    _still_a_following_standby(app, follower, epoch, index)


def test_a_standby_that_does_not_know_its_lag_promotes_with_force(
    standby: tuple[Any, TestClient, dict[str, str], int],
) -> None:
    app, client, headers, epoch = standby
    follower: Follower = app.state.follower

    promoted = _promote(client, headers, force=True)

    assert promoted.status_code == 200, promoted.text
    assert promoted.json()["epoch"] == epoch + 1
    assert app.state.follower is None
    assert not follower.running


def test_a_standby_with_no_root_to_follow_says_so(tmp_path: Path) -> None:
    """No active URL at all: the lag can never be known, and the words say
    why rather than blaming a root it was never told about."""
    _seed(tmp_path)
    app = create_app(settings_for(tmp_path / "standby", role="standby", active_url=None))
    with TestClient(app) as client:
        token = client.post("/v1/auth/login", json={"passphrase": PASSPHRASE}).json()[
            "sessionToken"
        ]
        refused = _promote(client, {"Authorization": f"Bearer {token}"})
        assert refused.status_code == 409, refused.text
        assert _problem(refused)["title"] == "How far behind is unknown"
        assert "no active root to follow" in _problem(refused)["detail"]
        assert app.state.machine.role == "standby"


def test_a_standby_caught_up_promotes_without_force(
    standby: tuple[Any, TestClient, dict[str, str], int],
) -> None:
    app, client, headers, epoch = standby
    follower: Follower = app.state.follower
    # Its last pull found it level with the root.
    follower.status.local_index = app.state.machine.state.index
    follower.status.active_applied_index = app.state.machine.state.index
    assert follower.status.lag_entries == 0

    promoted = _promote(client, headers)

    assert promoted.status_code == 200, promoted.text
    assert promoted.json()["epoch"] == epoch + 1


def test_a_standby_known_to_be_behind_still_needs_force(
    standby: tuple[Any, TestClient, dict[str, str], int],
) -> None:
    app, client, headers, epoch = standby
    follower: Follower = app.state.follower
    index = app.state.machine.state.index
    follower.status.local_index = index
    follower.status.active_applied_index = index + 3

    refused = _promote(client, headers)
    assert refused.status_code == 409, refused.text
    assert _problem(refused)["title"] == "Standby is behind"
    assert "3 entries behind" in _problem(refused)["detail"]
    _still_a_following_standby(app, follower, epoch, index)

    promoted = _promote(client, headers, force=True)
    assert promoted.status_code == 200, promoted.text


def test_an_unreadable_sealed_key_leaves_the_standby_following(
    standby: tuple[Any, TestClient, dict[str, str], int],
) -> None:
    app, client, headers, epoch = standby
    follower: Follower = app.state.follower
    machine: StateMachine = app.state.machine
    index = machine.state.index
    # The token key, sealed under some other key than this passphrase's.
    wrong = security.seal_b64(b"not this install's key", b"\x01" * 32)
    machine._state = replace(
        machine.state, identity=replace(machine.state.identity, sealedSigningKey=wrong)
    )

    refused = _promote(client, headers, force=True)

    assert refused.status_code == 500, refused.text
    assert _problem(refused)["title"] == "Sealed token key unreadable"
    _still_a_following_standby(app, follower, epoch, index)


def test_a_passphrase_change_during_promotion_leaves_the_standby_following(
    standby: tuple[Any, TestClient, dict[str, str], int],
) -> None:
    app, client, headers, epoch = standby
    follower: Follower = app.state.follower
    index = app.state.machine.state.index
    # What a change that lands between the check and the switch looks like
    # to the unlock: the salt it derived with is no longer the install's.
    app.state.auth_state.publish_master_key = lambda *_a, **_k: False

    refused = _promote(client, headers, force=True)

    assert refused.status_code == 409, refused.text
    assert _problem(refused)["title"] == "Passphrase changed"
    _still_a_following_standby(app, follower, epoch, index)


def test_a_promotion_that_cannot_be_written_leaves_the_standby_following(
    standby: tuple[Any, TestClient, dict[str, str], int],
) -> None:
    app, client, headers, epoch = standby
    follower: Follower = app.state.follower
    machine: StateMachine = app.state.machine
    index = machine.state.index

    def full_disk(*_a: Any, **_k: Any) -> Any:
        raise OSError(28, "No space left on device")

    machine.append = full_disk  # type: ignore[method-assign]

    refused = _promote(client, headers, force=True)

    assert refused.status_code == 500, refused.text
    assert _problem(refused)["title"] == "Promotion not written"
    assert "No space left on device" in _problem(refused)["detail"]
    _still_a_following_standby(app, follower, epoch, index)
