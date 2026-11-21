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

import asyncio
import json
import re
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
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


# --------------------------------------------------------------------------- the review


def _snapshots_fail(machine: StateMachine) -> None:
    def full_disk(*_a: Any, **_k: Any) -> Any:
        raise OSError(28, "No space left on device")

    machine._store.write_snapshot = full_disk  # type: ignore[method-assign]


def _promoted_root(app: Any, follower: Follower, epoch: int) -> None:
    machine: StateMachine = app.state.machine
    assert machine.role == "control"
    assert machine.state.epoch == epoch + 1
    assert app.state.follower is None
    assert not follower.running, "a root never follows"
    assert app.state.node_poller is not None, "the root's tasks did not start"


def test_a_snapshot_that_fails_after_the_promote_entry_is_not_a_failed_promotion(
    standby: tuple[Any, TestClient, dict[str, str], int], tmp_path: Path
) -> None:
    """The promote entry is the one that reaches the compaction threshold,
    and the snapshot `append` writes after committing it fails. The entry is
    in the log, so the next start would start this node as the root: the
    promotion stands, it is not reported as not written."""
    app, client, headers, epoch = standby
    follower: Follower = app.state.follower
    machine: StateMachine = app.state.machine
    machine._entries_since_snapshot = machine._compact_threshold - 1
    _snapshots_fail(machine)

    promoted = _promote(client, headers, force=True)

    assert promoted.status_code == 200, promoted.text
    assert promoted.json()["epoch"] == epoch + 1
    _promoted_root(app, follower, epoch)
    log_file = tmp_path / "standby" / "state" / "log.jsonl"
    last = json.loads(log_file.read_text(encoding="utf-8").splitlines()[-1])
    assert (last["op"], last["epoch"]) == ("promote", epoch + 1)


def test_a_snapshot_that_fails_after_promoting_still_starts_the_root(
    standby: tuple[Any, TestClient, dict[str, str], int],
) -> None:
    """The snapshot a promotion takes at once fails: the root still starts
    its tasks and answers success, because the promotion is in the log."""
    app, client, headers, epoch = standby
    follower: Follower = app.state.follower
    _snapshots_fail(app.state.machine)

    promoted = _promote(client, headers, force=True)

    assert promoted.status_code == 200, promoted.text
    _promoted_root(app, follower, epoch)


def test_a_follower_that_will_not_stop_does_not_hold_up_the_root(
    standby: tuple[Any, TestClient, dict[str, str], int],
) -> None:
    app, client, headers, epoch = standby
    follower: Follower = app.state.follower
    pause = follower.pause

    async def stuck() -> None:
        await pause()
        raise RuntimeError("the token provider would not close")

    follower.stop = stuck  # type: ignore[method-assign]

    promoted = _promote(client, headers, force=True)

    assert promoted.status_code == 200, promoted.text
    _promoted_root(app, follower, epoch)


def test_a_standby_that_follows_nothing_needs_force(
    standby: tuple[Any, TestClient, dict[str, str], int],
) -> None:
    app, client, headers, _epoch = standby
    client.portal.call(app.state.follower.stop)
    app.state.follower = None

    refused = _promote(client, headers)
    assert refused.status_code == 409, refused.text
    assert _problem(refused)["title"] == "How far behind is unknown"
    assert "replication is not running" in _problem(refused)["detail"]
    assert app.state.machine.role == "standby"

    promoted = _promote(client, headers, force=True)
    assert promoted.status_code == 200, promoted.text


def test_the_promoted_line_says_how_old_the_lag_reading_is(
    standby: tuple[Any, TestClient, dict[str, str], int], caplog: pytest.LogCaptureFixture
) -> None:
    app, client, headers, _epoch = standby
    follower: Follower = app.state.follower
    index = app.state.machine.state.index
    follower.status.local_index = index
    follower.status.active_applied_index = index
    follower.status.last_contact_at = (datetime.now(UTC) - timedelta(seconds=40)).isoformat()

    with caplog.at_level("WARNING", logger="eugene_plexus_control.routes.control"):
        promoted = _promote(client, headers)

    assert promoted.status_code == 200, promoted.text
    line = next(r.getMessage() for r in caplog.records if "PROMOTED" in r.getMessage())
    assert re.search(r"lag 0 entries, read 4[01] s before", line), line


# --------------------------------------------------------------------------- two at once


def _loops() -> list[asyncio.Task[Any]]:
    return [t for t in asyncio.all_tasks() if t.get_name() == "control-follower" and not t.done()]


async def _two_promotes(tmp_path: Path, fail: str) -> tuple[Any, list[int], int]:
    """Two promotions sent at once to one standby. `fail`: "both" fail to
    write, or only the "first" does. Returns the app, the two answers and
    how many pull loops run afterwards."""
    _seed(tmp_path)
    app = create_app(settings_for(tmp_path / "standby", role="standby", active_url=DEAD_ROOT))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            login = await client.post("/v1/auth/login", json={"passphrase": PASSPHRASE})
            headers = {"Authorization": f"Bearer {login.json()['sessionToken']}"}
            machine: StateMachine = app.state.machine
            real_append = machine.append
            calls = {"n": 0}

            def flaky(*a: Any, **k: Any) -> Any:
                calls["n"] += 1
                if fail == "both" or calls["n"] == 1:
                    raise OSError(28, "No space left on device")
                return real_append(*a, **k)

            machine.append = flaky  # type: ignore[method-assign]

            async def one() -> int:
                answer = await client.post(
                    "/v1/control/promote",
                    json={"passphrase": PASSPHRASE, "force": True},
                    headers=headers,
                )
                return answer.status_code

            codes = await asyncio.gather(one(), one())
            await asyncio.sleep(0.2)
            loops = len(_loops())
            for task in _loops():
                task.cancel()
            return app, sorted(codes), loops


async def test_two_failing_promotions_leave_one_pull_loop(tmp_path: Path) -> None:
    app, codes, loops = await _two_promotes(tmp_path, "both")
    assert codes == [500, 500]
    assert app.state.machine.role == "standby"
    assert loops == 1, f"{loops} pull loops"


async def test_a_failed_then_a_good_promotion_leave_a_root_that_follows_nothing(
    tmp_path: Path,
) -> None:
    app, codes, loops = await _two_promotes(tmp_path, "first")
    assert codes == [200, 500]
    assert app.state.machine.role == "control"
    assert app.state.follower is None
    assert loops == 0, f"{loops} pull loops"


async def test_pause_and_resume_keep_one_loop_on_their_own(tmp_path: Path) -> None:
    """Without the handler's lock: two pauses of one loop, each followed by a
    resume, leave one loop; a pause always returns with none running."""
    follower = Follower(standby_at(tmp_path / "copy"), active_url=DEAD_ROOT, interval_seconds=0.05)
    follower.start()
    try:

        async def pause_then_resume() -> None:
            await follower.pause()
            assert not follower.running, "pause returned with a loop running"
            follower.resume()

        await asyncio.gather(pause_then_resume(), pause_then_resume())
        assert len(_loops()) == 1
        follower.resume()
        assert len(_loops()) == 1
    finally:
        await follower.stop()
    assert _loops() == []
