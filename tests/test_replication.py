"""Row 4b: a standby whose copy differs from the root's takes a fresh one.

`docs/design/rows/row-04.md` (specs), LT-OLDSB. A standby that ran code
older than LT1 applied every entry that registered a machine key and kept
none of the keys: its apply ignored fields it did not know. Updated, it
followed on from the same index, so nothing ever noticed that its node
records held no `tlsPins`. Promoted, it could reach no node over TLS.

Each log page now carries the SHA-256 of the root's canonical state at
its applied index (`LogPage.appliedDigest`). A caught-up standby on the
same version compares it with its own copy's, and when they differ it
says so and takes a fresh snapshot, at most once in 10 minutes. The root
hears the standby's version and its copy's digest on every pull
(`StandbyStatus.version`, `StandbyStatus.matches`).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_control import __version__, replication
from eugene_plexus_control.app import create_app
from eugene_plexus_control.applied import OP_ENROLL_NODE, OP_PATCH_CONFIG
from eugene_plexus_control.replication import Follower
from eugene_plexus_control.state_machine import StateMachine

from .conftest import PASSPHRASE, NodeKeys, enroll, settings_for, standby_at
from .test_replication_follower import _LiveServer

PIN = "P" * 43


@pytest.fixture
def root(tmp_path: Path) -> Iterator[tuple[_LiveServer, StateMachine, str]]:
    """A live active root holding one node with a machine key on record."""
    app = create_app(settings_for(tmp_path / "active"))
    with _LiveServer(app) as server:
        client = httpx.Client(base_url=server.url)
        assert (
            client.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
        )
        token = client.post("/v1/auth/login", json={"passphrase": PASSPHRASE}).json()[
            "sessionToken"
        ]
        client.close()
        machine: StateMachine = app.state.machine
        machine.append(
            OP_ENROLL_NODE,
            {
                "name": "gpu",
                "role": "agent",
                "url": "http://192.0.2.7:8079",
                "publicKey": "Z3B1LXB1YmxpYy1rZXktYnl0ZXMtMzItY2hhcnMtaz0=",
                "enrolledAt": "2026-10-20T10:00:00+00:00",
                "machineUrl": "https://192.0.2.7:8083",
                "tlsPins": [PIN],
            },
        )
        yield server, machine, token


def _without_pins(machine: StateMachine) -> dict[str, Any]:
    """The root's state as a pre-LT1 standby applied it: every node, no key."""
    document = json.loads(machine.canonical())
    for node in document["nodes"]:
        node["tlsPins"] = []
        node["machineUrl"] = None
    return dict(document)


async def test_replica_that_differs_takes_a_snapshot(
    root: tuple[_LiveServer, StateMachine, str],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    server, machine, token = root
    standby = standby_at(tmp_path / "standby")
    standby.install_snapshot(_without_pins(machine))
    assert standby.state.index == machine.state.index, "caught up, and wrong"
    assert standby.state.nodes["gpu"].tlsPins == ()

    follower = Follower(
        standby, active_url=server.url, interval_seconds=0.01, token_provider=lambda: token
    )
    try:
        with caplog.at_level(logging.WARNING, logger=replication.log.name):
            await follower.tick()
        assert standby.state.nodes["gpu"].tlsPins == (PIN,), "the fresh copy holds the pins"
        assert standby.canonical() == machine.canonical()
        said = [r.getMessage() for r in caplog.records if r.name == replication.log.name]
        assert said == [
            f"this copy differs from the root at index {machine.state.index}; "
            "taking a fresh snapshot"
        ], said
        await follower.tick()
        assert follower.status.matches is True

        # It differs again at once: no second snapshot inside 10 minutes,
        # and the copy says it does not match.
        standby.install_snapshot(_without_pins(machine))
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=replication.log.name):
            await follower.tick()
        assert standby.state.nodes["gpu"].tlsPins == ()
        assert follower.status.matches is False
        assert not [r for r in caplog.records if r.name == replication.log.name]
    finally:
        await follower.stop()


async def test_a_standby_on_another_version_does_not_compare(
    root: tuple[_LiveServer, StateMachine, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two versions may write the same state in different bytes: a copy is
    judged only against a root on its own version, or every rolling update
    would replace it and loop."""
    server, machine, token = root
    monkeypatch.setattr(replication, "__version__", "0.0.0-older")
    standby = standby_at(tmp_path / "standby")
    standby.install_snapshot(_without_pins(machine))
    follower = Follower(
        standby, active_url=server.url, interval_seconds=0.01, token_provider=lambda: token
    )
    try:
        await follower.tick()
        assert standby.state.nodes["gpu"].tlsPins == ()
        assert follower.status.matches is None
    finally:
        await follower.stop()


def test_a_log_page_carries_the_digest_of_the_root_s_state(active_client: TestClient) -> None:
    machine: StateMachine = active_client.app.state.machine  # type: ignore[attr-defined]
    machine.append(OP_PATCH_CONFIG, {"values": {"uiTheme": "dark"}})
    page = active_client.get(f"/v1/control/log?after={machine.state.index}")
    assert page.status_code == 200, page.text
    assert page.json()["appliedDigest"] == hashlib.sha256(machine.canonical()).hexdigest()
    assert page.headers["eugene-plexus-version"] == __version__


def test_the_root_reports_the_standby_s_version_and_whether_its_copy_matches(
    active_client: TestClient,
) -> None:
    spare, joined = enroll(active_client, "spare")
    assert joined.status_code == 201, joined.text
    assert active_client.put("/v1/nodes/spare/standby").status_code == 202
    machine: StateMachine = active_client.app.state.machine  # type: ignore[attr-defined]
    index = machine.state.index
    digest = hashlib.sha256(machine.canonical()).hexdigest()

    def pull(version: str | None, copy: str | None, after: int = index) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {_standby_token(spare)}"}
        if version is not None:
            headers["Eugene-Plexus-Version"] = version
        if copy is not None:
            headers["Eugene-Plexus-Copy-Digest"] = copy
        got = active_client.get(f"/v1/control/log?after={after}", headers=headers)
        assert got.status_code == 200, got.text
        [status] = active_client.get("/v1/control/status").json()["standbys"]
        return dict(status)

    old = pull(None, None)
    assert old.get("version") is None and old.get("matches") is None, (
        "an older standby says neither"
    )
    same = pull(__version__, digest)
    assert (same["version"], same["matches"]) == (__version__, True)
    differs = pull(__version__, "0" * 64)
    assert differs["matches"] is False
    behind = pull(__version__, digest, after=index - 1)
    assert behind["matches"] is False, "a pull behind the root's index says nothing new"
    other = pull("0.0.0-older", digest)
    assert other["version"] == "0.0.0-older" and other.get("matches") is None, (
        "another version is not compared"
    )


def _standby_token(keys: NodeKeys) -> str:
    return keys.service_token(sub="standby")
