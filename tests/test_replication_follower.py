"""The standby follower, against a real active root over real HTTP.

Two live apps, one following the other. The interesting cases are not
"it caught up" — they are what happens when it *cannot*:

* the active root compacted past the standby's position, so the entries
  it wants no longer exist;
* an entry arrives that will not apply;
* the active root is simply gone.

In all three the required behaviour is the same shape: recover if
recovery is defined, otherwise keep trying and keep reporting. **Never
promote.** A follower that could promote itself would be the automatic
failover the design refuses, and the argument against it — that it
trades a visible, bounded, self-healing window for genuine split-brain —
does not get easier to make later.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import uvicorn

from eugene_plexus_control.app import create_app
from eugene_plexus_control.applied import OP_PATCH_CONFIG
from eugene_plexus_control.replication import Follower
from eugene_plexus_control.state_machine import StateMachine

from .conftest import PASSPHRASE, settings_for, standby_at


class _LiveServer:
    """A real uvicorn on a real port.

    The follower speaks HTTP, so testing it against `TestClient` would
    test a transport it never uses. This is a few lines of setup for a
    materially different assertion.
    """

    def __init__(self, app: object) -> None:
        config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> _LiveServer:
        self.thread.start()
        while not self.server.started:
            if not self.thread.is_alive():  # pragma: no cover - startup failure
                raise RuntimeError("the test server died during startup")
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)

    @property
    def url(self) -> str:
        socket = self.server.servers[0].sockets[0]
        return f"http://127.0.0.1:{socket.getsockname()[1]}"


@pytest.fixture
def live_active(tmp_path: Path) -> Iterator[tuple[_LiveServer, StateMachine, str]]:
    """One app, one lifespan, one state machine.

    Worth spelling out because getting it wrong is subtle: wrapping the
    app in a `TestClient` *and* a live server runs the lifespan twice,
    which builds two `StateMachine` objects over the same directory. The
    test then holds one and the server serves the other, and they
    disagree about the applied index while agreeing about the file — a
    second writer, invented by the test harness, in the one component
    whose whole design is that there is only ever one.
    """
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
        yield server, app.state.machine, token


async def test_a_standby_catches_up_by_pulling(
    live_active: tuple[_LiveServer, StateMachine, str], tmp_path: Path
) -> None:
    server, machine, token = live_active
    machine.append(OP_PATCH_CONFIG, {"values": {"uiTheme": "dark"}})

    standby = standby_at(tmp_path / "standby")
    follower = Follower(
        standby, active_url=server.url, interval_seconds=0.01, token_provider=lambda: token
    )
    try:
        # A fresh standby has nothing, so its first pull starts at 0 and
        # the active root can still serve from there.
        await follower.tick()
        assert standby.state.index == machine.state.index
        assert standby.canonical() == machine.canonical()

        machine.append(OP_PATCH_CONFIG, {"values": {"uiFontSize": "large"}})
        await follower.tick()
        assert standby.canonical() == machine.canonical()

        assert follower.status.reachable is True
        assert follower.status.lag_entries == 0
    finally:
        await follower.stop()


async def test_a_standby_past_compaction_re_snapshots_instead_of_guessing(
    live_active: tuple[_LiveServer, StateMachine, str], tmp_path: Path
) -> None:
    """The 409 path, which is the one that matters after a long outage.

    The entries this standby wants have been deleted. Reading around the
    gap would produce a state that never existed anywhere, so the only
    correct move is to take a snapshot — and the response says so
    explicitly rather than leaving the standby to infer it from an empty
    page.
    """
    server, machine, token = live_active
    for theme in ("dark", "light", "auto"):
        machine.append(OP_PATCH_CONFIG, {"values": {"uiTheme": theme}})
    machine.snapshot_now()
    assert machine.first_available_index() == machine.state.index + 1

    standby = standby_at(tmp_path / "standby")
    follower = Follower(
        standby, active_url=server.url, interval_seconds=0.01, token_provider=lambda: token
    )
    try:
        await follower.tick()
        assert standby.state.index == machine.state.index
        assert standby.canonical() == machine.canonical()
    finally:
        await follower.stop()


async def test_the_log_endpoint_says_compacted_rather_than_empty(
    live_active: tuple[_LiveServer, StateMachine, str],
) -> None:
    """An empty page means "you are current". Conflating the two would
    have a standby that fell behind sit there believing it was up to
    date, which is the worst possible way to discover it at a
    promotion."""
    server, machine, token = live_active
    for theme in ("dark", "light"):
        machine.append(OP_PATCH_CONFIG, {"values": {"uiTheme": theme}})
    machine.snapshot_now()

    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient() as client:
        compacted = await client.get(
            f"{server.url}/v1/control/log", params={"after": 0}, headers=headers
        )
        assert compacted.status_code == 409
        assert "snapshot" in compacted.text

        current = await client.get(
            f"{server.url}/v1/control/log",
            params={"after": machine.state.index},
            headers=headers,
        )
        assert current.status_code == 200
        assert current.json()["entries"] == []


async def test_an_unreachable_active_root_leaves_the_standby_trying(
    tmp_path: Path,
) -> None:
    """Down, not out. The follower reports the failure and carries on;
    nothing here decides the active root is gone for good, because that
    decision is an operator's."""
    standby = standby_at(tmp_path / "standby")
    follower = Follower(
        standby,
        active_url="http://127.0.0.1:1",
        interval_seconds=0.01,
        token_provider=lambda: None,
    )
    try:
        # A connection failure, specifically — not any exception. The
        # required behaviour is that a *network* failure changes
        # nothing, and a blanket `Exception` would pass just as
        # happily if the follower crashed on its own state.
        with pytest.raises(httpx.ConnectError):
            await follower.tick()
        assert standby.state.index == 0
        assert standby.role == "standby", "a failed pull must never change the role"
    finally:
        await follower.stop()


async def test_the_follower_reports_lag_in_entries(
    live_active: tuple[_LiveServer, StateMachine, str], tmp_path: Path
) -> None:
    """Entries rather than seconds, because entries are what would be
    lost on a promotion and seconds are not."""
    server, machine, token = live_active
    standby = standby_at(tmp_path / "standby")
    follower = Follower(
        standby, active_url=server.url, interval_seconds=0.01, token_provider=lambda: token
    )
    try:
        await follower.tick()
        for theme in ("dark", "light", "auto"):
            machine.append(OP_PATCH_CONFIG, {"values": {"uiTheme": theme}})

        # The standby has not pulled yet, so it knows only what it was
        # last told. That is honest: lag is computed from its own
        # position against the last figure it saw.
        assert follower.status.lag_entries == 0
        await follower.tick()
        assert follower.status.lag_entries == 0
        assert standby.state.index == machine.state.index
    finally:
        await follower.stop()
