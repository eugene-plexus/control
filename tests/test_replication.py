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

import contextlib
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


# --------------------------------------------------------------------------- the review


def test_an_incomplete_foreign_or_older_snapshot_is_refused(tmp_path: Path) -> None:
    """A standby replaces its copy by itself now, so what it takes is
    checked first (row 4 review, blocker 4). Nothing changes on a refusal."""
    from eugene_plexus_control.applied import ApplyError

    from .conftest import machine_at

    root = machine_at(tmp_path / "root")
    root.seed_identity({"salt": "c2FsdA==", "passphraseVerifier": "v", "controlPublicKey": "k"})
    root.append(OP_PATCH_CONFIG, {"values": {"joinTokenTtlSeconds": 60}})
    copy = standby_at(tmp_path / "copy")
    copy.install_snapshot(root.snapshot_document())
    before = copy.canonical()

    with pytest.raises(ApplyError, match="incomplete"):
        copy.install_snapshot({"index": 100})
    foreign = root.snapshot_document() | {"index": 100, "controlPublicKey": "another"}
    with pytest.raises(ApplyError, match="another install"):
        copy.install_snapshot(foreign)
    assert copy.canonical() == before
    copy.promote(node=None)
    assert copy.state.epoch == 2
    before = copy.canonical()
    older = root.snapshot_document() | {"index": 100}
    with pytest.raises(ApplyError, match="epoch 1"):
        copy.install_snapshot(older)
    assert copy.canonical() == before


async def test_a_root_that_gives_no_usable_snapshot_is_not_asked_on_every_pull(
    tmp_path: Path,
) -> None:
    """A snapshot with no install identity (the root is not initialized),
    or one that fails, holds the next one off: three pulls ask once (row 4
    review, blocker 4)."""
    from eugene_plexus_control.applied import AppliedState, to_canonical

    asked: list[str] = []
    answer = {"snapshot": 200}

    def handle(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.path)
        if request.url.path.endswith("/snapshot"):
            if answer["snapshot"] != 200:
                return httpx.Response(answer["snapshot"], json={"detail": "no"})
            return httpx.Response(200, json=to_canonical(AppliedState()))
        return httpx.Response(200, json={"entries": [], "appliedIndex": 0})

    for status in (200, 500):
        answer["snapshot"] = status
        asked.clear()
        follower = Follower(
            standby_at(tmp_path / f"copy-{status}"),
            active_url="http://root.test",
            interval_seconds=0.01,
            token_provider=lambda: None,
        )
        follower._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        try:
            for _ in range(3):
                with contextlib.suppress(httpx.HTTPError):
                    await follower.tick()
            assert asked.count("/v1/control/snapshot") == 1, (status, asked)
        finally:
            await follower.stop()


# --------------------------------------------------------------------------- the re-review


def _copy_of(tmp_path: Path) -> tuple[StateMachine, StateMachine]:
    from .conftest import machine_at

    root = machine_at(tmp_path / "root")
    root.seed_identity(
        {
            "salt": "c2FsdA==",
            "passphraseVerifier": "v",
            "controlPublicKey": "k",
            "sealedSigningKey": "s",
            "sealedControlKey": "c",
            "rootTokenPublicKey": "r",
        }
    )
    root.append(
        "enrollNode",
        {
            "name": "gpu",
            "role": "agent",
            "url": "http://192.0.2.7:8079",
            "publicKey": "Z3B1LXB1YmxpYy1rZXktYnl0ZXMtMzItY2hhcnMtaz0=",
            "enrolledAt": "2026-10-20T10:00:00+00:00",
        },
    )
    copy = standby_at(tmp_path / "copy")
    copy.install_snapshot(root.snapshot_document())
    return root, copy


def test_a_snapshot_with_a_null_or_mistyped_field_is_refused(tmp_path: Path) -> None:
    """Presence is not enough: the parser reads `nodes: null` as no nodes,
    so a snapshot saying so would clear the registry (row 4 re-review,
    blocker 4)."""
    from eugene_plexus_control.applied import ApplyError

    root, copy = _copy_of(tmp_path)
    before = copy.canonical()
    good = root.snapshot_document()
    for name, bad in (
        ("nodes", None),
        ("nodes", {}),
        ("config", []),
        ("revokedSessions", None),
        ("people", "everyone"),
        ("salt", 7),
        ("epoch", "1"),
        ("nodes", [{"role": "agent"}]),
    ):
        with pytest.raises(ApplyError, match="refusing a snapshot"):
            copy.install_snapshot(good | {name: bad})
        assert copy.canonical() == before, name
    assert set(copy.state.nodes) == {"gpu"}


#: The top-level `Snapshot` fields a v0.2.0 root writes (control 3c378a1,
#: `applied.to_canonical`); v0.2 is the oldest supported version.
_WRITTEN_BY_V0_2_0 = frozenset(
    {
        "index",
        "epoch",
        "nodes",
        "components",
        "runtimes",
        "config",
        "clientKeys",
        "clientAdmission",
        "salt",
        "passphraseVerifier",
        "sealedSigningKey",
        "rootTokenPublicKey",
        "revokedSessions",
        "sealedControlKey",
        "controlPublicKey",
        "sealedRecoveryKey",
        "signingKeyId",
        "people",
        "sites",
        "nodeHelpers",
        "oidcClients",
        "oidcKeys",
        "revokedSignIns",
    }
)


def test_a_snapshot_from_a_root_that_writes_fewer_fields_is_taken(tmp_path: Path) -> None:
    """A field added after the oldest supported root is optional: a root
    that does not write it is followed, and its absence reads as empty (row
    4 re-review)."""
    root, copy = _copy_of(tmp_path)
    document = root.snapshot_document()
    older = {name: value for name, value in document.items() if name in _WRITTEN_BY_V0_2_0}
    assert set(document) - set(older), "the root writes fields v0.2.0 did not"
    copy.install_snapshot(older | {"index": older["index"] + 1})
    assert set(copy.state.nodes) == {"gpu"}


async def test_a_failed_fresh_snapshot_is_retried_after_the_short_wait(
    root: tuple[_LiveServer, StateMachine, str], tmp_path: Path
) -> None:
    """The 10 minutes between fresh snapshots start only once one was taken:
    one that failed is tried again after the 30 s that follow any failure
    (row 4 re-review)."""
    server, machine, token = root
    standby = standby_at(tmp_path / "standby")
    standby.install_snapshot(_without_pins(machine))
    follower = Follower(
        standby, active_url=server.url, interval_seconds=0.01, token_provider=lambda: token
    )
    asked: list[str] = []
    real_get = follower._client.get
    failing = {"yes": True}

    async def get(url: str, **kwargs: Any) -> httpx.Response:
        asked.append(str(url))
        if url.endswith("/snapshot") and failing["yes"]:
            return httpx.Response(
                500, json={"detail": "not now"}, request=httpx.Request("GET", url)
            )
        return await real_get(url, **kwargs)

    follower._client.get = get  # type: ignore[method-assign]
    try:
        with contextlib.suppress(httpx.HTTPError):
            await follower.tick()
        assert sum(u.endswith("/snapshot") for u in asked) == 1
        follower._snapshot_after = 0.0  # the 30 s have passed
        failing["yes"] = False
        await follower.tick()
        assert sum(u.endswith("/snapshot") for u in asked) == 2, "not held off for 10 minutes"
        assert standby.state.nodes["gpu"].tlsPins == (PIN,)
    finally:
        await follower.stop()
