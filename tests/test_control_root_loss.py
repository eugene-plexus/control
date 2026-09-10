"""REQUIRED DELIVERABLE — what survives the control root going away.

> *"Kill the control root and assert that a chat completion still
> succeeds through the gateway to a running engine."* — M5 design, §5

**The chat completion half of that is not in this repo, and cannot
honestly be.** It needs a gateway, a driver and a real engine, and a
version of it that stood up stubs for all three would assert that our
stubs do not call us. It lives in
[`specs/scripts/m5-acceptance.sh`](https://github.com/eugene-plexus/specs/blob/main/scripts/m5-acceptance.sh),
which runs the real five-process chain against a real model, kills this
component mid-run, and asserts a real completion still comes back.

What this repo owns, and what is asserted here, is the other half: that
losing the control root loses **management only**, and that management
comes back correctly.

* An unclean kill loses nothing a caller was told succeeded.
* A torn tail is discarded and reported rather than treated as
  corruption or, worse, silently kept.
* A standby that was replicating can be promoted from the passphrase
  alone — no key copied between hosts — and can then write.
* The old root, if it comes back, is fenced by epoch. Permanently, and
  without any node having to agree with any other node.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_control.app import create_app
from eugene_plexus_control.applied import OP_PATCH_CONFIG, ApplyError, apply
from eugene_plexus_control.log_store import LogCorruption, LogStore
from eugene_plexus_control.state_machine import StateMachine

from .conftest import PASSPHRASE, machine_at, settings_for, standby_at

# --------------------------------------------------------------------------- #
# An unclean kill
# --------------------------------------------------------------------------- #


def test_an_unclean_kill_leaves_a_loadable_log(tmp_path: Path) -> None:
    """SIGKILL a process mid-append; the log still loads and is gapless.

    A real subprocess and a real kill, not a simulated one. The point is
    the fsync: an entry a caller was told succeeded has to survive the
    machine going away, because the caller may have been an operator
    enrolling a host and the alternative is an install that disagrees
    with the person who configured it.
    """
    state_dir = tmp_path / "state"
    script = tmp_path / "writer.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {str(Path(__file__).resolve().parents[1] / "src")!r})
            from pathlib import Path
            from eugene_plexus_control.log_store import LogStore
            from eugene_plexus_control.state_machine import StateMachine

            machine = StateMachine(LogStore(Path({str(state_dir)!r})))
            machine.load()
            index = 0
            while True:
                index += 1
                machine.append("patchConfig", {{"values": {{"uiTheme": "dark"}}}})
                print(index, flush=True)
            """
        ).strip(),
        encoding="utf-8",
    )

    process = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        # Wait until it has certainly written several entries, then kill
        # it without warning at an arbitrary point in its loop.
        deadline = time.time() + 20
        acknowledged = 0
        assert process.stdout is not None
        while time.time() < deadline:
            line = process.stdout.readline()
            if not line:
                break
            acknowledged = int(line.strip())
            if acknowledged >= 5:
                break
        process.kill()
        process.wait(timeout=10)
    finally:
        if process.poll() is None:  # pragma: no cover - only on a hung child
            process.kill()

    assert acknowledged >= 5, "the child never got far enough to be worth killing"

    recovered = machine_at(state_dir)
    # Everything the child printed was acknowledged to it, so all of it
    # must be here. It may legitimately have written more than it
    # printed; losing an unacknowledged entry is fine, losing an
    # acknowledged one is not.
    assert recovered.state.index >= acknowledged


def test_a_torn_tail_is_discarded_and_reported(tmp_path: Path) -> None:
    """A half-written final line is normal after a kill, not corruption.

    Constructed rather than raced for, because a kill landing exactly
    between `write` and `fsync` is not something a test can arrange
    reliably — and the behaviour under test is what happens *given* the
    condition, not how often it arises.
    """
    machine = machine_at(tmp_path / "state")
    machine.append(OP_PATCH_CONFIG, {"values": {"uiTheme": "dark"}})
    machine.append(OP_PATCH_CONFIG, {"values": {"uiFontSize": "large"}})

    log_path = tmp_path / "state" / "log.jsonl"
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write('{"index":3,"epoch":1,"op":"patchCon')

    recovered = machine_at(tmp_path / "state")
    assert recovered.state.index == 2
    assert recovered.state.config["uiFontSize"] == "large"
    # Surfaced, not swallowed: an operator learns about it from
    # /healthz rather than from the logs of a process that has since
    # restarted.
    assert recovered.dropped_tail_lines == 1


def test_a_hole_in_the_middle_of_the_log_is_corruption(tmp_path: Path) -> None:
    """A bad line that is not the last one is a different thing entirely.

    Refusing here rather than reading around it is the same rule as
    refusing a gap at apply time: a state assembled from part of a file
    is a state that never existed anywhere.
    """
    machine = machine_at(tmp_path / "state")
    machine.append(OP_PATCH_CONFIG, {"values": {"uiTheme": "dark"}})
    machine.append(OP_PATCH_CONFIG, {"values": {"uiTheme": "light"}})

    log_path = tmp_path / "state" / "log.jsonl"
    lines = log_path.read_text(encoding="utf-8").splitlines()
    log_path.write_text("not json at all\n" + "\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(LogCorruption):
        StateMachine(LogStore(tmp_path / "state")).load()


def test_a_gap_on_disk_is_corruption(tmp_path: Path) -> None:
    machine = machine_at(tmp_path / "state")
    machine.append(OP_PATCH_CONFIG, {"values": {"uiTheme": "dark"}})
    machine.append(OP_PATCH_CONFIG, {"values": {"uiTheme": "light"}})

    log_path = tmp_path / "state" / "log.jsonl"
    entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    entries[1]["index"] = 4
    log_path.write_text(
        "".join(json.dumps(e, sort_keys=True) + "\n" for e in entries), encoding="utf-8"
    )

    with pytest.raises(LogCorruption, match="gap in the log"):
        StateMachine(LogStore(tmp_path / "state")).load()


# --------------------------------------------------------------------------- #
# Promotion, from a passphrase and nothing else
# --------------------------------------------------------------------------- #


def _replicate(active: StateMachine, standby_dir: Path) -> StateMachine:
    standby = standby_at(standby_dir)
    standby.install_snapshot(active.snapshot_document())
    for entry in active.read_page(standby.state.index, 10_000):
        standby.accept_replicated(entry)
    return standby


def test_a_replicating_standby_is_promotable_from_the_passphrase_alone(tmp_path: Path) -> None:
    """The whole point of the replication set, end to end.

    No key is copied between hosts. The standby holds the salt, the
    verifier and three sealed values it cannot open; the operator
    supplies the passphrase at promotion, which is the one moment it is
    needed — and that is what resolves "does the master key cross a host
    boundary" in the good direction.

    After promoting it must be able to *write*, because a promoted root
    that can read and not write is a control plane that is still down.
    """
    active_app = create_app(settings_for(tmp_path / "active"))
    with TestClient(active_app) as active:
        assert (
            active.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
        )
        active_machine: StateMachine = active_app.state.machine
        active_machine.append(
            "enrollNode",
            {
                "name": "attic",
                "role": "standby",
                "url": "http://127.0.0.1:8079",
                "publicKey": "cHVibGljLWtleS1ieXRlcy0zMi1jaGFyYWN0ZXJzLWs=",
                "enrolledAt": "2026-09-09T10:00:00+00:00",
            },
        )
        active_machine.append(OP_PATCH_CONFIG, {"values": {"uiTheme": "dark"}})
        epoch_before = active_machine.state.epoch
        # Note the path: `settings_for` puts the log under `<dir>/state`,
        # so a replica built for an app has to be seeded there.
        _replicate(active_machine, tmp_path / "standby" / "state")

    # The active root is gone now — its app is closed. The standby comes
    # up on its own replicated state, with no contact with anything.
    standby_app = create_app(settings_for(tmp_path / "standby", role="standby"))
    with TestClient(standby_app) as standby:
        status = standby.get("/healthz")
        assert status.status_code == 200
        assert status.json()["details"]["role"] == "standby"

        # Locked, not un-initialized. The distinction is the difference
        # between "log in" and "wipe the install", and an operator meets
        # it at the worst possible moment.
        locked = standby.get("/v1/control/status")
        assert locked.status_code == 503
        assert "Locked" in locked.text

        # A standby is loggable-into, using the replicated verifier and
        # salt. It has to be: deciding whether to promote means reading
        # `GET /v1/control/status` first, and that needs a token.
        token = standby.post("/v1/auth/login", json={"passphrase": PASSPHRASE}).json()[
            "sessionToken"
        ]
        headers = {"Authorization": f"Bearer {token}"}

        readable = standby.get("/v1/control/status", headers=headers)
        assert readable.status_code == 200
        assert readable.json()["role"] == "standby"
        assert readable.json()["epoch"] == epoch_before

        # Logged in and still not a writer. Reading is not writing, and
        # the role gate is what says so.
        rejected = standby.patch("/v1/config", json={"uiTheme": "light"}, headers=headers)
        assert rejected.status_code == 409, rejected.text
        assert "not the active root" in rejected.text.lower()

        refused = standby.post(
            "/v1/control/promote", json={"passphrase": "not-the-passphrase"}, headers=headers
        )
        assert refused.status_code == 401

        promoted = standby.post(
            "/v1/control/promote", json={"passphrase": PASSPHRASE}, headers=headers
        )
        assert promoted.status_code == 200, promoted.text
        body = promoted.json()
        assert body["role"] == "control"
        assert body["epoch"] == epoch_before + 1, "promotion must raise the epoch"

        # It can write now, which is the difference between promoted and
        # merely relabelled.
        patched = standby.patch("/v1/config", json={"uiTheme": "light"}, headers=headers)
        assert patched.status_code == 200, patched.text
        assert patched.json()["applied"] == ["uiTheme"]
        assert standby_app.state.machine.state.config["uiTheme"] == "light"


def test_a_standby_with_nothing_replicated_refuses_to_promote(tmp_path: Path) -> None:
    """*"A standby that cannot prove its state is worse than no standby."*

    An empty standby holds no verifier, so there is nothing to check the
    passphrase against. Refusing is the only honest answer — promoting
    would produce a control root with an empty node registry that
    believes it is authoritative.
    """
    app = create_app(settings_for(tmp_path / "empty", role="standby"))
    with TestClient(app) as client:
        # Nothing replicated means no verifier, so there is not even a
        # login to get a token from. 503 "Setup required" is the honest
        # answer: from this host's point of view there is no install.
        response = client.post("/v1/control/promote", json={"passphrase": PASSPHRASE})
        assert response.status_code == 503
        assert "Setup required" in response.text


def test_the_active_root_refuses_to_promote_itself(tmp_path: Path) -> None:
    app = create_app(settings_for(tmp_path / "active"))
    with TestClient(app) as client:
        assert (
            client.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
        )
        token = client.post("/v1/auth/login", json={"passphrase": PASSPHRASE}).json()[
            "sessionToken"
        ]
        response = client.post(
            "/v1/control/promote",
            json={"passphrase": PASSPHRASE},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 409
        assert "already" in response.text.lower()


# --------------------------------------------------------------------------- #
# Epoch fencing
# --------------------------------------------------------------------------- #


def test_a_returning_old_root_cannot_append_behind_its_successor(tmp_path: Path) -> None:
    """§6, at the log layer.

    The old root comes back believing it is still root. Its entries
    carry the superseded epoch, and every replica declines them
    independently — no election, no quorum, and no replica having to
    agree with any other replica. Each just refuses a downgrade.
    """
    active = machine_at(tmp_path / "active")
    active.append(OP_PATCH_CONFIG, {"values": {"uiTheme": "dark"}})

    standby = _replicate(active, tmp_path / "standby")
    standby.promote(node=None)
    assert standby.state.epoch == active.state.epoch + 1

    stale_entry = {
        "index": standby.state.index + 1,
        "epoch": active.state.epoch,
        "op": OP_PATCH_CONFIG,
        "payload": {"values": {"uiTheme": "light"}},
    }
    with pytest.raises(ApplyError, match="below the highest seen"):
        apply(standby.state, stale_entry)


def test_promotion_is_never_automatic(tmp_path: Path) -> None:
    """No timer, no threshold, no follower decision.

    Asserted because the temptation to close §6's partition window with
    automatic promotion will recur, and the whole argument against it —
    that it trades a visible, bounded, self-healing window for genuine
    split-brain — is easier to hold when something fails if the code
    drifts toward it.
    """
    from eugene_plexus_control import replication

    source = Path(replication.__file__).read_text(encoding="utf-8")
    assert "promote(" not in source, (
        "the follower must never promote. Falling behind, losing contact and the active "
        "root vanishing all mean the same thing here: keep trying, keep reporting. "
        "Promotion is an operator act."
    )
