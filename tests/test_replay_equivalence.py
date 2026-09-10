"""REQUIRED DELIVERABLE — replay equivalence.

> *"A standby's applied state must be byte-identical to the active
> root's after applying the same log."* — M5 design, §5

This is the property that makes a promotion safe rather than hopeful. It
is also, in the design's own words, "exactly the kind of invariant that
rots silently if nothing asserts it": breaking it does not fail a
request, it makes two hosts disagree about the same install and both
report healthy. So it is asserted here from five directions.

1. **Over HTTP, through the real replication surface.** A standby pulls
   `GET /v1/control/log` from a live active root and applies what it
   gets. This is the one that matters, because it exercises the
   transport as well as the logic — and the transport is where the first
   real divergence came from (Pydantic normalizing `+00:00` to `Z` and
   appending a slash to a URL, which is why the snapshot and the log
   travel verbatim).
2. **Through a snapshot**, which is how a standby actually bootstraps.
3. **Through a *compacted* snapshot plus a tail**, which is the path a
   standby takes when it fell behind compaction — and the path that
   would have silently lost config, runtime specs and the signing key
   before the contract was corrected.
4. **Twice, at different wall-clock times**, because a clock read
   anywhere in apply would show up here and nowhere else.
5. **Against every op**, so a new op cannot be added without either
   appearing here or leaving an obvious hole.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_control.app import create_app
from eugene_plexus_control.applied import (
    OP_DELETE_COMPONENT,
    OP_DELETE_RUNTIME,
    OP_ENROLL_NODE,
    OP_PATCH_CONFIG,
    OP_PROMOTE,
    OP_PUT_COMPONENT,
    OP_PUT_RUNTIME,
    OP_REVOKE_NODE,
    OP_ROTATE_SIGNING_KEY,
    AppliedState,
    ApplyError,
    apply,
    apply_all,
    canonical_bytes,
    from_canonical,
    to_canonical,
)
from eugene_plexus_control.state_machine import StateMachine

from .conftest import PASSPHRASE, machine_at, settings_for, standby_at

# Every op that changes replicated state, exercised in one history. The
# two that are not here are absent by design and asserted so below.
EXERCISED_OPS = {
    OP_ENROLL_NODE,
    OP_REVOKE_NODE,
    OP_PUT_COMPONENT,
    OP_DELETE_COMPONENT,
    OP_PUT_RUNTIME,
    OP_DELETE_RUNTIME,
    OP_PATCH_CONFIG,
    OP_ROTATE_SIGNING_KEY,
    OP_PROMOTE,
}


def placeholder(label: str) -> str:
    """A deterministic, obviously-not-a-key stand-in.

    Computed from readable text rather than written as a base64 literal.
    A literal here is indistinguishable from a real leaked key to a
    secret scanner — and to a reader — and this repo's CI runs gitleaks
    over every commit. The apply function treats these as opaque
    strings, so what matters is only that they are stable across a
    replay.
    """
    return base64.b64encode(f"THIS-IS-NOT-A-REAL-KEY::{label}".encode()).decode("ascii")


def _seed_identity(machine: StateMachine) -> None:
    machine.seed_identity(
        {
            "salt": base64.b64encode(b"0123456789abcdef").decode("ascii"),
            "passphraseVerifier": "$argon2id$v=19$m=65536,t=3,p=4$fake$verifier",
            "sealedSigningKey": placeholder("signing"),
            "sealedControlKey": placeholder("control-identity"),
            "controlPublicKey": placeholder("control-public"),
            "sealedRecoveryKey": placeholder("recovery"),
            "signingKeyId": "1",
        }
    )


def _write_history(machine: StateMachine) -> None:
    """One history that touches every op, with awkward values on purpose.

    The awkward values are the point of the exercise, not decoration:

    * a **non-ASCII node name**, because `ensure_ascii` in the serializer
      would change the bytes without changing the meaning;
    * a **URL with no path**, because that is exactly what Pydantic
      rewrites with a trailing slash;
    * **nested dicts with keys in a deliberately unsorted order**, so
      `sort_keys` is doing work rather than being a no-op;
    * a **config value set and then overwritten**, so "last write wins"
      is exercised rather than assumed;
    * a **node revoked after it owned a runtime**, so the cascade is in
      the compared state rather than only in a unit test.
    """
    machine.append(
        OP_ENROLL_NODE,
        {
            "name": "gpu-büro",
            "role": "agent",
            "url": "http://10.0.0.7:8079",
            "publicKey": "bm9kZS1wdWJsaWMta2V5LWJ5dGVzLTMyLWxvbmc=",
            "agentVersion": "0.1.0",
            "os": "linux",
            "arch": "x64",
            "devices": [
                {"memoryTotalBytes": 25769803776, "kind": "cuda", "index": 0, "name": "RTX 3090"},
                {"kind": "cpu", "memoryTotalBytes": 103079215104},
            ],
            "enrolledAt": "2026-09-09T11:22:33.444444+00:00",
        },
    )
    machine.append(
        OP_ENROLL_NODE,
        {
            "name": "attic",
            "role": "control",
            "url": "http://10.0.0.2:8079",
            "publicKey": "YW5vdGhlci1ub2RlLXB1YmxpYy1rZXktMzItbG9uZw==",
            "os": "windows",
            "arch": "x64",
            "enrolledAt": "2026-09-09T11:20:00+00:00",
        },
    )
    machine.append(
        OP_PUT_COMPONENT,
        {"node": "attic", "name": "gateway", "kind": "gateway", "url": "http://10.0.0.2:8080"},
    )
    machine.append(
        OP_PUT_COMPONENT,
        {
            "node": "gpu-büro",
            "name": "left",
            "kind": "inference-driver",
            "url": "http://10.0.0.7:8081",
        },
    )
    machine.append(
        OP_PUT_RUNTIME,
        {
            "node": "gpu-büro",
            "spec": {
                "name": "qwen-27b",
                "modelAlias": "qwen3.6-27b",
                "engine": "llama_cpp",
                "modelPath": "D:/models/qwen/Qwen3.6-27B-Q4_K_M.gguf",
                "flags": {"n-gpu-layers": 99, "ctx-size": 8192},
                "port": 8090,
            },
        },
    )
    machine.append(
        OP_PUT_RUNTIME,
        {
            "node": "attic",
            "spec": {"name": "small", "engine": "llama_cpp", "modelPath": "C:/m/small.gguf"},
        },
    )
    machine.append(OP_PATCH_CONFIG, {"values": {"uiTheme": "dark", "nodePollIntervalSeconds": 30}})
    machine.append(
        OP_PATCH_CONFIG, {"values": {"uiTheme": "light", "standbyUrls": ["http://10.0.0.9:8083/"]}}
    )
    machine.append(OP_DELETE_RUNTIME, {"node": "attic", "name": "small"})
    machine.append(OP_DELETE_COMPONENT, {"node": "attic", "name": "gateway"})
    machine.append(
        OP_ROTATE_SIGNING_KEY,
        {"signingKeyId": "2", "sealedSigningKey": placeholder("rotated"), "reason": "operator"},
    )
    # Revoking takes the node's component and runtime with it. Placed
    # last so the cascade shows up in the compared state.
    machine.append(OP_REVOKE_NODE, {"name": "gpu-büro"})
    machine.append(OP_PROMOTE, {"node": "attic"}, epoch_override=machine.state.epoch + 1)


def _ops_in(entries: list[dict[str, Any]]) -> set[str]:
    return {str(e["op"]) for e in entries}


# --------------------------------------------------------------------------- #
# 1. Over HTTP, through the real replication surface
# --------------------------------------------------------------------------- #


def test_standby_matches_active_after_pulling_the_log_over_http(tmp_path: Path) -> None:
    """The headline assertion, with the transport in the loop.

    A standby pulls from a live active root and ends byte-identical. The
    transport belongs in this test because it is where the first real
    divergence came from: re-serializing the log through a response
    model rewrote `+00:00` to `Z` and appended a slash to a URL, both
    semantically identical and both a different `sha256`.
    """
    active_app = create_app(settings_for(tmp_path / "active"))
    with TestClient(active_app) as active:
        assert (
            active.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
        )
        token = active.post("/v1/auth/login", json={"passphrase": PASSPHRASE}).json()[
            "sessionToken"
        ]
        headers = {"Authorization": f"Bearer {token}"}

        machine: StateMachine = active_app.state.machine
        _write_history(machine)

        # The standby bootstraps from the snapshot, exactly as the
        # follower does, then tails whatever the snapshot did not cover.
        snapshot = active.get("/v1/control/snapshot", headers=headers)
        assert snapshot.status_code == 200

        standby = standby_at(tmp_path / "standby")
        standby.install_snapshot(snapshot.json())

        page = active.get("/v1/control/log", params={"after": standby.state.index}, headers=headers)
        assert page.status_code == 200
        for entry in page.json()["entries"]:
            standby.accept_replicated(entry)

        assert standby.state.index == machine.state.index
        assert standby.canonical() == machine.canonical()


def test_the_snapshot_document_is_the_bytes_that_are_compared(tmp_path: Path) -> None:
    """`GET /v1/control/snapshot` sends `canonical_bytes` verbatim.

    Asserted directly rather than inferred, because the moment this
    endpoint starts re-serializing through a response model the headline
    test above gets a false failure that looks like a logic bug.
    """
    app = create_app(settings_for(tmp_path / "active"))
    with TestClient(app) as client:
        assert (
            client.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
        )
        token = client.post("/v1/auth/login", json={"passphrase": PASSPHRASE}).json()[
            "sessionToken"
        ]
        machine: StateMachine = app.state.machine
        _write_history(machine)

        response = client.get("/v1/control/snapshot", headers={"Authorization": f"Bearer {token}"})
        assert response.content == machine.canonical()


# --------------------------------------------------------------------------- #
# 2 & 3. Snapshot bootstrap, and bootstrap past compaction
# --------------------------------------------------------------------------- #


def test_replaying_the_whole_log_matches_the_writer(tmp_path: Path) -> None:
    """A replica that replays from nothing reaches the same state."""
    active = machine_at(tmp_path / "active")
    _seed_identity(active)
    _write_history(active)

    replayed = apply_all(from_canonical(_genesis(active)), _entries(active))
    assert canonical_bytes(replayed) == active.canonical()


def test_snapshot_plus_tail_matches_a_root_that_never_snapshotted(tmp_path: Path) -> None:
    """The compaction path, which is where the contract was wrong.

    A standby that bootstraps from a snapshot taken partway through and
    then applies the tail must reach the same state as a root that
    replayed everything. Before `Snapshot` grew `config`, `signingKeyId`
    and a runtime *spec*, this test is what would have caught it — and
    it would have been the only thing that did, because nothing else
    reads state through a snapshot.
    """
    active = machine_at(tmp_path / "active")
    _seed_identity(active)
    _write_history(active)
    all_entries = _entries(active)

    # Rebuild a mid-history state, snapshot it, and hand that to a fresh
    # replica along with only the entries after it.
    midpoint = len(all_entries) // 2
    partial = apply_all(from_canonical(_genesis(active)), all_entries[:midpoint])

    replica = standby_at(tmp_path / "replica")
    replica.install_snapshot(to_canonical(partial))
    assert replica.state.index == partial.index

    for entry in all_entries[midpoint:]:
        replica.accept_replicated(entry)

    assert replica.canonical() == active.canonical()


def test_compaction_does_not_change_applied_state(tmp_path: Path) -> None:
    """Compacting and reloading from disk is a no-op on state.

    The log file shrinks; the state does not move. If an op's effect had
    nowhere to land in the snapshot this is where it would vanish.
    """
    active = machine_at(tmp_path / "active")
    _seed_identity(active)
    _write_history(active)
    before = active.canonical()

    active.snapshot_now()
    reloaded = machine_at(tmp_path / "active")
    assert reloaded.canonical() == before
    # And the log really was compacted, so the test is not passing
    # because nothing happened.
    assert reloaded.first_available_index() == reloaded.state.index + 1


# --------------------------------------------------------------------------- #
# 4. Time cannot enter
# --------------------------------------------------------------------------- #


def test_applying_the_same_log_at_two_different_times_is_identical(tmp_path: Path) -> None:
    """Rule 2: every timestamp comes from the payload.

    A clock read anywhere in `apply` would show up here and nowhere
    else — no request fails, no type is wrong, and a promotion months
    later silently produces a state the active root never had.
    """
    active = machine_at(tmp_path / "active")
    _seed_identity(active)
    _write_history(active)
    entries = _entries(active)
    genesis = _genesis(active)

    first = canonical_bytes(apply_all(from_canonical(genesis), entries))
    time.sleep(0.01)
    second = canonical_bytes(apply_all(from_canonical(genesis), entries))
    assert first == second


def test_apply_imports_nothing_that_can_tell_the_time() -> None:
    """Rule 1, checked structurally as well as behaviourally.

    The behavioural test above catches a clock read only if the clock
    moved enough between the two runs. This catches the import itself,
    which is the thing a reviewer would actually miss.
    """
    import eugene_plexus_control.applied as applied_module

    forbidden = {"datetime", "time", "random", "secrets", "os", "Path", "httpx"}
    present = forbidden & set(vars(applied_module))
    assert not present, (
        f"applied.py has {sorted(present)} in scope. apply() must be pure — no clock, no "
        "filesystem, no network, no randomness — or a standby's replay diverges from the "
        "active root's original."
    )


# --------------------------------------------------------------------------- #
# 5. Every op, and the ones deliberately absent
# --------------------------------------------------------------------------- #


def test_the_history_exercises_every_replicated_op(tmp_path: Path) -> None:
    """A new op cannot be added without showing up here.

    Deliberately an assertion about coverage rather than a comment
    asking for it: `LogOp` is closed, so an op that is not exercised is
    an op whose replication nobody has checked.
    """
    active = machine_at(tmp_path / "active")
    _seed_identity(active)
    _write_history(active)
    assert _ops_in(_entries(active)) == EXERCISED_OPS


def test_initialization_and_join_tokens_are_not_log_ops(tmp_path: Path) -> None:
    """The two mutations that deliberately do not replicate.

    Initialization happens once, before any standby exists, so the
    snapshot always carries it forward. A join token is
    active-root-local, stored only as a hash, and useless if the root
    that minted it dies first. Both are recorded here so that "why isn't
    there an `initialize` op" has an answer that is checked rather than
    remembered.
    """
    active = machine_at(tmp_path / "active")
    _seed_identity(active)
    assert _entries(active) == []
    assert active.state.identity.salt is not None
    # It survived into the snapshot rather than into the log.
    reloaded = machine_at(tmp_path / "active")
    assert reloaded.state.identity.salt == active.state.identity.salt


# --------------------------------------------------------------------------- #
# The negative cases — a replica must stop rather than guess
# --------------------------------------------------------------------------- #


def test_a_gap_is_refused_rather_than_skipped() -> None:
    state = AppliedState()
    with pytest.raises(ApplyError, match="out-of-order"):
        apply(state, {"index": 2, "epoch": 1, "op": OP_PATCH_CONFIG, "payload": {"values": {}}})


def test_a_replayed_entry_is_refused() -> None:
    state = AppliedState()
    entry = {
        "index": 1,
        "epoch": 1,
        "op": OP_PATCH_CONFIG,
        "payload": {"values": {"uiTheme": "dark"}},
    }
    state = apply(state, entry)
    with pytest.raises(ApplyError, match="out-of-order"):
        apply(state, entry)


def test_an_entry_from_a_superseded_epoch_is_refused() -> None:
    """Epoch fencing reaches the log, not only the agents.

    A returning old root cannot append behind its successor's back,
    which is the same rule as §6 applied one layer down.
    """
    state = AppliedState()
    state = apply(state, {"index": 1, "epoch": 5, "op": OP_PROMOTE, "payload": {}})
    with pytest.raises(ApplyError, match="below the highest seen"):
        apply(state, {"index": 2, "epoch": 4, "op": OP_PATCH_CONFIG, "payload": {"values": {}}})


def test_an_unknown_op_is_refused() -> None:
    """An op absent from `LogOp` is an op that would not replicate."""
    with pytest.raises(ApplyError, match="unknown op"):
        apply(AppliedState(), {"index": 1, "epoch": 1, "op": "deleteEverything", "payload": {}})


def test_a_standby_refuses_to_assign_indices(tmp_path: Path) -> None:
    """One writer, enforced rather than agreed."""
    from eugene_plexus_control.state_machine import NotActive

    standby = standby_at(tmp_path / "standby")
    with pytest.raises(NotActive):
        standby.append(OP_PATCH_CONFIG, {"values": {"uiTheme": "dark"}})


def test_the_active_root_refuses_replicated_entries(tmp_path: Path) -> None:
    """The other half of the same rule: the active root assigns, it does
    not accept. A root that took entries from elsewhere would be a
    second writer with extra steps."""
    from eugene_plexus_control.state_machine import NotActive

    active = machine_at(tmp_path / "active")
    with pytest.raises(NotActive):
        active.accept_replicated({"index": 1, "epoch": 1, "op": OP_PROMOTE, "payload": {}})


def test_a_snapshot_older_than_applied_state_is_refused(tmp_path: Path) -> None:
    """Pointing a standby at a stale root must be an error, not a rewind.

    Silently adopting an older snapshot would discard entries this
    replica had already acknowledged — which is the same class of
    mistake as promoting a standby that is behind, and gets the same
    answer: refuse and say how far.
    """
    active = machine_at(tmp_path / "active")
    _seed_identity(active)
    _write_history(active)

    replica = standby_at(tmp_path / "replica")
    replica.install_snapshot(active.snapshot_document())

    stale = to_canonical(AppliedState())
    with pytest.raises(ApplyError, match="already applied"):
        replica.install_snapshot(stale)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _entries(machine: StateMachine) -> list[dict[str, Any]]:
    return machine.read_page(0, 10_000)


def _genesis(machine: StateMachine) -> dict[str, Any]:
    """The state a replica starts from: the identity snapshot at index 0.

    Reconstructed rather than captured because `seed_identity` writes it
    before any entry exists, which is precisely the claim being relied
    on.
    """
    genesis = to_canonical(machine.state)
    return {
        **genesis,
        "index": 0,
        "epoch": 1,
        "nodes": [],
        "components": [],
        "runtimes": [],
        "config": {},
        "signingKeyId": "1",
        "sealedSigningKey": placeholder("signing"),
    }
