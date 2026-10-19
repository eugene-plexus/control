"""What v0.1 wrote and v0.2 retired is refused, by name, and never half-applied.

v0.3.0 converted it once, on the root that held it. This version does not: a log
or a snapshot that still carries `putNodeHelper`, an `enrollNode` granting `files`,
a person's `helperGrants` or a snapshot's `nodeHelpers` raises `ApplyError`, whose
words name what was found and say to start v0.3.0 once on that root. The refusal is
the replica's ordinary one for an entry it cannot apply (see `applied.ApplyError`),
and nothing is persisted before it.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_control import applied
from eugene_plexus_control.app import create_app
from eugene_plexus_control.applied import AppliedState, ApplyError
from eugene_plexus_control.log_store import LOG_FILENAME, LogStore
from eugene_plexus_control.state_machine import ROLE_ACTIVE, ROLE_STANDBY, StateMachine

from .conftest import settings_for

WAY_OUT = (
    "was retired in v0.2; start Eugene v0.3.0 once on the active root to convert it, then update."
)


def _entry(index: int, op: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"index": index, "epoch": 1, "op": op, "payload": payload, "at": "2026-10-05T12:00:00Z"}


def _enroll(grants: list[str]) -> dict[str, Any]:
    return {
        "name": "oldsite",
        "role": "agent",
        "grants": grants,
        "enrolledAt": "2026-10-05T12:00:00+00:00",
    }


def test_a_node_folder_entry_is_refused() -> None:
    entry = _entry(1, "putNodeHelper", {"helper": {"node": "n", "enabled": True, "folders": []}})
    with pytest.raises(ApplyError) as refused:
        applied.apply(AppliedState(), entry)
    assert str(refused.value) == (
        "entry 1 (putNodeHelper, an operator-managed node folder) " + WAY_OUT
    )


def test_a_job_site_enrolled_as_a_node_is_refused() -> None:
    with pytest.raises(ApplyError) as refused:
        applied.apply(AppliedState(), _entry(1, "enrollNode", _enroll(["files"])))
    assert str(refused.value) == (
        "entry 1 (an enrollNode with the grant `files`, a Job Site as a node) " + WAY_OUT
    )
    # An ordinary node still enrolls.
    state = applied.apply(AppliedState(), _entry(1, "enrollNode", _enroll(["node"])))
    assert "oldsite" in state.nodes


def test_a_person_with_folder_grants_is_refused() -> None:
    with pytest.raises(ApplyError) as refused:
        applied._person_record({"helperGrants": []})
    assert str(refused.value) == "a person record with folder grants (helperGrants) " + WAY_OUT


def test_a_snapshot_with_node_folders_or_a_files_node_is_refused() -> None:
    document = applied.to_canonical(AppliedState())
    assert "nodeHelpers" not in document, "nothing writes it now; no reader of v0.3.0 needs it"
    assert applied.from_canonical(document) == AppliedState()
    # An empty list is what v0.3.0 writes still: it is read, not refused.
    assert applied.from_canonical({**document, "nodeHelpers": []}) == AppliedState()

    with pytest.raises(ApplyError) as refused:
        applied.from_canonical({**document, "nodeHelpers": [{"node": "n"}]})
    assert str(refused.value) == "a snapshot with node folders (nodeHelpers) " + WAY_OUT

    node = {"name": "oldsite", "role": "agent", "grants": ["files"]}
    with pytest.raises(ApplyError) as refused:
        applied.from_canonical({**copy.deepcopy(document), "nodes": [node]})
    assert str(refused.value) == ("a snapshot node 'oldsite' with the grant `files` " + WAY_OUT)


def test_a_root_that_replays_one_refuses_to_load_and_changes_nothing(tmp_path: Path) -> None:
    """`StateMachine.load` does not catch it: the root does not come up on a log it
    cannot apply, and the log is left as it was."""
    store = LogStore(tmp_path)
    store.append(_entry(1, "enrollNode", _enroll(["files"])))
    before = (tmp_path / LOG_FILENAME).read_bytes()

    machine = StateMachine(LogStore(tmp_path), role=ROLE_ACTIVE)
    with pytest.raises(ApplyError, match=r"start Eugene v0\.3\.0 once on the active root"):
        machine.load()
    assert (tmp_path / LOG_FILENAME).read_bytes() == before


def test_a_standby_refuses_a_replicated_one_before_it_persists_it(tmp_path: Path) -> None:
    machine = StateMachine(LogStore(tmp_path), role=ROLE_STANDBY)
    machine.load()
    with pytest.raises(ApplyError, match=r"start Eugene v0\.3\.0 once on the active root"):
        machine.accept_replicated(_entry(1, "enrollNode", _enroll(["files"])))
    assert not (tmp_path / LOG_FILENAME).exists()
    assert machine.state.index == 0


def test_a_standby_holding_one_sets_its_records_aside_and_starts_empty(tmp_path: Path) -> None:
    """v0.3.0 converted only the active root. A standby that followed an old root keeps
    its records beside it, renamed, and starts empty to bootstrap from the root."""
    state = tmp_path / "standby" / "state"
    LogStore(state).append(_entry(1, "enrollNode", _enroll(["files"])))
    before = (state / LOG_FILENAME).read_bytes()

    app = create_app(settings_for(tmp_path / "standby", role="standby", active_url=None))
    with TestClient(app):
        assert app.state.machine.state.index == 0
    assert not (state / LOG_FILENAME).exists()
    assert (state / (LOG_FILENAME + ".retired-v0.1")).read_bytes() == before


def test_an_active_root_holding_one_does_not_start(tmp_path: Path) -> None:
    state = tmp_path / "root" / "state"
    LogStore(state).append(_entry(1, "enrollNode", _enroll(["files"])))
    app = create_app(settings_for(tmp_path / "root"))
    with pytest.raises(ApplyError, match=r"once on the active root"), TestClient(app):
        pass
    assert (state / LOG_FILENAME).exists()
