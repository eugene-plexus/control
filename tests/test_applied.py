"""Row 70: a group's policies in the replicated log and the snapshot."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from eugene_plexus_control import permissions
from eugene_plexus_control.applied import (
    OP_PUT_PERMISSION_GROUP,
    OP_PUT_PERSON,
    AppliedState,
    ApplyError,
    apply_all,
    canonical_bytes,
    from_canonical,
    to_canonical,
)

WHEN = "2026-10-21T00:00:00+00:00"


def _old_group(group_id: str = "g1", name: str = "Night shift") -> dict[str, Any]:
    """A group as a root before row 70 wrote it: no `policies` key."""
    return {
        "id": group_id,
        "name": name,
        "description": "",
        "apps": None,
        "permissions": [],
        "createdAt": WHEN,
        "updatedAt": WHEN,
    }


def _person(groups: list[str]) -> dict[str, Any]:
    return {
        "id": "p1",
        "name": "Ada",
        "passwordVerifier": "$argon2id$v=19$m=8,t=1,p=1$c2FsdHNhbHQ$aGFzaGhhc2g",
        "apps": None,
        "groups": groups,
        "disabled": False,
        "createdAt": WHEN,
        "passwordChangedAt": WHEN,
    }


def _entries(*groups: dict[str, Any]) -> list[dict[str, Any]]:
    entries = [
        {"index": i + 1, "epoch": 1, "op": OP_PUT_PERMISSION_GROUP, "payload": {"group": g}}
        for i, g in enumerate(groups)
    ]
    entries.append(
        {
            "index": len(entries) + 1,
            "epoch": 1,
            "op": OP_PUT_PERSON,
            "payload": {"person": _person([g["id"] for g in groups])},
        }
    )
    return entries


def test_an_old_group_entry_without_policies_loads() -> None:
    state = apply_all(AppliedState(), _entries(_old_group()))
    assert "policies" not in state.permission_groups["g1"]
    assert permissions.effective_policies(state, state.people["p1"]) == {}
    # And an old snapshot without them round-trips to the same bytes.
    snapshot = to_canonical(state)
    assert "policies" not in snapshot["permissionGroups"][0]
    assert canonical_bytes(from_canonical(snapshot)) == canonical_bytes(state)


def test_a_group_with_policies_replicates_and_round_trips() -> None:
    group = {**_old_group(), "policies": {"queueBand": "background", "summaryContextPercent": 70}}
    state = apply_all(AppliedState(), _entries(group))
    assert state.permission_groups["g1"]["policies"] == {
        "queueBand": "background",
        "summaryContextPercent": 70,
    }
    restored = from_canonical(to_canonical(state))
    assert canonical_bytes(restored) == canonical_bytes(state)
    assert permissions.effective_policies(restored, restored.people["p1"])["queueBand"] == (
        "background"
    )


@pytest.mark.parametrize(
    "policies",
    [
        {},  # kept only when a policy is set
        {"queueBand": None},  # not set is absent, never null, in the log
        {"queueBand": "urgent"},
        {"queueBand": "interactive", "routing": "fast"},
        {"summaryContextPercent": 49},
        {"summaryContextPercent": 96},
        {"summaryContextPercent": 80.0},
        {"summaryContextPercent": True},
        {"summaryContextPercent": "80"},
        ["queueBand"],
        "interactive",
    ],
)
def test_a_bad_policy_never_enters_the_log(policies: Any) -> None:
    group = {**_old_group(), "policies": policies}
    with pytest.raises(ApplyError, match="permission group polic"):
        apply_all(AppliedState(), _entries(group))
    good = apply_all(AppliedState(), _entries(_old_group()))
    snapshot = deepcopy(to_canonical(good))
    snapshot["permissionGroups"][0]["policies"] = policies
    with pytest.raises(ApplyError, match="permission group polic"):
        from_canonical(snapshot)
