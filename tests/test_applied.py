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


# --------------------------------------------------------------------------- #
# row 37: models and routing, and a person's active group
# --------------------------------------------------------------------------- #


def test_a_group_entry_without_the_new_policies_loads() -> None:
    """A row 70 group (queue band, percent) loads as no routing and no model
    limit: absent is the old behaviour."""
    group = {**_old_group(), "policies": {"queueBand": "background"}}
    state = apply_all(AppliedState(), _entries(group))
    assert permissions.model_policies(state, state.people["p1"]) == {}
    assert "roleRouting" not in permissions.effective_policies(state, state.people["p1"])


def test_a_group_with_routing_and_models_replicates_and_round_trips() -> None:
    policies = {"roleRouting": {"chat": ["m2", "m1"]}, "allowedModels": ["m1", "m2", "qwen3-*"]}
    state = apply_all(AppliedState(), _entries({**_old_group(), "policies": policies}))
    assert state.permission_groups["g1"]["policies"] == policies
    restored = from_canonical(to_canonical(state))
    assert canonical_bytes(restored) == canonical_bytes(state)
    group = {"id": "g1", "name": "Night shift"}
    assert permissions.model_policies(restored, restored.people["p1"]) == {
        "roleRouting": policies["roleRouting"],
        "routingGroup": group,
        "modelLimits": [{"group": group, "models": policies["allowedModels"]}],
    }


@pytest.mark.parametrize(
    "policies",
    [
        {"roleRouting": {}},  # kept only when a role is routed
        {"roleRouting": None},
        {"roleRouting": {"typo": ["m1"]}},
        {"roleRouting": {"chat": []}},
        {"roleRouting": {"chat": ["m1", "m1"]}},
        {"roleRouting": {"chat": [""]}},
        {"roleRouting": {"chat": ["m"] * 17}},
        {"roleRouting": {"chat": "m1"}},
        {"roleRouting": ["chat"]},
        {"allowedModels": None},
        {"allowedModels": "m1"},
        {"allowedModels": ["m1", "m1"]},
        {"allowedModels": [""]},
        {"allowedModels": [f"m{i}" for i in range(65)]},
        {"allowedModels": ["x" * 257]},
        {"allowedModels": ["*a*b*c*d*e*f*g*h*i"]},  # nine *: past the review's cap of 8
    ],
)
def test_a_bad_routing_or_model_policy_never_enters_the_log(policies: Any) -> None:
    with pytest.raises(ApplyError, match="permission group polic"):
        apply_all(AppliedState(), _entries({**_old_group(), "policies": policies}))


def _with_active(active: str | None, groups: list[str]) -> list[dict[str, Any]]:
    entries = _entries(_old_group("g1", "Sales"), _old_group("g2", "Ops"))
    person = {**_person(groups)}
    if active is not None:
        person["activeGroup"] = active
    entries[-1]["payload"]["person"] = person
    return entries


def test_an_active_group_replicates_and_round_trips() -> None:
    state = apply_all(AppliedState(), _with_active("g2", ["g1", "g2"]))
    assert state.people["p1"]["activeGroup"] == "g2"
    restored = from_canonical(to_canonical(state))
    assert canonical_bytes(restored) == canonical_bytes(state)
    assert restored.people["p1"]["activeGroup"] == "g2"


def test_a_person_cannot_be_given_a_group_they_are_not_in() -> None:
    with pytest.raises(ApplyError, match="active group"):
        apply_all(AppliedState(), _with_active("g2", ["g1"]))
    with pytest.raises(ApplyError, match="active group"):
        apply_all(AppliedState(), _with_active("g2", None))  # type: ignore[arg-type]
    with pytest.raises(ApplyError, match="active group"):
        apply_all(AppliedState(), _with_active("", ["g1"]))


def test_deleting_the_active_group_clears_the_choice() -> None:
    from eugene_plexus_control.applied import OP_DELETE_PERMISSION_GROUP

    entries = _with_active("g2", ["g1", "g2"])
    entries.append(
        {
            "index": len(entries) + 1,
            "epoch": 1,
            "op": OP_DELETE_PERMISSION_GROUP,
            "payload": {"id": "g2"},
        }
    )
    state = apply_all(AppliedState(), entries)
    assert state.people["p1"]["groups"] == ["g1"]
    assert "activeGroup" not in state.people["p1"]


def _choose(entries: list[dict[str, Any]], active: Any) -> list[dict[str, Any]]:
    from eugene_plexus_control.applied import OP_SET_PERSON_ACTIVE_GROUP

    return [
        *entries,
        {
            "index": len(entries) + 1,
            "epoch": 1,
            "op": OP_SET_PERSON_ACTIVE_GROUP,
            "payload": {"id": "p1", "activeGroup": active},
        },
    ]


def test_a_choice_entry_changes_only_the_active_group() -> None:
    """Row 37 review B1: the person's own choice is its own entry, applied to
    the record as it is then, so it cannot write back a disabled flag, a
    membership, a grant or a password the owner changed meanwhile."""
    entries = _with_active(None, ["g1", "g2"])
    later = {**entries[-1]["payload"]["person"], "disabled": True, "groups": ["g2"]}
    entries.append(
        {"index": len(entries) + 1, "epoch": 1, "op": OP_PUT_PERSON, "payload": {"person": later}}
    )
    state = apply_all(AppliedState(), _choose(entries, "g2"))
    person = state.people["p1"]
    assert person["activeGroup"] == "g2"
    assert person["disabled"] is True and person["groups"] == ["g2"]
    cleared = apply_all(AppliedState(), _choose(_choose(entries, "g2"), None))
    assert "activeGroup" not in cleared.people["p1"]
    # A group they are no longer in is refused, as in a whole record.
    with pytest.raises(ApplyError, match="active group"):
        apply_all(AppliedState(), _choose(entries, "g1"))
