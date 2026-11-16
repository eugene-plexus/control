"""Row 70: the one function that says which group policies apply to a person."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from eugene_plexus_control import permissions


def _group(group_id: str, name: str, **policies: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": group_id,
        "name": name,
        "description": "",
        "apps": [],
        "permissions": [],
        "createdAt": "2026-10-21T00:00:00+00:00",
        "updatedAt": "2026-10-21T00:00:00+00:00",
    }
    if policies:
        record["policies"] = policies
    return record


def _state(*groups: dict[str, Any]) -> Any:
    return SimpleNamespace(permission_groups={g["id"]: g for g in groups}, people={}, config={})


def test_effective_policies_take_the_most_generous_value() -> None:
    state = _state(
        _group("g1", "Night shift", queueBand="background", summaryContextPercent=70),
        _group("g2", "Desk", queueBand="interactive", summaryContextPercent=85),
        _group("g3", "Batch", queueBand="sheddable"),
        _group("g4", "Plain"),
    )
    both = permissions.effective_policies(state, {"groups": ["g1", "g2"]})
    assert both == {
        "queueBand": "interactive",
        "summaryContextPercent": 85,
        "setBy": {"queueBand": ["g2"], "summaryContextPercent": ["g2"]},
    }
    # The order the person lists their groups in does not matter.
    assert permissions.effective_policies(state, {"groups": ["g2", "g1"]}) == both
    # Background beats sheddable; a group with no percent leaves another's.
    assert permissions.effective_policies(state, {"groups": ["g3", "g1"]}) == {
        "queueBand": "background",
        "summaryContextPercent": 70,
        "setBy": {"queueBand": ["g1"], "summaryContextPercent": ["g1"]},
    }
    # None set gives unset: no policy, so the reader's default applies.
    assert permissions.effective_policies(state, {"groups": ["g4"]}) == {}
    assert permissions.effective_policies(state, {"groups": []}) == {}
    # Individual access (groups null or absent) takes no group's policy.
    assert permissions.effective_policies(state, {"groups": None}) == {}
    assert permissions.effective_policies(state, {}) == {}
    # A dangling group id (older or corrupt state) is skipped, not trusted.
    assert permissions.effective_policies(state, {"groups": ["gone"]}) == {}


def test_every_group_that_sets_the_winning_value_is_named() -> None:
    state = _state(
        _group("b", "Beta", queueBand="interactive"),
        _group("a", "Alpha", queueBand="interactive", summaryContextPercent=60),
        _group("c", "Gamma", queueBand="background", summaryContextPercent=60),
    )
    assert permissions.effective_policies(state, {"groups": ["b", "c", "a"]}) == {
        "queueBand": "interactive",
        "summaryContextPercent": 60,
        # Sorted by group name, as the claim and the console show them.
        "setBy": {"queueBand": ["a", "b"], "summaryContextPercent": ["a", "c"]},
    }


def test_groups_in_effect_are_sorted_by_name_and_bounded() -> None:
    state = _state(_group("z", "zulu"), _group("a", "Alpha"), _group("m", "mike"))
    person = {"groups": ["z", "m", "a", "gone"]}
    assert permissions.groups_in_effect(state, person) == [
        {"id": "a", "name": "Alpha"},
        {"id": "m", "name": "mike"},
        {"id": "z", "name": "zulu"},
    ]
    assert permissions.groups_in_effect(state, {"groups": None}) == []


# --------------------------------------------------------------------------- #
# row 37: routing from the active group; model lists from every group
# --------------------------------------------------------------------------- #


def test_role_routing_follows_the_active_group() -> None:
    """R37-FIRST (Troy, 2026-10-22): routing comes from one group, the
    person's active group: their choice, else the first in their list."""
    state = _state(
        _group("g1", "Sales", roleRouting={"chat": ["m2", "m1"]}),
        _group("g2", "Ops", roleRouting={"chat": ["m3"]}),
        _group("g3", "Plain", queueBand="background"),
    )
    first = permissions.effective_policies(state, {"groups": ["g2", "g1"]})
    assert first["activeGroup"] == "g2"
    assert first["roleRouting"] == {"chat": ["m3"]}
    assert first["setBy"]["roleRouting"] == ["g2"]
    chosen = permissions.effective_policies(state, {"groups": ["g2", "g1"], "activeGroup": "g1"})
    assert chosen["activeGroup"] == "g1"
    assert chosen["roleRouting"] == {"chat": ["m2", "m1"]}
    assert chosen["setBy"]["roleRouting"] == ["g1"]
    # Not merged: an active group that routes nothing leaves the install's.
    plain = permissions.effective_policies(state, {"groups": ["g3", "g1"]})
    assert plain == {"queueBand": "background", "setBy": {"queueBand": ["g3"]}}
    assert permissions.active_group(state, {"groups": ["g3", "g1"]})["id"] == "g3"
    # A choice the person is no longer in (older state) falls back to the first.
    stale = permissions.effective_policies(state, {"groups": ["g2"], "activeGroup": "g1"})
    assert stale["activeGroup"] == "g2" and stale["roleRouting"] == {"chat": ["m3"]}
    # Individual access: no group, so none of it.
    assert permissions.effective_policies(state, {"groups": None, "activeGroup": "g1"}) == {}


def test_choosing_a_group_with_no_list_does_not_widen_the_models() -> None:
    """R37-SELF, Choice 3 (default built): a person in Sales (m1) and
    Everyone (no list) who chooses Everyone still reaches only m1."""
    state = _state(_group("g1", "Sales", allowedModels=["m1"]), _group("g2", "Everyone"))
    person = {"groups": ["g1", "g2"], "activeGroup": "g2"}
    found = permissions.effective_policies(state, person)
    assert found["modelLimits"] == [{"group": {"id": "g1", "name": "Sales"}, "models": ["m1"]}]
    answer = permissions.model_policies(state, person)
    assert answer["modelLimits"] == [{"group": {"id": "g1", "name": "Sales"}, "models": ["m1"]}]
    assert "roleRouting" not in answer and "routingGroup" not in answer


def test_two_model_lists_both_apply() -> None:
    state = _state(
        _group("g1", "Sales", allowedModels=["m1", "qwen3-*"]),
        _group("g2", "Night", allowedModels=["*-8b", "m1"]),
        _group("g3", "Open"),
        _group("g4", "Nothing", allowedModels=[]),
    )
    both = permissions.effective_policies(state, {"groups": ["g2", "g1", "g3"]})
    # Sorted by group name; a model must be in every list.
    assert both["modelLimits"] == [
        {"group": {"id": "g2", "name": "Night"}, "models": ["*-8b", "m1"]},
        {"group": {"id": "g1", "name": "Sales"}, "models": ["m1", "qwen3-*"]},
    ]
    assert "modelLimits" not in permissions.effective_policies(state, {"groups": ["g3"]})
    # An empty list is a limit: no model.
    assert permissions.effective_policies(state, {"groups": ["g4", "g3"]})["modelLimits"] == [
        {"group": {"id": "g4", "name": "Nothing"}, "models": []}
    ]


def test_leaving_a_group_widens_the_models() -> None:
    state = _state(_group("g1", "Sales", allowedModels=["m1"]), _group("g2", "Everyone"))
    assert permissions.model_policies(state, {"groups": ["g1", "g2"]})["modelLimits"]
    assert permissions.model_policies(state, {"groups": ["g2"]}) == {}


def test_the_claim_keeps_only_the_row_70_policies() -> None:
    """A Workbench from before row 37 refuses `eugene_policies` with fields it
    does not know, so the claim carries the row 70 policies alone."""
    state = _state(
        _group(
            "g1",
            "Sales",
            queueBand="background",
            roleRouting={"chat": ["m1"]},
            allowedModels=["m1"],
        ),
    )
    claim = permissions.claim_policies(state, {"groups": ["g1"]})
    assert claim == {"queueBand": "background", "setBy": {"queueBand": ["g1"]}}
