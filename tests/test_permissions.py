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
