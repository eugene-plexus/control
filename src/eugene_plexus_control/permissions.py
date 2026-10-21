"""What each person may do with job sites (J77, `job-sites-own-enrollment.md` §3.3).

Kept here at the root and set where People are managed. A person's
`permissions` is exactly the list given; null is the install's defaults, the
settings `peopleMayAddJobSites` and `peopleMayUseJobSites`, both on, so a
household needs no setup and an organisation turns them off and grants
person by person.

They only narrow. Whatever they say, a site still needs the person's OS
account on its machine, a link made there and their own key on their own
changes, so a root that sets one wrongly gives no one more than the machine
allows, and the site checks no signature for them.

- `add-job-sites`: add a machine of their own as a job site (J9).
- `use-job-sites`: link themselves at a site's machine, and keep workspaces
  of their own there, and call into them. A folder the site's owner shares
  with them is the owner's grant and is not governed by it. Taking it away
  leaves a link inert, not removed.
"""

from __future__ import annotations

from typing import Any

ADD_JOB_SITES = "add-job-sites"
USE_JOB_SITES = "use-job-sites"
#: Each permission and the setting that is its default.
DEFAULTS: dict[str, str] = {
    ADD_JOB_SITES: "peopleMayAddJobSites",
    USE_JOB_SITES: "peopleMayUseJobSites",
}


def default_on(state: Any, permission: str) -> bool:
    """The install's default for `permission`: on unless set off."""
    return state.config.get(DEFAULTS[permission]) is not False


def in_effect(state: Any, person: dict[str, Any]) -> list[str]:
    """What applies to `person` now: their own list, or the defaults."""
    if person.get("groups") is not None:
        granted = {p for group in _groups(state, person) for p in group["permissions"]}
        return [p for p in DEFAULTS if p in granted]
    chosen = person.get("permissions")
    if chosen is None:
        return [p for p in DEFAULTS if default_on(state, p)]
    return [p for p in DEFAULTS if p in chosen]


def has(state: Any, subject: str, permission: str) -> bool:
    """Whether person `subject` holds `permission`. Eugene's owner and
    anyone not on the People list hold none."""
    person = state.people.get(subject)
    return (
        person is not None and not person.get("disabled") and permission in in_effect(state, person)
    )


def refusal(permission: str) -> str:
    """The sentence a person is told when they lack `permission`."""
    if permission == ADD_JOB_SITES:
        return "Eugene's owner has not let you add job sites. They can, on Eugene's People page."
    return (
        "Eugene's owner has not let you use job sites as yourself. They can, on Eugene's "
        "People page. A folder a site's owner shares with you still works."
    )


def _groups(state: Any, person: dict[str, Any]) -> list[dict[str, Any]]:
    # Fail closed if corrupt/older state ever refers to an absent group.
    return [
        state.permission_groups[g]
        for g in person.get("groups") or []
        if g in state.permission_groups
    ]


def apps_in_effect(state: Any, person: dict[str, Any]) -> list[str] | None:
    """Null grants every app; an explicit empty group selection grants none."""
    if person.get("groups") is None:
        apps: list[str] | None = person.get("apps")
        return apps
    groups = _groups(state, person)
    if any(g["apps"] is None for g in groups):
        return None
    return sorted({a for g in groups for a in g["apps"]})


# --------------------------------------------------------------------------- #
# group policies (row 70)
# --------------------------------------------------------------------------- #

#: Queue bands, lowest first: the most generous is the last.
QUEUE_BANDS: tuple[str, ...] = ("sheddable", "background", "interactive")
SUMMARY_PERCENT_MIN = 50
SUMMARY_PERCENT_MAX = 95
#: The policies a group may set (`GroupPolicies`).
GROUP_POLICIES: tuple[str, ...] = ("queueBand", "summaryContextPercent")
#: At most this many groups in the `eugene_groups` claim (`Person.groups`'s own bound).
MAX_CLAIM_GROUPS = 64


def _by_name(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(groups, key=lambda g: (g["name"].casefold(), g["id"]))


def _generosity(policy: str, value: Any) -> int:
    return QUEUE_BANDS.index(value) if policy == "queueBand" else int(value)


def effective_policies(state: Any, person: dict[str, Any]) -> dict[str, Any]:
    """The group policies that apply to `person` now, as `PoliciesInEffect`.

    The most generous value of each policy across the groups they are in:
    the highest queue band, the larger summary percent. `setBy` names every
    group that sets the value in effect, sorted by group name. A policy no
    group sets is absent: the reader's default applies. Individual access
    (`groups` null) takes no group's policy; a group id the state no longer
    has is skipped. This is the one place the combination rule lives: the
    console, the sign-in claims and the admission check all read it.
    """
    groups = _by_name(_groups(state, person))
    found: dict[str, Any] = {}
    set_by: dict[str, list[str]] = {}
    for policy in GROUP_POLICIES:
        values = [
            (g["id"], value)
            for g in groups
            if (value := (g.get("policies") or {}).get(policy)) is not None
        ]
        if not values:
            continue
        best = max(_generosity(policy, value) for _, value in values)
        winners = [(gid, value) for gid, value in values if _generosity(policy, value) == best]
        found[policy] = winners[0][1]
        set_by[policy] = [gid for gid, _ in winners]
    if set_by:
        found["setBy"] = set_by
    return found


def groups_in_effect(state: Any, person: dict[str, Any]) -> list[dict[str, str]]:
    """The groups `person` is in, as `{id, name}` sorted by name, for the
    `eugene_groups` claim. Group names are not secret to their members."""
    return [{"id": g["id"], "name": g["name"]} for g in _by_name(_groups(state, person))][
        :MAX_CLAIM_GROUPS
    ]
