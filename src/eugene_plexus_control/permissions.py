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
#: The policies combined across all of a person's groups, most generous wins (row 70).
GENEROUS_POLICIES: tuple[str, ...] = ("queueBand", "summaryContextPercent")
#: Row 37: routing (from the active group alone) and the model list (every group narrows).
ROW_37_POLICIES: tuple[str, ...] = ("roleRouting", "allowedModels")
#: The policies a group may set (`GroupPolicies`).
GROUP_POLICIES: tuple[str, ...] = GENEROUS_POLICIES + ROW_37_POLICIES
#: Row 37: at most this many models in a role's list, and in a group's model list.
MAX_ROLE_MODELS = 16
MAX_ALLOWED_MODELS = 64
MAX_MODEL_ID = 256
#: Row 37 review: at most this many `*` in one model-list entry.
MAX_STARS = 8


def standby_behind(app: Any) -> str | None:
    """Row 37: the standby's node when it applies a log version below row
    37's, so a group record with `roleRouting` or `allowedModels`, a person
    with `activeGroup`, or a `setPersonActiveGroup` entry would stop its
    copy; None otherwise (row 5's rule, as row 38's alerts)."""
    from .applied import ROW_37_LOG_VERSION
    from .routes.control import standby_short_of

    return standby_short_of(app, ROW_37_LOG_VERSION)


def standby_refusal(node: str) -> str:
    return (
        f"The standby on {node} is older than this root and could not copy a group's models "
        "or routing, or a person's chosen group. Update it, then save again."
    )


#: At most this many groups in the `eugene_groups` claim (`Person.groups`'s own bound).
MAX_CLAIM_GROUPS = 64


def _by_name(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(groups, key=lambda g: (g["name"].casefold(), g["id"]))


def _generosity(policy: str, value: Any) -> int:
    return QUEUE_BANDS.index(value) if policy == "queueBand" else int(value)


def effective_policies(state: Any, person: dict[str, Any]) -> dict[str, Any]:
    """The group policies that apply to `person` now, as `PoliciesInEffect`.

    The most generous value of each row 70 policy across the groups they are
    in: the highest queue band, the larger summary percent. Row 37's
    `roleRouting` comes from their active group alone (`active_group`),
    which `activeGroup` names; `modelLimits` holds the model list of every
    group of theirs that sets one (call R37-SELF, Choice 3: a model must be
    in all of them, whichever group they work from). `setBy` names every
    group that sets the value in effect, sorted by group name. A policy no
    group sets is absent: the reader's default applies. Individual access
    (`groups` null) takes no group's policy; a group id the state no longer
    has is skipped. This is the one place the combination rule lives: the
    console, the sign-in claims and the admission check all read it.
    """
    groups = _by_name(_groups(state, person))
    found: dict[str, Any] = {}
    set_by: dict[str, list[str]] = {}
    for policy in GENEROUS_POLICIES:
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
    # Row 37 (R37-FIRST): routing is one group's, never merged. `activeGroup`
    # is said only when it routes a role, so a person whose groups set
    # nothing of row 37 reads exactly what a root before row 37 said.
    active = active_group(state, person)
    routing = (active.get("policies") or {}).get("roleRouting") if active else None
    if active is not None and routing is not None:
        found["activeGroup"] = active["id"]
        found["roleRouting"] = routing
        set_by["roleRouting"] = [active["id"]]
    limits = model_limits(state, person)
    if limits:
        found["modelLimits"] = [
            {"group": {"id": g["id"], "name": g["name"]}, "models": g["models"]} for g in limits
        ]
    if set_by:
        found["setBy"] = set_by
    return found


def active_group(state: Any, person: dict[str, Any]) -> dict[str, Any] | None:
    """The group whose `roleRouting` applies to `person` (row 37): the one
    they chose, while they are still in it, else the first
    of their groups (in their list's order) the state still has. None for
    individual access or no group."""
    groups = _groups(state, person)
    chosen = person.get("activeGroup")
    for group in groups:
        if group["id"] == chosen:
            return group
    return groups[0] if groups else None


def claim_policies(state: Any, person: dict[str, Any]) -> dict[str, Any]:
    """`eugene_policies` for the sign-in claims: `effective_policies` with
    only the row 70 policies. A Workbench from before row 37 refuses the
    claim when it holds a field it does not know, and an app reads its
    routing from the gateway, not from its claims."""
    found = effective_policies(state, person)
    kept: dict[str, Any] = {p: found[p] for p in GENEROUS_POLICIES if p in found}
    set_by = {p: ids for p, ids in (found.get("setBy") or {}).items() if p in GENEROUS_POLICIES}
    if set_by:
        kept["setBy"] = set_by
    return kept


def model_limits(state: Any, person: dict[str, Any]) -> list[dict[str, Any]]:
    """Row 37 (call R37-SELF, Choice 3): `{id, name, models}` for every group
    of `person`'s that sets a model list, sorted by name. A model is theirs
    only when every list allows it, so choosing another active group never
    widens it; a group with no list adds no limit."""
    return [
        {"id": g["id"], "name": g["name"], "models": list(models)}
        for g in _by_name(_groups(state, person))
        if (models := (g.get("policies") or {}).get("allowedModels")) is not None
    ]


def model_policies(state: Any, person: dict[str, Any] | None) -> dict[str, Any]:
    """Row 37: what the admission answer carries for a verified person:
    `roleRouting` and the `routingGroup` it comes from (their active group),
    and `modelLimits` (every group's list), each only when set. Read from the
    root's state at each check."""
    if person is None:
        return {}
    found: dict[str, Any] = {}
    group = active_group(state, person)
    routing = (group.get("policies") or {}).get("roleRouting") if group else None
    if group is not None and routing is not None:
        found["roleRouting"] = routing
        found["routingGroup"] = {"id": group["id"], "name": group["name"]}
    limits = model_limits(state, person)
    if limits:
        found["modelLimits"] = [
            {"group": {"id": g["id"], "name": g["name"]}, "models": g["models"]} for g in limits
        ]
    return found


def groups_in_effect(state: Any, person: dict[str, Any]) -> list[dict[str, str]]:
    """The groups `person` is in, as `{id, name}` sorted by name, for the
    `eugene_groups` claim. Group names are not secret to their members."""
    return [{"id": g["id"], "name": g["name"]} for g in _by_name(_groups(state, person))][
        :MAX_CLAIM_GROUPS
    ]
