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
    chosen = person.get("permissions")
    if chosen is None:
        return [p for p in DEFAULTS if default_on(state, p)]
    return [p for p in DEFAULTS if p in chosen]


def has(state: Any, subject: str, permission: str) -> bool:
    """Whether person `subject` holds `permission`. Eugene's owner and
    anyone not on the People list hold none."""
    person = state.people.get(subject)
    return person is not None and permission in in_effect(state, person)


def refusal(permission: str) -> str:
    """The sentence a person is told when they lack `permission`."""
    if permission == ADD_JOB_SITES:
        return "Eugene's owner has not let you add job sites. They can, on Eugene's People page."
    return (
        "Eugene's owner has not let you use job sites as yourself. They can, on Eugene's "
        "People page. A folder a site's owner shares with you still works."
    )
