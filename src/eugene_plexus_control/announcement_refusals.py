"""Why a machine's address announcement was refused (row 36).

An observation, as `lastError` and `machineKeyChange` are: kept in memory on
the active root (`app.state.announcement_refusals`), never in the replicated
log and never sent to a standby. A refusal must not grow the log. A standby,
or a root that restarted, learns it at the machine's next attempt (the agent
asks again every 15 minutes).

Two things are kept per machine, and they differ on purpose:

* **The last refusal of an announcement that verified against the machine's
  signing key**, in the root's words. Only the machine can have sent it, so
  its address may be shown to the owner.
* **A count and a time for the announcements that did not verify.** Anyone
  who can reach this root can send one. It must not put words on the owner's
  screen, so nothing it carried is kept.

Both are bounded by the number of enrolled machines: an announcement for a
name that is not enrolled is a 404 and keeps nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

#: The reasons, as `NodeAnnouncementRefusalReason` in `control.yaml`.
OPEN_INTERNET = "open_internet"
ANOTHERS_ADDRESS = "anothers_address"
NOT_A_NODE = "not_a_node"
MACHINE_PORT = "machine_port"
STALE_SEQUENCE = "stale_sequence"
NO_SIGNING_KEY = "no_signing_key"


@dataclass(frozen=True)
class Refusal:
    at: datetime
    reason: str
    title: str
    detail: str
    announced_url: str | None


@dataclass(frozen=True)
class Unverified:
    count: int
    last_at: datetime


class AnnouncementRefusals:
    """What this root last refused of each machine's announcements."""

    def __init__(self) -> None:
        self._refused: dict[str, Refusal] = {}
        self._unverified: dict[str, Unverified] = {}

    def refuse(
        self, name: str, *, reason: str, title: str, detail: str, announced_url: str | None
    ) -> None:
        """Keep the refusal of an announcement the machine signed."""
        self._refused[name] = Refusal(
            at=datetime.now(UTC),
            reason=reason,
            title=title,
            detail=detail,
            # Nothing on record verifies what a `no_signing_key` request
            # carried, so none of it is kept.
            announced_url=None if reason == NO_SIGNING_KEY else announced_url,
        )

    def count_unverified(self, name: str) -> None:
        """Count an announcement that was malformed or not signed by the machine."""
        before = self._unverified.get(name)
        self._unverified[name] = Unverified(
            count=(before.count if before else 0) + 1, last_at=datetime.now(UTC)
        )

    def clear(self, name: str) -> None:
        """The machine's address was accepted or confirmed, it was removed, or
        it enrolled again: what was refused no longer applies."""
        self._refused.pop(name, None)
        self._unverified.pop(name, None)

    def refusal(self, name: str) -> Refusal | None:
        return self._refused.get(name)

    def unverified(self, name: str) -> Unverified | None:
        return self._unverified.get(name)

    def is_empty(self) -> bool:
        return not self._refused and not self._unverified
