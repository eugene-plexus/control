"""The standby side: pull the log, apply it, stay promotable.

**Standbys pull; the active root never pushes.** Three reasons, and all
three matter:

  * It keeps exactly one writer. A pushing root would be reaching into a
    replica's state, and the moment two processes can write a replica's
    log the single-writer property is a convention rather than a fact.
  * A standby measures its own lag honestly. Being told your lag by the
    host you might have to replace is the wrong direction for that
    information.
  * The standby initiates the connection, which is the direction that
    survives a firewall between buildings.

Catching up has two paths and the choice between them is not a
heuristic. A standby whose position is still in the active root's log
follows the log. One whose position predates `firstAvailableIndex` —
because the root compacted past it — **must** take a snapshot; the
entries it wants no longer exist, and reading around the gap would
produce a state that never existed anywhere.

A 409 from `GET /v1/control/log` is exactly that signal, and the
follower's response is to re-snapshot rather than to retry, because
retrying a request for entries that were deleted is a loop.

This runs as a background task with no timer-driven decisions in it.
Nothing here promotes anything: falling behind, losing contact and the
active root vanishing all produce the same behaviour — keep trying, keep
reporting. **Promotion is an operator act**, and a follower that could
promote itself would be the automatic failover the design refuses,
because automatic promotion without quorum is the definition of
split-brain.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ._http import internal_client
from .applied import ApplyError
from .state_machine import StateMachine

log = logging.getLogger(__name__)

# One page per pull. Large enough that a standby that was offline for a
# while catches up in a few round trips, small enough that a single
# response is not a surprise on a slow link between buildings.
PAGE_LIMIT = 500


@dataclass
class FollowerStatus:
    """What the standby knows about its own position and its last
    contact. An observation, so it never enters applied state."""

    active_url: str | None = None
    reachable: bool = False
    last_contact_at: str | None = None
    active_applied_index: int | None = None
    local_index: int = 0
    last_error: str | None = None

    @property
    def lag_entries(self) -> int | None:
        """How far behind, in entries rather than seconds.

        Entries are what would be lost on a promotion; seconds are not.
        An operator deciding whether to promote needs the first number,
        and `GET /v1/control/status` is where they read it — before a
        failure rather than during one.
        """
        if self.active_applied_index is None:
            return None
        return max(0, self.active_applied_index - self.local_index)


class Follower:
    """Pulls from the active root until told to stop.

    Owns no state of its own beyond its status: the machine applies,
    the machine persists, and this only decides what to ask for next.
    """

    def __init__(
        self,
        machine: StateMachine,
        *,
        active_url: str | None,
        interval_seconds: float,
        token_provider: object | None = None,
    ) -> None:
        self._machine = machine
        self._interval = interval_seconds
        # See `_http`: shared SSL context, and no proxy between a
        # standby and the active root.
        self._client = internal_client()
        self._token_provider = token_provider
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self.status = FollowerStatus(active_url=active_url)

    # ----- lifecycle --------------------------------------------------

    def start(self) -> None:
        if self.status.active_url is None:
            log.warning(
                "this control root is a standby but EUGENE_PLEXUS_CONTROL_ACTIVE_URL is "
                "unset, so it has nothing to replicate from. It will stay at index %d and "
                "must not be promoted.",
                self._machine.state.index,
            )
            return
        self._task = asyncio.create_task(self._run(), name="control-follower")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
        await self._client.aclose()

    # ----- the loop ---------------------------------------------------

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Never let the loop die. A standby that stopped
                # replicating but kept answering /healthz would be the
                # worst possible failure here: it looks like insurance
                # and is not.
                self.status.reachable = False
                self.status.last_error = f"{type(exc).__name__}: {exc}"
                log.warning("replication tick failed: %s", self.status.last_error)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
            except TimeoutError:
                continue

    async def tick(self) -> None:
        """One pull. Separated from the loop so tests drive it directly.

        Applies whatever arrives, one entry at a time, through the
        machine — which persists each before committing it, so a standby
        killed mid-catch-up restarts from a state that matches what it
        had durably accepted.
        """
        url = self.status.active_url
        if url is None:
            return

        # A standby that has never replicated starts with a snapshot,
        # not with the log. This is not an optimization — the install
        # identity (salt, verifier, the sealed keys) is established
        # *before* index 1 and is therefore in no log entry at all, so a
        # standby that only tailed the log would catch up to the right
        # index holding none of the material a promotion needs. It would
        # look perfectly healthy and be unpromotable, which is the exact
        # failure this component exists to prevent.
        if self._machine.state.identity.salt is None:
            log.info("standby has no install identity yet; bootstrapping from a snapshot")
            await self._install_snapshot(url)

        after = self._machine.state.index
        try:
            page = await self._get_page(url, after)
        except _Compacted:
            log.info(
                "standby at index %d fell behind the active root's compaction; "
                "re-snapshotting rather than reading around the gap",
                after,
            )
            await self._install_snapshot(url)
            return

        self.status.reachable = True
        self.status.last_contact_at = datetime.now(UTC).isoformat()
        self.status.last_error = None
        reported = page.get("appliedIndex")
        if isinstance(reported, int):
            self.status.active_applied_index = reported

        entries = page.get("entries") or []
        applied = 0
        for entry in entries:
            try:
                self._machine.accept_replicated(entry)
            except ApplyError as exc:
                # An entry that will not apply is fatal for this
                # replica. Re-snapshot; never skip. Skipping is how two
                # roots end up disagreeing while both report healthy.
                log.error("rejecting replicated entry: %s", exc)
                self.status.last_error = str(exc)
                await self._install_snapshot(url)
                return
            applied += 1

        self.status.local_index = self._machine.state.index
        if applied:
            log.debug(
                "applied %d replicated entr(ies), now at %d",
                applied,
                self._machine.state.index,
            )

    async def _get_page(self, url: str, after: int) -> dict[str, Any]:
        response = await self._client.get(
            f"{url.rstrip('/')}/v1/control/log",
            params={"after": after, "limit": PAGE_LIMIT},
            headers=self._headers(),
            timeout=max(5.0, self._interval * 2),
        )
        if response.status_code == 409:
            raise _Compacted()
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("malformed log page")
        return body

    async def _install_snapshot(self, url: str) -> None:
        response = await self._client.get(
            f"{url.rstrip('/')}/v1/control/snapshot",
            headers=self._headers(),
            timeout=max(10.0, self._interval * 4),
        )
        response.raise_for_status()
        document = response.json()
        if not isinstance(document, dict):
            raise ValueError("malformed snapshot")
        self._machine.install_snapshot(document)
        self.status.local_index = self._machine.state.index
        self.status.reachable = True
        self.status.last_contact_at = datetime.now(UTC).isoformat()

    def _headers(self) -> dict[str, str]:
        provider = self._token_provider
        token = provider() if callable(provider) else None
        return {"Authorization": f"Bearer {token}"} if token else {}


class _Compacted(Exception):
    """The entries this standby wants have been compacted away."""
