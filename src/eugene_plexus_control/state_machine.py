"""The one writer. Every control-state change passes through here.

`Writer.append()` is the choke point the whole design rests on: one
writer, one ordered mutation path, a monotonic index. If mutations ever
start writing files from ten places, replication is over — so this class
is the only thing in the component that holds a `LogStore` for writing,
and every route that changes anything calls it.

Two entry points, and the role gates which one is legal:

  * `append(op, payload)` — **active root only.** Assigns the next
    index, applies to a candidate state, makes the entry durable, then
    swaps the new state in. That order is why a rejected entry never
    reaches the log and a failed write leaves state untouched.

  * `accept_replicated(entry)` — **standby only.** Applies an entry the
    active root already assigned. Refuses a gap, refuses a regressed
    epoch, and never renumbers anything.

A standby calling `append` is the bug this class exists to make
impossible, so it raises rather than returning an error a caller might
ignore.

**Promotion** is the one transition between the two, and it is an
operator act with a passphrase — never a timer, never a config PATCH.
It stamps `epoch + 1` on its own `promote` entry, which is how every
replica and every agent learns the new generation by replaying rather
than by being told out of band.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .applied import (
    ROLE_CONTROL,
    AppliedState,
    ApplyError,
    apply,
    apply_all,
    canonical_bytes,
    from_canonical,
    to_canonical,
)
from .log_store import LogStore

log = logging.getLogger(__name__)

ROLE_ACTIVE = "control"
ROLE_STANDBY = "standby"

# Compact once the log is longer than this. Chosen so an operator's
# ordinary session never triggers one, and a long-lived install does not
# accumulate an unbounded file. Compaction is the reason a snapshot
# exists, and the reason `firstAvailableIndex` is on every log page.
DEFAULT_COMPACT_THRESHOLD = 512


class NotActive(Exception):
    """A write was attempted on a standby.

    Not an HTTP concern and not recoverable by retrying: the caller
    reached the wrong host. Routes turn it into a 409 that names the
    active root, because the useful response is "go there", not "try
    again"."""


class AlreadyActive(Exception):
    """Promotion was attempted on the root that is already active."""


@dataclass(frozen=True)
class ReplicationPosition:
    """How far a replica has got. What a standby reports about itself and
    what `GET /v1/control/status` reports about its standbys."""

    applied_index: int
    epoch: int


class StateMachine:
    """Applied state, the log, and the rule that only one thing writes.

    Holds a single lock. Reads take it too, briefly, so a caller can
    never observe state between "durable" and "applied" — a window that
    would let `GET /v1/nodes` return a node the log does not yet
    describe.
    """

    def __init__(
        self,
        store: LogStore,
        *,
        role: str = ROLE_ACTIVE,
        compact_threshold: int = DEFAULT_COMPACT_THRESHOLD,
    ) -> None:
        self._store = store
        self._lock = threading.RLock()
        self._state = AppliedState()
        self._role = role
        self._compact_threshold = compact_threshold
        self._entries_since_snapshot = 0
        self._dropped_tail_lines = 0

    # ----- lifecycle --------------------------------------------------

    def load(self) -> None:
        """Rebuild state from the snapshot plus every entry after it.

        This is the same code path a standby uses to bootstrap, run
        against local files instead of HTTP. Deliberately so: if restart
        recovery and standby bootstrap were two implementations, only
        one of them would be exercised regularly.
        """
        with self._lock:
            result = self._store.load()
            self._dropped_tail_lines = result.dropped_tail_lines
            base = (
                from_canonical(result.snapshot) if result.snapshot is not None else AppliedState()
            )
            self._state = apply_all(base, result.entries)
            self._entries_since_snapshot = len(result.entries)
            log.info(
                "loaded control state: index %d, epoch %d, %d node(s), role %s",
                self._state.index,
                self._state.epoch,
                len(self._state.nodes),
                self._role,
            )

    # ----- reads ------------------------------------------------------

    @property
    def state(self) -> AppliedState:
        with self._lock:
            return self._state

    @property
    def role(self) -> str:
        with self._lock:
            return self._role

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._role == ROLE_ACTIVE

    @property
    def dropped_tail_lines(self) -> int:
        """Non-zero after an unclean kill. Surfaced on `/healthz` so an
        operator learns about it from the dashboard rather than from the
        logs of a process that has since restarted."""
        with self._lock:
            return self._dropped_tail_lines

    def position(self) -> ReplicationPosition:
        with self._lock:
            return ReplicationPosition(applied_index=self._state.index, epoch=self._state.epoch)

    def first_available_index(self) -> int:
        with self._lock:
            return self._store.first_available_index(self._state.index)

    def read_page(self, after: int, limit: int) -> list[dict[str, Any]]:
        """Entries after `after`, for a standby that is pulling.

        Standbys pull and the active root never pushes. That keeps
        exactly one writer, lets a standby measure its own lag honestly,
        and puts the connection in the direction that survives a
        firewall between buildings.
        """
        with self._lock:
            return self._store.read_after(after, limit)

    def snapshot_document(self) -> dict[str, Any]:
        """Applied state as canonical data, for `GET /v1/control/snapshot`.

        The same bytes the replay-equivalence test compares, which is
        the point: a standby bootstraps from exactly what is compared,
        so a divergence cannot hide in the difference between the
        comparison form and the wire form.
        """
        with self._lock:
            return to_canonical(self._state)

    def canonical(self) -> bytes:
        with self._lock:
            return canonical_bytes(self._state)

    # ----- the write path ---------------------------------------------

    def append(
        self,
        op: str,
        payload: dict[str, Any],
        *,
        epoch_override: int | None = None,
    ) -> dict[str, Any]:
        """Assign the next index, apply, persist, commit. Active only.

        The ordering is the interesting part:

        1. Apply to a **candidate** state. An entry that cannot be
           applied never enters the log, so a replica never has to cope
           with an entry the writer itself could not use.
        2. Append durably. If this raises, committed state is unchanged
           and the caller gets an error rather than a lie.
        3. Swap the candidate in.

        A crash between 2 and 3 is harmless: the entry is durable and
        replay applies it on the next boot, which is the same state by
        definition because apply is deterministic.
        """
        with self._lock:
            if self._role != ROLE_ACTIVE:
                raise NotActive(
                    f"this control root is a {self._role}; writes belong to the active root"
                )

            index = self._state.index + 1
            epoch = epoch_override if epoch_override is not None else self._state.epoch
            entry: dict[str, Any] = {
                "index": index,
                "epoch": epoch,
                "op": op,
                "payload": payload,
                # Informational only, never used for ordering. Stamped
                # here — once, by the writer — so replay uses this value
                # rather than the replica's clock.
                "at": datetime.now(UTC).isoformat(),
            }

            candidate = apply(self._state, entry)
            self._store.append(entry)
            self._state = candidate
            self._entries_since_snapshot += 1

            self._maybe_compact_locked()
            return entry

    def accept_replicated(self, entry: dict[str, Any]) -> None:
        """Apply an entry the active root assigned. Standby only.

        Persisted before it is committed, in the same order as `append`,
        so a standby that is killed mid-catch-up restarts from a state
        that matches what it had acknowledged.
        """
        with self._lock:
            if self._role == ROLE_ACTIVE:
                raise NotActive(
                    "the active root does not accept replicated entries; it assigns them"
                )
            candidate = apply(self._state, entry)
            self._store.append(entry)
            self._state = candidate
            self._entries_since_snapshot += 1
            self._maybe_compact_locked()

    def install_snapshot(self, document: dict[str, Any]) -> None:
        """Adopt a snapshot wholesale. How a standby starts, and how one
        that fell behind compaction recovers.

        Refuses to move backwards. A snapshot older than what this
        replica has already applied would silently discard entries it
        has acknowledged, and "the operator pointed a standby at a stale
        root" should be an error rather than a quiet rewind.
        """
        with self._lock:
            incoming = from_canonical(document)
            if incoming.index < self._state.index:
                raise ApplyError(
                    f"refusing a snapshot at index {incoming.index}: this replica has "
                    f"already applied {self._state.index}"
                )
            self._store.write_snapshot(document, compact_through=incoming.index)
            self._state = incoming
            self._entries_since_snapshot = 0
            log.info("installed snapshot at index %d, epoch %d", incoming.index, incoming.epoch)

    def snapshot_now(self) -> None:
        """Write a snapshot of current state and compact the log."""
        with self._lock:
            document = to_canonical(self._state)
            self._store.write_snapshot(document, compact_through=self._state.index)
            self._entries_since_snapshot = 0

    def _maybe_compact_locked(self) -> None:
        if self._entries_since_snapshot < self._compact_threshold:
            return
        document = to_canonical(self._state)
        self._store.write_snapshot(document, compact_through=self._state.index)
        self._entries_since_snapshot = 0
        log.info("snapshot written at index %d; log compacted", self._state.index)

    # ----- identity, established once ---------------------------------

    def seed_identity(self, identity_fields: dict[str, Any]) -> None:
        """Write the install identity and snapshot it. Called by
        `POST /v1/auth/initialize`, exactly once.

        **Not a log entry, and that is deliberate.** Initialization
        happens before any standby can exist, so the snapshot always
        carries it forward and it never needs to replicate as an entry.
        `LogOp` therefore stays closed at nine — see §10 of the M5
        design doc, which records this alongside the reason a join token
        is not a mutation either.
        """
        with self._lock:
            if self._role != ROLE_ACTIVE:
                raise NotActive("only the active root can initialize an install")
            if self._state.identity.salt is not None:
                raise AlreadyActive("this install is already initialized")
            from dataclasses import replace as dc_replace

            from .applied import InstallIdentity

            identity = InstallIdentity(**identity_fields)
            self._state = dc_replace(self._state, identity=identity)
            self.snapshot_now()

    # ----- promotion --------------------------------------------------

    def promote(self, *, node: str | None) -> dict[str, Any]:
        """Become the active root at a higher epoch.

        Flips the role first so `append` is legal, then writes a
        `promote` entry stamped `epoch + 1`. If the append fails the
        role is put back, because a root that believes it is active
        without having recorded it is exactly the split-brain this whole
        mechanism exists to prevent.

        Agents learn the new epoch on their next contact and fence the
        old root by refusing its lower one — no election, no quorum, and
        no agent needing to agree with any other agent.
        """
        with self._lock:
            if self._role == ROLE_ACTIVE:
                raise AlreadyActive("this control root is already active")
            new_epoch = self._state.epoch + 1
            previous_role = self._role
            self._role = ROLE_ACTIVE
            try:
                entry = self.append(
                    "promote", {"node": node} if node else {}, epoch_override=new_epoch
                )
            except BaseException:
                self._role = previous_role
                raise
            # Snapshot immediately. A promotion is the moment an
            # operator is most likely to lose the host again, and a
            # standby of the *new* root should be able to bootstrap
            # without replaying the old root's whole history.
            self.snapshot_now()
            log.warning(
                "promoted to active control root at epoch %d (was %s)", new_epoch, previous_role
            )
            return entry

    def force_role(self, role: str) -> None:
        """Set the role without promoting. Startup and tests only.

        Exists because `role` is configuration — nothing here votes —
        and a standby has to come up as one before it has replayed
        anything. Not reachable over HTTP: a role that a config PATCH
        could change would be a second writer reachable from the
        network.
        """
        with self._lock:
            if role not in (ROLE_ACTIVE, ROLE_STANDBY):
                raise ValueError(f"unknown control role {role!r}")
            self._role = role


def active_node_name(state: AppliedState) -> str | None:
    """Which node the registry says is the active root, if any."""
    for record in state.nodes.values():
        if record.role == ROLE_CONTROL:
            return record.name
    return None
