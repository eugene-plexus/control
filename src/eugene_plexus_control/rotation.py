"""Signing-key rotation, and why revocation is one.

**Revoking a node is a rotation, not a deletion.** This is the sharp
edge of the whole design and the part that is easiest to get wrong: a
revoked node still *holds* the service-token signing key, so removing
its registry entry does not stop it authenticating to other components.
Revoking therefore mints a new key, redistributes it to every remaining
node, and bumps nothing else — the entry in the log is what makes the
new key the install's key.

Three consequences the implementation has to carry rather than assume
away:

**It has to be idempotent and resumable.** A node that is `down` during
a rotation holds a superseded key and is refused until it reconnects and
is re-keyed. That means a rotation is not a single atomic act, so the
state is *named phases* rather than a percentage, and `nodesPending`
names the hosts rather than counting them — "which host is stale" is the
question an operator actually has.

**A restart must stop being a re-key.** Through M4 tokens were "rotated
on each watchdog restart", harmless with one process and an outage with
N nodes. The key is persisted sealed under the master key and survives a
restart; only this operation changes it.

**Two concurrent rotations have no correct answer**, so the second is
refused with 409. Ending up with two new keys, each distributed to a
different subset of hosts, is a mixed-key state nothing can resolve.

Progress lives here in memory and is **not** replicated: it is one
root's observation of an operation in flight, and the durable fact — the
new key — is in the log. It persists until the next rotation starts so a
UI that reconnects afterwards still learns how the last one ended.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

log = logging.getLogger(__name__)

STATE_MINTING = "minting"
STATE_DISTRIBUTING = "distributing"
STATE_DONE = "done"
STATE_FAILED = "failed"

REASON_OPERATOR = "operator"
REASON_REVOCATION = "revocation"


class RotationInFlight(Exception):
    """A rotation is already running. The caller gets 409."""


@dataclass(frozen=True)
class RotationProgress:
    """One rotation, as an operator needs to read it."""

    state: str
    reason: str
    revoked_node: str | None = None
    nodes_total: int = 0
    nodes_rekeyed: int = 0
    nodes_pending: tuple[str, ...] = ()
    error: str | None = None
    started_at: str = ""
    finished_at: str | None = None
    key_id: str | None = None

    @property
    def in_flight(self) -> bool:
        return self.state in (STATE_MINTING, STATE_DISTRIBUTING)


@dataclass
class RotationTracker:
    """Holds the current or most recent rotation.

    A lock rather than an event-loop assumption, because a rotation is
    advanced from a background task while a route reads it.
    """

    _current: RotationProgress | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def current(self) -> RotationProgress | None:
        with self._lock:
            return self._current

    def begin(self, *, reason: str, revoked_node: str | None, nodes: list[str]) -> RotationProgress:
        with self._lock:
            if self._current is not None and self._current.in_flight:
                raise RotationInFlight(
                    "a key rotation is already in flight; two concurrent rotations would "
                    "leave the install in a mixed-key state with no single correct answer"
                )
            progress = RotationProgress(
                state=STATE_MINTING,
                reason=reason,
                revoked_node=revoked_node,
                nodes_total=len(nodes),
                nodes_pending=tuple(sorted(nodes)),
                started_at=datetime.now(UTC).isoformat(),
            )
            self._current = progress
            return progress

    def minted(self, key_id: str) -> None:
        with self._lock:
            if self._current is None:
                return
            self._current = replace(self._current, state=STATE_DISTRIBUTING, key_id=key_id)

    def rekeyed(self, node: str) -> None:
        """Mark one node as holding the new key.

        Idempotent: re-keying a node twice is a no-op rather than an
        error, because a resumable rotation will legitimately retry a
        host it already reached.
        """
        with self._lock:
            if self._current is None:
                return
            pending = tuple(n for n in self._current.nodes_pending if n != node)
            if len(pending) == len(self._current.nodes_pending):
                return
            self._current = replace(
                self._current,
                nodes_pending=pending,
                nodes_rekeyed=self._current.nodes_rekeyed + 1,
            )

    def finish(self) -> None:
        """Close the rotation out.

        A rotation with nodes still pending finishes as `failed` and
        names them. That is the honest report: those hosts hold a
        superseded key and will be refused until they reconnect, and
        calling it `done` would tell an operator the install is
        consistent when it is not.
        """
        with self._lock:
            if self._current is None:
                return
            pending = self._current.nodes_pending
            self._current = replace(
                self._current,
                state=STATE_FAILED if pending else STATE_DONE,
                error=(
                    f"{len(pending)} node(s) were unreachable and still hold the previous "
                    f"key: {', '.join(pending)}. They are re-keyed on reconnect and refused "
                    "until then. Re-run the rotation once they are back."
                    if pending
                    else None
                ),
                finished_at=datetime.now(UTC).isoformat(),
            )
            if pending:
                log.warning(
                    "key rotation finished with %d node(s) stale: %s",
                    len(pending),
                    ", ".join(pending),
                )
            else:
                log.info("key rotation complete across %d node(s)", self._current.nodes_total)

    def fail(self, error: str) -> None:
        with self._lock:
            if self._current is None:
                return
            self._current = replace(
                self._current,
                state=STATE_FAILED,
                error=error,
                finished_at=datetime.now(UTC).isoformat(),
            )
            log.error("key rotation failed: %s", error)
