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
    OP_CHANGE_PASSPHRASE,
    OP_PUT_OIDC_KEY,
    OP_ROTATE_SIGNING_KEY,
    OP_SET_INSTALL_CA,
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

#: The `Snapshot` fields every supported root writes (row 4 re-review): a
#: snapshot without one would clear what it holds. A field added later is
#: optional, so a root older than the standby is still followed, and its
#: absence reads as empty.
_REQUIRED_SNAPSHOT_KEYS = frozenset(
    {
        "index",
        "epoch",
        "nodes",
        "components",
        "runtimes",
        "config",
        "salt",
        "passphraseVerifier",
        "sealedSigningKey",
        "controlPublicKey",
        "sealedControlKey",
        "rootTokenPublicKey",
        "revokedSessions",
    }
)
#: Fields that are lists, and fields that are objects, wherever they appear.
_LIST_KEYS = frozenset(
    {
        "nodes",
        "components",
        "runtimes",
        "clientKeys",
        "accessAudit",
        "revokedSessions",
        "people",
        "permissionGroups",
        "sites",
        "oidcClients",
        "oidcKeys",
        "revokedSignIns",
    }
)
_OBJECT_KEYS = frozenset({"config", "clientAdmission", "browserSignIn"})
_TEXT_OR_NONE_KEYS = frozenset(
    {
        "salt",
        "passphraseVerifier",
        "sealedSigningKey",
        "rootTokenPublicKey",
        "sealedControlKey",
        "controlPublicKey",
        "sealedRecoveryKey",
        "signingKeyId",
        "installCaCertificate",
        "sealedInstallCaKey",
    }
)


def _unusable_snapshot(document: Any) -> str | None:
    """Why `document` cannot replace a replica's state, or None (row 4
    re-review). Presence alone is not enough: the parser reads `nodes: null`
    as no nodes, so a snapshot saying so would clear the registry."""
    if not isinstance(document, dict):
        return "is not an object"
    missing = sorted(_REQUIRED_SNAPSHOT_KEYS - set(document))
    if missing:
        return f"is incomplete: it has no {', '.join(missing)}"
    for name in ("index", "epoch"):
        value = document[name]
        if type(value) is not int or value < (1 if name == "epoch" else 0):
            return f"has an unusable {name} ({value!r})"
    for name in sorted(_LIST_KEYS & set(document)):
        if not isinstance(document[name], list):
            return f"has {name} that is not a list ({type(document[name]).__name__})"
    for name in sorted(_OBJECT_KEYS & set(document)):
        if not isinstance(document[name], dict):
            return f"has {name} that is not an object ({type(document[name]).__name__})"
    for name in sorted(_TEXT_OR_NONE_KEYS & set(document)):
        if document[name] is not None and not isinstance(document[name], str):
            return f"has {name} that is not text"
    for node in document["nodes"]:
        if not isinstance(node, dict) or not isinstance(node.get("name"), str) or not node["name"]:
            return "lists a node with no name"
    return None


# Compact once the log is longer than this. Chosen so an operator's
# ordinary session never triggers one, and a long-lived install does not
# accumulate an unbounded file. Compaction is the reason a snapshot
# exists, and the reason `firstAvailableIndex` is on every log page.
DEFAULT_COMPACT_THRESHOLD = 512

#: Ops after which every root writes a snapshot and compacts at once (row 5):
#: the entries before a passphrase change hold values sealed under the key
#: it replaced, and compaction is what takes them off the disk.
COMPACT_AFTER = frozenset({OP_CHANGE_PASSPHRASE})

#: Ops whose payload carries a value sealed under the master key (row 5).
#: The writer appends one only while the install's salt is the one the value
#: was sealed under (`append(sealed_under=)`), so a value sealed just before
#: a passphrase change is refused, never stored under a key nobody derives.
SEALED_OPS = frozenset(
    {
        OP_CHANGE_PASSPHRASE,
        OP_PUT_OIDC_KEY,
        OP_ROTATE_SIGNING_KEY,
        OP_SET_INSTALL_CA,
        "putExternalProvider",
    }
)

#: What a refused sealed write tells the person who made it.
SAVED_AGAIN = "The passphrase changed while this was saved; save it again."

_UNSET: Any = object()


class SealedUnderOldKey(Exception):
    """A sealed write named a salt that is no longer the install's: the
    passphrase changed between sealing and appending. Nothing was stored."""

    def __init__(self) -> None:
        super().__init__(SAVED_AGAIN)


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

    A single lock serializes durable mutations. Ordinary reads observe the
    immutable last committed snapshot without waiting for fsync; operations
    that need a multi-field snapshot still acquire the lock. Publication follows
    the durable append, so readers cannot see uncommitted state.
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
        #: Row 5: the index of a passphrase change whose old copies are not
        #: compacted away yet (`compact_pending`). None when there is none.
        self._pending_change: int | None = None
        #: `applied_digest`'s last answer, for the state object it hashed.
        self._digest: tuple[AppliedState, str] | None = None

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
            # **Entries the snapshot already holds are not replayed.** A kill
            # between a snapshot's rename and the compaction after it leaves a
            # snapshot at N beside a log that still starts at 1: both files are
            # good, and `LogStore.write_snapshot` promises that store "will replay
            # harmlessly". Replaying entry 1 onto state N is the out-of-order
            # error, and a root that cannot start is not harmless. They are
            # dropped from the file by the next compaction.
            pending = [e for e in result.entries if int(e["index"]) > base.index]
            self._state = apply_all(base, pending)
            self._entries_since_snapshot = len(pending)
            # Row 5: a passphrase change a crash left uncompacted: the log
            # still holds entries before it, or the snapshot predates it.
            changes = [int(e["index"]) for e in result.entries if e.get("op") in COMPACT_AFTER]
            if changes and (
                base.index < changes[-1] or int(result.entries[0]["index"]) < changes[-1]
            ):
                self._pending_change = changes[-1]
            log.info(
                "loaded control state: index %d, epoch %d, %d node(s), role %s",
                self._state.index,
                self._state.epoch,
                len(self._state.nodes),
                self._role,
            )
        if self._pending_change is not None:
            cause = self.compact_pending()
            if cause is None:
                log.warning(
                    "finished a passphrase change a restart interrupted: the copies sealed "
                    "under the old key are removed"
                )

    # ----- reads ------------------------------------------------------

    @property
    def state(self) -> AppliedState:
        # apply() builds an immutable candidate and append publishes it only
        # after fsync. A reader can safely keep the previous committed snapshot
        # while the writer waits for disk, without blocking the HTTP event loop.
        return self._state

    @property
    def role(self) -> str:
        return self._role

    @property
    def is_active(self) -> bool:
        return self._role == ROLE_ACTIVE

    @property
    def dropped_tail_lines(self) -> int:
        """Non-zero after an unclean kill. Surfaced on `/healthz` so an
        operator learns about it from the dashboard rather than from the
        logs of a process that has since restarted."""
        return self._dropped_tail_lines

    def position(self) -> ReplicationPosition:
        state = self._state
        return ReplicationPosition(applied_index=state.index, epoch=state.epoch)

    def admit(
        self,
        *,
        clock: Any,
        key_id: str,
        action: str,
        request_id: str,
        model: str | None,
        person_token: str | None = None,
        app_metadata_only: bool = False,
        purpose: str = "inference",
    ) -> dict[str, Any]:
        """Read, decide and commit under one lock, including policy edits."""
        from .applied import OP_PUT_CLIENT_ADMISSION, OP_TOUCH_CLIENT_KEY
        from .client_admission import decide, lineage_of
        from .person_access import check, queue_band

        with self._lock:
            if self._role != ROLE_ACTIVE:
                raise NotActive("Contact the active control root for admission.")
            ledger = self._state.client_admission
            # Under the same lock as policy writes and reservation commits. Release
            # remains possible after logout or revocation, without reauthorizing work.
            if app_metadata_only and (action != "check" or model or person_token):
                from .client_admission import AdmissionRefusal

                raise AdmissionRefusal(403, "App metadata access cannot authorize inference.")
            identity = (
                check(self._state, key_id, person_token)
                if action != "release" and not app_metadata_only
                else {}
            )
            result, candidate = decide(
                ledger,
                self._state.client_keys.get(key_id),
                key_id=key_id,
                action=action,
                request_id=request_id,
                model=model,
                now=clock.now(ledger["clock"]),
                principal=identity,
                purpose=purpose,
            )
            key = self._state.client_keys.get(key_id)
            used = _minute_after(key, action)
            if candidate is not None:
                bucket_id = lineage_of(key, key_id)
                payload = {
                    "clock": candidate["clock"],
                    "keyId": bucket_id,
                    "bucket": candidate["buckets"].get(bucket_id, {}),
                }
                if used is not None:
                    payload |= {"usedId": key_id, "usedAt": used}
                self.append(OP_PUT_CLIENT_ADMISSION, payload)
            elif used is not None:
                # A check writes at most one entry a minute for each key.
                self.append(OP_TOUCH_CLIENT_KEY, {"id": key_id, "usedAt": used})
            return {**result, **identity, **queue_band(self._state, identity)}

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

    def applied_digest(self) -> tuple[int, str]:
        """`(index, SHA-256 hex of canonical_bytes)` of the state now
        (`LogPage.appliedDigest`, row 4).

        Kept for the state object it was computed from, not for the index:
        a snapshot installed at the index already held replaces the state
        and keeps the index, and that is exactly when a standby's digest
        changes."""
        import hashlib

        with self._lock:
            state = self._state
            cached = self._digest
            if cached is not None and cached[0] is state:
                return state.index, cached[1]
            digest = hashlib.sha256(canonical_bytes(state)).hexdigest()
            self._digest = (state, digest)
            return state.index, digest

    # ----- the write path ---------------------------------------------

    def append(
        self,
        op: str,
        payload: dict[str, Any],
        *,
        epoch_override: int | None = None,
        actor: str | None = None,
        sealed_under: Any = _UNSET,
    ) -> dict[str, Any]:
        """Assign the next index, apply, persist, commit. Active only.

        **A sealed write names its key generation** (row 5): an op in
        `SEALED_OPS` must say which salt its sealed values were sealed
        under, and is refused (`SealedUnderOldKey`, nothing stored) unless
        that is the install's salt now. Checked here, under the writer's
        lock, so no passphrase change can land between the check and the
        write.

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
            if op in SEALED_OPS:
                if sealed_under is _UNSET:
                    raise TypeError(f"{op} carries a sealed value: name the salt it sealed under")
                if sealed_under != self._state.identity.salt:
                    raise SealedUnderOldKey()

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
                **({"actor": actor} if actor is not None else {}),
            }

            candidate = apply(self._state, entry)
            self._store.append(entry)
            self._state = candidate
            self._entries_since_snapshot += 1

            self._maybe_compact_locked(op)
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
            self._maybe_compact_locked(entry.get("op"))
        if self._pending_change is not None:
            # A standby that took a passphrase change compacts at once; a
            # failure is retried at every pull and at the next start.
            self.compact_pending()

    def install_snapshot(self, document: dict[str, Any]) -> None:
        """Adopt a snapshot wholesale. How a standby starts, and how one
        that fell behind compaction recovers.

        Refuses to move backwards. A snapshot older than what this
        replica has already applied would silently discard entries it
        has acknowledged, and "the operator pointed a standby at a stale
        root" should be an error rather than a quiet rewind.
        """
        with self._lock:
            # Row 4: a standby now replaces its copy by itself, so what it
            # takes is checked first. A snapshot that leaves out a field
            # would clear it (a document of `{"index": 100}` once cleared
            # the install's identity), and one from another install, or from
            # a root at a lower epoch, is not this install's state.
            why = _unusable_snapshot(document)
            if why is not None:
                raise ApplyError(f"refusing a snapshot that {why}")
            incoming = from_canonical(document)
            if incoming.index < self._state.index:
                raise ApplyError(
                    f"refusing a snapshot at index {incoming.index}: this replica has "
                    f"already applied {self._state.index}"
                )
            if incoming.epoch < self._state.epoch:
                raise ApplyError(
                    f"refusing a snapshot at epoch {incoming.epoch}: this replica holds "
                    f"epoch {self._state.epoch}"
                )
            held, offered = self._state.identity, incoming.identity
            # The identity key names the install. The salt names it too, but
            # only until the passphrase changes (row 5, `changePassphrase`
            # makes a new one and keeps the identity key), so it is the test
            # only for a replica that holds no identity key.
            names = ("controlPublicKey",) if held.controlPublicKey is not None else ("salt",)
            for name in names:
                mine, theirs = getattr(held, name), getattr(offered, name)
                if mine is not None and theirs != mine:
                    raise ApplyError(
                        f"refusing a snapshot of another install: its {name} is not this replica's"
                    )
            self._store.write_snapshot(document, compact_through=incoming.index)
            self._state = incoming
            self._entries_since_snapshot = 0
            self._pending_change = None
            log.info("installed snapshot at index %d, epoch %d", incoming.index, incoming.epoch)

    def snapshot_now(self) -> None:
        """Write a snapshot of current state and compact the log."""
        with self._lock:
            document = to_canonical(self._state)
            self._store.write_snapshot(document, compact_through=self._state.index)
            self._entries_since_snapshot = 0
            self._pending_change = None

    def _maybe_compact_locked(self, op: Any = None) -> None:
        if op in COMPACT_AFTER:
            # Row 5: not here. The entry is committed; the caller switches to
            # the new key first, then calls `compact_pending`, whose failure
            # cannot leave the old key in effect.
            self._pending_change = self._state.index
            return
        if self._entries_since_snapshot < self._compact_threshold:
            return
        document = to_canonical(self._state)
        self._store.write_snapshot(document, compact_through=self._state.index)
        self._entries_since_snapshot = 0
        self._pending_change = None
        log.info("snapshot written at index %d; log compacted", self._state.index)

    def compact_pending(self) -> str | None:
        """Row 5: drop the copies sealed under a key a passphrase change
        replaced: a snapshot of the state now, and every log entry before the
        change. The change itself stays (it holds only values sealed under
        the new key), so a standby one entry behind reads it from the log.

        None when done, or when nothing is pending; else the cause, said in
        the log once per try. Retried at every start, every hour on the
        root and at every pull on a standby."""
        with self._lock:
            pending = self._pending_change
            if pending is None:
                return None
            try:
                self._store.write_snapshot(to_canonical(self._state), compact_through=pending - 1)
            except OSError as exc:
                cause = str(exc) or type(exc).__name__
                log.warning(
                    "Passphrase changed; old copies could not be removed yet: %s. Trying "
                    "again later.",
                    cause,
                )
                return cause
            self._entries_since_snapshot = 0
            self._pending_change = None
            log.info(
                "snapshot written at index %d after a passphrase change; the copies sealed "
                "under the old key are removed",
                self._state.index,
            )
            return None

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


def _minute_after(key: dict[str, Any] | None, action: str) -> str | None:
    """The current minute, when a check or acquisition should move `lastUsedAt`.

    None when the key already carries this minute, so a poll loop writes one
    log entry a minute for each key, not one for each request.
    """
    if key is None or action not in ("check", "acquire"):
        return None
    minute = datetime.now(UTC).replace(second=0, microsecond=0)
    last = key.get("lastUsedAt")
    if last is not None and datetime.fromisoformat(last) >= minute:
        return None
    return minute.isoformat()
