"""One-time conversion of what an older build left in this root's state.

Two things were retired between v0.1.0 and v0.2: the operator-managed node
folders (`putNodeHelper`, a person's `helperGrants`, a snapshot's
`nodeHelpers`), and a Job Site joining as a node (`enrollNode` with the
grant `files`), which slice 2b.1 replaced with a Job Site's own enrollment.
A root that ran one of the builds in between can still hold them. A retired
Job Site is a node in its registry that is in no listing and no trust
bundle, and the files on disk still carry the old fields.

**Converted once, then only the current form is read.** At start, on the
active root, and only when it finds any of that on disk or in its registry,
this:

  1. keeps the files as they were: `log.jsonl` and `snapshot.json` are
     copied to `*.v0.1.bak`, once, never over an earlier copy;
  2. appends a `revokeNode` for each retired Job Site node. These are
     ordinary appends, so a standby that is following reads them like any
     other entry;
  3. takes a snapshot, which writes the current form and compacts the log,
     so nothing retired is left in either file. A standby that is behind the
     compaction takes the snapshot (the 409 that `GET /v1/control/log`
     answers), as it does after any compaction;
  4. writes `converted-from-v0.1.json`, the record that it was done.

**No entry is rewritten in place.** An entry's index is the revision of
what it changed, and a standby that has applied it would diverge from a
log that said something else under the same index. The log has no hash
chain and no signatures: what keeps it honest is a gapless index and an
epoch that never goes back, and appending keeps both.

**Every step can be killed.** The copies are written first, whole, through
a temporary name, so a kill leaves a complete copy or none. Each append is
durable on its own, a revoke that was already applied is not found again,
the snapshot is renamed into place before the log is compacted (and a
snapshot beside entries it already covers loads, `StateMachine.load`), and
the record is written last. The next start finds what is left and finishes:
by what is still retired on disk, or, for a kill after the compaction and
before the record, by copies that have no record beside them.

**A failure degrades.** `run` never raises: it names the step and what the
filesystem said, the root keeps serving what it has, `/healthz` says so,
and the next start tries again. Nothing is lost by a failure: the copies
come first, and whatever was appended is in the log like any other entry.

**A standby is not converted.** It cannot append, it was bootstrapped from
a root's snapshot (the current form) after the retirement, and a retired
node in its registry goes when the root's `revokeNode` or snapshot reaches
it.

**Kept for v0.2, deleted with this module in v0.3.** `applied.py` still
replays the retired entries, because the root that converts reads them
first, and a standby may follow it through them. v0.3 deletes those
handlers together with this module: by then every root that was going to
convert has.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import _private_files
from .applied import (
    OP_ENROLL_NODE,
    OP_PUT_NODE_HELPER,
    OP_PUT_PERSON,
    OP_REVOKE_NODE,
    is_retired_site,
)
from .log_store import LOG_FILENAME, SNAPSHOT_FILENAME, LogStore, _fsync_directory
from .state_machine import ROLE_ACTIVE, StateMachine

log = logging.getLogger(__name__)

BACKUP_SUFFIX = ".v0.1.bak"
"""Appended to the name of each file this keeps a copy of."""

MARKER_FILENAME = "converted-from-v0.1.json"
"""The record, in the state directory, that the conversion was done."""

_JOB_SITE_GRANT = "files"
_PERSON_FIELD = "helperGrants"
_SNAPSHOT_FIELD = "nodeHelpers"


class ConversionFailed(Exception):
    """A step failed. The message names the step and what was observed."""


@dataclass(frozen=True)
class Found:
    """What an older build left behind."""

    nodes: tuple[str, ...] = ()
    """Retired Job Site nodes in the applied registry, by name."""
    leftovers: dict[str, int] = field(default_factory=dict)
    """What else the files on disk still carry, counted by kind."""

    @property
    def any(self) -> bool:
        return bool(self.nodes or self.leftovers)

    def describe(self) -> str:
        parts = []
        if self.nodes:
            parts.append(f"{len(self.nodes)} retired Job Site node(s): {', '.join(self.nodes)}")
        parts.extend(f"{count} {what}" for what, count in self.leftovers.items())
        return "; ".join(parts)


@dataclass(frozen=True)
class Converted:
    """What one conversion did."""

    revoked: tuple[str, ...]
    copies: tuple[Path, ...]
    record: Path
    resumed: bool


def _holds_files(grants: Any) -> bool:
    return isinstance(grants, list) and _JOB_SITE_GRANT in grants


def find(store: LogStore, machine: StateMachine) -> Found:
    """Look for retired data in the registry and, raw, in both files.

    The files are read as documents, not through the applied state: the
    state drops what it does not keep, and this is looking for what is
    still on disk."""
    on_disk = store.load()
    leftovers: dict[str, int] = {}

    def note(what: str, count: int = 1) -> None:
        if count:
            leftovers[what] = leftovers.get(what, 0) + count

    snapshot = on_disk.snapshot or {}
    folders = snapshot.get(_SNAPSHOT_FIELD)
    note("node-folder record(s) in the snapshot", len(folders) if isinstance(folders, list) else 0)
    people = snapshot.get("people")
    note(
        "person record(s) with folder grants in the snapshot",
        sum(1 for p in people if isinstance(p, dict) and _PERSON_FIELD in p)
        if isinstance(people, list)
        else 0,
    )
    nodes = snapshot.get("nodes")
    note(
        "Job Site node(s) in the snapshot",
        sum(1 for n in nodes if isinstance(n, dict) and _holds_files(n.get("grants")))
        if isinstance(nodes, list)
        else 0,
    )
    for entry in on_disk.entries:
        payload = entry.get("payload")
        payload = payload if isinstance(payload, dict) else {}
        op = entry.get("op")
        if op == OP_PUT_NODE_HELPER:
            note("node-folder entry(ies) in the log")
        elif op == OP_ENROLL_NODE and _holds_files(payload.get("grants")):
            note("Job Site enrollment(s) in the log")
        elif op == OP_PUT_PERSON:
            person = payload.get("person")
            if isinstance(person, dict) and _PERSON_FIELD in person:
                note("person entry(ies) with folder grants in the log")

    retired = tuple(
        sorted(name for name, record in machine.state.nodes.items() if is_retired_site(record))
    )
    return Found(nodes=retired, leftovers=leftovers)


def _copies(directory: Path) -> list[Path]:
    return [directory / (name + BACKUP_SUFFIX) for name in (LOG_FILENAME, SNAPSHOT_FILENAME)]


def _interrupted(directory: Path) -> bool:
    """Copies were kept and nothing records the conversion: a run that was
    killed (or failed) after step 1."""
    return (not (directory / MARKER_FILENAME).exists()) and any(
        copy.exists() for copy in _copies(directory)
    )


def _keep_copies(store: LogStore) -> list[Path]:
    """Step 1. A copy that is already there is the files as they were, from
    an earlier run, and is kept: this run's files may already be partly
    converted."""
    kept: list[Path] = []
    for source in (store.log_path, store.snapshot_path):
        if not source.is_file():
            continue
        copy = source.with_name(source.name + BACKUP_SUFFIX)
        if copy.exists():
            if not copy.is_file():
                raise OSError(
                    f"{copy} is in the way: it is not a file, so the copy of {source.name} "
                    "cannot be kept there"
                )
            kept.append(copy)
            continue
        # Whole or not at all, and private: a copy of these files holds the
        # sealed keys, the salt and the passphrase verifier, as they do.
        _private_files.write_private_bytes(copy, source.read_bytes())
        kept.append(copy)
    # Names that must outlive a power cut before anything is compacted.
    _fsync_directory(store.directory)
    return kept


def convert(store: LogStore, machine: StateMachine) -> Converted | None:
    """Convert what is retired, once. None when there is nothing to do.

    Raises `ConversionFailed` naming the step; `run` is what a caller that
    must not crash uses."""
    if machine.role != ROLE_ACTIVE:
        return None
    directory = store.directory
    found = find(store, machine)
    resumed = _interrupted(directory)
    if not found.any and not resumed:
        return None
    log.info(
        "this root holds data an older build retired (%s); converting it once",
        found.describe() or "an earlier conversion stopped before it was recorded",
    )

    record = directory / MARKER_FILENAME
    revoked: list[str] = []
    step = "keeping a copy of the log and the snapshot"
    try:
        copies = _keep_copies(store)
        for name in found.nodes:
            step = f"revoking the retired Job Site node {name!r}"
            machine.append(OP_REVOKE_NODE, {"name": name})
            revoked.append(name)
        step = "writing the snapshot that folds the log"
        machine.snapshot_now()
        step = f"recording the conversion in {MARKER_FILENAME}"
        document = {
            "format": 1,
            "convertedFrom": "v0.1",
            "convertedAt": datetime.now(UTC).isoformat(),
            "found": {"nodes": list(found.nodes), **found.leftovers},
            "revokedNodes": revoked,
            "copies": [copy.name for copy in copies],
            "appliedIndex": machine.state.index,
            "resumedAfterAnInterruptedRun": resumed,
        }
        _private_files.write_private_text(
            record, json.dumps(document, indent=2, sort_keys=True) + "\n"
        )
    except Exception as exc:
        raise ConversionFailed(f"{step}: {type(exc).__name__}: {exc}") from exc
    return Converted(revoked=tuple(revoked), copies=tuple(copies), record=record, resumed=resumed)


def run(store: LogStore, machine: StateMachine) -> str | None:
    """Convert if there is anything to convert, and never raise. The cause
    of a failure, or None.

    A root that cannot convert is still the install: it keeps serving what
    it has and tries again at the next start."""
    try:
        done = convert(store, machine)
    except ConversionFailed as failure:
        cause = str(failure)
    except Exception as exc:
        cause = f"looking for retired data: {type(exc).__name__}: {exc}"
    else:
        if done is not None:
            log.info(
                "converted this root's retired data: revoked %d retired Job Site node(s)%s; "
                "the files as they were are kept as %s; recorded in %s",
                len(done.revoked),
                f" ({', '.join(done.revoked)})" if done.revoked else "",
                ", ".join(copy.name for copy in done.copies) or "(there were none)",
                done.record,
            )
        return None
    kept = [copy.name for copy in _copies(store.directory) if copy.is_file()]
    log.error(
        "could not convert this root's retired data (%s). It keeps serving what it has and "
        "tries again at the next start.%s",
        cause,
        f" The files as they were are kept as {', '.join(kept)}." if kept else "",
    )
    return cause
