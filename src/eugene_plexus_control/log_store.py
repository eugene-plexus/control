"""The log and the snapshot on disk.

Two files in one directory, with different write patterns, which is why
it is a directory and not a file:

  * `log.jsonl` — append-only, one entry per line, fsync'd per append.
  * `snapshot.json` — replaced atomically, so a crash mid-write leaves
    the previous snapshot rather than half of the new one.

**Durability is the whole job here.** An entry that a caller has been
told succeeded must survive an unclean kill, because the caller may have
been an operator enrolling a node and the alternative is an install that
disagrees with the person who configured it. So the append is
synchronous and fsync'd inside the writer's lock, and it is deliberately
*not* moved onto a thread: this is the serialization point of the whole
component, and concurrency here would buy nothing but a second writer.

**A torn tail is expected and handled.** A process killed between
`write` and `fsync` can leave a partial final line. `load` stops at the
first line it cannot parse and reports how many it dropped, which is the
correct reading: an entry that was not durable was never acknowledged,
so discarding it loses nothing a caller was promised. A parse failure in
the *middle* of the file is a different thing entirely — that is
corruption, and it raises.

Reading a page scans the file rather than seeking an offset index. That
is deliberate at this scale: control state is kilobytes of topology and
a node registry, not a database, because sealed component secrets are
already distributed to the hosts that read them. If the log ever grows
past what a scan can serve, compaction is the answer already in the
design, not an index.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import _private_files

log = logging.getLogger(__name__)

LOG_FILENAME = "log.jsonl"
SNAPSHOT_FILENAME = "snapshot.json"


class LogCorruption(Exception):
    """The log cannot be read as a log.

    Distinct from a torn tail, which is normal. This means an entry in
    the middle failed to parse or the indices are not gapless, and the
    only honest response is to stop rather than serve a state assembled
    from part of a file."""


@dataclass(frozen=True)
class LoadResult:
    """What `load` found on disk."""

    snapshot: dict[str, Any] | None
    entries: list[dict[str, Any]]
    dropped_tail_lines: int
    """Partial lines discarded from the end. Non-zero means this process
    was killed uncleanly last time; the entries were never acknowledged
    to a caller, so nothing was lost."""


class LogStore:
    """File-backed home for the log and the snapshot.

    Holds no lock of its own: serialization belongs to the one writer
    above it (`state_machine.StateMachine`), and a second lock here
    would imply there could be a second caller.
    """

    def __init__(self, directory: Path) -> None:
        self._dir = directory
        self._log_path = directory / LOG_FILENAME
        self._snapshot_path = directory / SNAPSHOT_FILENAME

    @property
    def directory(self) -> Path:
        return self._dir

    @property
    def log_path(self) -> Path:
        return self._log_path

    @property
    def snapshot_path(self) -> Path:
        return self._snapshot_path

    # ----- reading ----------------------------------------------------

    def load(self) -> LoadResult:
        """Read the snapshot and every entry after it."""
        snapshot: dict[str, Any] | None = None
        if self._snapshot_path.exists():
            try:
                raw = json.loads(self._snapshot_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise LogCorruption(f"snapshot {self._snapshot_path} is unreadable: {exc}") from exc
            if not isinstance(raw, dict):
                raise LogCorruption(f"snapshot {self._snapshot_path} is not an object")
            snapshot = raw

        entries: list[dict[str, Any]] = []
        dropped = 0
        if self._log_path.exists():
            lines = self._log_path.read_text(encoding="utf-8").splitlines()
            for position, line in enumerate(lines):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    parsed = json.loads(stripped)
                except ValueError as exc:
                    is_last = position == len(lines) - 1
                    if is_last:
                        # Torn tail: killed between write and fsync. The
                        # caller was never told this one succeeded.
                        dropped = 1
                        log.warning(
                            "discarding a partial final line in %s — this process was killed "
                            "uncleanly, and the entry was never acknowledged",
                            self._log_path,
                        )
                        break
                    raise LogCorruption(
                        f"{self._log_path} line {position + 1} is not valid JSON: {exc}"
                    ) from exc
                if not isinstance(parsed, dict):
                    raise LogCorruption(f"{self._log_path} line {position + 1} is not an object")
                entries.append(parsed)

        _assert_gapless(entries, self._log_path)
        return LoadResult(snapshot=snapshot, entries=entries, dropped_tail_lines=dropped)

    def read_after(self, after: int, limit: int) -> list[dict[str, Any]]:
        """Entries with `index` strictly greater than `after`.

        Returns what is present; the caller decides whether a gap
        between `after` and the oldest surviving entry means the
        requester has to re-snapshot. That judgement needs
        `first_available_index`, which is a property of the whole store
        rather than of one page.
        """
        result = self.load()
        return [e for e in result.entries if int(e.get("index", 0)) > after][:limit]

    def first_available_index(self, applied_index: int) -> int:
        """Oldest index still readable from the log.

        When the log is empty — a freshly compacted store, or one that
        has only ever snapshotted — this is `applied_index + 1`: nothing
        is available, and the next entry to be written is the first one
        that will be.
        """
        result = self.load()
        if result.entries:
            return int(result.entries[0].get("index", 0))
        return applied_index + 1

    # ----- writing ----------------------------------------------------

    def append(self, entry: dict[str, Any]) -> None:
        """Append one entry durably.

        `fsync` on the file and then on the directory: the second is
        what makes the *name* durable on POSIX, and skipping it is the
        classic way an append survives a crash on paper and not in
        practice. On Windows `os.fsync` on a directory handle is not
        available, so the directory sync is skipped there — the file
        sync is what NTFS needs.

        Opened `0600`: the log carries the salt and the passphrase
        verifier, which together are an offline attack on the passphrase.
        See `_private_files` for why that is set at creation.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        with _private_files.open_private_append(self._log_path) as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(self._dir)

    def write_snapshot(self, snapshot: dict[str, Any], *, compact_through: int | None) -> None:
        """Replace the snapshot atomically, then optionally compact.

        Order matters and is the reverse of what feels natural: the
        snapshot is durable *before* any entry is dropped, so a crash
        between the two leaves a store with both a good snapshot and
        entries it will replay harmlessly. Doing it the other way round
        would open a window where neither file holds the state.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

        # SIM115: the handle deliberately outlives its `with` block —
        # `os.replace` must happen after the file is closed, and the
        # name has to survive to be replaced. A context manager around
        # the whole thing cannot express an atomic rename.
        #
        # `tempfile` creates the file `0600` and `os.replace` keeps the
        # mode, which is what keeps the snapshot -- sealed keys, salt,
        # verifier -- private. Load-bearing, not incidental: a
        # `test_private_files` case watches for it, here and in `_compact`.
        handle = tempfile.NamedTemporaryFile(  # noqa: SIM115
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=self._dir,
            prefix=".snapshot-",
            suffix=".tmp",
            delete=False,
        )
        try:
            with handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(handle.name, self._snapshot_path)
        except BaseException:
            with contextlib_suppress():
                os.unlink(handle.name)
            raise
        _fsync_directory(self._dir)

        if compact_through is not None:
            self._compact(compact_through)

    def _compact(self, through_index: int) -> None:
        """Drop entries at or below `through_index`.

        Rewrite-and-rename rather than truncate-in-place, so a crash
        mid-compaction leaves the uncompacted log — which is correct but
        larger — instead of a file missing its head.
        """
        if not self._log_path.exists():
            return
        result = self.load()
        keep = [e for e in result.entries if int(e.get("index", 0)) > through_index]
        if len(keep) == len(result.entries):
            return

        handle = tempfile.NamedTemporaryFile(  # noqa: SIM115
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=self._dir,
            prefix=".log-",
            suffix=".tmp",
            delete=False,
        )
        try:
            with handle:
                for entry in keep:
                    handle.write(
                        json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                        + "\n"
                    )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(handle.name, self._log_path)
        except BaseException:
            with contextlib_suppress():
                os.unlink(handle.name)
            raise
        _fsync_directory(self._dir)
        log.info(
            "compacted %s through index %d (%d entries dropped, %d kept)",
            self._log_path,
            through_index,
            len(result.entries) - len(keep),
            len(keep),
        )


def _assert_gapless(entries: list[dict[str, Any]], path: Path) -> None:
    """Indices must be strictly increasing by one.

    Checked at load rather than only at apply because a hole in a file
    on disk is a different problem from a hole in a stream: the stream
    can be re-requested, the file cannot.
    """
    previous: int | None = None
    for entry in entries:
        index = entry.get("index")
        if not isinstance(index, int):
            raise LogCorruption(f"{path}: entry has non-integer index {index!r}")
        if previous is not None and index != previous + 1:
            raise LogCorruption(
                f"{path}: gap in the log — index {index} follows {previous}. "
                "Bootstrap from a snapshot rather than reading around it."
            )
        previous = index


def _fsync_directory(directory: Path) -> None:
    """Make a rename or a create durable. No-op where unsupported."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        # Windows cannot open a directory for reading, and does not need
        # this: the file-level fsync above is the durability barrier.
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def contextlib_suppress() -> Any:
    """`contextlib.suppress(OSError)`, named so the cleanup paths above
    read as intentional rather than as a bare except."""
    import contextlib

    return contextlib.suppress(OSError)
