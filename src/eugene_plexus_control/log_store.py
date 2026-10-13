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
`write` and `fsync` can leave a partial final line. `load` leaves out a
final line it cannot parse and reports that it did, which is the correct
reading: an entry that was not durable was never acknowledged, so
discarding it loses nothing a caller was promised. A parse failure in
the *middle* of the file is a different thing entirely — that is
corruption, and it raises.

**And it is cut off the file, not only out of memory.** The next
`append` writes at the end of the file, so bytes left there would be
glued to the front of the next entry: that entry, acknowledged to its
caller, would read back as the torn one and be dropped at the following
start, and with a second append the glued line would be in the middle,
which is corruption, and the root would not start. So `load` truncates
the file to the end of its last whole line (or, for a final line that
reads as a whole entry but has lost its newline, writes the newline) and
makes that durable, file and directory, before it returns, which is
before anything can append. Every role loads through here, so the active
root and a standby's own copy are repaired alike.

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
    to a caller, so nothing was lost. They are cut off the file as well,
    so only the `load` that found one reports it."""


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
        """Read the snapshot and every entry after it.

        Also leaves the log file ending on a whole line, which is what
        lets the next `append` follow it: a torn final line is cut off the
        file, not only left out of the result (see the module docstring).
        """
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
            scan = _scan_log(self._log_path.read_bytes(), self._log_path)
            entries = scan.entries
            dropped = scan.dropped
            # Checked before the file is touched: a log that is about to be
            # refused as corrupt is left exactly as it was found.
            _assert_gapless(entries, self._log_path)
            self._repair_tail(scan)

        return LoadResult(snapshot=snapshot, entries=entries, dropped_tail_lines=dropped)

    def _repair_tail(self, scan: _Scan) -> None:
        """Leave the file ending where its last good line does.

        Two ways the end of the file is wrong, and `append` writes straight
        after either:

          * the final line was left out (torn, or not readable): the file is
            truncated to the end of the line before it;
          * the final line was kept but has no newline, as when a power cut
            took the last byte of an append and nothing else: the newline is
            written, because the entry reads as whole and is kept.

        Then `fsync` on the file and on the directory, the same barrier
        `append` uses, so a second cut cannot bring the torn bytes back.
        Raises `OSError` naming the file if it cannot: an unrepaired log
        that an append is allowed to follow is the corruption this exists
        to prevent, so the root does not carry on over it.
        """
        try:
            if scan.keep < scan.size:
                log.warning(
                    "discarding a partial final line in %s (%d byte(s)) — this process was "
                    "killed uncleanly, and the entry was never acknowledged. The file is cut "
                    "back to its last whole line so the next entry does not start in the "
                    "middle of it",
                    self._log_path,
                    scan.size - scan.keep,
                )
                with open(self._log_path, "r+b") as handle:
                    handle.truncate(scan.keep)
                    handle.flush()
                    os.fsync(handle.fileno())
            elif scan.unterminated:
                log.warning(
                    "the final line of %s has no newline — this process was killed uncleanly "
                    "after the entry was written. It reads as a whole entry and is kept; the "
                    "newline is added so the next entry starts on its own line",
                    self._log_path,
                )
                with _private_files.open_private_append(self._log_path) as handle:
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            else:
                return
            _fsync_directory(self._dir)
        except OSError as exc:
            raise OSError(
                f"could not repair the end of {self._log_path} after an unclean stop: {exc}"
            ) from exc

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

        Written after the end of the file, which is where the last whole
        line ends only because `load` has already cut a torn tail off it.
        A caller appends to a store it has loaded; one that has not would
        glue this line onto the fragment.
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
        entries it already covers, which `StateMachine.load` skips and
        the next compaction drops. Doing it the other way round would
        open a window where neither file holds the state.
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


@dataclass(frozen=True)
class _Scan:
    """What one pass over the bytes of the log found."""

    entries: list[dict[str, Any]]
    dropped: int
    """1 when the final line could not be read and was left out, else 0."""
    size: int
    """Bytes in the file, as read."""
    keep: int
    """Bytes of it that are the log: `size`, or less when the final line was
    left out, in which case it is where that line starts."""
    unterminated: bool
    """The final line was kept (or is blank) and the file does not end in a
    newline."""


def _scan_log(raw: bytes, path: Path) -> _Scan:
    """Read the bytes of the log as entries, and find where its good part ends.

    A line is what lies between newlines, and only newlines. The writer ends
    every entry with exactly one and JSON escapes every other control
    character, so a newline in this file is always the end of an entry. That is
    not what `str.splitlines` does, which this used to call: it also splits on
    U+0085, U+2028 and U+2029, which JSON writes as they are inside a string,
    so an entry with one of them in a name was cut in two and the root could
    not start.

    One line may be left out, and only the last: a line that is not JSON (or
    not UTF-8, which is what a cut inside a multi-byte character leaves)
    because the process was killed part-way through writing it. Anywhere else
    that is corruption, and raises.
    """
    lines = raw.split(b"\n")
    if not lines[-1]:
        # The file ends on a newline, or is empty: nothing follows the last one.
        lines.pop()

    entries: list[dict[str, Any]] = []
    next_start = 0
    for position, line in enumerate(lines):
        start, next_start = next_start, next_start + len(line) + 1
        try:
            text = line.decode("utf-8").strip()
            if not text:
                continue
            parsed = json.loads(text)
        except ValueError as exc:
            if position == len(lines) - 1:
                # Torn tail: killed between write and fsync. The caller was
                # never told this one succeeded.
                return _Scan(entries, dropped=1, size=len(raw), keep=start, unterminated=False)
            raise LogCorruption(f"{path} line {position + 1} is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise LogCorruption(f"{path} line {position + 1} is not an object")
        entries.append(parsed)

    return _Scan(
        entries,
        dropped=0,
        size=len(raw),
        keep=len(raw),
        unterminated=bool(raw) and not raw.endswith(b"\n"),
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
