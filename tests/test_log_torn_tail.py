"""A torn log tail is cut off the file, so the next append is not glued onto it.

**The defect, from v0.1.0 (the pin d4886ef) on.** After a power cut in the
middle of an append, `load` dropped the partial final line from what it
returned and left its bytes in the file. The next `append` wrote straight
after them, so that entry shared a line with the fragment. The caller was
told "ok, index 10"; at the next start the glued line read as a torn tail
and was dropped, and the root came up at index 9. With a second append the
glued line was in the middle of the file, which is corruption: the root
raised `LogCorruption` and did not start at all.

These tests make that happen the way it happens (a fragment on disk, a
restart, appends, another restart) and then look at what a restart reads,
not only at the file.
"""

from __future__ import annotations

import errno
import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_control import log_store
from eugene_plexus_control.app import create_app
from eugene_plexus_control.applied import OP_PATCH_CONFIG
from eugene_plexus_control.log_store import LogCorruption, LogStore
from eugene_plexus_control.state_machine import StateMachine

from .conftest import PASSPHRASE, login, machine_at, read_log_file, settings_for, standby_at

# The front of the fourth entry, which a kill left behind.
TORN = b'{"index":4,"epoch":1,"op":"patchCon'


def _written(directory: Path, count: int = 3) -> StateMachine:
    """A root that has acknowledged `count` entries, indexes 1 to `count`."""
    machine = machine_at(directory)
    for theme in ("dark", "light", "dark", "light", "dark")[:count]:
        machine.append(OP_PATCH_CONFIG, {"values": {"uiTheme": theme}})
    return machine


def _log(directory: Path) -> Path:
    return directory / "log.jsonl"


def _tear(directory: Path, fragment: bytes = TORN) -> bytes:
    """Leave `fragment` after the last whole line, as a cut during an append
    does. Returns what the file held before."""
    clean = _log(directory).read_bytes()
    assert clean.endswith(b"\n"), "the file was not whole to begin with"
    _log(directory).write_bytes(clean + fragment)
    return clean


def _indexes(directory: Path) -> list[int]:
    """The indexes in the file, read a line at a time. A glued line is not
    JSON, so this raises on exactly the damage under test."""
    return [int(entry["index"]) for entry in read_log_file(directory)]


# --------------------------------------------------------------------------- #
# The reproduction: an entry acknowledged after a torn tail
# --------------------------------------------------------------------------- #


def test_an_entry_appended_after_a_torn_tail_is_there_after_the_next_restart(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "state"
    _written(directory, 3)
    _tear(directory)

    restarted = machine_at(directory)
    assert restarted.dropped_tail_lines == 1
    assert restarted.state.index == 3
    entry = restarted.append(OP_PATCH_CONFIG, {"values": {"uiTheme": "light"}})
    assert entry["index"] == 4

    # What a caller was told is what a restart reads: index 4, not 3.
    again = machine_at(directory)
    assert again.state.index == 4
    assert again.state.config["uiTheme"] == "light"
    assert again.canonical() == restarted.canonical()
    assert again.dropped_tail_lines == 0, "the fragment is gone from the file, not only from memory"
    assert _indexes(directory) == [1, 2, 3, 4]


def test_two_appends_after_a_torn_tail_leave_a_log_that_still_starts(tmp_path: Path) -> None:
    """The second append is what made the glued line a middle line, and a
    middle line that does not parse is corruption: the root did not start."""
    directory = tmp_path / "state"
    _written(directory, 3)
    _tear(directory)

    restarted = machine_at(directory)
    restarted.append(OP_PATCH_CONFIG, {"values": {"uiTheme": "light"}})
    restarted.append(OP_PATCH_CONFIG, {"values": {"uiTheme": "dark"}})

    again = machine_at(directory)  # raised LogCorruption before the fix
    assert again.state.index == 5
    assert again.state.config["uiTheme"] == "dark"
    assert again.canonical() == restarted.canonical()
    assert _indexes(directory) == [1, 2, 3, 4, 5]


@pytest.mark.parametrize(
    "fragment",
    [
        pytest.param(TORN, id="cut-in-the-middle-of-an-entry"),
        pytest.param(b"\x00" * 40, id="zero-filled-tail"),
        pytest.param(b"   " + TORN, id="cut-after-blanks"),
        pytest.param(b"not json at all\n", id="whole-line-that-is-not-json"),
    ],
)
def test_whatever_the_unreadable_last_line_is_it_is_cut_off_the_file(
    tmp_path: Path, fragment: bytes
) -> None:
    directory = tmp_path / "state"
    _written(directory, 3)
    clean = _tear(directory, fragment)

    store = LogStore(directory)
    loaded = store.load()
    assert [e["index"] for e in loaded.entries] == [1, 2, 3]
    assert loaded.dropped_tail_lines == 1
    assert _log(directory).read_bytes() == clean

    store.append({**loaded.entries[-1], "index": 4})
    again = store.load()
    assert [e["index"] for e in again.entries] == [1, 2, 3, 4]
    assert again.dropped_tail_lines == 0


def test_a_cut_inside_a_multibyte_character_is_a_torn_tail_too(tmp_path: Path) -> None:
    """Entries are written as UTF-8 with the characters as they are, so a cut
    can land inside one. That is not valid UTF-8, and it used to escape
    `load` as a `UnicodeDecodeError` rather than being read as the torn tail
    it is."""
    directory = tmp_path / "state"
    _written(directory, 3)
    store = LogStore(directory)
    clean = _log(directory).read_bytes()
    store.append({"index": 4, "epoch": 1, "op": OP_PATCH_CONFIG, "payload": {"name": "caf\u00e9"}})
    whole = _log(directory).read_bytes()
    _log(directory).write_bytes(whole[: whole.index("\u00e9".encode()) + 1])
    with pytest.raises(UnicodeDecodeError):
        _log(directory).read_bytes().decode("utf-8")

    loaded = store.load()

    assert [e["index"] for e in loaded.entries] == [1, 2, 3]
    assert loaded.dropped_tail_lines == 1
    assert _log(directory).read_bytes() == clean


# --------------------------------------------------------------------------- #
# A whole final line that lost only its newline
# --------------------------------------------------------------------------- #


def test_a_whole_final_line_that_lost_its_newline_is_kept_and_given_one(tmp_path: Path) -> None:
    """A cut can take the last byte of an append and nothing else. The line
    reads as a whole entry, so it is kept as it always was; truncating it
    would drop an entry `load` keeps. The next append must not be glued to it
    either, so the newline is written."""
    directory = tmp_path / "state"
    _written(directory, 3)
    whole = _log(directory).read_bytes()
    _log(directory).write_bytes(whole[:-1])

    restarted = machine_at(directory)
    assert restarted.state.index == 3, "kept: this is what load keeps today"
    assert restarted.dropped_tail_lines == 0
    assert _log(directory).read_bytes() == whole, "only the newline was written"

    entry = restarted.append(OP_PATCH_CONFIG, {"values": {"uiTheme": "light"}})
    assert entry["index"] == 4
    again = machine_at(directory)
    assert again.state.index == 4
    assert again.canonical() == restarted.canonical()
    assert _indexes(directory) == [1, 2, 3, 4]


def test_two_appends_after_an_unterminated_whole_line_leave_a_log_that_starts(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "state"
    _written(directory, 3)
    _log(directory).write_bytes(_log(directory).read_bytes()[:-1])

    restarted = machine_at(directory)
    restarted.append(OP_PATCH_CONFIG, {"values": {"uiTheme": "light"}})
    restarted.append(OP_PATCH_CONFIG, {"values": {"uiTheme": "dark"}})

    assert machine_at(directory).state.index == 5
    assert _indexes(directory) == [1, 2, 3, 4, 5]


# --------------------------------------------------------------------------- #
# What must not change
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "tail",
    [
        pytest.param(b"", id="no-torn-tail"),
        pytest.param(TORN, id="and-a-torn-tail"),
    ],
)
def test_a_corrupt_line_that_is_not_the_last_still_raises_and_nothing_is_cut(
    tmp_path: Path, tail: bytes
) -> None:
    """The torn tail is the one line `load` may leave out. A bad line before
    a good one is corruption, and the file is left exactly as found for
    whoever has to look at it."""
    directory = tmp_path / "state"
    _written(directory, 3)
    first, second, third = _log(directory).read_bytes().splitlines(keepends=True)
    damaged = first + b"this is not json\n" + second + third + tail
    _log(directory).write_bytes(damaged)

    with pytest.raises(LogCorruption, match="line 2 is not valid JSON"):
        LogStore(directory).load()
    with pytest.raises(LogCorruption):
        machine_at(directory)

    assert _log(directory).read_bytes() == damaged


def test_a_bad_line_just_before_a_torn_tail_is_corruption_too(tmp_path: Path) -> None:
    """Only the last line is the torn tail, so a bad one before it is in the
    middle, and nothing is truncated back to it."""
    directory = tmp_path / "state"
    _written(directory, 3)
    damaged = _log(directory).read_bytes() + b"this is not json\n" + TORN
    _log(directory).write_bytes(damaged)

    with pytest.raises(LogCorruption, match="line 4 is not valid JSON"):
        LogStore(directory).load()
    assert _log(directory).read_bytes() == damaged


def test_a_log_with_a_gap_is_refused_before_its_torn_tail_is_cut(tmp_path: Path) -> None:
    """A log that is about to be refused is not also edited."""
    directory = tmp_path / "state"
    _written(directory, 3)
    first, _second, third = _log(directory).read_bytes().splitlines(keepends=True)
    damaged = first + third + TORN
    _log(directory).write_bytes(damaged)

    with pytest.raises(LogCorruption, match="gap in the log"):
        LogStore(directory).load()
    assert _log(directory).read_bytes() == damaged


def test_a_clean_log_is_byte_identical_after_load_and_nothing_is_synced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "state"
    _written(directory, 3)
    before = _log(directory).read_bytes()
    synced: list[str] = []
    monkeypatch.setattr(os, "fsync", lambda fd: synced.append("file"))
    monkeypatch.setattr(log_store, "_fsync_directory", lambda _directory: synced.append("dir"))

    machine = machine_at(directory)
    LogStore(directory).load()
    machine.read_page(0, 10)
    machine.first_available_index()

    assert _log(directory).read_bytes() == before
    assert synced == [], "load wrote something to a file that needed nothing"


def test_a_missing_or_empty_log_is_left_as_it_is(tmp_path: Path) -> None:
    store = LogStore(tmp_path / "state")
    assert store.load().entries == []
    assert not store.log_path.exists(), "load does not create the log"

    store.directory.mkdir(parents=True)
    store.log_path.write_bytes(b"")
    assert store.load().entries == []
    assert store.log_path.read_bytes() == b""


def test_a_log_with_windows_line_endings_and_blank_lines_reads_as_it_did(tmp_path: Path) -> None:
    """What `load` keeps is unchanged: a CRLF file, and blank lines between
    entries, read the same and are not rewritten."""
    directory = tmp_path / "state"
    _written(directory, 3)
    first, second, third = _log(directory).read_bytes().splitlines()
    odd = first + b"\r\n\r\n" + second + b"\r\n   \n" + third + b"\r\n"
    _log(directory).write_bytes(odd)

    loaded = LogStore(directory).load()

    assert [e["index"] for e in loaded.entries] == [1, 2, 3]
    assert loaded.dropped_tail_lines == 0
    assert _log(directory).read_bytes() == odd


def test_separators_that_json_leaves_inside_a_string_do_not_split_an_entry(tmp_path: Path) -> None:
    """U+0085, U+2028 and U+2029 are line breaks to `str.splitlines` and plain
    characters to JSON, which writes them as they are. An entry with one in a
    name was read back as two broken lines and the root could not start."""
    store = LogStore(tmp_path / "state")
    names = ["plain", "line\u2028separator", "paragraph\u2029separator", "next\u0085line"]
    for index, name in enumerate(names, start=1):
        store.append({"index": index, "epoch": 1, "op": "x", "payload": {"name": name}})

    loaded = store.load()

    assert [e["payload"]["name"] for e in loaded.entries] == names
    assert loaded.dropped_tail_lines == 0


# --------------------------------------------------------------------------- #
# Durability: the repair is on disk before anything can append
# --------------------------------------------------------------------------- #


def _record_syncs(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int]]:
    """Every `fsync` the store makes, with the size the file had when it
    asked for it. A cut that was not made before the sync shows as the wrong
    size; a sync that was not made does not show at all."""
    events: list[tuple[str, int]] = []
    real_fsync = os.fsync

    def fsync(fd: int) -> None:
        events.append(("file", os.fstat(fd).st_size))
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(log_store, "_fsync_directory", lambda _d: events.append(("directory", 0)))
    return events


def test_the_cut_is_synced_to_the_file_and_the_directory_before_load_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "state"
    _written(directory, 3)
    clean = _tear(directory)
    events = _record_syncs(monkeypatch)

    LogStore(directory).load()

    assert events == [("file", len(clean)), ("directory", 0)]


def test_the_newline_is_synced_to_the_file_and_the_directory_before_load_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "state"
    _written(directory, 3)
    whole = _log(directory).read_bytes()
    _log(directory).write_bytes(whole[:-1])
    events = _record_syncs(monkeypatch)

    LogStore(directory).load()

    assert events == [("file", len(whole)), ("directory", 0)]


def test_a_repair_that_cannot_be_made_stops_the_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Carrying on over a tail that could not be cut is the corruption this
    prevents: the next append would be glued onto it."""
    directory = tmp_path / "state"
    _written(directory, 3)
    _tear(directory)

    def failing_fsync(_fd: int) -> None:
        raise OSError(errno.EIO, "the disk said no")

    monkeypatch.setattr(os, "fsync", failing_fsync)

    with pytest.raises(
        OSError, match=r"could not repair the end of .*log\.jsonl.*the disk said no"
    ):
        machine_at(directory)


# --------------------------------------------------------------------------- #
# Every role that loads a log file
# --------------------------------------------------------------------------- #


def test_a_standbys_own_copy_is_repaired_the_same_way(tmp_path: Path) -> None:
    active = _written(tmp_path / "active", 4)
    standby = standby_at(tmp_path / "standby")
    for entry in active.read_page(0, 3):
        standby.accept_replicated(entry)
    _tear(tmp_path / "standby")

    restarted = standby_at(tmp_path / "standby")
    assert restarted.dropped_tail_lines == 1
    assert restarted.state.index == 3
    for entry in active.read_page(restarted.state.index, 10):
        restarted.accept_replicated(entry)

    again = standby_at(tmp_path / "standby")
    assert again.state.index == 4, "the entry the standby acknowledged survived its restart"
    assert again.canonical() == active.canonical()
    assert _indexes(tmp_path / "standby") == [1, 2, 3, 4]


def test_the_apps_own_start_up_repairs_the_tail_before_the_first_write(tmp_path: Path) -> None:
    """Through the real start-up and the real endpoints, which is the path
    that the other tests reach only by calling the pieces."""
    settings = settings_for(tmp_path)

    def start() -> Any:
        return create_app(settings)

    first = start()
    with TestClient(first) as client:
        client.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).raise_for_status()
        client.headers["Authorization"] = f"Bearer {login(client)}"
        assert client.patch("/v1/config", json={"uiTheme": "dark"}).status_code == 200
    _tear(settings.state_dir)

    second = start()
    with TestClient(second) as client:
        health = client.get("/healthz").json()
        assert health["status"] == "degraded"
        assert health["details"]["discardedTornTailLines"] == 1
        client.headers["Authorization"] = f"Bearer {login(client)}"
        assert client.patch("/v1/config", json={"uiTheme": "light"}).status_code == 200

    third = start()
    with TestClient(third) as client:
        health = client.get("/healthz").json()
        assert "discardedTornTailLines" not in health["details"]
        # Initialization records the install CA (LT2, `setInstallCa`), entry 1.
        assert health["details"]["appliedIndex"] == 3
        assert health["status"] == "ok"
    assert third.state.machine.state.config["uiTheme"] == "light"
    assert _indexes(settings.state_dir) == [1, 2, 3]
