"""A failed append cannot corrupt later acknowledged entries (control#9)."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from eugene_plexus_control import _private_files, log_store
from eugene_plexus_control.applied import OP_PATCH_CONFIG
from eugene_plexus_control.log_store import LogStore

from .conftest import machine_at


@pytest.mark.parametrize("failure", ["write", "flush", "close", "fsync"])
@pytest.mark.parametrize("existing", [False, True])
def test_failed_append_rolls_back_before_another_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str, existing: bool
) -> None:
    machine = machine_at(tmp_path)
    if existing:
        machine.append(OP_PATCH_CONFIG, {"values": {"uiTheme": "dark"}})
    before = machine.canonical()
    index = machine.state.index
    path = tmp_path / "log.jsonl"
    clean = path.read_bytes() if path.exists() else b""
    real_open = _private_files.open_private_append

    class BrokenWriter:
        def __init__(self, handle: Any) -> None:
            self.handle = handle

        def write(self, value: str) -> int:
            if failure == "write":
                self.handle.write(value[:15])
                self.handle.flush()
                raise OSError("injected partial write")
            return int(self.handle.write(value))

        def flush(self) -> None:
            self.handle.flush()
            if failure == "flush":
                raise OSError("injected flush failure")

        def fileno(self) -> int:
            return int(self.handle.fileno())

    @contextmanager
    def broken_open(name: Path) -> Iterator[BrokenWriter]:
        with real_open(name) as handle:
            yield BrokenWriter(handle)
        if failure == "close":
            raise OSError("injected close failure")

    real_fsync = log_store.os.fsync
    calls = 0

    def fail_once(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected fsync failure")
        real_fsync(fd)

    with monkeypatch.context() as fault:
        fault.setattr(_private_files, "open_private_append", broken_open)
        if failure == "fsync":
            fault.setattr(log_store.os, "fsync", fail_once)
        with pytest.raises(OSError, match="injected"):
            machine.append(OP_PATCH_CONFIG, {"values": {"uiTheme": "light"}})

    assert machine.canonical() == before
    assert path.read_bytes() == clean
    machine.append(OP_PATCH_CONFIG, {"values": {"uiTheme": "light"}})
    machine.append(OP_PATCH_CONFIG, {"values": {"uiTheme": "dark"}})
    assert machine.state.index == index + 2
    assert machine_at(tmp_path).canonical() == machine.canonical()


def test_failed_rollback_refuses_writes_reads_and_compaction_until_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LogStore(tmp_path)
    store.load()
    store.append({"index": 1})

    def unavailable(fd: int) -> None:
        raise OSError("disk unavailable")

    with monkeypatch.context() as fault:
        fault.setattr(log_store.os, "fsync", unavailable)
        with pytest.raises(OSError, match="disk unavailable"):
            store.append({"index": 2})

    # Even if the filesystem recovers, an unconfirmed rollback cannot be
    # exposed to replicas or overwritten by a snapshot of older memory.
    for operation in (
        lambda: store.append({"index": 2}),
        store.load,
        lambda: store.read_after(0, 10),
        lambda: store.write_snapshot({"index": 1}, compact_through=1),
    ):
        with pytest.raises(OSError, match=r"restart.*disk unavailable"):
            operation()
    restarted = LogStore(tmp_path)
    assert [e["index"] for e in restarted.load().entries] == [1]
    restarted.append({"index": 2})
    assert [e["index"] for e in restarted.load().entries] == [1, 2]
