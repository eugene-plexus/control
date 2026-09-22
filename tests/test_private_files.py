"""Every file this component writes is readable by this account only.

The replicated log and the snapshot carry the sealed signing key, the
Argon2id salt and the passphrase verifier. The salt and the verifier
together are an offline attack on the operator's passphrase, so a log at
the umask's `0644` hands that attack to every account on the host.

Two kinds of test, because CI is Linux and this box is Windows. The
POSIX ones read the mode off the finished file -- the property itself.
The platform-independent ones watch `os.open` and assert every file was
*created* asking for `0600`, which is the only part of the mechanism a
Windows run can see; before the fix they fail on both, because
`Path.open` never calls `os.open` at all.
"""

from __future__ import annotations

import os
import stat
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_control import _private_files, config
from eugene_plexus_control.app import create_app
from eugene_plexus_control.log_store import LogStore

from .conftest import PASSPHRASE, login, settings_for

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="file modes are POSIX")


class _OpenSpy:
    """Records `(path, flags, mode)` for every `os.open`, then opens."""

    def __init__(self, real: Any) -> None:
        self.real = real
        self.calls: list[tuple[str, int, int]] = []

    def __call__(self, path: Any, flags: int, mode: int = 0o777, *args: Any, **kw: Any) -> int:
        self.calls.append((os.fspath(path), flags, mode))
        return self.real(path, flags, mode, *args, **kw)

    def modes_for(self, predicate: Any) -> list[int]:
        return [mode for path, flags, mode in self.calls if predicate(Path(path))]


@pytest.fixture
def open_spy(monkeypatch: pytest.MonkeyPatch) -> Iterator[_OpenSpy]:
    spy = _OpenSpy(os.open)
    monkeypatch.setattr(os, "open", spy)
    yield spy


def _initialize_and_patch(directory: Path) -> None:
    """The three writes a real install makes in its first minute: the
    identity entry, the snapshot `seed_identity` takes, and the config
    cache a PATCH writes through."""
    app = create_app(settings_for(directory))
    with TestClient(app) as client:
        response = client.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE})
        assert response.status_code == 204, response.text
        token = login(client)
        patched = client.patch(
            "/v1/config",
            json={"uiTheme": "dark"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert patched.status_code == 200, patched.text


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.fixture
def stock_umask() -> Iterator[None]:
    """The umask a stock Linux account has, under which `open()` gives
    `0644` -- so the test cannot pass by inheriting a strict one."""
    previous = os.umask(0o022)
    yield
    os.umask(previous)


@posix_only
@pytest.mark.usefixtures("stock_umask")
def test_the_log_the_snapshot_and_the_config_cache_are_0600(tmp_path: Path) -> None:
    directory = tmp_path / "install"
    _initialize_and_patch(directory)

    assert _mode(directory / "state" / "log.jsonl") == 0o600
    assert _mode(directory / "state" / "snapshot.json") == 0o600
    assert _mode(directory / "control.yaml") == 0o600


def test_every_secret_bearing_file_is_created_asking_for_0600(
    tmp_path: Path, open_spy: _OpenSpy
) -> None:
    directory = tmp_path / "install"
    _initialize_and_patch(directory)

    log_modes = open_spy.modes_for(lambda p: p.name == "log.jsonl")
    snapshot_modes = open_spy.modes_for(lambda p: p.name.startswith(".snapshot-"))
    cache_modes = open_spy.modes_for(
        lambda p: p.name.startswith(".control.yaml.") and p.name.endswith(".tmp")
    )

    assert log_modes, "the log was never opened through os.open"
    assert snapshot_modes, "the snapshot temp was never opened through os.open"
    assert cache_modes, "the config cache was never written through a private temp"
    assert set(log_modes + snapshot_modes + cache_modes) == {0o600}

    # And the cache is a replace, not a rewrite: the temp was exclusive.
    cache_flags = [
        flags for path, flags, _ in open_spy.calls if Path(path).name.startswith(".control.yaml.")
    ]
    assert all(flags & os.O_EXCL for flags in cache_flags)
    assert (directory / "control.yaml").read_text(encoding="utf-8").count("uiTheme: dark") == 1


@posix_only
@pytest.mark.usefixtures("stock_umask")
def test_an_existing_world_readable_log_is_narrowed_on_the_next_append(tmp_path: Path) -> None:
    """An install whose log predates the fix is exactly as readable as it
    was; the next append is the first chance to fix that."""
    store = LogStore(tmp_path)
    tmp_path.mkdir(exist_ok=True)
    store.log_path.write_text("", encoding="utf-8")
    store.log_path.chmod(0o644)

    store.append({"index": 1, "op": "noop"})

    assert _mode(store.log_path) == 0o600


def test_append_keeps_append_semantics(tmp_path: Path) -> None:
    store = LogStore(tmp_path)
    store.append({"index": 1, "op": "a"})
    store.append({"index": 2, "op": "b"})

    raw = store.log_path.read_bytes()
    assert raw.count(b"\n") == 2
    assert b"\r\n" not in raw
    assert [e["index"] for e in store.load().entries] == [1, 2]


def test_a_failed_replace_leaves_the_old_file_and_no_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "control.yaml"
    target.write_text("old: true\n", encoding="utf-8")

    def refuse(*_: Any) -> None:
        raise OSError("disk said no")

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(OSError, match="disk said no"):
        _private_files.write_private_text(target, "new: true\n")

    assert target.read_text(encoding="utf-8") == "old: true\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["control.yaml"]


def test_the_config_cache_stays_best_effort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache that could not be written must not fail the PATCH -- the
    entry is already in the log -- and must not leave its temp behind."""
    cache = config.BootstrapConfig(tmp_path / "control.yaml")

    def refuse(*_: Any) -> None:
        raise OSError("disk said no")

    monkeypatch.setattr(os, "replace", refuse)
    cache.write_through({"uiTheme": "dark"})

    assert list(tmp_path.iterdir()) == []
