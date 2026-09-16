"""S0 of the hobbyist UX plan, control's half: the keyring entry is scoped
per install, a legacy entry migrates once, the probe is measured, and
`GET /v1/auth/status` says unlocked and keyringAvailable.

The single-slot fake in `test_keyring_unlock.py` cannot tell two
usernames apart, which is exactly the property under test here, so this
file carries a dict-keyed fake of its own.
"""

from __future__ import annotations

import base64
import secrets
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_control import keyring_store
from eugene_plexus_control.app import create_app

from .conftest import PASSPHRASE, login, settings_for

SALT_A = base64.b64encode(b"A" * 16).decode("ascii")
SALT_B = base64.b64encode(b"B" * 16).decode("ascii")


class DictKeyring:
    """A keyring that remembers WHICH username a value went under."""

    def __init__(self) -> None:
        self.store: dict[tuple[str, str], str] = {}
        self.raise_on: set[str] = set()

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def get(service: str, username: str) -> str | None:
            if "get" in self.raise_on:
                raise keyring_store.keyring.errors.KeyringError("simulated")
            return self.store.get((service, username))

        def put(service: str, username: str, value: str) -> None:
            if "set" in self.raise_on:
                raise keyring_store.keyring.errors.KeyringError("simulated")
            self.store[(service, username)] = value

        def drop(service: str, username: str) -> None:
            if (service, username) not in self.store:
                raise keyring_store.keyring.errors.PasswordDeleteError("nothing stored")
            del self.store[(service, username)]

        monkeypatch.setattr(keyring_store.keyring, "get_password", get)
        monkeypatch.setattr(keyring_store.keyring, "set_password", put)
        monkeypatch.setattr(keyring_store.keyring, "delete_password", drop)


@pytest.fixture
def dict_keyring(monkeypatch: pytest.MonkeyPatch) -> DictKeyring:
    fake = DictKeyring()
    fake.install(monkeypatch)
    return fake


def _b64(key: bytes) -> str:
    return base64.b64encode(key).decode("ascii")


def _boot(directory: Path) -> TestClient:
    return TestClient(create_app(settings_for(directory)))


def _initialize_and_enable(client: TestClient) -> None:
    assert client.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
    token = login(client)
    client.headers["Authorization"] = f"Bearer {token}"
    assert client.patch("/v1/config", json={"securityMode": "os_keyring"}).status_code == 200


def test_two_installs_do_not_share_an_entry(dict_keyring: DictKeyring) -> None:
    a = keyring_store.install_id_for(SALT_A)
    b = keyring_store.install_id_for(SALT_B)
    assert a != b
    key_a, key_b = secrets.token_bytes(32), secrets.token_bytes(32)
    assert keyring_store.set_master_key(key_a, a)
    assert keyring_store.set_master_key(key_b, b)
    assert keyring_store.get_master_key(a) == key_a
    assert keyring_store.get_master_key(b) == key_b
    assert keyring_store.delete_master_key(a) is True
    assert keyring_store.get_master_key(a) is None
    assert keyring_store.get_master_key(b) == key_b


def test_a_second_install_on_the_machine_does_not_unlock_the_first(
    tmp_path: Path, dict_keyring: DictKeyring
) -> None:
    """The failure scoping exists for: install B's wizard stores B's key;
    install A restarts and must NOT find it - or, worse, find it, fail
    to open its seal, and discard what it thinks is its own entry."""
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    with _boot(a_dir) as a:
        _initialize_and_enable(a)
    with _boot(b_dir) as b:
        _initialize_and_enable(b)
    assert len(dict_keyring.store) == 2, "one entry per install"

    with _boot(a_dir) as restarted_a:
        # 401 = unlocked and this caller has no token; 503 = sealed.
        assert restarted_a.get("/v1/control/status").status_code == 401
    with _boot(b_dir) as restarted_b:
        assert restarted_b.get("/v1/control/status").status_code == 401
    assert len(dict_keyring.store) == 2, "neither restart discarded the other's key"


def test_a_legacy_entry_is_read_once_and_moved(dict_keyring: DictKeyring) -> None:
    install = keyring_store.install_id_for(SALT_A)
    key = secrets.token_bytes(32)
    dict_keyring.store[(keyring_store.SERVICE, keyring_store.LEGACY_USERNAME)] = _b64(key)
    assert keyring_store.get_master_key(install) == key
    assert (keyring_store.SERVICE, keyring_store.LEGACY_USERNAME) not in dict_keyring.store
    assert dict_keyring.store[(keyring_store.SERVICE, keyring_store.username_for(install))] == _b64(
        key
    )


def test_a_root_that_predates_scoping_still_auto_unlocks(
    tmp_path: Path, dict_keyring: DictKeyring
) -> None:
    """An upgrade must not turn a root that came back on its own into one
    that asks for the passphrase."""
    directory = tmp_path / "install"
    with _boot(directory) as client:
        _initialize_and_enable(client)
    # Move the stored key back to the pre-scoping slot, as an older build
    # would have left it.
    (((service, username), value),) = dict_keyring.store.items()
    dict_keyring.store = {(service, keyring_store.LEGACY_USERNAME): value}

    with _boot(directory) as restarted:
        assert restarted.get("/v1/control/status").status_code == 401
    assert (keyring_store.SERVICE, keyring_store.LEGACY_USERNAME) not in dict_keyring.store
    assert (service, username) in dict_keyring.store


def test_the_probe_is_a_round_trip_and_leaves_nothing_behind(dict_keyring: DictKeyring) -> None:
    keyring_store.reset_probe_cache()
    assert keyring_store.probe_sync() is True
    assert dict_keyring.store == {}
    dict_keyring.raise_on.add("set")
    assert keyring_store.probe_sync() is True, "memoised once per process"


def test_the_probe_says_no_in_a_container(dict_keyring: DictKeyring) -> None:
    keyring_store.reset_probe_cache()
    dict_keyring.raise_on.add("set")
    assert keyring_store.probe_sync() is False


def test_status_says_unlocked_and_whether_a_keyring_exists(
    tmp_path: Path, dict_keyring: DictKeyring
) -> None:
    keyring_store.reset_probe_cache()
    with _boot(tmp_path / "install") as client:
        fresh = client.get("/v1/auth/status").json()
        assert fresh == {"initialized": False, "unlocked": False, "keyringAvailable": True}
        assert (
            client.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
        )
        after = client.get("/v1/auth/status").json()
        assert after["initialized"] is True
        assert after["unlocked"] is True


def test_a_sealed_root_says_so_in_a_word(tmp_path: Path, dict_keyring: DictKeyring) -> None:
    """After a restart under prompt_on_startup the root is initialized and
    locked; the Issues list reads this instead of parsing a 503."""
    directory = tmp_path / "install"
    with _boot(directory) as client:
        assert (
            client.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
        )
    with _boot(directory) as restarted:
        body = restarted.get("/v1/auth/status").json()
        assert body["initialized"] is True
        assert body["unlocked"] is False
        assert restarted.get("/v1/control/status").status_code == 503
