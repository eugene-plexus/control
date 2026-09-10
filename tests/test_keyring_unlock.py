"""OS keyring auto-unlock for the trust root.

`securityMode` was declared on this component's config, `keyring` was in
its dependencies, and **nothing read either** — so choosing "OS keyring
auto-unlock" left the trust root prompting after a restart and an
install could not actually recover from a power cut. This is the
implementation, and these are the four things about it that are easy to
get wrong.

The `keyring` module is patched rather than driven for real: a test
suite must not write to the developer's Credential Manager, and CI has
no unlocked secret service at all. What is real here is the unseal —
real Argon2id, real sealed values — because the interesting failure is a
stored key that derives to something which does not open them.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_control import keyring_store
from eugene_plexus_control.app import create_app

from .conftest import PASSPHRASE, login, settings_for


class FakeKeyring:
    """An in-memory stand-in for one (service, username) slot."""

    def __init__(self) -> None:
        self.stored: str | None = None
        self.reads = 0
        self.fail_on_read: Exception | None = None

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def get(service: str, username: str) -> str | None:
            self.reads += 1
            if self.fail_on_read is not None:
                raise self.fail_on_read
            return self.stored

        def put(service: str, username: str, value: str) -> None:
            self.stored = value

        def drop(service: str, username: str) -> None:
            if self.stored is None:
                raise keyring_store.keyring.errors.PasswordDeleteError("nothing stored")
            self.stored = None

        monkeypatch.setattr(keyring_store.keyring, "get_password", get)
        monkeypatch.setattr(keyring_store.keyring, "set_password", put)
        monkeypatch.setattr(keyring_store.keyring, "delete_password", drop)


@pytest.fixture
def fake_keyring(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeKeyring]:
    fake = FakeKeyring()
    fake.install(monkeypatch)
    yield fake


def _boot(directory: Path) -> TestClient:
    return TestClient(create_app(settings_for(directory)))


def _initialize_and_enable(client: TestClient, fake: FakeKeyring) -> None:
    assert client.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
    token = login(client)
    client.headers["Authorization"] = f"Bearer {token}"
    assert client.patch("/v1/config", json={"securityMode": "os_keyring"}).status_code == 200
    assert fake.stored is not None, "flipping to os_keyring while unlocked must store the key"


def test_a_restart_auto_unlocks_when_the_operator_asked_for_it(
    tmp_path: Path, fake_keyring: FakeKeyring
) -> None:
    """The whole point: a trust root that comes back without a person.

    Without this, a restart leaves the root holding its keys sealed and
    every install-wide operation answering 503 until someone types the
    passphrase - which is correct behaviour for `prompt_on_startup` and
    a false promise under `os_keyring`.
    """
    directory = tmp_path / "install"
    with _boot(directory) as client:
        _initialize_and_enable(client, fake_keyring)

    # Same state directory, brand new process. Nobody logs in.
    with _boot(directory) as restarted:
        health = restarted.get("/healthz").json()
        assert health["status"] == "ok", health
        assert health["details"]["initialized"] is True
        # 401, not 503, is the assertion that matters. A locked root
        # answers 503 "Locked" to everything because it has no signing
        # key to verify against; 401 means the key is there and this
        # particular caller just did not present a token.
        assert restarted.get("/v1/control/status").status_code == 401
        assert fake_keyring.reads >= 1

        # And it is genuinely usable, not merely un-503ing.
        token = login(restarted)
        restarted.headers["Authorization"] = f"Bearer {token}"
        assert restarted.get("/v1/control/status").status_code == 200


def test_prompt_on_startup_never_reads_the_keyring(
    tmp_path: Path, fake_keyring: FakeKeyring
) -> None:
    """The default must not touch an OS secret store at all.

    Not just "does not use the value" - does not look. The mode's whole
    claim is that the key is never written to disk, and a read would
    mean there was something there to read.
    """
    directory = tmp_path / "install"
    with _boot(directory) as client:
        assert (
            client.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
        )
        login(client)
    assert fake_keyring.stored is None
    reads_before = fake_keyring.reads

    with _boot(directory) as restarted:
        assert restarted.get("/healthz").json()["details"]["initialized"] is True
    assert fake_keyring.reads == reads_before


def test_a_stale_stored_key_is_discarded_rather_than_trusted(
    tmp_path: Path, fake_keyring: FakeKeyring
) -> None:
    """The failure this codebase will actually meet.

    A keyring entry outlives the install directory. Wipe the state and
    re-initialize under a different passphrase - which is what every
    fresh test install is - and the old entry is still there, deriving
    to a key that opens nothing. Keeping it would leave a root that
    reports itself unlocked and hands out tokens signed with a key
    nothing accepts, so the entry is dropped and the operator is told to
    log in.
    """
    first = tmp_path / "first"
    with _boot(first) as client:
        _initialize_and_enable(client, fake_keyring)
    stale = fake_keyring.stored
    assert stale is not None

    # A different install, same machine, different passphrase - and the
    # stale entry still sitting in the keyring.
    second = tmp_path / "second"
    with _boot(second) as other:
        assert (
            other.post("/v1/auth/initialize", json={"passphrase": "a-different-one"}).status_code
            == 204
        )
        token = login(other, "a-different-one")
        other.headers["Authorization"] = f"Bearer {token}"
        other.patch("/v1/config", json={"securityMode": "os_keyring"})
    # Put the stale value back, as a real machine would have it.
    fake_keyring.stored = stale

    with _boot(second) as restarted:
        # Degraded, not unlocked, and not crashed either.
        health = restarted.get("/healthz").json()
        assert health["details"]["initialized"] is True
        assert restarted.get("/v1/control/status").status_code == 503
    # And the useless secret is gone rather than left to fail again.
    assert fake_keyring.stored is None


def test_leaving_os_keyring_wipes_the_stored_key(tmp_path: Path, fake_keyring: FakeKeyring) -> None:
    """Moving to the stronger boundary has to mean it.

    A root that still auto-unlocked after the operator chose
    `prompt_on_startup` would contradict the mode it reports, which is
    the worst kind of security setting.
    """
    with _boot(tmp_path / "install") as client:
        _initialize_and_enable(client, fake_keyring)
        assert (
            client.patch("/v1/config", json={"securityMode": "prompt_on_startup"}).status_code
            == 200
        )
    assert fake_keyring.stored is None


def test_a_broken_keyring_backend_leaves_a_usable_root(
    tmp_path: Path, fake_keyring: FakeKeyring
) -> None:
    """Headless Linux with no secret service, or a locked Keychain.

    The root must come up and be repairable. Refusing to start because
    an optional convenience was unavailable would lock the operator out
    of the endpoint they would fix it from - the degraded-mode rule,
    applied to the trust root.
    """
    directory = tmp_path / "install"
    with _boot(directory) as client:
        _initialize_and_enable(client, fake_keyring)

    fake_keyring.fail_on_read = RuntimeError("no secret service on this host")
    with _boot(directory) as restarted:
        # Locked, so degraded - but answering, and `/v1/auth/login` is
        # reachable, which is the way out.
        assert restarted.get("/healthz").status_code == 200
        assert restarted.post("/v1/auth/login", json={"passphrase": PASSPHRASE}).status_code == 200


def test_the_service_name_is_not_the_agents(tmp_path: Path) -> None:
    """An agent and a control root share a host and hold DIFFERENT keys.

    Same service and username would have each overwrite the other, and
    the symptom would be an auto-unlock that opens nothing - blamed on
    the passphrase, which would be the wrong place to look.
    """
    assert keyring_store.SERVICE == "eugene-plexus-control"
    assert keyring_store.SERVICE != "eugene-plexus-agent"
