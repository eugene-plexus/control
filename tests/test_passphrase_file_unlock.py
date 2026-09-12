"""Passphrase-file auto-unlock for the trust root.

`os_keyring` promised an install that comes back from a power cut with
nobody present, and **cannot keep that promise in a container**, because
a container has no Credential Manager, no Keychain and no Secret Service
daemon. A control plane in Docker therefore came back from every restart
initialized-but-locked, routing nothing, until a person opened a browser
— found on a real UnRAID install, 2026-09-12, on its first restart.

These are the things about the file path that are easy to get wrong, and
two of them are deliberate divergences from the keyring path rather than
copies of it.

The file is real here — `tmp_path`, not a fake — because every
interesting failure is a property of a real file: absent, wrong content,
a trailing newline somebody's shell added. The unseal is real too: real
Argon2id, real sealed values.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_control import passphrase_file
from eugene_plexus_control.app import create_app

from .conftest import PASSPHRASE, login, settings_for


def _boot(directory: Path, secret: Path | None = None) -> TestClient:
    return TestClient(create_app(settings_for(directory, passphrase_file=secret)))


def _initialize_and_enable(client: TestClient, passphrase: str = PASSPHRASE) -> None:
    assert client.post("/v1/auth/initialize", json={"passphrase": passphrase}).status_code == 204
    token = login(client, passphrase)
    client.headers["Authorization"] = f"Bearer {token}"
    assert client.patch("/v1/config", json={"securityMode": "passphrase_file"}).status_code == 200


def test_a_restart_auto_unlocks_from_a_mounted_file(tmp_path: Path) -> None:
    """The whole point: a trust root that comes back without a person, on
    a host where the keyring path cannot work.

    Asserts 401 rather than 503 on a protected route for the same reason
    the keyring test does — a locked root answers 503 to everything
    because it holds no signing key to verify against, so 401 is what
    proves the key is present and only the caller is anonymous.
    """
    directory = tmp_path / "install"
    secret = tmp_path / "passphrase"
    secret.write_text(PASSPHRASE, encoding="utf-8")

    with _boot(directory, secret) as client:
        _initialize_and_enable(client)

    # Same state directory, brand new process, nobody logs in.
    with _boot(directory, secret) as restarted:
        health = restarted.get("/healthz").json()
        assert health["status"] == "ok", health
        assert health["details"]["initialized"] is True
        assert restarted.get("/v1/control/status").status_code == 401

        # Genuinely usable, not merely un-503ing.
        token = login(restarted)
        restarted.headers["Authorization"] = f"Bearer {token}"
        assert restarted.get("/v1/control/status").status_code == 200


def test_prompt_on_startup_never_reads_the_file(tmp_path: Path) -> None:
    """The default must not consult the file at all — not "ignores the
    value", does not look. A mode whose claim is that the key never
    leaves process memory cannot be reading a secret off disk."""
    directory = tmp_path / "install"
    secret = tmp_path / "passphrase"
    secret.write_text(PASSPHRASE, encoding="utf-8")

    with _boot(directory, secret) as client:
        assert (
            client.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
        )
        login(client)

    reads: list[Path | None] = []
    original = passphrase_file.read_passphrase

    def spy(path: Path | None) -> str | None:
        reads.append(path)
        return original(path)

    passphrase_file.read_passphrase = spy  # type: ignore[assignment]
    try:
        with _boot(directory, secret) as restarted:
            assert restarted.get("/healthz").json()["details"]["initialized"] is True
            assert restarted.get("/v1/control/status").status_code == 503
    finally:
        passphrase_file.read_passphrase = original  # type: ignore[assignment]
    assert reads == [], "prompt_on_startup read the passphrase file"


def test_a_missing_file_leaves_a_usable_repairable_root(tmp_path: Path) -> None:
    """A secret declared but not mounted is the container mistake, and
    the root must come up anyway.

    Refusing to start because an opt-in convenience was unavailable would
    lock the operator out of the endpoint they would fix it from — the
    degraded-mode rule, applied to the trust root.
    """
    directory = tmp_path / "install"
    secret = tmp_path / "passphrase"
    secret.write_text(PASSPHRASE, encoding="utf-8")
    with _boot(directory, secret) as client:
        _initialize_and_enable(client)

    secret.unlink()
    with _boot(directory, secret) as restarted:
        assert restarted.get("/healthz").status_code == 200
        assert restarted.get("/v1/control/status").status_code == 503
        # The way out is reachable, which is the whole of "repairable".
        assert restarted.post("/v1/auth/login", json={"passphrase": PASSPHRASE}).status_code == 200


def test_the_mode_without_a_path_is_locked_not_crashed(tmp_path: Path) -> None:
    """Selecting the mode in the UI and forgetting the env var.

    The two halves live in different places on purpose — the mode is
    replicated config, the path is bootstrap — so this combination is
    reachable by an operator doing something reasonable.
    """
    directory = tmp_path / "install"
    secret = tmp_path / "passphrase"
    secret.write_text(PASSPHRASE, encoding="utf-8")
    with _boot(directory, secret) as client:
        _initialize_and_enable(client)

    with _boot(directory, None) as restarted:
        assert restarted.get("/healthz").status_code == 200
        assert restarted.get("/v1/control/status").status_code == 503


def test_a_wrong_passphrase_leaves_the_file_alone(tmp_path: Path) -> None:
    """**The deliberate divergence from the keyring path.**

    `keyring_store` discards a stored key that does not open the install,
    and is right to: the agent wrote that entry, it outlives the install
    directory, and a stale one fails as "auto-unlock appeared to work".

    This file is the operator's. It arrived on a secret mount or by hand
    and is quite possibly read-only. Deleting it would destroy
    configuration we did not create in order to report a mistake we can
    describe — so the root stays locked, says so, and the file survives
    untouched.
    """
    directory = tmp_path / "install"
    secret = tmp_path / "passphrase"
    secret.write_text(PASSPHRASE, encoding="utf-8")
    with _boot(directory, secret) as client:
        _initialize_and_enable(client)

    # What pointing a new container at an older install's secret looks like.
    secret.write_text("a-different-installs-passphrase", encoding="utf-8")
    with _boot(directory, secret) as restarted:
        assert restarted.get("/healthz").json()["details"]["initialized"] is True
        assert restarted.get("/v1/control/status").status_code == 503

    assert secret.exists(), "the operator's file must not be deleted"
    assert secret.read_text(encoding="utf-8") == "a-different-installs-passphrase"


def test_a_trailing_newline_is_tolerated(tmp_path: Path) -> None:
    """`echo secret > file` appends one and `printf` does not, so the
    same passphrase reaches us two ways and only one can be right.

    Left untreated this presents as "the passphrase in my file is not
    accepted" with nothing to distinguish it from a typo.
    """
    directory = tmp_path / "install"
    secret = tmp_path / "passphrase"
    secret.write_text(PASSPHRASE, encoding="utf-8")
    with _boot(directory, secret) as client:
        _initialize_and_enable(client)

    for written in (f"{PASSPHRASE}\n", f"{PASSPHRASE}\r\n"):
        secret.write_text(written, encoding="utf-8", newline="")
        with _boot(directory, secret) as restarted:
            assert restarted.get("/v1/control/status").status_code == 401, (
                f"{written!r} should unlock"
            )


def test_only_one_trailing_newline_is_stripped(tmp_path: Path) -> None:
    """`.strip()` is the obvious call and is wrong.

    A passphrase may legitimately begin or end with a space. Trimming it
    would mangle the credential silently and present as a wrong
    passphrase, which is the one diagnosis that sends an operator looking
    in the wrong place.
    """
    spaced = "  outer spaces matter  "
    directory = tmp_path / "install"
    secret = tmp_path / "passphrase"
    secret.write_text(spaced, encoding="utf-8")
    with _boot(directory, secret) as client:
        _initialize_and_enable(client, spaced)

    secret.write_text(f"{spaced}\n", encoding="utf-8", newline="")
    with _boot(directory, secret) as restarted:
        assert restarted.get("/v1/control/status").status_code == 401

    assert passphrase_file._strip_one_trailing_newline(f"{spaced}\n") == spaced
    assert passphrase_file._strip_one_trailing_newline(f"{spaced}\n\n") == f"{spaced}\n"


def test_an_oversized_file_is_refused_rather_than_read(tmp_path: Path) -> None:
    """A passphrase is a passphrase, not whatever got mounted there. The
    cap is what stops a wrong path turning into a large read."""
    secret = tmp_path / "not-a-passphrase"
    secret.write_bytes(b"x" * (passphrase_file.MAX_BYTES + 1))
    assert passphrase_file.read_passphrase(secret) is None


def test_an_empty_file_is_not_an_empty_passphrase(tmp_path: Path) -> None:
    """A secret mount that resolved to nothing must not be tried as a
    credential — the failure should read as "empty file", not as a
    rejected passphrase."""
    secret = tmp_path / "passphrase"
    secret.write_text("", encoding="utf-8")
    assert passphrase_file.read_passphrase(secret) is None
    secret.write_text("\n", encoding="utf-8", newline="")
    assert passphrase_file.read_passphrase(secret) is None


def test_a_locked_root_does_not_probe_its_nodes(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**A locked root cannot authenticate, so it must not call anybody.**

    `node_service_token` returns None without the install's signing key,
    and `probe_all` took a `str | None` and probed anyway — so every
    enrolled node answered `401 Unauthorized`, once per poll interval,
    forever.

    That is worse than not probing, because of where it sends the
    operator: you read `"GET /v1/node HTTP/1.1" 401 Unauthorized` in a
    worker's log and go looking at enrollment and tokens, which are fine.
    The cause is a sealed trust root on a different machine, and it said
    so nowhere. Reported exactly that way on 2026-09-12, next to the
    gateway's own "did not answer /v1/nodes" — one cause, two symptoms,
    neither naming it.

    **The install needs a node in it or this passes vacuously**, since
    the poller skips `probe_all` entirely when there are no targets.
    """
    import logging
    import time

    from eugene_plexus_control.nodes_client import NodesClient

    calls: list[object] = []

    async def spy(self: object, targets: object, token: object) -> dict[str, object]:
        calls.append(token)
        return {}

    monkeypatch.setattr(NodesClient, "probe_all", spy)

    directory = tmp_path / "install"
    secret = tmp_path / "passphrase"
    secret.write_text(PASSPHRASE, encoding="utf-8")

    with _boot(directory, secret) as client:
        _initialize_and_enable(client)
        # The poller re-reads this every pass, and the default is long
        # enough that the first (empty) pass would be the only one this
        # test ever saw.
        assert client.patch("/v1/config", json={"nodePollIntervalSeconds": 0.1}).status_code == 200
        minted = client.post("/v1/nodes/join-token", json={"nodeName": "worker"}).json()
        enrolled = client.post(
            "/v1/nodes/enroll",
            json={
                "token": minted["token"],
                "name": "worker",
                "publicKey": "0" * 43 + "=",
                "url": "http://192.0.2.10:8079",
            },
        )
        assert enrolled.status_code in (200, 201), enrolled.text

    # Now boot the same install with the secret gone: initialized,
    # locked, and still holding that node.
    secret.unlink()
    with (
        caplog.at_level(logging.WARNING, logger="eugene_plexus_control.app"),
        _boot(directory, secret) as locked,
    ):
        assert locked.get("/v1/control/status").status_code == 503
        time.sleep(0.4)

    assert calls == [], f"a locked root called its nodes with {calls!r}"

    # **The anti-vacuity guard.** The poller skips `probe_all` entirely
    # when there are no targets, so "it did not call anyone" is also true
    # of an empty install. The message has to say it had one node and
    # chose not to.
    said = [r.getMessage() for r in caplog.records]
    assert any("not polling 1 node(s)" in m for m in said), (
        f"expected the root to say it is locked and name the node it skipped; got {said!r}"
    )
