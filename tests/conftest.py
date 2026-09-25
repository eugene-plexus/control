"""Shared fixtures.

Every test that needs an app gets a **real** one: real log files in a
tmp_path, real Argon2id, real sealing. Nothing about the log core is
mocked, because the thing under test is what happens to bytes on disk
and a fake store would agree with whatever we believed when we wrote it.

The one concession to speed is that the passphrase is short. Argon2id at
the configured cost takes about a tenth of a second per call, and a
suite that logs in a hundred times would spend ten seconds proving
nothing about the KDF.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_control import sealing, tokens
from eugene_plexus_control.app import create_app
from eugene_plexus_control.log_store import LogStore
from eugene_plexus_control.settings import Settings
from eugene_plexus_control.state_machine import ROLE_ACTIVE, ROLE_STANDBY, StateMachine

PASSPHRASE = "correct-horse-battery-staple"


def settings_for(
    tmp_path: Path,
    *,
    role: str = "control",
    active_url: str | None = None,
    passphrase_file: Path | None = None,
) -> Settings:
    return Settings(
        config_file=tmp_path / "control.yaml",
        state_dir=tmp_path / "state",
        role=role,
        active_url=active_url,
        follow_interval_seconds=0.05,
        passphrase_file=passphrase_file,
    )


@pytest.fixture
def active_client(tmp_path: Path) -> Iterator[TestClient]:
    """An active control root, initialized and logged in.

    Yields a client whose Authorization header is already set, because
    almost every test needs one and the setup is four lines of noise
    otherwise.
    """
    app = create_app(settings_for(tmp_path / "active"))
    with TestClient(app) as client:
        response = client.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE})
        assert response.status_code == 204, response.text
        token = login(client)
        client.headers["Authorization"] = f"Bearer {token}"
        yield client


@pytest.fixture(autouse=True)
def _memoise_keyring_probe() -> Iterator[None]:
    """`GET /v1/auth/status` probes the OS keyring once per process; a test
    must not reach the developer's real one. Memoised to False for every
    test; a test about the probe resets the cache after installing its
    fake."""
    from eugene_plexus_control import keyring_store

    keyring_store._probe_result = False
    keyring_store._probe_done = True
    yield
    keyring_store.reset_probe_cache()


@pytest.fixture
def uninitialized_client(tmp_path: Path) -> Iterator[TestClient]:
    app = create_app(settings_for(tmp_path / "fresh"))
    with TestClient(app) as client:
        yield client


def login(client: TestClient, passphrase: str = PASSPHRASE) -> str:
    response = client.post("/v1/auth/login", json={"passphrase": passphrase})
    assert response.status_code == 200, response.text
    # `sessionToken`, from common.yaml's shared AuthLoginResponse - the
    # same field the agent returns. This used to be `token`, from a local
    # duplicate schema. `/v1/nodes/join-token` still returns `token`; it
    # is a JoinToken and a different thing.
    return str(response.json()["sessionToken"])


@dataclass(frozen=True)
class NodeKeys:
    """Everything an agent generates before it enrolls, private halves included.

    A test that enrolls a node holds the node's own token key, so it can
    mint exactly what that node could and nothing more (2026-09-25).
    """

    name: str
    public: str
    signing_private: str
    signing_public: str
    token: tokens.Signer

    def body(self, join_token: str, **extra: object) -> dict[str, object]:
        return {
            "token": join_token,
            "name": self.name,
            "publicKey": self.public,
            "signingPublicKey": self.signing_public,
            "tokenPublicKey": tokens.public_b64(self.token.key),
            **extra,
        }

    def service_token(
        self, *, sub: str = "agent", aud: tuple[str, ...] = ("control",), ttl: int = 600
    ) -> str:
        token, _ = self.token.mint(typ=tokens.TYP_SERVICE, sub=sub, aud=aud, ttl_seconds=ttl)
        return token


def node_keys(name: str) -> NodeKeys:
    identity = sealing.generate_control_identity()
    return NodeKeys(
        name=name,
        public=sealing.generate_sealing_keypair().public,
        signing_private=identity.private,
        signing_public=identity.public,
        token=tokens.Signer(key=tokens.generate_private_key(), issuer=f"node:{name}"),
    )


def mint_join_token(client: TestClient, **body: object) -> str:
    response = client.post("/v1/nodes/join-token", json=body or None)
    assert response.status_code == 201, response.text
    return str(response.json()["token"])


def enroll(client: TestClient, name: str, **extra: object) -> tuple[NodeKeys, Any]:
    """Enroll a node with fresh keys; returns the keys and the response."""
    keys = node_keys(name)
    grants = extra.pop("grants", None)
    join = mint_join_token(client, **({"grants": grants} if grants else {}))
    return keys, client.post("/v1/nodes/enroll", json=keys.body(join, **extra))


def session_after_history(app: Any, root_key: bytes) -> dict[str, str]:
    """A session signed by the key a written history rotated to.

    A history that rotates the root token key ends every session signed
    before it, exactly as a real rotation does, because verification reads
    the root's public key from applied state. The sealed key in that
    history is a placeholder no passphrase opens, so the test adopts the
    real key it wrote and signs with it directly.
    """
    app.state.auth_state.set_signing_key(root_key)
    signer = app.state.auth_state.signer()
    token, _ = signer.mint(
        typ=tokens.TYP_SESSION, sub="operator", aud=["control"], ttl_seconds=3600
    )
    return {"Authorization": f"Bearer {token}"}


def machine_at(directory: Path, *, role: str = ROLE_ACTIVE) -> StateMachine:
    """A bare state machine over a directory, with no HTTP in the way.

    Used where the point is the log itself rather than the surface over
    it — replay, compaction, torn tails.
    """
    machine = StateMachine(LogStore(directory), role=role)
    machine.load()
    return machine


def standby_at(directory: Path) -> StateMachine:
    return machine_at(directory, role=ROLE_STANDBY)


def read_log_file(directory: Path) -> list[dict[str, Any]]:
    path = directory / "log.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
