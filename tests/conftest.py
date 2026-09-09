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
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_control.app import create_app
from eugene_plexus_control.log_store import LogStore
from eugene_plexus_control.settings import Settings
from eugene_plexus_control.state_machine import ROLE_ACTIVE, ROLE_STANDBY, StateMachine

PASSPHRASE = "correct-horse-battery-staple"


def settings_for(
    tmp_path: Path, *, role: str = "control", active_url: str | None = None
) -> Settings:
    return Settings(
        config_file=tmp_path / "control.yaml",
        state_dir=tmp_path / "state",
        role=role,
        active_url=active_url,
        follow_interval_seconds=0.05,
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


@pytest.fixture
def uninitialized_client(tmp_path: Path) -> Iterator[TestClient]:
    app = create_app(settings_for(tmp_path / "fresh"))
    with TestClient(app) as client:
        yield client


def login(client: TestClient, passphrase: str = PASSPHRASE) -> str:
    response = client.post("/v1/auth/login", json={"passphrase": passphrase})
    assert response.status_code == 200, response.text
    return str(response.json()["token"])


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
