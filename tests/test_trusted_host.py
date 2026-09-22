"""First-run setup and sign-in answer only to a name a browser could mean.

**The attack is DNS rebinding, and first-run setup is first-come.** A web
page on `evil.example.com` resolves its own name to an address it
controls, then -- once the browser has loaded it -- re-resolves the same
name to `127.0.0.1`. The page is now same-origin with this machine's
loopback as far as the browser can tell, and the one thing it can reach
that has no credential in front of it is `POST /v1/auth/initialize`,
which gives the install to whoever calls it first. Login is the other
unauthenticated door, and it doubles as the unlock.

What the browser cannot hide is the name: the rebound request still
carries `Host: evil.example.com`. So those two routes refuse a name that
is not an address, a local network name or this machine's own -- and
nothing else in this component does, because agents and standbys dial it
by whatever `controlUrl` an operator typed, custom DNS names included.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from eugene_plexus_control.app import create_app

from .conftest import PASSPHRASE, login, settings_for

ALLOWED_ENV = "EUGENE_PLEXUS_CONTROL_ALLOWED_HOSTS"
EVIL = "evil.example.com"
SHORT = "elevenchars"
"""Refused for length (400) by `initialize` AFTER the host check, so an
allowed host answers 400 and a refused one 403 without paying for an
Argon2id derivation per case."""


@pytest.fixture(autouse=True)
def _no_ambient_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's own allowlist must not decide these tests."""
    monkeypatch.delenv(ALLOWED_ENV, raising=False)


@pytest.fixture
def fresh(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(settings_for(tmp_path / "fresh"))) as client:
        yield client


@pytest.fixture
def initialized(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(settings_for(tmp_path / "init"))) as client:
        assert (
            client.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
        )
        yield client


def _host(name: str) -> dict[str, str]:
    return {"host": name}


def test_a_rebound_name_cannot_claim_a_fresh_install(fresh: TestClient) -> None:
    response = fresh.post(
        "/v1/auth/initialize", json={"passphrase": PASSPHRASE}, headers=_host(EVIL)
    )

    assert response.status_code == 403, response.text
    problem = response.json()["detail"]
    assert problem["status"] == 403
    assert EVIL in problem["detail"]
    assert ALLOWED_ENV in problem["detail"]
    assert fresh.get("/v1/auth/status").json()["initialized"] is False


def test_a_rebound_name_cannot_sign_in_or_unlock(initialized: TestClient) -> None:
    response = initialized.post(
        "/v1/auth/login", json={"passphrase": PASSPHRASE}, headers=_host(f"{EVIL}:8083")
    )
    assert response.status_code == 403, response.text
    assert EVIL in response.json()["detail"]["detail"]


def test_the_port_and_a_trailing_dot_do_not_disguise_the_name(fresh: TestClient) -> None:
    for name in (f"{EVIL}:80", f"{EVIL}.", "EVIL.Example.COM"):
        response = fresh.post(
            "/v1/auth/initialize", json={"passphrase": SHORT}, headers=_host(name)
        )
        assert response.status_code == 403, name


@pytest.mark.parametrize(
    "name",
    [
        "127.0.0.1:8083",
        "192.168.16.252:8283",
        "[::1]",
        "[::1]:8083",
        "localhost",
        "localhost:8083",
        "ui.localhost",
        "tower",
        "tower:8283",
        "nas.local",
        "box.lan",
        "box.home.arpa",
        "control.internal",
        "x.tail1.ts.net",
        "X.Tail1.TS.NET:8083",
        "testserver",
    ],
)
def test_addresses_and_local_names_are_allowed(fresh: TestClient, name: str) -> None:
    response = fresh.post("/v1/auth/initialize", json={"passphrase": SHORT}, headers=_host(name))
    assert response.status_code == 400, f"{name}: {response.text}"
    assert "12 characters" in response.text


def test_an_allowed_host_really_initializes(fresh: TestClient) -> None:
    response = fresh.post(
        "/v1/auth/initialize", json={"passphrase": PASSPHRASE}, headers=_host("127.0.0.1:8083")
    )
    assert response.status_code == 204, response.text


def test_this_machines_own_name_is_allowed(
    fresh: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dotted hostname -- `tower.example.com` -- is this machine, and
    the browser that typed it is on it or next to it."""
    from eugene_plexus_control import trusted_host

    monkeypatch.setattr(trusted_host, "_machine_names", lambda: frozenset({"tower.example.com"}))
    response = fresh.post(
        "/v1/auth/initialize",
        json={"passphrase": SHORT},
        headers=_host("Tower.Example.com:8083"),
    )
    assert response.status_code == 400, response.text


def test_the_allowlist_admits_a_name_of_the_operators_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ALLOWED_ENV, " eugene.example.org , other.example.org:8083 ")
    with TestClient(create_app(settings_for(tmp_path / "listed"))) as client:
        ok = client.post(
            "/v1/auth/initialize", json={"passphrase": SHORT}, headers=_host("eugene.example.org")
        )
        also = client.post(
            "/v1/auth/initialize", json={"passphrase": SHORT}, headers=_host("other.example.org")
        )
        still = client.post("/v1/auth/initialize", json={"passphrase": SHORT}, headers=_host(EVIL))
    assert ok.status_code == 400, ok.text
    assert also.status_code == 400, also.text
    assert still.status_code == 403, still.text


def test_a_star_turns_the_check_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ALLOWED_ENV, "*")
    with TestClient(create_app(settings_for(tmp_path / "star"))) as client:
        response = client.post(
            "/v1/auth/initialize", json={"passphrase": SHORT}, headers=_host(EVIL)
        )
    assert response.status_code == 400, response.text


def test_a_request_with_no_host_header_is_let_through(fresh: TestClient) -> None:
    """HTTP/1.0 sends none and a browser always sends one, so an absent
    header is not the attack. No HTTP client here can omit it, so the
    dependency is called on a scope built without one."""
    from starlette.requests import Request

    from eugene_plexus_control.trusted_host import require_trusted_host

    bare = Request({"type": "http", "headers": [], "app": fresh.app})
    assert require_trusted_host(bare) is None

    named = Request({"type": "http", "headers": [(b"host", EVIL.encode())], "app": fresh.app})
    with pytest.raises(HTTPException) as refused:
        require_trusted_host(named)
    assert refused.value.status_code == 403


def test_the_machine_routes_answer_any_name(initialized: TestClient) -> None:
    """Agents and standbys dial this root by an operator-typed
    `controlUrl`, which may be any DNS name at all."""
    token = login(initialized)
    evil = _host(EVIL)

    assert initialized.get("/healthz", headers=evil).status_code == 200
    assert initialized.get("/v1/auth/status", headers=evil).status_code == 200
    nodes = initialized.get("/v1/nodes", headers={**evil, "Authorization": f"Bearer {token}"})
    assert nodes.status_code == 200, nodes.text
    status = initialized.get(
        "/v1/control/status", headers={**evil, "Authorization": f"Bearer {token}"}
    )
    assert status.status_code == 200, status.text
