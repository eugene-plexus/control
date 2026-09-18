"""R1.2: the trust root's login limiter keys on the browser, not the proxy.

Roadmap `docs/design/release-roadmap.md` §2.2, finding review §6.1 #1.

The root is reached through a node agent's `/api/proxy/control/...`, so
its peer is loopback for every caller there has ever been. The checks
that matter run through the *actual* `ProxyHeadersMiddleware` uvicorn
installs, configured from the *actual* `uvicorn.Config` this package's
`main()` builds: the defect lives in that pair and a bare `TestClient`
cannot see it.

**And this root's login is the one that matters most.** Since
2026-09-13 signing in unlocks a sealed control root, so a limiter that
one mistyping person can empty for the whole install is a limiter that
locks everyone out of the screen they reach when nothing else works.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from eugene_plexus_control import peer
from eugene_plexus_control.app import create_app

from .conftest import PASSPHRASE, settings_for

WRONG = "not the passphrase"


def entrypoint_kwargs(tmp_path: Path) -> dict[str, Any]:
    """Whatever this package's own `main()` hands uvicorn.

    Driving `main()` rather than reading the source: the subject is what
    the entrypoint passes, and a text check goes green the moment
    somebody writes the kwarg into a comment.
    """
    from eugene_plexus_control import __main__ as entry

    captured: dict[str, Any] = {}
    original = entry.uvicorn.run
    env = dict(os.environ)
    try:
        entry.uvicorn.run = lambda app, **kwargs: captured.update(kwargs)  # type: ignore[assignment]
        os.environ["EUGENE_PLEXUS_CONTROL_CONFIG_FILE"] = str(tmp_path / "entry" / "control.yaml")
        os.environ["EUGENE_PLEXUS_CONTROL_STATE_DIR"] = str(tmp_path / "entry" / "state")
        os.environ["EUGENE_PLEXUS_CONTROL_BIND_PORT"] = "18083"
        entry.main()
    finally:
        entry.uvicorn.run = original  # type: ignore[assignment]
        os.environ.clear()
        os.environ.update(env)
    return captured


def test_the_entrypoint_trusts_no_forwarding_header(tmp_path: Path) -> None:
    """uvicorn's default is `forwarded_allow_ips="127.0.0.1"` — which is
    every proxied caller in the install."""
    assert entrypoint_kwargs(tmp_path)["forwarded_allow_ips"] == []


def served(tmp_path: Path, peer_host: str) -> Iterator[TestClient]:
    """The app wrapped the way this package's own entrypoint serves it.

    `trusted_hosts` comes from the entrypoint rather than from a literal
    here, so the reproduction is of the deployed pair and not of an
    argument this file chose.
    """
    app = create_app(settings_for(tmp_path / "root"))
    wrapped = ProxyHeadersMiddleware(
        app, trusted_hosts=entrypoint_kwargs(tmp_path).get("forwarded_allow_ips", "127.0.0.1")
    )
    with TestClient(wrapped, client=(peer_host, 43210)) as client:
        assert (
            client.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
        )
        yield client


def _login(client: TestClient, passphrase: str, **headers: str) -> int:
    return client.post(
        "/v1/auth/login", json={"passphrase": passphrase}, headers=headers or None
    ).status_code


def test_a_supplied_forwarded_header_does_not_buy_a_fresh_bucket(tmp_path: Path) -> None:
    """Five wrong passphrases each claiming a different origin, then a
    sixth. Before this slice uvicorn rewrote the peer from
    `X-Forwarded-For` — the peer being loopback, which through the
    agent's proxy it always is — so those were five buckets of one."""
    for client in served(tmp_path, "127.0.0.1"):
        for i in range(5):
            assert _login(client, WRONG, **{"x-forwarded-for": f"203.0.113.{i}"}) == 401
        assert _login(client, WRONG, **{"x-forwarded-for": "203.0.113.99"}) == 429


def test_our_own_header_does_separate_the_buckets(tmp_path: Path) -> None:
    """The half that needs no attacker: one browser's mistakes stop
    being every browser's lockout."""
    for client in served(tmp_path, "127.0.0.1"):
        for _ in range(5):
            assert _login(client, WRONG, **{peer.PEER_HEADER: "192.168.1.50"}) == 401
        assert _login(client, WRONG, **{peer.PEER_HEADER: "192.168.1.50"}) == 429
        assert _login(client, WRONG, **{peer.PEER_HEADER: "192.168.1.51"}) == 401
        assert _login(client, PASSPHRASE, **{peer.PEER_HEADER: "192.168.1.51"}) == 200


def test_an_off_host_caller_cannot_name_its_own_bucket(tmp_path: Path) -> None:
    """Off-host the TCP peer is the evidence, which is what makes a
    forged header worthless from anywhere it could be forged."""
    for client in served(tmp_path, "198.51.100.7"):
        for i in range(5):
            assert _login(client, WRONG, **{peer.PEER_HEADER: f"10.0.0.{i}"}) == 401
        assert _login(client, WRONG, **{peer.PEER_HEADER: "10.0.0.99"}) == 429


@pytest.mark.parametrize(
    ("tcp", "supplied", "expected"),
    [
        ("127.0.0.1", "192.168.1.9", "192.168.1.9"),
        ("::1", "192.168.1.9", "192.168.1.9"),
        ("::ffff:127.0.0.1", "192.168.1.9", "192.168.1.9"),
        ("198.51.100.7", "192.168.1.9", "198.51.100.7"),
        ("127.0.0.1", "not-an-address", "127.0.0.1"),
        ("127.0.0.1", None, "127.0.0.1"),
        ("testclient", "192.168.1.9", "testclient"),
    ],
)
def test_the_header_is_read_only_where_it_can_only_be_ours(
    tcp: str, supplied: str | None, expected: str
) -> None:
    assert peer.peer_of(tcp, supplied) == expected


def test_this_component_reads_no_forwarding_header(tmp_path: Path) -> None:
    """The trust root proxies nothing and must never learn to believe
    one of these. Stated as a check because the failure mode is somebody
    reaching for `X-Forwarded-For` the next time a peer reads wrong."""
    source = Path(__file__).resolve().parents[1] / "src" / "eugene_plexus_control"
    # Quoted, because the English word "forwarded" is all over this
    # codebase's prose and a header is only a header where it is a
    # string somebody could look up.
    named = re.compile(
        "[\"']({})[\"']".format("|".join(re.escape(h) for h in sorted(peer.FORWARDING_HEADERS))),
        re.IGNORECASE,
    )
    offenders = [
        path.relative_to(source).as_posix()
        for path in source.rglob("*.py")
        if path.name != "peer.py" and named.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
