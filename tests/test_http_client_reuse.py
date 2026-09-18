"""One context, one clock — the control root's half.

This component already owned its clients per instance, so the defect
here was narrower than in the gateway or the agent and no less real:
every `httpx.AsyncClient()` built without `verify=` parses certifi's PEM
bundle at ~104 ms of synchronous CPU, and both of these are built during
startup, before anything can be served.

The proxy half matters more. A standby follower polls the active root
and the node probe polls every enrolled agent — all of it this install's
own LAN, none of it reachable through a corporate `HTTP_PROXY` — and the
trust root is the component whose silence looks like every node being
down at once.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from eugene_plexus_control import _http
from eugene_plexus_control.nodes_client import NodesClient


def test_the_node_client_declines_an_ambient_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://corp.proxy:3128")
    monkeypatch.setenv("HTTPS_PROXY", "http://corp.proxy:3128")
    client = NodesClient(timeout_provider=lambda: 5.0)
    assert client._client._mounts == {}, "node probes routed through the user's proxy"


def test_the_ssl_context_is_built_once_per_process() -> None:
    assert _http.ssl_context() is _http.ssl_context()


def test_building_a_client_does_not_reparse_the_bundle() -> None:
    _http.ssl_context()
    started = time.perf_counter()
    NodesClient(timeout_provider=lambda: 5.0)
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert elapsed_ms < 50, (
        f"a client cost {elapsed_ms:.0f} ms; a certifi parse is ~104 ms, so the "
        "shared context is not being used"
    )


def test_no_duration_in_this_component_is_measured_with_monotonic() -> None:
    """`monotonic()` is `GetTickCount64` on the Python both installers
    provision: a 15.6 ms grid under anything reported as a duration."""
    root = Path(__file__).resolve().parents[1] / "src" / "eugene_plexus_control"
    offenders = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if "_generated" not in path.parts and "time.monotonic()" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"these measure with monotonic(): {offenders}"
