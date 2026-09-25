"""The union views and runtime forwarding, against fake agents.

Fake agents rather than real ones, and the seam is chosen deliberately:
what is under test here is how the control root **assembles and
attributes** what nodes tell it, not whether an agent supervises
correctly. The agent's own suite owns that.

What the fakes make possible is the case that matters and that a real
pair of hosts makes awkward to arrange: **one node answers and the other
does not.** A dashboard showing fewer runtimes than exist, with no
indication why, is worse than one that names the host it could not
reach — and nothing gets un-declared because a poll failed, because that
is `down`, not `out`.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI, Response
from fastapi.testclient import TestClient

from eugene_plexus_control import tokens


def _fake_agent(
    *,
    runtimes: list[dict[str, Any]],
    components: list[dict[str, Any]],
    refuse_create: str | None = None,
) -> FastAPI:
    """The three endpoints the control root calls on an agent.

    `refuse_create` is a parameter rather than a second route
    registration because FastAPI matches the first route it has for a
    path — registering a second POST /v1/runtimes silently does
    nothing, which is a fake that quietly agrees with you.
    """
    app = FastAPI()
    accepted: list[dict[str, Any]] = []
    app.state.accepted = accepted

    @app.get("/v1/runtimes")
    async def list_runtimes() -> dict[str, Any]:
        return {"runtimes": runtimes}

    @app.get("/v1/components")
    async def list_components() -> dict[str, Any]:
        return {"components": components}

    @app.post("/v1/runtimes")
    async def create_runtime(spec: dict[str, Any], response: Response) -> dict[str, Any]:
        if refuse_create is not None:
            response.status_code = 400
            return {"detail": refuse_create}
        accepted.append(spec)
        response.status_code = 201
        return {
            "name": spec["name"],
            "engine": spec.get("engine", "llama_cpp"),
            "modelPath": spec.get("modelPath", "/m.gguf"),
            "modelAlias": spec.get("modelAlias"),
            "status": "starting",
            "url": "http://127.0.0.1:8090",
        }

    return app


class _Server:
    def __init__(self, app: FastAPI) -> None:
        self.app = app
        config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> _Server:
        self.thread.start()
        while not self.server.started:
            if not self.thread.is_alive():  # pragma: no cover
                raise RuntimeError("fake agent died during startup")
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.servers[0].sockets[0].getsockname()[1]}"


@pytest.fixture
def agent_a() -> Iterator[_Server]:
    app = _fake_agent(
        runtimes=[
            {
                "name": "qwen",
                "engine": "llama_cpp",
                "modelPath": "/m/qwen.gguf",
                "modelAlias": "qwen3.6-27b",
                "status": "ready",
                "url": "http://10.0.0.7:8090",
                # A node the agent claims for itself, which must be
                # ignored: an agent supervises only its own host, so a
                # node it reported could contradict reality.
                "node": "a-lie",
            }
        ],
        components=[
            {
                "name": "left",
                "kind": "inference-driver",
                "url": "http://10.0.0.7:8081",
                "status": "ready",
            }
        ],
    )
    with _Server(app) as server:
        yield server


def _enroll(client: TestClient, name: str, url: str) -> None:
    import base64

    # The agent's address rides on the enrollment from M7. Before that the
    # exchange never carried it and this helper appended a second
    # `enrollNode` entry by hand — which is how every really-enrolled node
    # came to have no URL without any test noticing.
    token = client.post("/v1/nodes/join-token").json()["token"]
    response = client.post(
        "/v1/nodes/enroll",
        json={
            "token": token,
            "name": name,
            "publicKey": base64.b64encode(name.encode().ljust(32, b"0")).decode(),
            "signingPublicKey": base64.b64encode(name.encode().ljust(32, b"1")).decode(),
            "tokenPublicKey": tokens.public_b64(tokens.generate_private_key()),
            "url": url,
        },
    )
    assert response.status_code == 201, response.text


def test_the_union_view_tags_every_runtime_with_the_node_that_reported_it(
    active_client: TestClient, agent_a: _Server
) -> None:
    """And overrides whatever the agent said about `node`."""
    _enroll(active_client, "gpu-box", agent_a.url)

    response = active_client.get("/v1/runtimes")
    assert response.status_code == 200
    body = response.json()
    assert len(body["runtimes"]) == 1
    runtime = body["runtimes"][0]
    assert runtime["node"] == "gpu-box", "the node we asked, never the node it claimed"
    assert runtime["modelAlias"] == "qwen3.6-27b"
    assert body["unreachableNodes"] == []


def test_an_unreachable_node_is_named_rather_than_omitted(
    active_client: TestClient, agent_a: _Server
) -> None:
    """The whole reason the response has an `unreachableNodes` field.

    One host answers, one does not. The view is honestly partial instead
    of quietly short.
    """
    _enroll(active_client, "gpu-box", agent_a.url)
    _enroll(active_client, "attic", "http://127.0.0.1:1")

    body = active_client.get("/v1/runtimes").json()
    assert [r["node"] for r in body["runtimes"]] == ["gpu-box"]
    assert body["unreachableNodes"] == ["attic"]


def test_no_node_reachable_is_503_and_not_an_empty_list(active_client: TestClient) -> None:
    """Empty means nothing is declared; 503 means we cannot tell.
    Conflating them would make a total network outage look like a
    freshly wiped install."""
    _enroll(active_client, "attic", "http://127.0.0.1:1")
    response = active_client.get("/v1/components")
    assert response.status_code == 503
    assert "cannot currently see" in response.text


def test_a_declared_component_a_node_does_not_report_is_still_shown(
    active_client: TestClient, agent_a: _Server
) -> None:
    """A real discrepancy an operator should see, not a row to hide."""
    _enroll(active_client, "gpu-box", agent_a.url)
    machine = active_client.app.state.machine  # type: ignore[attr-defined]
    machine.append(
        "putComponent",
        {"node": "gpu-box", "name": "ghost", "kind": "gateway", "url": "http://10.0.0.7:8080"},
    )

    body = active_client.get("/v1/components").json()
    by_name = {c["name"]: c for c in body["components"]}
    assert by_name["left"]["status"] == "ready"
    assert by_name["ghost"]["status"] == "undeclared_on_node"


def test_creating_a_runtime_forwards_to_the_node(
    active_client: TestClient, agent_a: _Server
) -> None:
    """Accepted here, forwarded there. That is what makes "the operator
    talks to the control root" true rather than aspirational."""
    _enroll(active_client, "gpu-box", agent_a.url)

    response = active_client.post(
        "/v1/runtimes",
        json={
            "node": "gpu-box",
            "spec": {
                "name": "gemma",
                "engine": "llama_cpp",
                "modelPath": "/m/gemma.gguf",
                "modelAlias": "gemma-9b",
            },
        },
    )
    assert response.status_code == 201, response.text
    assert response.json()["node"] == "gpu-box"
    assert agent_a.app.state.accepted[0]["name"] == "gemma"

    # And the declaration is now in the log, so a standby inherits it.
    machine = active_client.app.state.machine  # type: ignore[attr-defined]
    assert machine.state.runtimes["gpu-box/gemma"].spec["modelPath"] == "/m/gemma.gguf"


def test_a_runtime_the_agent_refuses_is_not_declared(active_client: TestClient) -> None:
    """Forward first, record second.

    Validation that needs the host — does this model path exist, is this
    port free — can only happen on the agent, so recording a declaration
    the agent then refused would put a runtime in the install's topology
    that never existed anywhere.
    """
    refusing = _fake_agent(runtimes=[], components=[], refuse_create="no such model path")

    with _Server(refusing) as server:
        _enroll(active_client, "gpu-box", server.url)
        machine = active_client.app.state.machine  # type: ignore[attr-defined]
        before = machine.state.index

        response = active_client.post(
            "/v1/runtimes",
            json={"node": "gpu-box", "spec": {"name": "nope", "modelPath": "/missing.gguf"}},
        )
        assert response.status_code == 502
        assert machine.state.index == before, "a refused runtime must not be declared"


def test_declaring_on_an_unknown_node_is_a_404(active_client: TestClient) -> None:
    response = active_client.post(
        "/v1/runtimes", json={"node": "nowhere", "spec": {"name": "x", "modelPath": "/m.gguf"}}
    )
    assert response.status_code == 404


def test_a_runtime_spec_without_a_name_is_refused(
    active_client: TestClient, agent_a: _Server
) -> None:
    """`name` is what the declaration is keyed by, so a spec without one
    has nowhere to live in applied state."""
    _enroll(active_client, "gpu-box", agent_a.url)
    response = active_client.post(
        "/v1/runtimes", json={"node": "gpu-box", "spec": {"modelPath": "/m.gguf"}}
    )
    assert response.status_code == 400
