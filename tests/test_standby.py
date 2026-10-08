"""The warm standby's credential (docs/design/warm-standby.md, SB1-SB4).

One node, given the `standby` grant by an owner, reads the replication
surface with its own `sub: standby` token, and nothing else does. Every
case here fails against the root before control#5: the replication
routes took an operator session only, and no grant or route existed.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_control.app import create_app
from eugene_plexus_control.applied import (
    OP_PATCH_CONFIG,
    OP_PROMOTE,
    OP_SET_STANDBY,
    ApplyError,
    standby_node,
)
from eugene_plexus_control.nodes_client import NodeProbe
from eugene_plexus_control.replication import Follower, StandbyToken
from eugene_plexus_control.state_machine import StateMachine

from .conftest import PASSPHRASE, NodeKeys, enroll, login, settings_for, standby_at
from .test_replication_follower import _LiveServer

REPLICATION = ("/v1/control/log?after=0", "/v1/control/snapshot")


def _as(client: TestClient, path: str, token: str) -> httpx.Response:
    return client.get(path, headers={"Authorization": f"Bearer {token}"})


@pytest.fixture
def install(active_client: TestClient) -> dict[str, Any]:
    spare, joined = enroll(active_client, "spare")
    assert joined.status_code == 201, joined.text
    gpu, joined = enroll(active_client, "gpu")
    assert joined.status_code == 201, joined.text
    edge, joined = enroll(active_client, "edge", grants=["gateway"])
    assert joined.status_code == 201, joined.text
    return {"client": active_client, "spare": spare, "gpu": gpu, "edge": edge}


def test_only_the_granted_node_reads_replication_with_its_standby_token(
    install: dict[str, Any],
) -> None:
    client: TestClient = install["client"]
    spare: NodeKeys = install["spare"]
    gpu: NodeKeys = install["gpu"]
    edge: NodeKeys = install["edge"]
    standby_token = spare.service_token(sub="standby")
    for path in REPLICATION:
        assert _as(client, path, standby_token).status_code == 401, "no grant yet"

    made = client.put("/v1/nodes/spare/standby")
    assert made.status_code == 202, made.text
    assert (made.json()["reason"], made.json()["standbyNode"]) == ("standby", "spare")
    assert "standby" in client.get("/v1/nodes/spare").json()["grants"]

    for path in REPLICATION:
        assert _as(client, path, standby_token).status_code == 200, path
    refused = {
        "the standby's own agent token": spare.service_token(sub="agent"),
        "another node's standby token": gpu.service_token(sub="standby"),
        "another node's agent token": gpu.service_token(sub="agent"),
        "a granted gateway token": edge.service_token(sub="gateway"),
    }
    for label, token in refused.items():
        for path in REPLICATION:
            answer = _as(client, path, token)
            assert answer.status_code == 401, (label, path, answer.text)

    stopped = client.delete("/v1/nodes/spare/standby")
    assert stopped.status_code == 202, stopped.text
    assert stopped.json().get("standbyNode") is None
    assert "standby" not in client.get("/v1/nodes/spare").json()["grants"]
    # The same token, minutes of life left, is refused at once.
    for path in REPLICATION:
        assert _as(client, path, standby_token).status_code == 401, path


def test_the_grant_is_checked_against_applied_state_on_every_pull(
    install: dict[str, Any],
) -> None:
    """A bundle that still listed the grant must not open the door: the
    route checks the applied state itself (SB2)."""
    client: TestClient = install["client"]
    spare: NodeKeys = install["spare"]
    assert client.put("/v1/nodes/spare/standby").status_code == 202
    token = spare.service_token(sub="standby")
    machine: StateMachine = client.app.state.machine  # type: ignore[attr-defined]
    view = client.app.state.trust.view_for(machine.state)  # type: ignore[attr-defined]
    machine.append(OP_SET_STANDBY, {"node": "spare", "on": False})
    client.app.state.trust.view_for = lambda state: view  # type: ignore[attr-defined]
    answer = _as(client, REPLICATION[0], token)
    assert answer.status_code == 401 and "no machine is the standby" in answer.text


def test_one_standby_at_a_time_never_the_roots_own_machine(install: dict[str, Any]) -> None:
    client: TestClient = install["client"]
    machine: StateMachine = client.app.state.machine  # type: ignore[attr-defined]
    assert client.put("/v1/nodes/spare/standby").status_code == 202
    index = machine.state.index
    second = client.put("/v1/nodes/gpu/standby")
    assert second.status_code == 409 and "'spare' is the standby" in second.text
    again = client.put("/v1/nodes/spare/standby")
    assert again.status_code == 202 and again.json()["standbyNode"] == "spare"
    assert machine.state.index == index, "nothing appended for no change"
    assert client.delete("/v1/nodes/gpu/standby").status_code == 202
    assert machine.state.index == index, "gpu was not the standby"
    assert client.put("/v1/nodes/nobody/standby").status_code == 404

    assert client.delete("/v1/nodes/spare/standby").status_code == 202
    client.app.state.node_probes["gpu"] = NodeProbe(  # type: ignore[attr-defined]
        name="gpu", reachable=True, hosts_control=True
    )
    own = client.put("/v1/nodes/gpu/standby")
    assert own.status_code == 409 and "runs this control root" in own.text
    assert standby_node(machine.state) is None


def test_making_a_standby_takes_a_session(install: dict[str, Any]) -> None:
    client: TestClient = install["client"]
    spare: NodeKeys = install["spare"]
    bare = TestClient(client.app)
    assert bare.put("/v1/nodes/spare/standby").status_code == 401
    agent = {"Authorization": f"Bearer {spare.service_token(sub='agent')}"}
    assert bare.put("/v1/nodes/spare/standby", headers=agent).status_code in (401, 403)
    assert standby_node(client.app.state.machine.state) is None  # type: ignore[attr-defined]


def test_status_reports_the_standby_from_its_own_pulls(install: dict[str, Any]) -> None:
    """SB4: nothing is probed; the standby's `after` is its position."""
    client: TestClient = install["client"]
    spare: NodeKeys = install["spare"]
    assert client.put("/v1/nodes/spare/standby").status_code == 202
    [unheard] = client.get("/v1/control/status").json()["standbys"]
    assert unheard["node"] == "spare" and unheard["reachable"] is False
    assert "lastContactAt" not in unheard or unheard["lastContactAt"] is None
    machine: StateMachine = client.app.state.machine  # type: ignore[attr-defined]
    behind = machine.state.index - 2
    token = spare.service_token(sub="standby")
    assert _as(client, f"/v1/control/log?after={behind}", token).status_code == 200
    [heard] = client.get("/v1/control/status").json()["standbys"]
    assert (heard["appliedIndex"], heard["lagEntries"], heard["reachable"]) == (behind, 2, True)
    assert heard["lastContactAt"]
    # An operator reading the log is not the standby.
    assert client.get("/v1/control/log?after=0").status_code == 200
    [still] = client.get("/v1/control/status").json()["standbys"]
    assert still["appliedIndex"] == behind


def test_set_standby_refuses_what_the_route_refuses(tmp_path: Path) -> None:
    """On the writer the route refuses first; on a replica these mean a log
    this root did not write."""
    from eugene_plexus_control.applied import AppliedState, NodeRecord, apply

    state = AppliedState()
    for name in ("a", "b"):
        state = _with_node(state, NodeRecord(name=name, url=f"http://{name}:8079"))

    def entry(index: int, payload: dict[str, Any]) -> dict[str, Any]:
        return {"index": index, "epoch": 1, "op": OP_SET_STANDBY, "payload": payload}

    with pytest.raises(ApplyError, match="unknown node"):
        apply(state, entry(1, {"node": "zz", "on": True}))
    with pytest.raises(ApplyError, match="on: true or false"):
        apply(state, entry(1, {"node": "a", "on": "yes"}))
    held = apply(state, entry(1, {"node": "a", "on": True}))
    assert held.nodes["a"].grants == ("node", "standby")
    with pytest.raises(ApplyError, match="already the standby"):
        apply(held, entry(2, {"node": "b", "on": True}))
    promoted = apply(held, {"index": 2, "epoch": 2, "op": OP_PROMOTE, "payload": {"node": "a"}})
    assert promoted.nodes["a"].grants == ("node",), "a root follows nobody"
    assert standby_node(promoted) is None


def _with_node(state: Any, record: Any) -> Any:
    from dataclasses import replace

    return replace(state, nodes={**state.nodes, record.name: record})


# --------------------------------------------------------------------------- the follower


@pytest.fixture
def live(tmp_path: Path) -> Iterator[dict[str, Any]]:
    app = create_app(settings_for(tmp_path / "active"))
    with _LiveServer(app) as server:
        client = TestClient(app)
        assert (
            client.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
        )
        client.headers["Authorization"] = f"Bearer {login(client)}"
        spare, joined = enroll(client, "spare")
        assert joined.status_code == 201, joined.text
        yield {"server": server, "client": client, "spare": spare, "machine": app.state.machine}


def _agent(spare: NodeKeys, seen: list[str], *, refuse: bool = False) -> httpx.AsyncClient:
    """This machine's agent: trades the spawn token for a standby token,
    as `POST /v1/auth/service-token` does."""

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/auth/service-token"
        assert request.headers["authorization"] == "Bearer spawn-token"
        seen.append(request.content.decode())
        if refuse:
            return httpx.Response(
                403,
                json={"detail": {"title": "Not granted", "detail": "this node holds no standby"}},
            )
        expires = datetime.now(UTC) + timedelta(minutes=15)
        token = spare.service_token(sub="standby", ttl=900)
        return httpx.Response(200, json={"token": token, "expiresAt": expires.isoformat()})

    return httpx.AsyncClient(transport=httpx.MockTransport(handle))


async def test_the_standby_follows_with_the_token_its_agent_trades_it(
    live: dict[str, Any], tmp_path: Path
) -> None:
    client: TestClient = live["client"]
    machine: StateMachine = live["machine"]
    assert client.put("/v1/nodes/spare/standby").status_code == 202
    machine.append(OP_PATCH_CONFIG, {"values": {"uiTheme": "dark"}})
    seen: list[str] = []
    credential = StandbyToken(
        agent_url="http://agent.test", spawn_token="spawn-token", client=_agent(live["spare"], seen)
    )
    standby = standby_at(tmp_path / "standby")
    follower = Follower(
        standby, active_url=live["server"].url, interval_seconds=0.01, token_provider=credential
    )
    try:
        await follower.tick()
        assert standby.canonical() == machine.canonical(), follower.status.last_error
        await follower.tick()
        # Traded once, then reused while it has time left.
        assert len(seen) == 1 and '"audience":"control"' in seen[0].replace(" ", "")
        [report] = client.get("/v1/control/status").json()["standbys"]
        assert report["appliedIndex"] == machine.state.index and report["reachable"] is True

        # The grant goes: the next pull is refused, in the root's words.
        assert client.delete("/v1/nodes/spare/standby").status_code == 202
        with pytest.raises(Exception, match="refused this standby's log read \\(401\\)"):
            await follower.tick()
    finally:
        await follower.stop()


async def test_an_agent_that_gives_no_token_is_named_as_the_cause(
    live: dict[str, Any], tmp_path: Path
) -> None:
    seen: list[str] = []
    credential = StandbyToken(
        agent_url="http://agent.test",
        spawn_token="spawn-token",
        client=_agent(live["spare"], seen, refuse=True),
    )
    follower = Follower(
        standby_at(tmp_path / "standby"),
        active_url=live["server"].url,
        interval_seconds=0.01,
        token_provider=credential,
    )
    try:
        with pytest.raises(Exception, match="agent gave the standby no token \\(403\\)"):
            await follower.tick()
    finally:
        await follower.stop()


def test_the_node_its_agent_names_is_the_one_a_promotion_names(tmp_path: Path) -> None:
    """The agent tells the standby its node (NODE_NAME), so a promotion
    marks that node the root and takes its standby grant (warm-standby.md)."""
    settings = settings_for(tmp_path / "standby", role="standby", active_url=None)
    settings.node_name = "spare"
    app = create_app(settings)
    with TestClient(app):
        assert app.state.node_name == "spare"


def test_a_standby_started_by_its_agent_follows_with_the_token_it_was_given(
    tmp_path: Path,
) -> None:
    """SB3: the agent's spawn token reaches the follower; a hand-started
    standby with none sends nothing, and is refused."""
    settings = settings_for(tmp_path / "given", role="standby", active_url=None)
    settings.agent_url = "http://127.0.0.1:1"
    settings.service_token = "spawn-token"
    given = create_app(settings)
    with TestClient(given):
        assert isinstance(given.state.follower._token_provider, StandbyToken)
    bare = create_app(settings_for(tmp_path / "bare", role="standby", active_url=None))
    with TestClient(bare):
        assert bare.state.follower._token_provider is None


async def test_the_probe_reads_whether_a_node_hosts_the_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from eugene_plexus_control.nodes_client import NodesClient

    client = NodesClient(timeout_provider=lambda: 1.0)
    answers = iter([{"hostsControl": True}, {"hostsControl": "yes"}, {}])

    async def fake_get(_url: str, _path: str, _token: str | None) -> Any:
        return next(answers)

    monkeypatch.setattr(client, "_get", fake_get)
    try:
        seen = [(await client.probe("n", "http://n:8079", None)).hosts_control for _ in range(3)]
    finally:
        await client.aclose()
    assert seen == [True, False, False], "only an explicit true counts"
