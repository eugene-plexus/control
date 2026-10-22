"""Row 4a: a promoted root tells every machine where it is.

`docs/design/rows/row-04.md` (specs), WS-ADDR. Before this row a promotion
pushed a bundle at the new epoch and nothing else: the bundle carries no
address, every agent kept pulling from the root that stopped, and only the
promoted machine could reach the new root.

The signed machine list now names the root (`MachineList.control`: its
epoch, its node and the plain address of its machine's port). A promotion
pushes the bundle and then the list, and the promoted copy starts polling
the nodes like any active root, pushing both again to a node that is
behind.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import jwt
from fastapi.testclient import TestClient

from eugene_plexus_control import _http, machine_list
from eugene_plexus_control.app import create_app, push_to_nodes_behind
from eugene_plexus_control.applied import OP_ENROLL_NODE, AppliedState, NodeRecord
from eugene_plexus_control.nodes_client import NodeProbe
from eugene_plexus_control.state_machine import StateMachine

from .conftest import PASSPHRASE, enroll, settings_for
from .test_control_root_loss import _replicate

PIN_ATTIC = "A" * 43
PIN_GPU = "G" * 43


def _node(name: str, **fields: Any) -> NodeRecord:
    return NodeRecord(name=name, **fields)


def test_the_machine_list_names_the_root_its_node_and_epoch(tmp_path: Path) -> None:
    from dataclasses import replace

    state = replace(
        AppliedState(),
        epoch=3,
        nodes={
            "attic": _node("attic", url="http://10.0.0.9:8079", machineUrl="https://10.0.0.9:8083")
        },
    )
    door = tmp_path / "machine_door.json"
    door.write_text(
        json.dumps({"pins": [PIN_ATTIC], "url": "https://10.0.0.9:8283"}), encoding="utf-8"
    )

    def block(**kwargs: Any) -> dict[str, Any]:
        options: dict[str, Any] = {"node": "attic", "door_file": None, "nodes_origin": None}
        options.update(kwargs)
        return machine_list.control_block(state, **options)

    # The door file its agent writes is what the machine port says now.
    assert block(door_file=str(door)) == {
        "epoch": 3,
        "node": "attic",
        "url": "http://10.0.0.9:8283",
    }
    # Else what its own machine announced.
    assert block()["url"] == "http://10.0.0.9:8083"
    # Else the nodes name; else nothing a node could move to.
    assert block(node="other", nodes_origin="https://nodes.example.test/")["url"] == (
        "https://nodes.example.test"
    )
    assert block(node=None) == {"epoch": 3, "node": None, "url": None}

    signed = machine_list.content(state, [PIN_ATTIC], control=block())
    assert signed["control"] == {"epoch": 3, "node": "attic", "url": "http://10.0.0.9:8083"}


class _Pushes:
    """The nodes as a root's node client sees them: every POST is kept."""

    def __init__(self, probes: dict[str, NodeProbe] | None = None) -> None:
        self.posts: list[tuple[str, str, dict[str, Any]]] = []
        self.probes = probes or {}

    async def post_json(self, url: str, path: str, token: str | None, body: dict[str, Any]) -> Any:
        self.posts.append((url, path, body))
        return {}

    async def probe_all(self, targets: Any, token_for: Any) -> dict[str, NodeProbe]:
        return dict(self.probes)

    async def aclose(self) -> None:
        return None

    def paths(self) -> list[str]:
        return [path for _url, path, _body in self.posts]


def _claims(jws: str) -> dict[str, Any]:
    return dict(jwt.decode(jws, options={"verify_signature": False}))


def _wait(check: Any, timeout: float = 10.0) -> Any:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        found = check()
        if found:
            return found
        time.sleep(0.05)
    raise AssertionError("timed out")


def test_a_promotion_pushes_the_bundle_then_a_list_that_names_the_new_root(
    tmp_path: Path,
) -> None:
    active_app = create_app(settings_for(tmp_path / "active"))
    with TestClient(active_app) as active:
        assert (
            active.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
        )
        machine: StateMachine = active_app.state.machine
        for name, ip, pin in (("attic", "10.0.0.9", PIN_ATTIC), ("gpu", "10.0.0.7", PIN_GPU)):
            machine.append(
                OP_ENROLL_NODE,
                {
                    "name": name,
                    "role": "agent",
                    "url": f"http://{ip}:8079",
                    "publicKey": "cHVibGljLWtleS1ieXRlcy0zMi1jaGFyYWN0ZXJzLWs=",
                    "enrolledAt": "2026-10-20T10:00:00+00:00",
                    "machineUrl": f"https://{ip}:8083",
                    "tlsPins": [pin],
                },
            )
        epoch = machine.state.epoch
        _replicate(machine, tmp_path / "standby" / "state")

    settings = settings_for(tmp_path / "standby", role="standby")
    settings.node_name = "attic"
    app = create_app(settings)
    with TestClient(app) as standby:
        pushes = _Pushes()
        app.state.nodes_client = pushes
        assert app.state.node_poller is None, "a standby polls nobody"
        token = standby.post("/v1/auth/login", json={"passphrase": PASSPHRASE}).json()[
            "sessionToken"
        ]
        promoted = standby.post(
            "/v1/control/promote",
            json={"passphrase": PASSPHRASE},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert promoted.status_code == 200, promoted.text
        assert promoted.json()["epoch"] == epoch + 1

        lists = _wait(
            lambda: [b for _u, p, b in pushes.posts if p == "/v1/node/machine-list"] or None
        )
        first_list = pushes.paths().index("/v1/node/machine-list")
        assert "/v1/node/trust-bundle" in pushes.paths()[:first_list], "the epoch goes first"
        control = _claims(lists[0]["jws"])["control"]
        assert control == {"epoch": epoch + 1, "node": "attic", "url": "http://10.0.0.9:8083"}
        targets = {u for u, p, _b in pushes.posts if p == "/v1/node/machine-list"}
        assert targets == {"http://10.0.0.9:8079", "http://10.0.0.7:8079"}
        # It reaches the nodes over TLS to the keys it holds for them, and
        # it polls them now, as an active root does.
        assert _http.machine_route("http://10.0.0.7:8079/v1/node") == (
            "https://10.0.0.7:8083/agent/v1/node"
        )
        assert app.state.node_poller is not None


def test_the_root_pushes_to_a_node_behind_at_most_once_a_minute(
    active_client: TestClient,
) -> None:
    app: Any = active_client.app
    for name in ("gpu", "edge", "fresh"):
        _keys, joined = enroll(active_client, name, url=f"http://192.0.2.{len(name)}:8079")
        assert joined.status_code == 201, joined.text
    epoch = app.state.machine.state.epoch
    probes = {
        # It missed the epoch.
        "gpu": NodeProbe(name="gpu", reachable=True, epoch=epoch - 1),
        # It holds the epoch, but its root address came with an older one.
        "edge": NodeProbe(name="edge", reachable=True, epoch=epoch, control_url_epoch=epoch - 1),
        # Up to date, or an agent older than row 4 that says nothing.
        "fresh": NodeProbe(name="fresh", reachable=True, epoch=epoch),
    }
    # Somewhere a node could move to: without one, a node behind only on
    # its address could do nothing with the list.
    app.state.settings.nodes_origin = "https://nodes.example.test"
    pushes = _Pushes()
    app.state.nodes_client = pushes
    clock = 1000.0

    def push(now: float) -> None:
        active_client.portal.call(lambda: push_to_nodes_behind(app, probes, now=now))

    push(clock)
    pushed = sorted((u.rstrip("/"), p) for u, p, _b in pushes.posts)
    assert pushed == [
        ("http://192.0.2.3:8079", "/v1/node/machine-list"),
        ("http://192.0.2.3:8079", "/v1/node/trust-bundle"),
        ("http://192.0.2.4:8079", "/v1/node/machine-list"),
        ("http://192.0.2.4:8079", "/v1/node/trust-bundle"),
    ]
    jws = next(b["jws"] for _u, p, b in pushes.posts if p == "/v1/node/machine-list")
    assert _claims(jws)["control"]["epoch"] == epoch

    pushes.posts.clear()
    push(clock + 59)
    assert pushes.posts == [], "not again within the minute"
    push(clock + 61)
    assert len(pushes.posts) == 4
