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
import pytest
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


# --------------------------------------------------------------------------- the review


def test_root_tokens_carry_the_epoch_and_an_older_one_is_refused(
    active_client: TestClient,
) -> None:
    """A promotion keeps the root's token key. Every token the root signs
    says the epoch it was signed at, so a verifier refuses the root it no
    longer follows (row 4 review, blocker 1)."""
    from eugene_plexus_control import tokens

    app: Any = active_client.app
    machine: StateMachine = app.state.machine
    session = active_client.headers["Authorization"].removeprefix("Bearer ")
    assert _claims(session)["ep"] == machine.state.epoch
    _keys, joined = enroll(active_client, "gpu")
    assert joined.status_code == 201, joined.text
    bundle = app.state.trust.current
    assert bundle is not None and bundle.token_epochs is not None
    assert bundle.token_epochs[1] == machine.state.epoch
    assert "tokenEpochs" not in active_client.get("/v1/config").json().get("values", {})

    signer = app.state.auth_state.signer()
    stale = tokens.Signer(key=signer.key, issuer=signer.issuer, epoch=machine.state.epoch - 1)
    old, _ = stale.mint(typ=tokens.TYP_SESSION, sub="operator", aud=["control"], ttl_seconds=600)
    refused = active_client.get("/v1/nodes", headers={"Authorization": f"Bearer {old}"})
    assert refused.status_code == 401
    assert "no longer follows (epoch 0; it follows epoch 1)" in refused.text


def test_a_copy_promoted_here_starts_as_the_root(tmp_path: Path) -> None:
    """A copy that crashed after its promotion, before its agent heard of
    it, is started as a standby again; its own log says it is the root at
    a higher epoch than the agent holds, so it starts as the root (row 4
    review, blocker 3). A copy whose agent holds that epoch, or one that
    names another machine, stays a standby."""
    from eugene_plexus_control.state_machine import ROLE_ACTIVE

    from .conftest import machine_at, standby_at

    root = machine_at(tmp_path / "root")
    root.seed_identity({"salt": "c2FsdA==", "passphraseVerifier": "v", "controlPublicKey": "k"})
    root.append(
        OP_ENROLL_NODE,
        {
            "name": "attic",
            "role": "agent",
            "url": "http://10.0.0.9:8079",
            "publicKey": "cHVibGljLWtleS1ieXRlcy0zMi1jaGFyYWN0ZXJzLWs=",
            "enrolledAt": "2026-10-20T10:00:00+00:00",
        },
    )
    for name, follow, node, active in (
        ("crashed", 1, "attic", True),
        ("heard", 2, "attic", False),
        ("other", 1, "shed", False),
    ):
        copy = standby_at(tmp_path / name / "state")
        copy.install_snapshot(root.snapshot_document())
        copy.promote(node="attic")
        assert copy.state.epoch == 2
        settings = settings_for(tmp_path / name, role="standby")
        settings.node_name = node
        settings.follow_epoch = follow
        app = create_app(settings)
        with TestClient(app):
            assert (app.state.machine.role == ROLE_ACTIVE) is active, name
            assert (app.state.follower is None) is active, name
            assert (app.state.node_poller is not None) is active, name


def test_one_slow_machine_does_not_hold_up_the_others(active_client: TestClient) -> None:
    """Each machine's bundle and list on its own, with a deadline: one that
    never answers is given up on, and the rest have both at once (row 4
    review)."""
    import asyncio

    from eugene_plexus_control import trust

    app: Any = active_client.app
    for name, ip in (("slow", "192.0.2.21"), ("quick", "192.0.2.22")):
        _keys, joined = enroll(active_client, name, url=f"http://{ip}:8079")
        assert joined.status_code == 201, joined.text

    class _Client(_Pushes):
        async def post_json(
            self, url: str, path: str, token: str | None, body: dict[str, Any]
        ) -> Any:
            if "192.0.2.21" in url:
                await asyncio.Event().wait()
            return await super().post_json(url, path, token, body)

    pushes = _Client()
    app.state.nodes_client = pushes
    bundle = app.state.trust.current
    started = time.perf_counter()
    taken = active_client.portal.call(lambda: trust.push_pairs(app, bundle, "a-list", deadline=0.5))
    assert time.perf_counter() - started < 5
    assert taken == {"slow": False, "quick": True}
    assert [p for u, p, _b in pushes.posts if "192.0.2.22" in u] == [
        "/v1/node/trust-bundle",
        "/v1/node/machine-list",
    ]


# --------------------------------------------------------------------------- the re-review


def test_a_promotion_ends_the_old_roots_tokens_without_an_epoch(tmp_path: Path) -> None:
    """A root promoted from a copy never signed a token without `ep`. When it
    is the first to record since when roots mark theirs (the root it replaced
    ran code older than row 4), it names the epoch it replaced, so that
    root's tokens without `ep` are refused at once (row 4 re-review,
    blocker 1b)."""
    from eugene_plexus_control import tokens

    from .conftest import standby_at

    root_app = create_app(settings_for(tmp_path / "root"))
    with TestClient(root_app) as root:
        assert root.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
        document = root_app.state.machine.snapshot_document()
    # The root it replaced never wrote the mark.
    document["config"] = {k: v for k, v in document["config"].items() if k != "tokenEpochs"}
    standby_at(tmp_path / "copy" / "state").install_snapshot(document)

    app = create_app(settings_for(tmp_path / "copy", role="standby"))
    with TestClient(app) as client:
        token = client.post("/v1/auth/login", json={"passphrase": PASSPHRASE}).json()[
            "sessionToken"
        ]
        assert app.state.machine.state.config.get("tokenEpochs") is None, "no mark yet"
        promoted = client.post(
            "/v1/control/promote",
            json={"passphrase": PASSPHRASE},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert promoted.status_code == 200, promoted.text
        bundle = app.state.trust.current
        assert bundle is not None and bundle.epoch == 2
        assert bundle.token_epochs is not None and bundle.token_epochs[1] == 1
        legacy = tokens.Signer(key=app.state.auth_state.signer().key, issuer="control")
        before, _ = legacy.mint(
            typ=tokens.TYP_SESSION,
            sub="operator",
            aud=["control"],
            ttl_seconds=600,
            now=bundle.token_epochs[0] - 30,
        )
        with pytest.raises(tokens.TokenError, match="no longer follows"):
            tokens.verify(before, bundle=bundle, recipient="control", classes=(tokens.TYP_SESSION,))


def test_the_root_says_when_it_is_locked(tmp_path: Path) -> None:
    """`/healthz` says a root that holds no token key is locked: its agent
    tells Home (row 4 re-review, blocker 3)."""
    settings = settings_for(tmp_path / "root")
    first = create_app(settings)
    with TestClient(first) as client:
        assert (
            client.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
        )
        assert client.get("/healthz").json()["details"]["locked"] is False
    again = create_app(settings)
    with TestClient(again) as client:
        assert client.get("/healthz").json()["details"]["locked"] is True
        client.post("/v1/auth/login", json={"passphrase": PASSPHRASE})
        assert client.get("/healthz").json()["details"]["locked"] is False
