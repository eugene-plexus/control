"""Trust-bundle distribution, root-key rotation, and per-node sealing.

Until 2026-09-25 this file was about rotating the one key every node
held, and the mixed-key state a partial redistribution left behind. No
node holds a key another could use now
(`specs/docs/design/per-node-token-keys.md`), so a revocation or a
rotation only publishes a new **trust bundle**. These tests are about
that bundle reaching nodes, being refused by a node that pins a
different root, and naming the nodes that do not hold it yet.

Sealing is the other half of the blast-radius argument: one compromised
GPU box in another building must not yield every secret in the install.
"""

from __future__ import annotations

import base64
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI, Response
from fastapi.testclient import TestClient

from eugene_plexus_control import sealing, tokens
from eugene_plexus_control.app import create_app
from eugene_plexus_control.applied import NodeRecord
from eugene_plexus_control.sealing import SealError
from eugene_plexus_control.state_machine import StateMachine

from .conftest import PASSPHRASE, enroll, login, settings_for, standby_at

# --------------------------------------------------------------------------- #
# Distribution
# --------------------------------------------------------------------------- #


def _bundle_agent() -> tuple[FastAPI, list[tokens.TrustBundle]]:
    """A fake agent that does what a real one does with a pushed bundle:
    verify it against the control identity it pinned at enrollment, and
    refuse anything else with 401. A fake that recorded whatever arrived
    is how the unsigned re-key survived M5."""
    received: list[tokens.TrustBundle] = []
    app = FastAPI()
    app.state.control_public_key = None

    @app.post("/v1/node/trust-bundle")
    async def take(body: dict[str, Any], response: Response) -> dict[str, str]:
        pinned = app.state.control_public_key
        try:
            bundle = tokens.parse_bundle(str(body["jws"]), authority=pinned or "")
        except (tokens.BundleError, KeyError, ValueError):
            response.status_code = 401
            return {"detail": "signature rejected"}
        received.append(bundle)
        return {"ok": "true"}

    @app.get("/v1/runtimes")
    async def runtimes() -> dict[str, Any]:
        return {"runtimes": []}

    return app, received


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
def agent() -> Iterator[tuple[_Server, list[tokens.TrustBundle]]]:
    app, received = _bundle_agent()
    with _Server(app) as server:
        yield server, received


def _pin(server: _Server, client: TestClient) -> None:
    """What a real agent records at enrollment as `controlPublicKey`."""
    machine = client.app.state.machine  # type: ignore[attr-defined]
    server.app.state.control_public_key = machine.state.identity.controlPublicKey


def _await(received: list[tokens.TrustBundle], client: TestClient, version: int) -> None:
    """Let the background push finish. Any request drives the app's loop."""
    deadline = time.time() + 20
    while time.time() < deadline:
        if any(b.version >= version for b in received):
            return
        client.get("/healthz")
        time.sleep(0.02)


def _root_kids(bundle: tokens.TrustBundle) -> set[str]:
    return {k.kid for k in bundle.keys.values() if k.issuer == "control"}


def test_a_rotation_replaces_only_the_root_key_and_moves_nothing_private(
    active_client: TestClient, agent: tuple[_Server, list[tokens.TrustBundle]]
) -> None:
    """D10. The new key is sealed into the log and named in a bundle; node
    keys are untouched; the push carries public material only."""
    server, received = agent
    _pin(server, active_client)
    keys, _ = enroll(active_client, "gpu-box", url=server.url)
    machine = active_client.app.state.machine  # type: ignore[attr-defined]
    auth = active_client.app.state.auth_state  # type: ignore[attr-defined]
    before = active_client.app.state.trust.current  # type: ignore[attr-defined]

    response = active_client.post("/v1/control/rotate-key")
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["reason"] == "rotation"

    last = machine.read_page(machine.state.index - 1, 1)[0]
    assert last["op"] == "rotateSigningKey"
    assert "PRIVATE KEY" not in str(last)

    # The caller signed in under the old key, so it is signed out: that is
    # what the emergency lever is for.
    assert active_client.get("/v1/nodes").status_code == 401
    active_client.headers["Authorization"] = f"Bearer {login(active_client)}"

    _await(received, active_client, body["version"])
    pushed = received[-1]
    assert pushed.version == body["version"]
    assert _root_kids(pushed) != _root_kids(before)
    assert _root_kids(pushed) == {auth.signer().kid}
    node_kids = {k.kid for k in pushed.keys.values() if k.issuer == "node:gpu-box"}
    assert node_kids == {keys.token.kid}, "a node's key survives a root rotation"


def test_a_revocation_pushes_a_bundle_without_the_node_and_names_who_is_behind(
    active_client: TestClient, agent: tuple[_Server, list[tokens.TrustBundle]]
) -> None:
    """The test §12 asked for, re-read: revoke a node while another is
    offline. The offline one is named, and it catches up by pulling."""
    server, received = agent
    _pin(server, active_client)
    enroll(active_client, "gpu-box", url=server.url)
    enroll(active_client, "attic", url="http://127.0.0.1:1")
    enroll(active_client, "shed", url=server.url)

    response = active_client.delete("/v1/nodes/shed")
    assert response.status_code == 202, response.text
    version = response.json()["version"]
    _await(received, active_client, version)
    assert all("node:shed" not in {k.issuer for k in b.keys.values()} for b in received[-1:])

    listed = active_client.get("/v1/nodes").json()["nodes"]
    assert {n["name"] for n in listed} == {"gpu-box", "attic"}
    deadline = time.time() + 10
    behind: list[str] = []
    while time.time() < deadline:
        from eugene_plexus_control import trust

        behind = trust.nodes_behind(active_client.app, version)
        if behind == ["attic"]:
            break
        active_client.get("/healthz")
        time.sleep(0.05)
    assert behind == ["attic"]


def test_a_node_that_pins_another_root_refuses_the_bundle(
    active_client: TestClient, agent: tuple[_Server, list[tokens.TrustBundle]]
) -> None:
    """The other half of the signature: a node enrolled with a *different*
    root refuses this one's bundle, and is named behind rather than
    counted as holding it."""
    server, received = agent
    server.app.state.control_public_key = sealing.generate_control_identity().public
    enroll(active_client, "gpu-box", url=server.url)
    response = active_client.post("/v1/control/rotate-key")
    assert response.status_code == 202
    active_client.headers["Authorization"] = f"Bearer {login(active_client)}"
    time.sleep(0.5)
    active_client.get("/healthz")
    assert received == []
    from eugene_plexus_control import trust

    assert trust.nodes_behind(active_client.app, response.json()["version"]) == ["gpu-box"]


def test_the_bundle_is_public_and_a_sealed_root_still_serves_it(tmp_path: Any) -> None:
    """Every agent pulls it, including while this root is sealed after a
    restart: the signature is the credential, and the last one is kept."""
    directory = tmp_path / "install"
    app = create_app(settings_for(directory))
    with TestClient(app) as client:
        assert (
            client.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
        )
        first = client.get("/v1/trust/bundle")
        assert first.status_code == 200, first.text
        authority = app.state.machine.state.identity.controlPublicKey
        assert tokens.parse_bundle(first.json()["jws"], authority=authority)

    restarted = create_app(settings_for(directory))
    with TestClient(restarted) as client:
        assert client.get("/v1/nodes").status_code == 503, "sealed"
        kept = client.get("/v1/trust/bundle")
        assert kept.status_code == 200
        assert kept.json()["jws"] == first.json()["jws"]


def test_a_fresh_root_has_no_bundle_to_give(tmp_path: Any) -> None:
    app = create_app(settings_for(tmp_path / "fresh"))
    with TestClient(app) as client:
        assert client.get("/v1/trust/bundle").status_code == 503


def _replicate(active: StateMachine, standby_dir: Any) -> StateMachine:
    standby = standby_at(standby_dir)
    standby.install_snapshot(active.snapshot_document())
    for entry in active.read_page(standby.state.index, 10_000):
        standby.accept_replicated(entry)
    return standby


def test_promotion_announces_the_new_epoch_with_the_same_keys(
    tmp_path: Any, agent: tuple[_Server, list[tokens.TrustBundle]]
) -> None:
    """The promoted root pushes a bundle at the new epoch, signed by the
    identity promotion preserved and naming the same root key, so nodes
    fence the old root and nobody is signed out."""
    server, received = agent
    active_app = create_app(settings_for(tmp_path / "active"))
    with TestClient(active_app) as active:
        assert (
            active.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
        )
        active.headers["Authorization"] = f"Bearer {login(active)}"
        _pin(server, active)
        enroll(active, "gpu-box", url=server.url)
        active_machine: StateMachine = active_app.state.machine
        root_kid = active_app.state.auth_state.signer().kid
        epoch_before = active_machine.state.epoch
        _replicate(active_machine, tmp_path / "standby" / "state")
    received.clear()

    standby_app = create_app(settings_for(tmp_path / "standby", role="standby"))
    with TestClient(standby_app) as standby:
        token = standby.post("/v1/auth/login", json={"passphrase": PASSPHRASE}).json()[
            "sessionToken"
        ]
        promoted = standby.post(
            "/v1/control/promote",
            json={"passphrase": PASSPHRASE},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert promoted.status_code == 200, promoted.text
        assert promoted.json()["epoch"] == epoch_before + 1
        deadline = time.time() + 20
        while not any(b.epoch == epoch_before + 1 for b in received) and time.time() < deadline:
            standby.get("/healthz")
            time.sleep(0.05)

    announced = [b for b in received if b.epoch == epoch_before + 1]
    assert announced, "the promoted root never announced its epoch"
    assert _root_kids(announced[-1]) == {root_kid}, "a promotion changes the epoch, never the key"


# --------------------------------------------------------------------------- #
# Sealing
# --------------------------------------------------------------------------- #


def test_a_secret_opens_for_the_node_and_for_recovery_and_nobody_else() -> None:
    """The blast-radius table, asserted.

    One node yields that node's secrets only; the recovery recipient
    reaches them too, and it is itself sealed under the passphrase.
    """
    node = sealing.generate_sealing_keypair()
    other_node = sealing.generate_sealing_keypair()
    recovery = sealing.generate_sealing_keypair()

    sealed = sealing.seal_to_recipients(
        "hf_the_operators_token",
        {sealing.node_recipient_label("gpu-box"): node.public, "recovery": recovery.public},
    )

    assert (
        sealing.open_as_recipient(sealed, label="node:gpu-box", private_b64=node.private)
        == "hf_the_operators_token"
    )
    assert (
        sealing.open_as_recipient(sealed, label="recovery", private_b64=recovery.private)
        == "hf_the_operators_token"
    )
    with pytest.raises(SealError):
        sealing.open_as_recipient(sealed, label="node:gpu-box", private_b64=other_node.private)
    with pytest.raises(SealError, match="not sealed to"):
        sealing.open_as_recipient(sealed, label="node:attic", private_b64=other_node.private)


def test_the_recipients_of_a_secret_are_readable_without_any_key() -> None:
    """So an operator can see whether a secret is still reachable after
    a node was revoked, without holding anything."""
    node = sealing.generate_sealing_keypair()
    recovery = sealing.generate_sealing_keypair()
    sealed = sealing.seal_to_recipients(
        "x", {sealing.node_recipient_label("gpu-box"): node.public, "recovery": recovery.public}
    )
    assert sealing.recipients_of(sealed) == ["node:gpu-box", "recovery"]


def test_sealing_is_not_deterministic_which_is_why_it_is_replicated_verbatim() -> None:
    """Recorded as a test because it is the reason key material travels
    as a payload the writer stamped rather than something a replica
    recomputes. A replica that re-sealed would produce different bytes
    and break replay equivalence."""
    node = sealing.generate_sealing_keypair()
    recipients = {sealing.node_recipient_label("gpu-box"): node.public}
    assert sealing.seal_to_recipients("x", recipients) != sealing.seal_to_recipients(
        "x", recipients
    )


def test_a_secret_with_no_recipients_is_refused() -> None:
    with pytest.raises(SealError, match="nobody can read"):
        sealing.seal_to_recipients("x", {})


def test_sealing_to_a_node_requires_that_node_to_have_a_key() -> None:
    from eugene_plexus_control.routes.nodes import seal_for_node

    keyless = NodeRecord(name="gpu-box", publicKey=None)
    with pytest.raises(Exception, match="no identity public key"):
        seal_for_node("secret", keyless, None)


def test_a_malformed_sealed_value_is_refused_rather_than_ignored() -> None:
    with pytest.raises(SealError):
        sealing.recipients_of("not base64 at all !!")
    with pytest.raises(SealError, match="unsupported seal alg"):
        sealing.recipients_of(base64.b64encode(b'{"alg":"rot13","recipients":{}}').decode())


def test_the_control_identity_is_a_signing_key_not_a_sealing_one() -> None:
    """Different jobs: nodes check that an epoch change came from a root
    they recognise, which is a signature question, not a secrecy one."""
    identity = sealing.generate_control_identity()
    raw = base64.b64decode(identity.public)
    assert len(raw) == 32
    # It is not usable as a sealing recipient, which is the point of
    # keeping them separate.
    with pytest.raises(SealError):
        sealing.open_as_recipient(
            sealing.seal_to_recipients("x", {"recovery": identity.public}),
            label="recovery",
            private_b64=identity.private,
        )
