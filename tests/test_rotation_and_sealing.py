"""Key rotation and per-node sealing.

Rotation is the sharp edge: *"if rotation is incomplete — a node `down`
during it, a partial redistribution — the install is left in a
mixed-key state where some legitimate calls fail and a revoked node may
still be trusted somewhere."* The design asked specifically for a test
that revokes a node while another is offline, and that is the one below.

Sealing is the other half of the blast-radius argument: one compromised
GPU box in another building must not yield every secret in the install.
"""

from __future__ import annotations

import base64
import threading
from collections.abc import Iterator
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_control import sealing
from eugene_plexus_control.applied import NodeRecord
from eugene_plexus_control.rotation import RotationTracker
from eugene_plexus_control.sealing import SealError

# --------------------------------------------------------------------------- #
# Rotation
# --------------------------------------------------------------------------- #


def _rekeyable_agent() -> tuple[FastAPI, list[dict[str, Any]]]:
    received: list[dict[str, Any]] = []
    app = FastAPI()

    @app.post("/v1/node/rekey")
    async def rekey(body: dict[str, Any]) -> dict[str, str]:
        received.append(body)
        return {"ok": "true"}

    @app.get("/v1/runtimes")
    async def runtimes() -> dict[str, Any]:
        return {"runtimes": []}

    return app, received


class _Server:
    def __init__(self, app: FastAPI) -> None:
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
def rekeyable() -> Iterator[tuple[_Server, list[dict[str, Any]]]]:
    app, received = _rekeyable_agent()
    with _Server(app) as server:
        yield server, received


def _enroll(client: TestClient, name: str, url: str) -> None:
    token = client.post("/v1/nodes/join-token").json()["token"]
    public = base64.b64encode(name.encode().ljust(32, b"0")).decode()
    assert (
        client.post(
            "/v1/nodes/enroll", json={"token": token, "name": name, "publicKey": public}
        ).status_code
        == 201
    )
    machine = client.app.state.machine  # type: ignore[attr-defined]
    record = machine.state.nodes[name]
    machine.append(
        "enrollNode",
        {
            "name": name,
            "role": record.role,
            "url": url,
            "publicKey": record.publicKey,
            "enrolledAt": record.enrolledAt,
        },
    )


def test_rotation_mints_logs_and_redistributes_in_that_order(
    active_client: TestClient, rekeyable: tuple[_Server, list[dict[str, Any]]]
) -> None:
    """The new key is durable **before** any node hears about it.

    A crash midway then leaves an install whose recorded key is the new
    one and whose hosts are a mix, which re-running the rotation fixes.
    The other order would leave nodes holding a key no surviving root
    knows about, which nothing fixes.
    """
    server, received = rekeyable
    _enroll(active_client, "gpu-box", server.url)
    machine = active_client.app.state.machine  # type: ignore[attr-defined]

    response = active_client.post("/v1/control/rotate-key")
    assert response.status_code == 202, response.text
    assert response.json()["reason"] == "operator"

    # The log carries the sealed key, not the plaintext one.
    last = machine.read_page(machine.state.index - 1, 1)[0]
    assert last["op"] == "rotateSigningKey"
    auth = active_client.app.state.auth_state  # type: ignore[attr-defined]
    assert base64.b64encode(auth.signing_key).decode() not in str(last)

    # And the caller is now logged out, because one key signs both
    # service tokens and operator sessions. Correct rather than
    # convenient: a host that held a service token could have held an
    # operator session too, and the usual reason to rotate is that some
    # host should no longer be trusted.
    assert active_client.get("/v1/control/rotate-key").status_code == 401
    _relogin(active_client)

    _await_rotation(active_client)
    progress = active_client.get("/v1/control/rotate-key").json()
    assert progress["state"] in ("distributing", "done")
    assert received and base64.b64decode(received[0]["signingKey"]) == auth.signing_key


def test_a_node_that_is_down_during_a_rotation_is_named_not_hidden(
    active_client: TestClient, rekeyable: tuple[_Server, list[dict[str, Any]]]
) -> None:
    """The test §12 asked for: revoke a node while another is offline.

    The rotation finishes as `failed` with the offline host **named**,
    because "which host is stale" is the question an operator actually
    has and a percentage cannot answer it. Calling it `done` would tell
    them the install is consistent when it is not.
    """
    server, _ = rekeyable
    _enroll(active_client, "gpu-box", server.url)
    _enroll(active_client, "attic", "http://127.0.0.1:1")
    _enroll(active_client, "shed", server.url)

    response = active_client.delete("/v1/nodes/shed")
    assert response.status_code == 202

    tracker: RotationTracker = active_client.app.state.rotation  # type: ignore[attr-defined]
    _relogin(active_client)
    _await_rotation(active_client)
    final = tracker.current
    assert final is not None
    assert final.state == "failed"
    assert final.nodes_pending == ("attic",)
    assert final.error is not None and "attic" in final.error
    assert "re-keyed on reconnect" in final.error


def test_two_concurrent_rotations_are_refused(active_client: TestClient) -> None:
    """Two new keys, each distributed to a different subset of hosts, is
    a mixed-key state with no single correct answer."""
    tracker: RotationTracker = active_client.app.state.rotation  # type: ignore[attr-defined]
    tracker.begin(reason="operator", revoked_node=None, nodes=["a", "b"])
    response = active_client.post("/v1/control/rotate-key")
    assert response.status_code == 409
    assert "mixed-key" in response.text


def test_rotation_progress_persists_after_it_ends(active_client: TestClient) -> None:
    """So a UI that reconnects afterwards still learns how it ended."""
    assert active_client.get("/v1/control/rotate-key").status_code == 404
    assert active_client.post("/v1/control/rotate-key").status_code == 202
    _relogin(active_client)
    _await_rotation(active_client)
    later = active_client.get("/v1/control/rotate-key")
    assert later.status_code == 200
    assert later.json()["state"] in ("done", "distributing")


def _relogin(client: TestClient) -> None:
    """Get a session signed with the new key.

    A rotation invalidates every token issued under the old one, this
    caller's included. A UI would show a login prompt here; a test says
    so out loud so that the behaviour is asserted rather than worked
    around by accident.
    """
    from .conftest import PASSPHRASE

    token = client.post("/v1/auth/login", json={"passphrase": PASSPHRASE}).json()["token"]
    client.headers["Authorization"] = f"Bearer {token}"


def _await_rotation(client: TestClient) -> None:
    """Let the background redistribution finish.

    A poll rather than a sleep: the task is scheduled on the app's loop
    and the length of a timeout is not the thing under test.
    """
    import time

    tracker: RotationTracker = client.app.state.rotation  # type: ignore[attr-defined]
    deadline = time.time() + 20
    while time.time() < deadline:
        current = tracker.current
        if current is not None and not current.in_flight:
            return
        # Any request drives the loop the task is scheduled on.
        client.get("/healthz")
        time.sleep(0.02)


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
