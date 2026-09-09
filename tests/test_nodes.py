"""Enrollment, revocation-as-rotation, and the observation boundary.

The two things this file is really for:

**A join token is the only credential a new host has**, so every way it
can be misused gets an assertion — replay, expiry, and presenting one
bound to a different node.

**Revoking a node rotates the signing key.** That is the sharp edge of
the whole design: a revoked node still *holds* the key, so removing its
registry entry would not stop it authenticating to other components. A
test that only checked the node disappeared would pass while the install
stayed compromised.
"""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_control import sealing
from eugene_plexus_control.join_tokens import JoinTokenError, JoinTokenStore

NODE_PUBLIC = base64.b64encode(b"x" * 32).decode("ascii")


def _mint(client: TestClient, **body: object) -> str:
    response = client.post("/v1/nodes/join-token", json=body or None)
    assert response.status_code == 201, response.text
    return str(response.json()["token"])


def _enroll(client: TestClient, token: str, name: str, **extra: object) -> object:
    return client.post(
        "/v1/nodes/enroll",
        json={"token": token, "name": name, "publicKey": NODE_PUBLIC, **extra},
    )


def test_enrollment_hands_back_the_installs_signing_key(active_client: TestClient) -> None:
    """The thing a single-watchdog install could not do.

    A driver spawned on one host used to reject a gateway's token from
    another, because each watchdog minted its own key. Enrollment
    handing out *the install's* key is the fix, and it is the whole
    reason the trust root had to become singular.
    """
    response = _enroll(active_client, _mint(active_client), "gpu-box")
    assert response.status_code == 201, response.text
    body = response.json()

    assert body["name"] == "gpu-box"
    assert body["epoch"] == 1
    auth = active_client.app.state.auth_state  # type: ignore[attr-defined]
    assert base64.b64decode(body["signingKey"]) == auth.signing_key
    assert body["controlPublicKey"]
    # The second recipient every secret on that node will be sealed to.
    assert body["recoveryPublicKey"] == sealing.public_from_private(auth.recovery_private_key)


def test_a_join_token_cannot_be_replayed(active_client: TestClient) -> None:
    """409 rather than 401, because the two mean different things to
    whoever is holding the token: "wrong credential" versus "that
    enrollment already happened, go look at the node list"."""
    token = _mint(active_client)
    assert _enroll(active_client, token, "first").status_code == 201
    replayed = _enroll(active_client, token, "second")
    assert replayed.status_code == 409
    assert "already" in replayed.text


def test_a_token_bound_to_one_node_refuses_another(active_client: TestClient) -> None:
    """What binding buys: a leaked token can only do the thing it was
    minted for."""
    token = _mint(active_client, nodeName="attic")
    refused = _enroll(active_client, token, "shed")
    assert refused.status_code == 401
    assert "bound to node" in refused.text
    assert _enroll(active_client, token, "attic").status_code == 201


def test_an_expired_token_is_refused() -> None:
    """Checked against the wall clock, which is fine here precisely
    because join tokens are not replicated state — no other host has to
    agree with this one about when a token died."""
    store = JoinTokenStore()
    # A zero TTL rather than sleeping through a real one. The route
    # clamps the minimum; the store takes what it is given, which is
    # what makes this expressible without reaching past the API.
    minted = store.mint(ttl_seconds=0, node_name=None)
    with pytest.raises(JoinTokenError):
        store.consume(minted.token, node_name="anything")


def test_a_bad_public_key_is_refused(active_client: TestClient) -> None:
    response = active_client.post(
        "/v1/nodes/enroll",
        json={"token": _mint(active_client), "name": "gpu-box", "publicKey": "not base64!!"},
    )
    assert response.status_code == 400


def test_the_node_list_separates_applied_state_from_observation(
    active_client: TestClient,
) -> None:
    """`reachable` defaults to false because nothing has polled yet, and
    false means `down`, never `out`. Nothing is reassigned on it."""
    _enroll(active_client, _mint(active_client), "gpu-box", os="linux", arch="x64")

    listed = active_client.get("/v1/nodes")
    assert listed.status_code == 200
    node = listed.json()["nodes"][0]
    assert node["name"] == "gpu-box"
    assert node["os"] == "linux"
    assert node["reachable"] is False
    assert node["lastSeenEpoch"] is None
    assert node["enrolledAt"] is not None


def test_revoking_a_node_rotates_the_signing_key(active_client: TestClient) -> None:
    """The assertion that matters, and the one it would be easy to skip.

    Checking only that the node disappeared would pass while the revoked
    host kept a key that every component in the install still trusts.
    """
    _enroll(active_client, _mint(active_client), "gpu-box")
    auth = active_client.app.state.auth_state  # type: ignore[attr-defined]
    key_before = auth.signing_key
    id_before = active_client.app.state.machine.state.identity.signingKeyId  # type: ignore[attr-defined]

    response = active_client.delete("/v1/nodes/gpu-box")
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["reason"] == "revocation"
    assert body["revokedNode"] == "gpu-box"

    assert auth.signing_key != key_before, "revocation must re-key the install"
    identity = active_client.app.state.machine.state.identity  # type: ignore[attr-defined]
    assert identity.signingKeyId != id_before
    assert "gpu-box" not in active_client.app.state.machine.state.nodes  # type: ignore[attr-defined]


def test_revoking_a_node_undeclares_what_it_hosted(active_client: TestClient) -> None:
    """The honest consequence of removing a host: the model files on its
    disk are untouched, but nothing in this install will route to them."""
    _enroll(active_client, _mint(active_client), "gpu-box")
    machine = active_client.app.state.machine  # type: ignore[attr-defined]
    machine.append(
        "putRuntime",
        {
            "node": "gpu-box",
            "spec": {"name": "qwen", "engine": "llama_cpp", "modelPath": "/m.gguf"},
        },
    )
    assert machine.state.runtimes

    assert active_client.delete("/v1/nodes/gpu-box").status_code == 202
    assert machine.state.runtimes == {}


def test_revoking_an_unknown_node_is_a_404(active_client: TestClient) -> None:
    assert active_client.delete("/v1/nodes/nowhere").status_code == 404


def test_re_enrolling_the_same_name_replaces_rather_than_conflicts(
    active_client: TestClient,
) -> None:
    """ "Reinstall the OS on the GPU box" has to have a path through the
    API. A rebuilt host presents a new keypair under the same
    operator-chosen name, and refusing that would leave the operator
    revoking and re-adding a node to say the same thing."""
    assert _enroll(active_client, _mint(active_client), "gpu-box").status_code == 201
    replacement = base64.b64encode(b"y" * 32).decode("ascii")
    response = active_client.post(
        "/v1/nodes/enroll",
        json={"token": _mint(active_client), "name": "gpu-box", "publicKey": replacement},
    )
    assert response.status_code == 201
    nodes = active_client.app.state.machine.state.nodes  # type: ignore[attr-defined]
    assert len(nodes) == 1
    assert nodes["gpu-box"].publicKey == replacement
