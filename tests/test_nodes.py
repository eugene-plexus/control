"""Enrollment, revocation, and the observation boundary.

The things this file is really for:

**A join token is the only credential a new host has**, so every way it
can be misused gets an assertion — replay, expiry, and presenting one
bound to a different node.

**Enrollment moves no private key** (2026-09-25). The node generates its
own token key and sends the public half; the answer carries a trust
bundle that already lists it. Until then this answer carried the
install's signing key in plaintext, which made every node the install.

**Revoking a node removes its key from the bundle, and nothing else.**
A test that only checked the node disappeared would pass while the
revoked host's tokens kept verifying; these check the tokens.
"""

from __future__ import annotations

import base64
import pathlib

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_control import sealing, tokens
from eugene_plexus_control.join_tokens import JoinTokenError, JoinTokenStore

from .conftest import NodeKeys, enroll, mint_join_token, node_keys


def _enroll_keys(client: TestClient, keys: NodeKeys, token: str, **extra: object) -> object:
    return client.post("/v1/nodes/enroll", json=keys.body(token, **extra))


def _bundle(client: TestClient, response: object) -> tokens.TrustBundle:
    authority = client.app.state.machine.state.identity.controlPublicKey  # type: ignore[attr-defined]
    jws = response.json()["trustBundle"]["jws"]  # type: ignore[attr-defined]
    return tokens.parse_bundle(jws, authority=authority)


def test_enrollment_moves_no_private_key_and_lists_the_node(active_client: TestClient) -> None:
    """The whole point of row 3. The answer carries public material only,
    and a bundle signed by the key the node is about to pin."""
    keys, response = enroll(active_client, "gpu-box")
    assert response.status_code == 201, response.text
    body = response.json()

    assert body["name"] == "gpu-box"
    assert body["epoch"] == 1
    assert "signingKey" not in body and "signingKeyId" not in body
    auth = active_client.app.state.auth_state  # type: ignore[attr-defined]
    assert base64.b64encode(auth.signing_key).decode() not in response.text
    assert body["recoveryPublicKey"] == sealing.public_from_private(auth.recovery_private_key)

    bundle = _bundle(active_client, response)
    issuers = {k.issuer: k for k in bundle.keys.values()}
    assert issuers["node:gpu-box"].kid == keys.token.kid
    assert issuers["node:gpu-box"].grants == frozenset({"node"})
    assert tokens.GRANT_AUTHORITY in issuers["control"].grants


def test_a_join_token_can_grant_the_gateway_role_and_a_node_cannot_ask(
    active_client: TestClient,
) -> None:
    """D4: the gateway grant comes from the operator's join token, never
    from the node. An enrollment body that names grants is ignored."""
    _, granted = enroll(active_client, "nas", grants=["gateway"])
    assert granted.status_code == 201, granted.text
    keys = node_keys("worker")
    asked = active_client.post(
        "/v1/nodes/enroll",
        json={**keys.body(mint_join_token(active_client)), "grants": ["gateway"]},
    )
    assert asked.status_code == 201, asked.text

    bundle = _bundle(active_client, asked)
    by_issuer = {k.issuer: k.grants for k in bundle.keys.values()}
    assert by_issuer["node:nas"] == frozenset({"node", "gateway"})
    assert by_issuer["node:worker"] == frozenset({"node"})
    listed = {n["name"]: n for n in active_client.get("/v1/nodes").json()["nodes"]}
    assert listed["nas"]["grants"] == ["gateway", "node"]


def test_enrollment_records_where_the_agent_is(active_client: TestClient) -> None:
    """Without `url` every really-enrolled node had no address (M7)."""
    _, response = enroll(active_client, "gpu-box", url="http://100.64.0.7:8079")
    assert response.status_code == 201, response.text

    node = active_client.get("/v1/nodes/gpu-box").json()
    assert node["url"].rstrip("/") == "http://100.64.0.7:8079"
    record = active_client.app.state.machine.state.nodes["gpu-box"]  # type: ignore[attr-defined]
    assert record.url == "http://100.64.0.7:8079/"
    assert node["tokenPublicKey"] == record.tokenPublicKey

    _, shed = enroll(active_client, "shed")
    assert shed.status_code == 201
    assert active_client.get("/v1/nodes/shed").json().get("url") is None


def test_a_join_token_cannot_be_replayed(active_client: TestClient) -> None:
    """409 rather than 401: "wrong credential" versus "that enrollment
    already happened, go look at the node list"."""
    token = mint_join_token(active_client)
    assert _enroll_keys(active_client, node_keys("first"), token).status_code == 201  # type: ignore[attr-defined]
    replayed = _enroll_keys(active_client, node_keys("second"), token)
    assert replayed.status_code == 409  # type: ignore[attr-defined]
    assert "already" in replayed.text  # type: ignore[attr-defined]


def test_a_token_bound_to_one_node_refuses_another(active_client: TestClient) -> None:
    token = mint_join_token(active_client, nodeName="attic")
    refused = _enroll_keys(active_client, node_keys("shed"), token)
    assert refused.status_code == 401  # type: ignore[attr-defined]
    assert "bound to node" in refused.text  # type: ignore[attr-defined]
    assert _enroll_keys(active_client, node_keys("attic"), token).status_code == 201  # type: ignore[attr-defined]


def test_an_expired_token_is_refused() -> None:
    store = JoinTokenStore()
    minted = store.mint(ttl_seconds=0, node_name=None)
    with pytest.raises(JoinTokenError):
        store.consume(minted.token, node_name="anything")


@pytest.mark.parametrize("field", ["publicKey", "signingPublicKey", "tokenPublicKey"])
def test_a_bad_public_key_is_refused(active_client: TestClient, field: str) -> None:
    body = node_keys("gpu-box").body(mint_join_token(active_client))
    body[field] = "not base64!!"
    assert active_client.post("/v1/nodes/enroll", json=body).status_code == 400


def test_a_token_key_another_node_holds_is_refused(active_client: TestClient) -> None:
    """A copied `node.yaml` is not a new node: one key, two issuers, would
    let one machine's tokens be read as the other's."""
    keys, first = enroll(active_client, "gpu-box")
    assert first.status_code == 201
    clone = NodeKeys(
        name="impostor",
        public=keys.public,
        signing_private=keys.signing_private,
        signing_public=keys.signing_public,
        token=tokens.Signer(key=keys.token.key, issuer="node:impostor"),
    )
    refused = _enroll_keys(active_client, clone, mint_join_token(active_client))
    assert refused.status_code == 400  # type: ignore[attr-defined]
    assert "already holds" in refused.text  # type: ignore[attr-defined]


def test_a_locked_root_refuses_enrollment_without_spending_the_token(
    active_client: TestClient,
) -> None:
    """Found 2026-09-24: the token was consumed before the lock check."""
    token = mint_join_token(active_client)
    auth = active_client.app.state.auth_state  # type: ignore[attr-defined]
    saved = auth.control_private_key
    auth.control_private_key = None
    assert _enroll_keys(active_client, node_keys("gpu-box"), token).status_code == 503  # type: ignore[attr-defined]
    auth.control_private_key = saved
    assert _enroll_keys(active_client, node_keys("gpu-box"), token).status_code == 201  # type: ignore[attr-defined]


def test_the_node_list_separates_applied_state_from_observation(
    active_client: TestClient,
) -> None:
    enroll(active_client, "gpu-box", os="linux", arch="x64")
    node = active_client.get("/v1/nodes").json()["nodes"][0]
    assert node["name"] == "gpu-box"
    assert node["os"] == "linux"
    assert node["reachable"] is False
    assert node["lastSeenEpoch"] is None
    assert node["trustBundleVersion"] is None
    assert node["enrolledAt"] is not None


# --------------------------------------------------------------------------- #
# Revocation
# --------------------------------------------------------------------------- #


def _verify_at_control(client: TestClient, token: str) -> tokens.Claims:
    machine = client.app.state.machine  # type: ignore[attr-defined]
    view = client.app.state.trust.view_for(machine.state)  # type: ignore[attr-defined]
    return tokens.verify(
        token, bundle=view, recipient="control", classes=[tokens.TYP_SERVICE, tokens.TYP_SESSION]
    )


def test_revoking_a_node_ends_its_tokens_and_nobody_elses(active_client: TestClient) -> None:
    """The assertion that matters. Checking only that the node disappeared
    would pass while its tokens kept verifying; and the operator's own
    session must survive it, because nothing else changed."""
    keys, _ = enroll(active_client, "gpu-box")
    other, _ = enroll(active_client, "attic")
    token = keys.service_token()
    assert _verify_at_control(active_client, token).issuer_node == "gpu-box"
    auth = active_client.app.state.auth_state  # type: ignore[attr-defined]
    key_before = auth.signing_key

    response = active_client.delete("/v1/nodes/gpu-box")
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["reason"] == "revocation"
    assert body["revokedNode"] == "gpu-box"
    assert body["version"] >= 1

    with pytest.raises(tokens.TokenError, match="not in the trust bundle"):
        _verify_at_control(active_client, token)
    assert _verify_at_control(active_client, other.service_token()).issuer_node == "attic"
    assert auth.signing_key == key_before, "nothing is rotated"
    assert active_client.get("/v1/nodes").status_code == 200, "the operator is still signed in"


def test_a_revocation_is_one_revoke_node_entry(active_client: TestClient) -> None:
    enroll(active_client, "gpu-box")
    machine = active_client.app.state.machine  # type: ignore[attr-defined]
    index_before = machine.state.index
    assert active_client.delete("/v1/nodes/gpu-box").status_code == 202
    entries = machine.read_page(index_before, 100)
    assert [e["op"] for e in entries] == ["revokeNode"]


def test_revoking_a_node_undeclares_what_it_hosted(active_client: TestClient) -> None:
    enroll(active_client, "gpu-box")
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


def test_a_revocation_the_root_cannot_publish_revokes_nothing(active_client: TestClient) -> None:
    """A revocation no node would hear of is worse than none: it would read
    as done. With the identity key out of memory nothing is appended."""
    enroll(active_client, "gpu-box")
    machine = active_client.app.state.machine  # type: ignore[attr-defined]
    auth = active_client.app.state.auth_state  # type: ignore[attr-defined]
    index_before = machine.state.index
    auth.control_private_key = None
    response = active_client.delete("/v1/nodes/gpu-box")
    assert response.status_code == 503, response.text
    assert "gpu-box" in machine.state.nodes
    assert machine.state.index == index_before


def test_revoking_an_unknown_node_is_a_404(active_client: TestClient) -> None:
    assert active_client.delete("/v1/nodes/nowhere").status_code == 404


def test_re_enrolling_the_same_name_replaces_rather_than_conflicts(
    active_client: TestClient,
) -> None:
    """ "Reinstall the OS on the GPU box" has to have a path through the
    API, and the old key must stop verifying when it does."""
    old, first = enroll(active_client, "gpu-box")
    assert first.status_code == 201
    new, second = enroll(active_client, "gpu-box")
    assert second.status_code == 201
    nodes = active_client.app.state.machine.state.nodes  # type: ignore[attr-defined]
    assert len(nodes) == 1
    assert nodes["gpu-box"].publicKey == new.public
    with pytest.raises(tokens.TokenError):
        _verify_at_control(active_client, old.service_token())
    assert _verify_at_control(active_client, new.service_token())


# --------------------------------------------------------------------------- #
# Re-advertising: a node whose address changed (M9)
# --------------------------------------------------------------------------- #


def _announce(client: TestClient, name: str, *, url: str, sequence: int, private: str) -> object:
    message = sealing.address_message(name=name, sequence=sequence, url=url)
    return client.patch(
        f"/v1/nodes/{name}",
        json={
            "url": url,
            "sequence": sequence,
            "signature": sealing.sign_address(private, message),
        },
    )


def test_a_node_that_moved_can_say_so_and_the_root_records_it(
    active_client: TestClient,
) -> None:
    keys, response = enroll(active_client, "gpu-box", url="http://100.64.0.7:8079")
    assert response.status_code == 201

    moved = _announce(
        active_client,
        "gpu-box",
        url="http://100.64.0.9:8079",
        sequence=1,
        private=keys.signing_private,
    )
    assert moved.status_code == 200, moved.text  # type: ignore[attr-defined]
    body = moved.json()  # type: ignore[attr-defined]
    assert body["changed"] is True
    assert body["url"].rstrip("/") == "http://100.64.0.9:8079"
    assert body["sequence"] == 1
    node = active_client.get("/v1/nodes/gpu-box").json()
    assert node["url"].rstrip("/") == "http://100.64.0.9:8079"
    assert node["advertiseSequence"] == 1


def test_an_unchanged_announcement_writes_nothing_to_the_log(
    active_client: TestClient,
) -> None:
    keys, _ = enroll(active_client, "gpu-box", url="http://100.64.0.7:8079")
    machine = active_client.app.state.machine  # type: ignore[attr-defined]
    before = machine.state.index
    response = _announce(
        active_client,
        "gpu-box",
        url="http://100.64.0.7:8079",
        sequence=1,
        private=keys.signing_private,
    )
    assert response.status_code == 200, response.text  # type: ignore[attr-defined]
    assert response.json()["changed"] is False  # type: ignore[attr-defined]
    assert machine.state.index == before


def test_an_announcement_signed_by_the_wrong_node_is_refused(
    active_client: TestClient,
) -> None:
    """The signature covers `name`, so one node's announcement cannot be
    replayed against another's record."""
    keys, _ = enroll(active_client, "gpu-box", url="http://100.64.0.7:8079")
    stranger = node_keys("elsewhere")
    response = _announce(
        active_client,
        "gpu-box",
        url="http://100.64.0.66:8079",
        sequence=1,
        private=stranger.signing_private,
    )
    assert response.status_code == 401, response.text  # type: ignore[attr-defined]
    assert active_client.get("/v1/nodes/gpu-box").json()["url"].rstrip("/") == (
        "http://100.64.0.7:8079"
    )
    right = _announce(
        active_client,
        "gpu-box",
        url="http://100.64.0.66:8079",
        sequence=1,
        private=keys.signing_private,
    )
    assert right.status_code == 200  # type: ignore[attr-defined]


def test_a_replayed_announcement_cannot_move_a_node_back(active_client: TestClient) -> None:
    keys, _ = enroll(active_client, "gpu-box", url="http://100.64.0.7:8079")
    old = sealing.address_message(name="gpu-box", sequence=1, url="http://100.64.0.7:8079")
    old_signature = sealing.sign_address(keys.signing_private, old)
    moved = _announce(
        active_client,
        "gpu-box",
        url="http://100.64.0.9:8079",
        sequence=2,
        private=keys.signing_private,
    )
    assert moved.status_code == 200  # type: ignore[attr-defined]
    replay = active_client.patch(
        "/v1/nodes/gpu-box",
        json={"url": "http://100.64.0.7:8079", "sequence": 1, "signature": old_signature},
    )
    assert replay.status_code == 409, replay.text
    assert active_client.get("/v1/nodes/gpu-box").json()["url"].rstrip("/") == (
        "http://100.64.0.9:8079"
    )


def test_re_enrollment_resets_the_sequence(active_client: TestClient) -> None:
    keys, _ = enroll(active_client, "gpu-box", url="http://a:8079")
    assert (
        _announce(
            active_client, "gpu-box", url="http://b:8079", sequence=7, private=keys.signing_private
        ).status_code  # type: ignore[attr-defined]
        == 200
    )
    keys2, second = enroll(active_client, "gpu-box", url="http://c:8079")
    assert second.status_code == 201
    assert active_client.get("/v1/nodes/gpu-box").json()["advertiseSequence"] == 0
    assert (
        _announce(
            active_client, "gpu-box", url="http://d:8079", sequence=1, private=keys2.signing_private
        ).status_code  # type: ignore[attr-defined]
        == 200
    )


def test_announcing_to_an_unknown_node_is_404(active_client: TestClient) -> None:
    keys = node_keys("nobody")
    response = _announce(
        active_client, "nobody", url="http://a:8079", sequence=1, private=keys.signing_private
    )
    assert response.status_code == 404, response.text  # type: ignore[attr-defined]


def test_the_node_list_says_why_a_probe_failed(active_client: TestClient) -> None:
    from eugene_plexus_control.nodes_client import NodeProbe

    enroll(active_client, "gpu-box", os="windows", arch="x64")
    active_client.app.state.node_probes = {  # type: ignore[attr-defined]
        "gpu-box": NodeProbe(
            name="gpu-box",
            reachable=False,
            error="HTTP 401: Invalid token: Bearer token rejected: The token is not yet valid (iat)",
        )
    }
    node = active_client.get("/v1/nodes").json()["nodes"][0]
    assert node["reachable"] is False
    assert "not yet valid (iat)" in node["lastError"]

    active_client.app.state.node_probes = {  # type: ignore[attr-defined]
        "gpu-box": NodeProbe(name="gpu-box", reachable=True, trust_bundle_version=7)
    }
    node = active_client.get("/v1/nodes").json()["nodes"][0]
    assert node["lastError"] is None
    assert node["trustBundleVersion"] == 7


def test_a_node_nothing_has_polled_yet_is_distinguishable_from_one_that_is_down() -> None:
    from eugene_plexus_control.applied import NodeRecord
    from eugene_plexus_control.nodes_client import NodeProbe
    from eugene_plexus_control.routes.nodes import _to_node

    assert NodeProbe(name="x", reachable=False, error="connection refused").error
    unpolled = _to_node(NodeRecord(name="gpu-box", url="http://a:8079"), None)
    assert unpolled.reachable is False
    assert unpolled.lastError is None
    assert unpolled.lastSeenAt is None


def test_a_locked_root_notices_an_unlock_without_waiting_a_poll_interval() -> None:
    from eugene_plexus_control.app import _LOCKED_POLL_SLEEP_SECONDS

    assert _LOCKED_POLL_SLEEP_SECONDS <= 1.0
    source = (
        pathlib.Path(__file__).resolve().parents[1] / "src" / "eugene_plexus_control" / "app.py"
    ).read_text(encoding="utf-8")
    locked_branch = source.split("announced_locked = True", 1)[1].split("continue", 1)[0]
    assert "_LOCKED_POLL_SLEEP_SECONDS" in locked_branch
    assert "min(" in locked_branch


def test_a_probe_error_carries_the_refusing_agents_own_words() -> None:
    import httpx

    from eugene_plexus_control.nodes_client import _describe

    request = httpx.Request("GET", "http://node:8079/v1/node")
    problem = httpx.Response(
        401,
        json={
            "detail": {
                "title": "Invalid token",
                "detail": "Bearer token rejected: The token is not yet valid (iat)",
                "status": 401,
            }
        },
        request=request,
    )
    described = _describe(httpx.HTTPStatusError("401", request=request, response=problem))
    assert (
        described
        == "HTTP 401: Invalid token: Bearer token rejected: The token is not yet valid (iat)"
    )
    bare = httpx.Response(502, text="<html>bad gateway</html>", request=request)
    assert _describe(httpx.HTTPStatusError("502", request=request, response=bare)) == "HTTP 502"


# --------------------------------------------------------------------------- #
# Join tokens
# --------------------------------------------------------------------------- #


def test_a_minted_token_can_be_listed_and_withdrawn(active_client: TestClient) -> None:
    minted = active_client.post(
        "/v1/nodes/join-token", json={"nodeName": "attic", "grants": ["gateway"]}
    )
    assert minted.status_code == 201, minted.text
    token_id = minted.json()["id"]
    token = minted.json()["token"]
    assert minted.json()["grants"] == ["gateway"]

    rows = active_client.get("/v1/nodes/join-tokens").json()["tokens"]
    assert [r["id"] for r in rows] == [token_id]
    assert rows[0]["nodeName"] == "attic"
    assert rows[0]["used"] is False
    assert rows[0]["grants"] == ["gateway"]

    assert active_client.delete(f"/v1/nodes/join-tokens/{token_id}").status_code == 204
    assert active_client.get("/v1/nodes/join-tokens").json()["tokens"] == []
    assert _enroll_keys(active_client, node_keys("attic"), token).status_code == 401  # type: ignore[attr-defined]


def test_the_listing_carries_no_token(active_client: TestClient) -> None:
    token = mint_join_token(active_client)
    body = active_client.get("/v1/nodes/join-tokens").text
    assert token not in body
    assert "token" not in active_client.get("/v1/nodes/join-tokens").json()["tokens"][0]


def test_an_id_is_not_derived_from_the_token(active_client: TestClient) -> None:
    minted = active_client.post("/v1/nodes/join-token").json()
    token, token_id = minted["token"], minted["id"]
    assert token_id not in token
    assert token[: len(token_id)] != token_id
    from eugene_plexus_control import security

    assert token_id not in security.hash_join_token(token)


def test_a_revoked_token_is_forgotten_rather_than_flagged(active_client: TestClient) -> None:
    minted = active_client.post("/v1/nodes/join-token").json()
    active_client.delete(f"/v1/nodes/join-tokens/{minted['id']}")
    withdrawn = _enroll_keys(active_client, node_keys("attic"), minted["token"])
    invented = _enroll_keys(active_client, node_keys("attic"), "eptk_" + "0" * 40)
    assert withdrawn.status_code == invented.status_code == 401  # type: ignore[attr-defined]
    assert withdrawn.json()["detail"] == invented.json()["detail"]  # type: ignore[attr-defined]


def test_revoking_an_unknown_id_is_404_not_a_silent_success(active_client: TestClient) -> None:
    response = active_client.delete("/v1/nodes/join-tokens/deadbeefdeadbeef")
    assert response.status_code == 404
    assert "expired" in response.text


def test_a_spent_token_is_listed_and_can_be_cleared(active_client: TestClient) -> None:
    token = mint_join_token(active_client)
    assert _enroll_keys(active_client, node_keys("gpu-box"), token).status_code == 201  # type: ignore[attr-defined]
    rows = active_client.get("/v1/nodes/join-tokens").json()["tokens"]
    assert len(rows) == 1 and rows[0]["used"] is True
    assert active_client.delete(f"/v1/nodes/join-tokens/{rows[0]['id']}").status_code == 204
    assert active_client.get("/v1/nodes/gpu-box").status_code == 200


def test_listing_and_revoking_are_operator_only(active_client: TestClient) -> None:
    """A node's own service token, even the gateway's, is not an operator."""
    keys, _ = enroll(active_client, "nas", grants=["gateway"])
    for sub in ("agent", "gateway"):
        headers = {"Authorization": f"Bearer {keys.service_token(sub=sub)}"}
        assert active_client.get("/v1/nodes/join-tokens", headers=headers).status_code == 401
        assert active_client.delete("/v1/nodes/join-tokens/x", headers=headers).status_code == 401
        assert active_client.delete("/v1/nodes/nas", headers=headers).status_code == 401
