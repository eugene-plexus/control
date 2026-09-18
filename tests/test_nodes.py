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
import pathlib

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


def test_enrollment_records_where_the_agent_is_and_names_the_key_generation(
    active_client: TestClient,
) -> None:
    """The two fields M7 added to the exchange, and why. Without `url`
    every really-enrolled node had no address: the poller reported "no
    url recorded", forwarding answered 502, the gateway fell back to
    loopback. Without `signingKeyId` a node could not tell a newer re-key
    from a replay."""
    response = _enroll(active_client, _mint(active_client), "gpu-box", url="http://100.64.0.7:8079")
    assert response.status_code == 201, response.text
    assert response.json()["signingKeyId"] == "1"

    node = active_client.get("/v1/nodes/gpu-box").json()
    assert node["url"].rstrip("/") == "http://100.64.0.7:8079"
    # Normalized once on the way into the log, the way every URL is.
    record = active_client.app.state.machine.state.nodes["gpu-box"]  # type: ignore[attr-defined]
    assert record.url == "http://100.64.0.7:8079/"

    # An enrollment without one is accepted and produces a node this root
    # cannot reach — honest, and visible.
    assert _enroll(active_client, _mint(active_client), "shed").status_code == 201
    assert active_client.get("/v1/nodes/shed").json().get("url") is None


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


# --------------------------------------------------------------------------- #
# Re-advertising: a node whose address changed (M9)
# --------------------------------------------------------------------------- #


def _node_keypair() -> tuple[str, str]:
    """`(private seed, public)` base64 Ed25519 — the shape an agent mints
    for signing, alongside the X25519 pair it has sealed to."""
    import nacl.signing

    signing = nacl.signing.SigningKey.generate()
    return (
        base64.b64encode(bytes(signing)).decode("ascii"),
        base64.b64encode(bytes(signing.verify_key)).decode("ascii"),
    )


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
    """The M9 defect, closed.

    Before this the address was announced once, at enrollment, and a host
    that rebooted onto a new tailnet IP left the root holding an address
    nobody was listening on — with no way back, because the only address
    the root had was the stale one.
    """
    private, public = _node_keypair()
    assert (
        _enroll(
            active_client,
            _mint(active_client),
            "gpu-box",
            url="http://100.64.0.7:8079",
            signingPublicKey=public,
        ).status_code
        == 201
    )

    response = _announce(
        active_client, "gpu-box", url="http://100.64.0.9:8079", sequence=1, private=private
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["changed"] is True
    assert body["url"].rstrip("/") == "http://100.64.0.9:8079"
    assert body["sequence"] == 1

    node = active_client.get("/v1/nodes/gpu-box").json()
    assert node["url"].rstrip("/") == "http://100.64.0.9:8079"
    assert node["advertiseSequence"] == 1


def test_an_unchanged_announcement_writes_nothing_to_the_log(
    active_client: TestClient,
) -> None:
    """The common case is a restart that announces the address it already
    had. If that appended, every reboot of every node would grow the log
    for no information — so it must be visibly free, not merely
    idempotent."""
    private, public = _node_keypair()
    _enroll(
        active_client,
        _mint(active_client),
        "gpu-box",
        url="http://100.64.0.7:8079",
        signingPublicKey=public,
    )
    machine = active_client.app.state.machine  # type: ignore[attr-defined]
    before = machine.state.index

    response = _announce(
        active_client, "gpu-box", url="http://100.64.0.7:8079", sequence=1, private=private
    )
    assert response.status_code == 200, response.text
    assert response.json()["changed"] is False
    assert machine.state.index == before


def test_an_announcement_signed_by_the_wrong_node_is_refused(
    active_client: TestClient,
) -> None:
    """A service token names a *kind*, not a host, which is why this is
    signed at all. The signature covers `name`, so one node's
    announcement cannot be replayed against another's record."""
    private_a, public_a = _node_keypair()
    private_b, _public_b = _node_keypair()
    _enroll(
        active_client,
        _mint(active_client),
        "gpu-box",
        url="http://100.64.0.7:8079",
        signingPublicKey=public_a,
    )
    assert public_a  # the key the root holds for gpu-box

    response = _announce(
        active_client, "gpu-box", url="http://evil:8079", sequence=1, private=private_b
    )
    assert response.status_code == 401, response.text
    assert active_client.get("/v1/nodes/gpu-box").json()["url"].rstrip("/") == (
        "http://100.64.0.7:8079"
    )
    # And the right key still works, so the refusal was about the signer.
    assert (
        _announce(
            active_client, "gpu-box", url="http://evil:8079", sequence=1, private=private_a
        ).status_code
        == 200
    )


def test_a_replayed_announcement_cannot_move_a_node_back(active_client: TestClient) -> None:
    """Without the sequence, a captured announcement pins a node to an
    address it has left — a denial of service that costs an attacker
    nothing but a packet capture."""
    private, public = _node_keypair()
    _enroll(
        active_client,
        _mint(active_client),
        "gpu-box",
        url="http://100.64.0.7:8079",
        signingPublicKey=public,
    )
    old = sealing.address_message(name="gpu-box", sequence=1, url="http://100.64.0.7:8079")
    old_signature = sealing.sign_address(private, old)

    assert (
        _announce(
            active_client, "gpu-box", url="http://100.64.0.9:8079", sequence=2, private=private
        ).status_code
        == 200
    )

    replay = active_client.patch(
        "/v1/nodes/gpu-box",
        json={
            "url": "http://100.64.0.7:8079",
            "sequence": 1,
            "signature": old_signature,
        },
    )
    assert replay.status_code == 409, replay.text
    assert active_client.get("/v1/nodes/gpu-box").json()["url"].rstrip("/") == (
        "http://100.64.0.9:8079"
    )


def test_a_node_enrolled_before_signing_keys_is_told_to_re_enroll(
    active_client: TestClient,
) -> None:
    """`signingPublicKey` is optional on the wire so an older agent still
    enrolls. The cost lands here, and it is named rather than left to be
    inferred from a bare 401."""
    private, _public = _node_keypair()
    _enroll(active_client, _mint(active_client), "gpu-box", url="http://100.64.0.7:8079")
    assert active_client.get("/v1/nodes/gpu-box").json().get("signingPublicKey") is None

    response = _announce(
        active_client, "gpu-box", url="http://100.64.0.9:8079", sequence=1, private=private
    )
    assert response.status_code == 401, response.text
    assert "re-enroll" in response.json()["detail"]["detail"].lower()


def test_re_enrollment_resets_the_sequence(active_client: TestClient) -> None:
    """A rebuilt host presents a fresh identity file with its counter at
    zero. If the root kept the old high-water mark, that host could never
    re-advertise — so `enrollNode` replacing the record wholesale is
    load-bearing, not incidental."""
    private, public = _node_keypair()
    _enroll(
        active_client,
        _mint(active_client),
        "gpu-box",
        url="http://a:8079",
        signingPublicKey=public,
    )
    assert (
        _announce(
            active_client, "gpu-box", url="http://b:8079", sequence=7, private=private
        ).status_code
        == 200
    )

    private2, public2 = _node_keypair()
    assert (
        _enroll(
            active_client,
            _mint(active_client),
            "gpu-box",
            url="http://c:8079",
            signingPublicKey=public2,
        ).status_code
        == 201
    )
    assert active_client.get("/v1/nodes/gpu-box").json()["advertiseSequence"] == 0
    assert (
        _announce(
            active_client, "gpu-box", url="http://d:8079", sequence=1, private=private2
        ).status_code
        == 200
    )


def test_announcing_to_an_unknown_node_is_404(active_client: TestClient) -> None:
    private, _public = _node_keypair()
    response = _announce(active_client, "nobody", url="http://a:8079", sequence=1, private=private)
    assert response.status_code == 404, response.text


def test_the_node_list_says_why_a_probe_failed(active_client: TestClient) -> None:
    """Found 2026-09-15: a worker refused this root's tokens as "not yet
    valid (iat)" over half a second of clock skew, the probe client
    recorded exactly that, and the list said only `reachable: false`.
    The reason travels now, as an observation beside `lastSeenAt`."""
    from eugene_plexus_control.nodes_client import NodeProbe

    _enroll(active_client, _mint(active_client), "gpu-box", os="windows", arch="x64")
    active_client.app.state.node_probes = {
        "gpu-box": NodeProbe(
            name="gpu-box",
            reachable=False,
            error="HTTP 401: Invalid token: Bearer token rejected: The token is not yet valid (iat)",
        )
    }
    node = active_client.get("/v1/nodes").json()["nodes"][0]
    assert node["reachable"] is False
    assert "not yet valid (iat)" in node["lastError"]

    active_client.app.state.node_probes = {"gpu-box": NodeProbe(name="gpu-box", reachable=True)}
    assert active_client.get("/v1/nodes").json()["nodes"][0]["lastError"] is None


def test_a_node_nothing_has_polled_yet_is_distinguishable_from_one_that_is_down() -> None:
    """Reported 2026-09-17: update the container, unlock the root, and
    every node reads as down for the next fifteen seconds.

    `reachable: false` is the honest default for the field -- nothing is
    reassigned on the strength of it -- but a console cannot print
    "down" from it alone, because a root that has never polled says
    exactly the same thing as one whose probes all failed. The signature
    that separates them is on the wire and is asserted here, because it
    is what the Nodes page renders as "checking" rather than "down":

        never polled   reachable=false, lastError=null, lastSeenAt=null
        probe failed   reachable=false, lastError=<why>

    A failed probe ALWAYS carries a reason -- every `reachable=False`
    return in `nodes_client` sets one -- which is what makes the absence
    of a reason mean "no observation" rather than "no explanation".
    """
    from eugene_plexus_control.nodes_client import NodeProbe

    assert NodeProbe(name="x", reachable=False, error="connection refused").error
    # And the one the route synthesises when there is no probe at all.
    from eugene_plexus_control.applied import NodeRecord
    from eugene_plexus_control.routes.nodes import _to_node

    unpolled = _to_node(NodeRecord(name="gpu-box", url="http://a:8079"), None)
    assert unpolled.reachable is False
    assert unpolled.lastError is None
    assert unpolled.lastSeenAt is None


def test_a_locked_root_notices_an_unlock_without_waiting_a_poll_interval() -> None:
    """The other half of the same report.

    While locked the poller does nothing but re-read a local variable,
    and sleeping the full interval there is the operator's wait AFTER
    unlocking before any node has been observed at all. A second of
    no-op iterations costs nothing and covers every unlock route there
    is, including ones added later -- an event set by the login handler
    would have to be remembered by each of them.
    """
    from eugene_plexus_control.app import _LOCKED_POLL_SLEEP_SECONDS

    assert _LOCKED_POLL_SLEEP_SECONDS <= 1.0
    source = (
        pathlib.Path(__file__).resolve().parents[1] / "src" / "eugene_plexus_control" / "app.py"
    ).read_text(encoding="utf-8")
    locked_branch = source.split("announced_locked = True", 1)[1].split("continue", 1)[0]
    # The locked branch must not sleep the poll interval on its own.
    assert "_LOCKED_POLL_SLEEP_SECONDS" in locked_branch
    assert "min(" in locked_branch


def test_a_probe_error_carries_the_refusing_agents_own_words() -> None:
    """`_describe` used to stop at the status code."""
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


def test_a_minted_token_can_be_listed_and_withdrawn(active_client: TestClient) -> None:
    """The half that did not exist until 2026-09-17.

    Minting was the only verb: a token pasted into the wrong window
    stayed live for the rest of its TTL and the only remedy was to wait.
    """
    minted = active_client.post("/v1/nodes/join-token", json={"nodeName": "attic"})
    assert minted.status_code == 201, minted.text
    token_id = minted.json()["id"]
    token = minted.json()["token"]

    listed = active_client.get("/v1/nodes/join-tokens")
    assert listed.status_code == 200, listed.text
    rows = listed.json()["tokens"]
    assert [r["id"] for r in rows] == [token_id]
    assert rows[0]["nodeName"] == "attic"
    assert rows[0]["used"] is False

    assert active_client.delete(f"/v1/nodes/join-tokens/{token_id}").status_code == 204
    assert active_client.get("/v1/nodes/join-tokens").json()["tokens"] == []
    # And it is dead, not merely hidden.
    assert _enroll(active_client, token, "attic").status_code == 401


def test_the_listing_carries_no_token(active_client: TestClient) -> None:
    """The store keeps a hash and never the secret, so there is nothing
    here to steal and nothing to re-read a lost token from. A listing
    that leaked one would quietly undo the property the whole store is
    built on."""
    token = _mint(active_client)
    body = active_client.get("/v1/nodes/join-tokens").text
    assert token not in body
    assert "token" not in active_client.get("/v1/nodes/join-tokens").json()["tokens"][0]


def test_an_id_is_not_derived_from_the_token(active_client: TestClient) -> None:
    """A handle that were a prefix or a hash of the secret would mean
    listing moves the credential's verifier around. It is a separate
    random value, and this is the assertion that says so."""
    minted = active_client.post("/v1/nodes/join-token").json()
    token, token_id = minted["token"], minted["id"]
    assert token_id not in token
    assert token[: len(token_id)] != token_id
    from eugene_plexus_control import security

    assert token_id not in security.hash_join_token(token)


def test_a_revoked_token_is_forgotten_rather_than_flagged(active_client: TestClient) -> None:
    """Presenting a withdrawn token is indistinguishable from presenting
    one that never existed. A revoked credential should not confirm it
    was ever real."""
    minted = active_client.post("/v1/nodes/join-token").json()
    active_client.delete(f"/v1/nodes/join-tokens/{minted['id']}")

    withdrawn = _enroll(active_client, minted["token"], "attic")
    invented = _enroll(active_client, "eptk_" + "0" * 40, "attic")
    assert withdrawn.status_code == invented.status_code == 401
    assert withdrawn.json()["detail"] == invented.json()["detail"]


def test_revoking_an_unknown_id_is_404_not_a_silent_success(active_client: TestClient) -> None:
    """ "Already gone" and "you are revoking the wrong thing" want
    different next moves, and the operator is looking at a list this
    root just served."""
    response = active_client.delete("/v1/nodes/join-tokens/deadbeefdeadbeef")
    assert response.status_code == 404
    assert "expired" in response.text


def test_a_spent_token_is_listed_and_can_be_cleared(active_client: TestClient) -> None:
    """The sweep keeps a used token until it expires on purpose, so a
    replay answers 409 rather than 401 — which means the listing has to
    show it, and revoking it has to be allowed and pointless rather than
    refused. Membership is in the replicated log and is untouched."""
    token = _mint(active_client)
    assert _enroll(active_client, token, "gpu-box").status_code == 201

    rows = active_client.get("/v1/nodes/join-tokens").json()["tokens"]
    assert len(rows) == 1 and rows[0]["used"] is True

    assert active_client.delete(f"/v1/nodes/join-tokens/{rows[0]['id']}").status_code == 204
    assert active_client.get("/v1/nodes/gpu-box").status_code == 200


def test_listing_and_revoking_are_operator_only(active_client: TestClient) -> None:
    """What a token is bound to and when it dies is a picture of who is
    being invited into this install, which is not a service's business."""
    from eugene_plexus_control import security

    auth = active_client.app.state.auth_state  # type: ignore[attr-defined]
    # Argument order is load-bearing for the SECRET SCAN, which is a silly
    # sentence and is true: gitleaks' `generic-api-key` rule matches a
    # `*_key=` keyword followed by more arguments on the same line, so this
    # call read as a hard-coded credential and turned this repo's CI red on
    # 2026-09-17 without anyone noticing. `signing_key` last is the shape
    # `test_auth.py` already uses and the scan already accepts. An allowlist
    # would have been the worse fix -- it is the rule earning its keep.
    service = security.issue_service_token(
        kind="gateway", ttl_seconds=60, signing_key=auth.signing_key
    )
    headers = {"Authorization": f"Bearer {service}"}
    assert active_client.get("/v1/nodes/join-tokens", headers=headers).status_code == 401
    assert active_client.delete("/v1/nodes/join-tokens/x", headers=headers).status_code == 401
