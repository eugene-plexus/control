"""What address a node is allowed to name for itself (R2.4, review §6.2 #15).

The signature and the sequence were never the defect. A **correctly
authenticated** worker could name any URL at all, with `is_loopback_host`
as the only filter — so `http://169.254.169.254/` was a valid address for
a node, after which the root probes it every poll and the console proxy
dials it carrying the operator's own bearer token.

Two rules under test, and they fail differently on purpose: an address
that can never be a node is a **400**, and a node moving itself somewhere
further away than it was enrolled is a **409** with a remedy.
"""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_control import node_address, sealing, tokens

NODE_PUBLIC = base64.b64encode(b"x" * 32).decode("ascii")


def _keypair() -> tuple[str, str]:
    """`(private seed, public)` base64 Ed25519 — the shape an agent mints."""
    import nacl.signing

    signing = nacl.signing.SigningKey.generate()
    return (
        base64.b64encode(bytes(signing)).decode("ascii"),
        base64.b64encode(bytes(signing.verify_key)).decode("ascii"),
    )


def _mint(client: TestClient) -> str:
    response = client.post("/v1/nodes/join-token")
    assert response.status_code == 201, response.text
    return str(response.json()["token"])


def _enroll(client: TestClient, name: str, *, url: str, public: str) -> object:
    return client.post(
        "/v1/nodes/enroll",
        json={
            "token": _mint(client),
            "name": name,
            "publicKey": NODE_PUBLIC,
            "url": url,
            "signingPublicKey": public,
            "tokenPublicKey": tokens.public_b64(tokens.generate_private_key()),
        },
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


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/",  # AWS/GCP/Azure instance metadata
        "http://[fe80::1]:8079",  # IPv6 link-local
        "http://0.0.0.0:8079",  # the bind wildcard, not an address
        "http://224.0.0.1:8079",  # multicast
        "file:///etc/shadow",
        "gopher://10.0.0.1:8079",
    ],
)
def test_a_node_cannot_name_an_address_that_is_never_a_node(
    active_client: TestClient, url: str
) -> None:
    """The refusal is 400 and the record does not move.

    Signed, sequenced, and from the right node — every authentication
    check passes, which is the whole point: this is not an authentication
    defect and cannot be fixed in the signature path.
    """
    private, public = _keypair()
    assert (
        _enroll(active_client, "gpu-box", url="http://10.0.0.7:8079", public=public).status_code
        == 201
    )

    response = _announce(active_client, "gpu-box", url=url, sequence=1, private=private)
    assert response.status_code == 400, response.text
    assert (
        active_client.get("/v1/nodes/gpu-box").json()["url"].rstrip("/") == "http://10.0.0.7:8079"
    )


def test_enrollment_refuses_the_same_addresses(active_client: TestClient) -> None:
    """Otherwise the rule is a rule about one endpoint rather than about
    what a node's address is, and a join token buys the metadata URL."""
    _private, public = _keypair()
    response = _enroll(active_client, "gpu-box", url="http://169.254.169.254/", public=public)
    assert response.status_code == 400, response.text


def test_a_refused_enrollment_does_not_burn_the_join_token(active_client: TestClient) -> None:
    """The order matters, and only a second attempt can show it.

    Added by the sabotage pass: moving the address check to *after*
    `store.consume` escaped every check here, because the refusal looks
    identical either way. A join token is single-use and
    operator-minted, so spending one on a request that was never going
    to be recorded means the operator mints another for no reason --
    on a host whose agent is sitting at a prompt.
    """
    _private, public = _keypair()
    token = _mint(active_client)

    bad = active_client.post(
        "/v1/nodes/enroll",
        json={
            "token": token,
            "name": "gpu-box",
            "publicKey": NODE_PUBLIC,
            "url": "http://169.254.169.254/",
            "signingPublicKey": public,
            "tokenPublicKey": tokens.public_b64(tokens.generate_private_key()),
        },
    )
    assert bad.status_code == 400, bad.text

    # The same token, now with an address that can be a node.
    good = active_client.post(
        "/v1/nodes/enroll",
        json={
            "token": token,
            "name": "gpu-box",
            "publicKey": NODE_PUBLIC,
            "url": "http://10.0.0.7:8079",
            "signingPublicKey": public,
            "tokenPublicKey": tokens.public_b64(tokens.generate_private_key()),
        },
    )
    assert good.status_code == 201, good.text


def test_a_node_cannot_move_itself_onto_the_open_internet(active_client: TestClient) -> None:
    """The transition a homelab node never makes by itself.

    A worker whose signing key leaked re-advertises to a host the
    attacker controls; from then on the root probes it and every console
    proxy hop hands it the operator's bearer. Refused, with the remedy
    named — re-enrollment, which needs an operator-minted join token and
    is therefore the confirmation this rule is asking for.
    """
    private, public = _keypair()
    assert (
        _enroll(active_client, "gpu-box", url="http://10.0.0.7:8079", public=public).status_code
        == 201
    )

    response = _announce(
        active_client, "gpu-box", url="http://8.8.8.8:8079", sequence=1, private=private
    )
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]["detail"].lower()
    assert "re-enroll" in detail
    assert (
        active_client.get("/v1/nodes/gpu-box").json()["url"].rstrip("/") == "http://10.0.0.7:8079"
    )

    # A hostname is public too: this root cannot know who resolves it.
    assert (
        _announce(
            active_client,
            "gpu-box",
            url="http://evil.example.com:8079",
            sequence=2,
            private=private,
        ).status_code
        == 409
    )


def test_the_announcements_every_install_actually_makes_still_work(
    active_client: TestClient,
) -> None:
    """The rule has to be invisible to the two moves that happen daily,
    or it is a rule against the product.

    loopback -> private is the S5 Reach switch, and it is a *click* on
    Home; private -> private is DHCP after a reboot. A node enrolled on
    the open internet — Troy's GPUs in another building — stays there.
    """
    private, public = _keypair()
    _enroll(active_client, "solo", url="http://127.0.0.1:8079", public=public)
    assert (
        _announce(
            active_client, "solo", url="http://192.168.16.75:8079", sequence=1, private=private
        ).status_code
        == 200
    )
    assert (
        _announce(
            active_client, "solo", url="http://192.168.16.90:8079", sequence=2, private=private
        ).status_code
        == 200
    )
    # A tailnet address is 100.64.0.0/10, which Python's `is_private`
    # answers False for. Classified off `is_global` instead, or the
    # Reach switch would be refused on the deployment this is built for.
    assert (
        _announce(
            active_client, "solo", url="http://100.64.0.9:8079", sequence=3, private=private
        ).status_code
        == 200
    )

    private2, public2 = _keypair()
    _enroll(active_client, "remote", url="http://9.9.9.9:8079", public=public2)
    assert (
        _announce(
            active_client, "remote", url="http://1.1.1.1:8079", sequence=1, private=private2
        ).status_code
        == 200
    )
    # And narrowing is always allowed: coming home onto the LAN is not a
    # move an attacker gains anything from.
    assert (
        _announce(
            active_client, "remote", url="http://10.0.0.4:8079", sequence=2, private=private2
        ).status_code
        == 200
    )


def test_the_classes_are_built_from_reachability_not_from_is_private() -> None:
    """The measurement that shaped the rule, pinned so it cannot drift.

    `ipaddress.ip_address("100.64.0.7").is_private` is **False** on the
    Python both installers provision, and that address is where a
    tailnet node lives.
    """
    import ipaddress

    assert ipaddress.ip_address("100.64.0.7").is_private is False
    # And the other half of the same trap, found by this test going red:
    # the documentation ranges are not global either, so a fixture built
    # from 203.0.113.x would have been a "public" address the rule reads
    # as private -- a fixture the detector cannot produce.
    assert ipaddress.ip_address("203.0.113.9").is_global is False
    assert node_address.classify("http://100.64.0.7:8079") == node_address.CLASS_PRIVATE
    assert node_address.classify("http://127.0.0.1:8079") == node_address.CLASS_LOOPBACK
    assert node_address.classify("http://localhost:8079") == node_address.CLASS_LOOPBACK
    assert node_address.classify("http://10.0.0.1:8079") == node_address.CLASS_PRIVATE
    assert node_address.classify("http://[fd00::1]:8079") == node_address.CLASS_PRIVATE
    assert node_address.classify("http://8.8.8.8:8079") == node_address.CLASS_PUBLIC
    assert node_address.classify("http://amish-station:8079") == node_address.CLASS_PUBLIC
