"""Machines says why a machine's announcement was refused (row 36).

Before this, a refused address left the machine reading "unreachable" with
the probe error of its OLD address, which sent the owner to firewalls and
keys. The root now keeps the last refusal of an announcement that verified
against the machine's signing key, and a count (never the text) of the ones
that did not.

An observation: held in memory on the active root, never in the log.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_control import sealing, tokens

NODE_PUBLIC = base64.b64encode(b"x" * 32).decode("ascii")
HOME = "http://192.0.2.7:59079"


@pytest.fixture(autouse=True)
def _no_node_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    from eugene_plexus_control import app

    async def no_poll(application: FastAPI) -> None:
        pass

    monkeypatch.setattr(app, "_poll_nodes", no_poll)


def _keypair() -> tuple[str, str]:
    import nacl.signing

    signing = nacl.signing.SigningKey.generate()
    return (
        base64.b64encode(bytes(signing)).decode("ascii"),
        base64.b64encode(bytes(signing.verify_key)).decode("ascii"),
    )


def _enroll(client: TestClient, name: str, *, url: str, public: str) -> None:
    token = client.post("/v1/nodes/join-token").json()["token"]
    response = client.post(
        "/v1/nodes/enroll",
        json={
            "token": token,
            "name": name,
            "publicKey": NODE_PUBLIC,
            "url": url,
            "signingPublicKey": public,
            "tokenPublicKey": tokens.public_b64(tokens.generate_private_key()),
        },
    )
    assert response.status_code == 201, response.text


def _announce(
    client: TestClient,
    name: str,
    *,
    url: str,
    sequence: int,
    private: str,
    machine: dict[str, str] | None = None,
) -> Any:
    message = sealing.address_message(name=name, sequence=sequence, url=url, machine=machine)
    return client.patch(
        f"/v1/nodes/{name}",
        json={
            "url": url,
            "sequence": sequence,
            "signature": sealing.sign_address(private, message),
            **(machine or {}),
        },
    )


def _node(client: TestClient, name: str = "nas") -> dict[str, Any]:
    response = client.get(f"/v1/nodes/{name}")
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def _refuse_each(client: TestClient, reason: str, private: str) -> Any:
    """Make `nas` (enrolled at HOME) send the announcement that `reason` names."""
    if reason == "open_internet":
        return _announce(client, "nas", url="http://8.8.8.8:8079", sequence=1, private=private)
    if reason == "anothers_address":
        return _announce(client, "nas", url="http://192.0.2.50:59079", sequence=1, private=private)
    if reason == "not_a_node":
        return _announce(client, "nas", url="http://169.254.169.254/", sequence=1, private=private)
    if reason == "machine_port":
        return _announce(
            client,
            "nas",
            url="http://192.0.2.8:59079",
            sequence=1,
            private=private,
            machine={"machineUrl": "http://192.0.2.8:8083", "tlsPin": "A" * 43},
        )
    if reason == "stale_sequence":
        assert (
            _announce(
                client, "nas", url="http://192.0.2.8:59079", sequence=1, private=private
            ).status_code
            == 200
        )
        return _announce(client, "nas", url="http://192.0.2.9:59079", sequence=1, private=private)
    raise AssertionError(reason)


@pytest.mark.parametrize(
    ("reason", "status", "in_detail"),
    [
        ("open_internet", 409, ["http://192.0.2.7:59079", "8.8.8.8"]),
        ("anothers_address", 409, ["192.0.2.50"]),
        ("not_a_node", 400, ["169.254.169.254"]),
        ("machine_port", 400, ["machine port"]),
        ("stale_sequence", 409, ["Sequence 1"]),
    ],
)
def test_a_signed_refusal_is_shown_on_the_node(
    active_client: TestClient, reason: str, status: int, in_detail: list[str]
) -> None:
    """Today `GET /v1/nodes/{name}` has no field at all: the refusal went to
    the machine and to the log, and to nobody who looks at Machines."""
    private, public = _keypair()
    _enroll(active_client, "other", url="http://192.0.2.50:59079", public=_keypair()[1])
    _enroll(active_client, "nas", url=HOME, public=public)
    assert _node(active_client)["announcementRefused"] is None

    refused = _refuse_each(active_client, reason, private)
    assert refused.status_code == status, refused.text
    sent = refused.json()["detail"]

    shown = _node(active_client)["announcementRefused"]
    assert shown["reason"] == reason
    # The root's own words, the ones the machine was sent.
    assert shown["title"] == sent["title"]
    assert shown["detail"] == sent["detail"]
    for text in in_detail:
        assert text in shown["detail"], shown
    assert shown["at"]
    # The list shows the same as the single view.
    listed = {n["name"]: n for n in active_client.get("/v1/nodes").json()["nodes"]}
    assert listed["nas"]["announcementRefused"] == shown
    assert listed["other"]["announcementRefused"] is None


def test_a_refusal_names_the_address_it_refused(active_client: TestClient) -> None:
    private, public = _keypair()
    _enroll(active_client, "nas", url=HOME, public=public)
    _announce(active_client, "nas", url="http://8.8.8.8:8079", sequence=1, private=private)
    shown = _node(active_client)["announcementRefused"]
    assert shown["announcedUrl"].rstrip("/") == "http://8.8.8.8:8079"


def test_a_machine_without_a_signing_key_is_a_fact_of_the_record(
    active_client: TestClient,
) -> None:
    """Recorded as `no_signing_key` whatever the request carried: nothing on
    record can verify it, so the answer is about the record, not the body."""
    machine = active_client.app.state.machine  # type: ignore[attr-defined]
    machine.append(
        "enrollNode",
        {"name": "old", "role": "agent", "url": HOME, "enrolledAt": "2026-01-01T00:00:00+00:00"},
    )
    private, _public = _keypair()
    refused = _announce(
        active_client, "old", url="http://192.0.2.8:59079", sequence=1, private=private
    )
    assert refused.status_code == 401, refused.text
    shown = _node(active_client, "old")["announcementRefused"]
    assert shown["reason"] == "no_signing_key"
    assert shown["title"] == refused.json()["detail"]["title"]
    # What the request carried is not shown: nothing verified it.
    assert shown.get("announcedUrl") is None


def test_an_unsigned_announcement_is_counted_not_shown(active_client: TestClient) -> None:
    """An announcement anyone can send must not put words on the owner's
    screen: a number and a time, and nothing it carried."""
    private, public = _keypair()
    _enroll(active_client, "nas", url=HOME, public=public)
    assert _node(active_client)["unverifiedAnnouncements"] is None

    forger, _ = _keypair()
    first = _announce(
        active_client, "nas", url="http://evil.example/SECRET-WORDS", sequence=1, private=forger
    )
    assert first.status_code == 401, first.text
    # A body that cannot be read as an announcement is counted too.
    malformed = active_client.patch(
        "/v1/nodes/nas", content=b"not json", headers={"Content-Type": "application/json"}
    )
    assert malformed.status_code in (400, 422), malformed.text
    wrong_type = active_client.patch(
        "/v1/nodes/nas", json={"url": 5, "sequence": 1, "signature": "x"}
    )
    assert wrong_type.status_code in (400, 422), wrong_type.text

    node = _node(active_client)
    unverified = node["unverifiedAnnouncements"]
    assert unverified["count"] >= 1
    assert unverified["lastAt"]
    # Not kept: no refusal, and none of the text it carried, anywhere.
    assert node["announcementRefused"] is None
    assert "SECRET-WORDS" not in json.dumps(node)
    assert "SECRET-WORDS" not in active_client.get("/v1/nodes").text

    # A good announcement from the real key is not counted against it.
    count = unverified["count"]
    refused = _announce(
        active_client, "nas", url="http://8.8.8.8:8079", sequence=1, private=private
    )
    assert refused.status_code == 409
    after = _node(active_client)
    assert after["unverifiedAnnouncements"]["count"] == count
    assert after["announcementRefused"]["reason"] == "open_internet"


def test_a_forged_announcement_does_not_replace_a_real_refusal(
    active_client: TestClient,
) -> None:
    private, public = _keypair()
    _enroll(active_client, "nas", url=HOME, public=public)
    _announce(active_client, "nas", url="http://8.8.8.8:8079", sequence=1, private=private)
    before = _node(active_client)["announcementRefused"]
    forger, _ = _keypair()
    _announce(active_client, "nas", url="http://169.254.169.254/", sequence=2, private=forger)
    assert _node(active_client)["announcementRefused"] == before


def test_a_refusal_is_not_written_to_the_log(active_client: TestClient) -> None:
    """A refusal must not grow the replicated log: it is an observation."""
    private, public = _keypair()
    _enroll(active_client, "nas", url=HOME, public=public)
    machine = active_client.app.state.machine  # type: ignore[attr-defined]
    before = machine.state.index
    _announce(active_client, "nas", url="http://8.8.8.8:8079", sequence=1, private=private)
    _announce(active_client, "nas", url="http://169.254.169.254/", sequence=2, private=private)
    forger, _ = _keypair()
    _announce(active_client, "nas", url=HOME, sequence=3, private=forger)
    assert machine.state.index == before


def test_the_latest_refusal_replaces_the_one_before(active_client: TestClient) -> None:
    private, public = _keypair()
    _enroll(active_client, "nas", url=HOME, public=public)
    _announce(active_client, "nas", url="http://8.8.8.8:8079", sequence=1, private=private)
    assert _node(active_client)["announcementRefused"]["reason"] == "open_internet"
    _announce(active_client, "nas", url="http://169.254.169.254/", sequence=2, private=private)
    assert _node(active_client)["announcementRefused"]["reason"] == "not_a_node"


def test_an_announcement_the_root_accepts_clears_the_refusal(
    active_client: TestClient,
) -> None:
    private, public = _keypair()
    _enroll(active_client, "nas", url=HOME, public=public)
    _announce(active_client, "nas", url="http://8.8.8.8:8079", sequence=1, private=private)
    forger, _ = _keypair()
    _announce(active_client, "nas", url=HOME, sequence=1, private=forger)
    assert _node(active_client)["announcementRefused"] is not None
    assert _node(active_client)["unverifiedAnnouncements"] is not None

    accepted = _announce(
        active_client, "nas", url="http://192.0.2.8:59079", sequence=1, private=private
    )
    assert accepted.status_code == 200, accepted.text
    node = _node(active_client)
    assert node["announcementRefused"] is None
    assert node["unverifiedAnnouncements"] is None


def test_an_unchanged_announcement_clears_the_refusal(active_client: TestClient) -> None:
    """The machine says where it is, and that is where the root has it."""
    private, public = _keypair()
    _enroll(active_client, "nas", url=HOME, public=public)
    _announce(active_client, "nas", url="http://8.8.8.8:8079", sequence=1, private=private)
    unchanged = _announce(active_client, "nas", url=HOME, sequence=2, private=private)
    assert unchanged.status_code == 200 and unchanged.json()["changed"] is False
    assert _node(active_client)["announcementRefused"] is None


def test_re_enrollment_clears_the_refusal(active_client: TestClient) -> None:
    """Re-enrollment is the way to confirm a move onto the open internet."""
    private, public = _keypair()
    _enroll(active_client, "nas", url=HOME, public=public)
    _announce(active_client, "nas", url="http://8.8.8.8:8079", sequence=1, private=private)
    forger, _ = _keypair()
    _announce(active_client, "nas", url=HOME, sequence=1, private=forger)
    assert _node(active_client)["announcementRefused"] is not None

    _enroll(active_client, "nas", url="http://8.8.8.8:8079", public=public)
    node = _node(active_client)
    assert node["announcementRefused"] is None
    assert node["unverifiedAnnouncements"] is None


def test_removal_clears_the_refusal(active_client: TestClient) -> None:
    private, public = _keypair()
    _enroll(active_client, "nas", url=HOME, public=public)
    _announce(active_client, "nas", url="http://8.8.8.8:8079", sequence=1, private=private)
    assert active_client.delete("/v1/nodes/nas").status_code == 202
    _enroll(active_client, "nas", url=HOME, public=public)
    node = _node(active_client)
    assert node["announcementRefused"] is None
    assert node["unverifiedAnnouncements"] is None


def test_an_announcement_for_no_machine_keeps_nothing(active_client: TestClient) -> None:
    """Anyone can name a machine that does not exist: that must not grow
    what the root holds."""
    forger, _ = _keypair()
    response = _announce(active_client, "ghost-1", url=HOME, sequence=1, private=forger)
    assert response.status_code == 404
    held = active_client.app.state.announcement_refusals  # type: ignore[attr-defined]
    assert held.is_empty()
