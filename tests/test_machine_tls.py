"""TLS between machines (LT1): what control records, signs and switches.

* A node's machine port and key ride its signed address announcement (or its
  enrolment), and the signature covers them: a key nobody signed is refused.
* The machine list is signed by the root's identity key, names every node's
  door and key and the root's own, and its `iat` only rises.
* A node's join token carries the root's key, and the root takes the token
  with or without it.
* A new install starts required; an upgraded one switches by itself once
  every machine can.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from types import SimpleNamespace

import jwt
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_control import machine_list, sealing, tokens
from eugene_plexus_control.app import _require_when_ready, create_app

from .conftest import PASSPHRASE, enroll, login, mint_join_token, node_keys, settings_for

PIN_A = base64.urlsafe_b64encode(hashlib.sha256(b"a").digest()).rstrip(b"=").decode()
PIN_B = base64.urlsafe_b64encode(hashlib.sha256(b"b").digest()).rstrip(b"=").decode()
ROOT_PIN = base64.urlsafe_b64encode(hashlib.sha256(b"root").digest()).rstrip(b"=").decode()


def _announce(
    client: TestClient,
    name: str,
    private: str,
    *,
    url: str,
    sequence: int,
    machine_url: str | None = None,
    pin: str | None = None,
    signed: dict | None = None,
) -> object:
    machine = {k: v for k, v in (("machineUrl", machine_url), ("tlsPin", pin)) if v is not None}
    message = sealing.address_message(
        name=name, sequence=sequence, url=url, machine=machine if signed is None else signed
    )
    return client.patch(
        f"/v1/nodes/{name}",
        json={
            "url": url,
            "sequence": sequence,
            "signature": sealing.sign_address(private, message),
            **machine,
        },
    )


def _machine_list(client: TestClient) -> dict:
    response = client.get("/v1/trust/machines", headers={"Authorization": ""})
    assert response.status_code == 200, response.text
    jws = response.json()["jws"]
    authority = client.app.state.machine.state.identity.controlPublicKey  # type: ignore[attr-defined]
    assert jwt.get_unverified_header(jws)["typ"] == machine_list.TYP_MACHINES
    return jwt.decode(jws, tokens.load_public(authority), algorithms=["EdDSA"])


def test_a_signed_announcement_registers_the_machine_port_and_its_key(
    active_client: TestClient,
) -> None:
    keys, _ = enroll(active_client, "gpu-box", url="http://100.64.0.7:8079")
    machine = active_client.app.state.machine  # type: ignore[attr-defined]
    before = machine.state.index
    response = _announce(
        active_client,
        "gpu-box",
        keys.signing_private,
        url="http://100.64.0.7:8079",
        sequence=1,
        machine_url="https://100.64.0.7:8083",
        pin=PIN_A,
    )
    assert response.status_code == 200, response.text  # type: ignore[attr-defined]
    # Same address, new key: that is a change, and it is in the log.
    assert response.json()["changed"] is True  # type: ignore[attr-defined]
    assert machine.state.index == before + 1
    node = active_client.get("/v1/nodes/gpu-box").json()
    assert node["machineUrl"].rstrip("/") == "https://100.64.0.7:8083"
    assert node["tlsPins"] == [PIN_A]
    listed = {n["name"]: n for n in _machine_list(active_client)["nodes"]}
    assert listed["gpu-box"]["machineUrl"] == "https://100.64.0.7:8083"
    assert listed["gpu-box"]["pins"] == [PIN_A]

    # The same statement again writes nothing.
    again = _announce(
        active_client,
        "gpu-box",
        keys.signing_private,
        url="http://100.64.0.7:8079",
        sequence=2,
        machine_url="https://100.64.0.7:8083",
        pin=PIN_A,
    )
    assert again.json()["changed"] is False  # type: ignore[attr-defined]


def test_a_key_the_signature_does_not_cover_is_refused(active_client: TestClient) -> None:
    """A party on the network that replays a node's signed address with its
    own key attached gets a 401, and the record keeps no key."""
    keys, _ = enroll(active_client, "gpu-box", url="http://100.64.0.7:8079")
    forged = _announce(
        active_client,
        "gpu-box",
        keys.signing_private,
        url="http://100.64.0.7:8079",
        sequence=1,
        machine_url="https://100.64.0.66:8083",
        pin=PIN_B,
        signed={},  # signed over the address alone, as an older node signs
    )
    assert forged.status_code == 401, forged.text  # type: ignore[attr-defined]
    node = active_client.get("/v1/nodes/gpu-box").json()
    assert node.get("machineUrl") is None and node["tlsPins"] == []
    stranger = node_keys("elsewhere")
    other = _announce(
        active_client,
        "gpu-box",
        stranger.signing_private,
        url="http://100.64.0.7:8079",
        sequence=1,
        machine_url="https://100.64.0.66:8083",
        pin=PIN_B,
    )
    assert other.status_code == 401  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("machine_url", "pin", "words"),
    [
        ("http://100.64.0.7:8083", PIN_A, "https"),
        ("https://169.254.169.254:8083", PIN_A, "link-local"),
        ("https://8.8.8.8:8083", PIN_A, "open internet"),
        ("https://100.64.0.7:8083", "not-a-pin", "base64url"),
        ("https://100.64.0.7:8083", None, "both are needed"),
    ],
)
def test_a_machine_port_that_cannot_be_one_is_refused(
    active_client: TestClient, machine_url: str, pin: str | None, words: str
) -> None:
    keys, _ = enroll(active_client, "gpu-box", url="http://100.64.0.7:8079")
    response = _announce(
        active_client,
        "gpu-box",
        keys.signing_private,
        url="http://100.64.0.7:8079",
        sequence=1,
        machine_url=machine_url,
        pin=pin,
    )
    # A pin not even shaped like one is the request schema's 422.
    assert response.status_code in (400, 422), response.text  # type: ignore[attr-defined]
    assert words in response.text or "pattern" in response.text  # type: ignore[attr-defined]


def test_an_announcement_without_a_key_clears_the_one_on_record(
    active_client: TestClient,
) -> None:
    """A node rolled back to an older version stops being dialled over TLS."""
    keys, _ = enroll(
        active_client,
        "gpu-box",
        url="http://100.64.0.7:8079",
        machineUrl="https://100.64.0.7:8083",
        tlsPin=PIN_A,
    )
    assert active_client.get("/v1/nodes/gpu-box").json()["tlsPins"] == [PIN_A]
    response = _announce(
        active_client, "gpu-box", keys.signing_private, url="http://100.64.0.7:8079", sequence=1
    )
    assert response.json()["changed"] is True  # type: ignore[attr-defined]
    node = active_client.get("/v1/nodes/gpu-box").json()
    assert node.get("machineUrl") is None and node["tlsPins"] == []
    assert node["machineTls"]["state"] == "no_key"


def test_the_machine_list_is_signed_and_its_iat_only_rises(active_client: TestClient) -> None:
    keys, _ = enroll(active_client, "gpu-box", url="http://100.64.0.7:8079")
    first = _machine_list(active_client)
    # A root with no machine port in front of it (this one) starts automatic;
    # `test_a_join_token_carries_the_root_s_key...` has one, and starts required.
    assert first["required"] is False
    assert first["controlPins"] == []
    _announce(
        active_client,
        "gpu-box",
        keys.signing_private,
        url="http://100.64.0.7:8079",
        sequence=1,
        machine_url="https://100.64.0.7:8083",
        pin=PIN_A,
    )
    second = _machine_list(active_client)
    assert second["version"] > first["version"]

    # Unchanged content is the same list, not a new signature.
    def fetch() -> str:
        return active_client.get("/v1/trust/machines", headers={"Authorization": ""}).json()["jws"]

    assert fetch() == fetch()


def test_a_sealed_root_serves_the_last_list_it_signed(tmp_path: Path) -> None:
    app = create_app(settings_for(tmp_path / "root"))
    with TestClient(app) as client:
        assert (
            client.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
        )
        signed = client.get("/v1/trust/machines").json()["jws"]
    with TestClient(create_app(settings_for(tmp_path / "root"))) as sealed:
        assert sealed.app.state.auth_state.control_private_key is None  # type: ignore[attr-defined]
        assert sealed.get("/v1/trust/machines").json()["jws"] == signed


def test_a_join_token_carries_the_root_s_key_and_works_either_way(tmp_path: Path) -> None:
    settings = settings_for(tmp_path / "root")
    settings.machine_pins = f"{ROOT_PIN},not a pin"
    with TestClient(create_app(settings)) as client:
        assert (
            client.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
        )
        client.headers["Authorization"] = f"Bearer {login(client)}"
        with_pin = mint_join_token(client)
        secret, _, pin = with_pin.partition(".")
        assert pin == ROOT_PIN and "." not in secret
        listed = _machine_list(client)
        assert listed["controlPins"] == [ROOT_PIN]
        assert listed["required"] is True, "a new install with a machine port starts required"
        keys = node_keys("a")
        assert client.post("/v1/nodes/enroll", json=keys.body(with_pin)).status_code == 201
        bare, _, _ = mint_join_token(client).partition(".")
        keys = node_keys("b")
        response = client.post("/v1/nodes/enroll", json=keys.body(bare))
        assert response.status_code == 201, response.text


def _record(name: str, *, pins: bool) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        url=f"http://10.0.0.{len(name)}:8079",
        machineUrl=f"https://10.0.0.{len(name)}:8083" if pins else None,
        tlsPins=(PIN_A,) if pins else (),
    )


def test_an_automatic_install_switches_once_every_machine_can(
    active_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = active_client.app
    machine = app.state.machine  # type: ignore[attr-defined]
    response = active_client.patch("/v1/config", json={"machineEncryption": "automatic"})
    assert response.status_code == 200, response.text
    enroll(active_client, "root-box", url="http://100.64.0.5:8079",
           machineUrl="https://100.64.0.5:8083", tlsPin=ROOT_PIN)  # fmt: skip
    enroll(active_client, "gpu-box", url="http://100.64.0.7:8079")
    probe = SimpleNamespace(reachable=True, error=None, pin_refused=None, hosts_control=False)
    own = SimpleNamespace(reachable=True, error=None, pin_refused=None, hosts_control=True)
    app.state.node_probes = {"root-box": own, "gpu-box": probe}  # type: ignore[attr-defined]
    monkeypatch.setattr("eugene_plexus_control._http.machine_route", lambda url: "https://x")
    _require_when_ready(app, machine)  # type: ignore[arg-type]
    assert machine.state.config["machineEncryption"] == "automatic", "gpu-box has no key yet"
    # gpu-box updates: its announcement carries a key.
    record = machine.state.nodes["gpu-box"]
    from eugene_plexus_control.applied import OP_UPDATE_NODE

    machine.append(
        OP_UPDATE_NODE,
        {"name": "gpu-box", "url": record.url, "sequence": 1,
         "machineUrl": "https://100.64.0.7:8083", "tlsPins": [PIN_A]},
    )  # fmt: skip
    app.state.node_probes["gpu-box"] = SimpleNamespace(  # type: ignore[attr-defined]
        reachable=False, error="connection refused", pin_refused=None, hosts_control=False
    )
    _require_when_ready(app, machine)  # type: ignore[arg-type]
    assert machine.state.config["machineEncryption"] == "automatic", "it does not answer yet"
    app.state.node_probes["gpu-box"] = probe  # type: ignore[attr-defined]
    _require_when_ready(app, machine)  # type: ignore[arg-type]
    assert machine.state.config["machineEncryption"] == "required"
    assert _machine_list(active_client)["required"] is True


def test_off_is_never_switched(active_client: TestClient) -> None:
    app = active_client.app
    machine = app.state.machine  # type: ignore[attr-defined]
    assert active_client.patch("/v1/config", json={"machineEncryption": "off"}).status_code == 200
    app.state.node_probes = {}  # type: ignore[attr-defined]
    enroll(active_client, "gpu-box", url="http://100.64.0.7:8079",
           machineUrl="https://100.64.0.7:8083", tlsPin=PIN_A)  # fmt: skip
    _require_when_ready(app, machine)  # type: ignore[arg-type]
    assert machine.state.config["machineEncryption"] == "off"


def test_the_root_reports_a_refused_key_as_a_security_event() -> None:
    record = _record("gpu-box", pins=True)
    refused = SimpleNamespace(reachable=False, error="x", pin_refused=PIN_B, hosts_control=False)
    link = machine_list.link(record, refused)
    assert link["state"] == "refused" and PIN_B in link["detail"]
    assert machine_list.link(_record("old", pins=False), None)["state"] == "no_key"
    assert machine_list.link(record, None)["state"] == "unknown"


def test_the_pins_survive_a_snapshot(active_client: TestClient) -> None:
    from eugene_plexus_control.applied import from_canonical, to_canonical

    enroll(active_client, "gpu-box", url="http://100.64.0.7:8079",
           machineUrl="https://100.64.0.7:8083", tlsPin=PIN_A)  # fmt: skip
    state = active_client.app.state.machine.state  # type: ignore[attr-defined]
    again = from_canonical(to_canonical(state))
    assert again.nodes["gpu-box"].machineUrl == "https://100.64.0.7:8083"
    assert again.nodes["gpu-box"].tlsPins == (PIN_A,)


# ---- the adversarial review (2026-10-20) ------------------------------------


def test_a_machine_port_on_another_host_is_refused(active_client: TestClient) -> None:
    keys, _ = enroll(active_client, "gpu-box", url="http://100.64.0.7:8079")
    response = _announce(
        active_client,
        "gpu-box",
        keys.signing_private,
        url="http://100.64.0.7:8079",
        sequence=1,
        machine_url="https://100.64.0.66:8083",
        pin=PIN_A,
    )
    assert response.status_code == 400, response.text  # type: ignore[attr-defined]
    assert "not on the host" in response.text  # type: ignore[attr-defined]


def test_a_door_another_node_registered_is_refused(active_client: TestClient) -> None:
    enroll(active_client, "gpu-box", url="http://100.64.0.7:8079",
           machineUrl="https://100.64.0.7:8083", tlsPin=PIN_A)  # fmt: skip
    thief, _ = enroll(active_client, "thief", url="http://100.64.0.7:8080")
    response = _announce(
        active_client,
        "thief",
        thief.signing_private,
        url="http://100.64.0.7:8080",
        sequence=1,
        machine_url="https://100.64.0.7:8083",
        pin=PIN_B,
    )
    assert response.status_code == 400, response.text  # type: ignore[attr-defined]
    assert "already registered" in response.text  # type: ignore[attr-defined]
    assert active_client.get("/v1/nodes/gpu-box").json()["tlsPins"] == [PIN_A]


def test_the_list_version_follows_the_log(active_client: TestClient) -> None:
    """A promoted standby, whose own file starts empty, still signs above what
    the root it replaced signed: the version rides the replicated index."""
    machine = active_client.app.state.machine  # type: ignore[attr-defined]
    first = _machine_list(active_client)
    assert first["version"] >= machine.state.index * 1000
    enroll(active_client, "gpu-box", url="http://100.64.0.7:8079")
    assert _machine_list(active_client)["version"] >= machine.state.index * 1000
