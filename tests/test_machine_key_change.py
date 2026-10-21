"""Row 3: a machine's TLS key changes across an overlap of two pins.

* A node announces its next key beside its current one (`tlsNextPin`), under
  its signature; the root records and lists both, and at most two.
* The drop (the new key alone) leaves the list, and a replay of the old
  announcement cannot bring the dropped key back.
* While a node holds two pins, its record says which machines this root sees
  holding an older list (`machineKeyChange`).
* The root's own keys come from the door file its agent writes, re-read while
  control runs: no restart, so no sealed root.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import jwt
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_control import machine_list, sealing, tokens
from eugene_plexus_control.app import create_app
from eugene_plexus_control.nodes_client import NodeProbe
from eugene_plexus_control.routes import nodes as node_routes

from .conftest import PASSPHRASE, enroll, login, settings_for


def _pin(seed: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(seed).digest()).rstrip(b"=").decode()


PIN_A, PIN_B, PIN_C = _pin(b"a"), _pin(b"b"), _pin(b"c")
DOOR = "https://100.64.0.7:8083"
URL = "http://100.64.0.7:8079"


def _body(name: str, private: str, sequence: int, fields: dict, signed: dict | None = None) -> dict:
    message = sealing.address_message(
        name=name, sequence=sequence, url=URL, machine=fields if signed is None else signed
    )
    return {
        "url": URL,
        "sequence": sequence,
        "signature": sealing.sign_address(private, message),
        **fields,
    }


def _machine_list(client: TestClient) -> dict:
    response = client.get("/v1/trust/machines", headers={"Authorization": ""})
    assert response.status_code == 200, response.text
    authority = client.app.state.machine.state.identity.controlPublicKey  # type: ignore[attr-defined]
    return jwt.decode(response.json()["jws"], tokens.load_public(authority), algorithms=["EdDSA"])


def test_next_pin_is_signed_recorded_and_listed(active_client: TestClient) -> None:
    keys, _ = enroll(active_client, "gpu-box", url=URL)
    private = keys.signing_private
    current = {"machineUrl": DOOR, "tlsPin": PIN_A}
    first = active_client.patch("/v1/nodes/gpu-box", json=_body("gpu-box", private, 1, current))
    assert first.status_code == 200, first.text
    assert _machine_list(active_client)["keyChange"] is True

    both = {**current, "tlsNextPin": PIN_B}
    # A signature that does not cover the next key is refused: nobody on the
    # network can add a key to a node's real announcement.
    forged = active_client.patch(
        "/v1/nodes/gpu-box", json=_body("gpu-box", private, 2, both, signed=current)
    )
    assert forged.status_code == 401, forged.text
    assert active_client.get("/v1/nodes/gpu-box").json()["tlsPins"] == [PIN_A]

    # A third key has no place in a record: one next key, a string, or the
    # body does not even parse.
    third = active_client.patch(
        "/v1/nodes/gpu-box",
        json=_body("gpu-box", private, 3, {**current, "tlsNextPin": [PIN_B, PIN_C]}),
    )
    assert third.status_code == 422, third.text
    same = active_client.patch(
        "/v1/nodes/gpu-box", json=_body("gpu-box", private, 4, {**current, "tlsNextPin": PIN_A})
    )
    assert same.status_code == 400, same.text
    alone = active_client.patch(
        "/v1/nodes/gpu-box", json=_body("gpu-box", private, 5, {"tlsNextPin": PIN_B})
    )
    assert alone.status_code == 400, alone.text

    overlap = _body("gpu-box", private, 6, both)
    response = active_client.patch("/v1/nodes/gpu-box", json=overlap)
    assert response.status_code == 200, response.text
    assert response.json()["changed"] is True
    node = active_client.get("/v1/nodes/gpu-box").json()
    assert node["tlsPins"] == sorted([PIN_A, PIN_B])
    assert node["tlsNextPin"] == PIN_B
    assert node["machineKeyChange"]["nextPin"] == PIN_B
    listed = {n["name"]: n for n in _machine_list(active_client)["nodes"]}
    assert sorted(listed["gpu-box"]["pins"]) == sorted([PIN_A, PIN_B])
    # The root's own table (its gateway's calls) takes either key at once.
    from eugene_plexus_control import _http

    assert _http.machine_table().machines["100.64.0.7:8083"] == frozenset({PIN_A, PIN_B})

    # The drop: the new key alone.
    dropped = active_client.patch(
        "/v1/nodes/gpu-box",
        json=_body("gpu-box", private, 7, {"machineUrl": DOOR, "tlsPin": PIN_B}),
    )
    assert dropped.status_code == 200, dropped.text
    node = active_client.get("/v1/nodes/gpu-box").json()
    assert node["tlsPins"] == [PIN_B]
    assert node.get("tlsNextPin") is None and node.get("machineKeyChange") is None
    listed = {n["name"]: n for n in _machine_list(active_client)["nodes"]}
    assert listed["gpu-box"]["pins"] == [PIN_B]

    # A replay of the overlap cannot bring the dropped key back.
    replay = active_client.patch("/v1/nodes/gpu-box", json=overlap)
    assert replay.status_code == 409, replay.text
    assert active_client.get("/v1/nodes/gpu-box").json()["tlsPins"] == [PIN_B]


def test_the_next_pin_survives_a_snapshot(active_client: TestClient) -> None:
    from eugene_plexus_control.applied import from_canonical, to_canonical

    keys, _ = enroll(active_client, "gpu-box", url=URL)
    both = {"machineUrl": DOOR, "tlsPin": PIN_A, "tlsNextPin": PIN_B}
    response = active_client.patch(
        "/v1/nodes/gpu-box", json=_body("gpu-box", keys.signing_private, 1, both)
    )
    assert response.status_code == 200, response.text
    state = active_client.app.state.machine.state  # type: ignore[attr-defined]
    again = from_canonical(json.loads(json.dumps(to_canonical(state))))
    assert again.nodes["gpu-box"].tlsNextPin == PIN_B
    assert again.nodes["gpu-box"].tlsPins == tuple(sorted([PIN_A, PIN_B]))


def test_the_root_says_which_machines_hold_an_older_list(active_client: TestClient) -> None:
    app = active_client.app
    keys, _ = enroll(active_client, "gpu-box", url=URL)
    for name, url in (("nas", "http://100.64.0.8:8079"), ("laptop", "http://100.64.0.9:8079")):
        other, _ = enroll(active_client, name, url=url)
        host = url.split("//")[1].split(":")[0]
        fields = {"machineUrl": f"https://{host}:8083", "tlsPin": _pin(name.encode())}
        message = sealing.address_message(name=name, sequence=1, url=url, machine=fields)
        body = {
            "url": url,
            "sequence": 1,
            "signature": sealing.sign_address(other.signing_private, message),
            **fields,
        }
        assert active_client.patch(f"/v1/nodes/{name}", json=body).status_code == 200
    both = {"machineUrl": DOOR, "tlsPin": PIN_A, "tlsNextPin": PIN_B}
    assert (
        active_client.patch(
            "/v1/nodes/gpu-box", json=_body("gpu-box", keys.signing_private, 1, both)
        ).status_code
        == 200
    )
    # Not signed into a list yet: every machine is waited on.
    change = active_client.get("/v1/nodes/gpu-box").json()["machineKeyChange"]
    assert change["listVersion"] is None
    assert sorted(change["waitingOn"]) == ["laptop", "nas"]
    version = _machine_list(active_client)["version"]
    change = active_client.get("/v1/nodes/gpu-box").json()["machineKeyChange"]
    assert change["listVersion"] == version

    app.state.node_probes = {  # type: ignore[attr-defined]
        "nas": NodeProbe(name="nas", reachable=True, list_version=version - 1),
        "laptop": NodeProbe(name="laptop", reachable=False, error="timed out"),
        "gpu-box": NodeProbe(name="gpu-box", reachable=True, list_version=version - 1),
    }
    change = active_client.get("/v1/nodes/gpu-box").json()["machineKeyChange"]
    # The changing machine itself is never waited on.
    assert change["waitingOn"] == ["nas"] and change["unreachable"] == ["laptop"]
    app.state.node_probes["nas"] = NodeProbe(  # type: ignore[attr-defined]
        name="nas", reachable=True, list_version=version
    )
    change = active_client.get("/v1/nodes/gpu-box").json()["machineKeyChange"]
    assert change["waitingOn"] == [] and change["unreachable"] == ["laptop"]

    # A list signed again for something else keeps the version that first
    # named both keys.
    enroll(active_client, "late", url="http://100.64.0.10:8079")
    assert _machine_list(active_client)["version"] > version
    change = active_client.get("/v1/nodes/gpu-box").json()["machineKeyChange"]
    assert change["listVersion"] == version


def test_machines_never_seen_with_a_list_version_are_not_ready(
    tmp_path: Path, active_client: TestClient
) -> None:
    """Review blocker 2: an agent older than row 3 that is off for the whole
    change of the root's key never takes the new key. The root names every
    machine it has never seen report a list version, and remembers the ones
    it has across a restart (its own file), even while they are off."""
    from eugene_plexus_control.app import _note_list_capable

    app = active_client.app
    keys, _ = enroll(active_client, "gpu-box", url=URL)
    for name, url in (("nas", "http://100.64.0.8:8079"), ("old", "http://100.64.0.9:8079")):
        other, _ = enroll(active_client, name, url=url)
        host = url.split("//")[1].split(":")[0]
        fields = {"machineUrl": f"https://{host}:8083", "tlsPin": _pin(name.encode())}
        message = sealing.address_message(name=name, sequence=1, url=url, machine=fields)
        body = {
            "url": url,
            "sequence": 1,
            "signature": sealing.sign_address(other.signing_private, message),
            **fields,
        }
        assert active_client.patch(f"/v1/nodes/{name}", json=body).status_code == 200
    both = {"machineUrl": DOOR, "tlsPin": PIN_A, "tlsNextPin": PIN_B}
    response = active_client.patch(
        "/v1/nodes/gpu-box", json=_body("gpu-box", keys.signing_private, 1, both)
    )
    assert response.status_code == 200, response.text
    change = active_client.get("/v1/nodes/gpu-box").json()["machineKeyChange"]
    assert sorted(change["notReady"]) == ["nas", "old"]

    # The NAS reports a list version once: it is ready, and stays so while
    # it is off; the old agent never reports one.
    probes = {
        "nas": NodeProbe(name="nas", reachable=True, list_version=3),
        "old": NodeProbe(name="old", reachable=True),
    }
    _note_list_capable(app, probes)  # what the poller does after each round
    app.state.node_probes = {  # type: ignore[attr-defined]
        "nas": NodeProbe(name="nas", reachable=False, error="timed out"),
        "old": NodeProbe(name="old", reachable=False, error="timed out"),
    }
    change = active_client.get("/v1/nodes/gpu-box").json()["machineKeyChange"]
    assert change["notReady"] == ["old"]
    assert sorted(change["unreachable"]) == ["nas", "old"]
    # Kept across a restart of this root.
    again = machine_list.MachineListPublisher(app.state.settings.state_dir)  # type: ignore[attr-defined]
    assert again.list_capable == {"nas"}


def test_the_probe_reads_the_list_version_a_node_holds() -> None:
    import asyncio

    import httpx

    from eugene_plexus_control.nodes_client import NodesClient

    def answer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"epoch": 1, "machine": {"listVersion": 7000}})

    client = NodesClient(timeout_provider=lambda: 5)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(answer))

    async def run() -> NodeProbe:
        try:
            return await client.probe("gpu-box", "http://100.64.0.7:8079", "t")
        finally:
            await client.aclose()

    assert asyncio.run(run()).list_version == 7000


def test_the_root_s_keys_come_from_the_door_file_its_agent_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    door_file = tmp_path / "machine_door.json"
    door_file.write_text(json.dumps({"pins": [PIN_A], "url": DOOR}), encoding="utf-8")
    settings = settings_for(tmp_path / "root")
    settings.machine_pins = PIN_C  # what an older agent hands over: the file wins
    settings.machine_door_file = str(door_file)
    monkeypatch.setattr(node_routes, "DOOR_RECHECK_SECONDS", 0.0)
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code
        client.headers["Authorization"] = f"Bearer {login(client)}"
        assert _machine_list(client)["controlPins"] == [PIN_A]
        token = client.post("/v1/nodes/join-token", json={}).json()["token"]
        assert token.endswith("." + PIN_A)

        # The change starts: both keys, the current one first, with no restart.
        door_file.write_text(json.dumps({"pins": [PIN_A, PIN_B], "url": DOOR}), encoding="utf-8")
        assert _machine_list(client)["controlPins"] == sorted([PIN_A, PIN_B])
        # Join codes keep the key the machine port presents until the switch.
        token = client.post("/v1/nodes/join-token", json={}).json()["token"]
        assert token.endswith("." + PIN_A)
        # After the switch the new key comes first.
        door_file.write_text(json.dumps({"pins": [PIN_B, PIN_A], "url": DOOR}), encoding="utf-8")
        token = client.post("/v1/nodes/join-token", json={}).json()["token"]
        assert token.endswith("." + PIN_B)
        door_file.write_text(json.dumps({"pins": [PIN_B], "url": DOOR}), encoding="utf-8")
        assert _machine_list(client)["controlPins"] == [PIN_B]
        # A file that cannot be read keeps the last keys read from it.
        door_file.write_text("{not json", encoding="utf-8")
        assert _machine_list(client)["controlPins"] == [PIN_B]
    assert machine_list.TYP_MACHINES == "ep-machines+jwt"
