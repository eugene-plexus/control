"""Install authority, durable replay, least privilege and signed legacy migration."""

from __future__ import annotations

import base64
import json
import time
from datetime import UTC, datetime

import jwt
import nacl.signing
from fastapi.testclient import TestClient

from eugene_plexus_control import tokens
from eugene_plexus_control.applied import canonical_bytes, from_canonical, to_canonical
from eugene_plexus_control.routes.client_keys import IMPORT_DOMAIN
from tests.conftest import enroll, machine_at


def test_mint_revoke_policy_and_restart(active_client: TestClient, tmp_path) -> None:
    client = active_client
    response = client.post("/v1/auth/client-keys", json={"name": " laptop "})
    assert response.status_code == 201, response.text
    made = response.json()
    claims = jwt.decode(made["token"], options={"verify_signature": False})
    assert jwt.get_unverified_header(made["token"])["typ"] == tokens.TYP_CLIENT
    assert claims["aud"] == ["gateway"] and claims["iss"] == "control"
    assert claims["jti"] == made["key"]["id"]
    assert claims["exp"] - claims["iat"] == 365 * 86400
    assert client.get("/v1/auth/client-keys").json()["scope"] == "install"
    assert made["token"] not in client.get("/v1/auth/client-keys").text
    path = "/v1/auth/client-keys/" + made["key"]["id"]
    assert client.delete(path).status_code == 204
    first = client.get("/v1/auth/client-keys/policy").json()
    assert first["keys"][0]["revokedAt"]
    assert client.delete(path).status_code == 204
    assert client.get("/v1/auth/client-keys/policy").json()["revision"] == first["revision"]
    machine = client.app.state.machine
    # Both snapshot bootstrap and log replay retain the exact revocation bytes.
    assert canonical_bytes(from_canonical(to_canonical(machine.state))) == machine.canonical()
    restarted = machine_at(tmp_path / "active" / "state")
    assert restarted.state.client_keys == machine.state.client_keys
    assert made["token"].encode() not in canonical_bytes(machine.state)


def test_policy_audiences_and_standby_refusal(active_client: TestClient) -> None:
    client = active_client
    keys, _ = enroll(client, "nas", grants=["gateway"])
    for kind, expected in (("agent", 200), ("gateway", 200), ("library", 401), ("control", 401)):
        header = {"Authorization": "Bearer " + keys.service_token(sub=kind)}
        assert client.get("/v1/auth/client-keys/policy", headers=header).status_code == expected
        assert (
            client.post("/v1/auth/client-keys", headers=header, json={"name": "x"}).status_code
            == 401
        )
        assert client.get("/v1/auth/client-keys", headers=header).status_code == 401
    client.app.state.machine.force_role("standby")
    assert client.get("/v1/auth/client-keys/policy").status_code == 503


def test_signed_import_is_idempotent_and_cannot_clear_revocation(active_client: TestClient) -> None:
    client = active_client
    keys, enrolled = enroll(client, "legacy")
    assert enrolled.status_code == 201
    signing = nacl.signing.SigningKey(base64.b64decode(keys.signing_private))
    key = {
        "id": "legacy-key",
        "name": "Old app",
        "tail": "sample",
        "createdAt": datetime.now(UTC).isoformat(),
        "expiresAt": datetime.fromtimestamp(time.time() + 3600, UTC).isoformat(),
    }

    def send(keys):
        canonical = json.dumps(
            {"node": "legacy", "keys": keys},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        signature = base64.b64encode(signing.sign(IMPORT_DOMAIN + canonical).signature).decode()
        return client.post(
            "/v1/nodes/legacy/client-keys/import", json={"keys": keys, "signature": signature}
        )

    assert send([key]).status_code == 200
    before = client.app.state.machine.state.index
    assert send([key]).status_code == 200
    assert client.app.state.machine.state.index == before
    assert client.get("/v1/auth/client-keys").json()["keys"][0].get("limits") is None
    assert (
        client.put(
            "/v1/auth/client-keys/legacy-key/limits",
            json={
                "limits": {
                    "allowedModels": ["chosen"],
                    "maxConcurrentRequests": 1,
                    "requestsPerMinute": 3,
                }
            },
        ).status_code
        == 200
    )
    assert client.delete("/v1/auth/client-keys/legacy-key").status_code == 204
    assert send([{**key, "name": "changed"}]).status_code == 200
    listed = client.get("/v1/auth/client-keys").json()["keys"][0]
    assert listed["revokedAt"] and listed["originNode"] == "legacy" and listed["migrated"]
    assert listed["limits"]["allowedModels"] == ["chosen"]
    assert listed["limits"]["requestsPerMinute"] == 3
    assert send([{**key, "tail": "other"}]).status_code == 409
    assert (
        client.post(
            "/v1/nodes/legacy/client-keys/import",
            json={"keys": [key], "signature": "not-a-signature"},
        ).status_code
        == 401
    )
