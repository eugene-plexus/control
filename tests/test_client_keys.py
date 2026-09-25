"""Install authority, durable replay and least privilege."""

from __future__ import annotations

import time
from datetime import UTC, datetime

import jwt
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_control import tokens
from eugene_plexus_control.applied import canonical_bytes, from_canonical, to_canonical
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


def test_a_record_that_claims_to_be_imported_is_refused(active_client: TestClient) -> None:
    """Per-node token keys retired the shared install key and, with it,
    the import of node-local key records (2026-09-25): a record that
    says `migrated` or names an `originNode` is not one this root makes,
    and the log refuses it rather than carrying it."""
    from eugene_plexus_control.applied import OP_PUT_CLIENT_KEY, ApplyError

    machine = active_client.app.state.machine
    key = {
        "id": "k1",
        "name": "App",
        "tail": "sample",
        "createdAt": datetime.now(UTC).isoformat(),
        "expiresAt": datetime.fromtimestamp(time.time() + 3600, UTC).isoformat(),
    }
    # A clean record is taken, so the refusals below are about the fields.
    machine.append(OP_PUT_CLIENT_KEY, {"key": key})
    assert "k1" in machine.state.client_keys
    for n, extra in enumerate(({"migrated": True}, {"originNode": "attic"})):
        with pytest.raises(ApplyError, match="unexpected client-key property"):
            machine.append(OP_PUT_CLIENT_KEY, {"key": {**key, "id": f"k{n + 2}", **extra}})
