"""A node updates where its own apps' sign-ins return to (2026-10-05).

Found on the first live move onto one HTTPS port: Workbench answered at its
new address and Eugene refused to send anyone back there, because its
sign-in client still named the old one. Fixing it took an operator restart
from a console the move had just made harder to reach. Now the agent updates
the return addresses at boot with its own token, and this root takes that
only for the clients of that node's own apps, changing nothing else.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from .conftest import enroll

NEW = ["https://workbench.example.org/oidc/callback"]


def _client(client: TestClient, owner: str | None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": "Workbench",
        "redirectUris": ["http://192.0.2.7:8300/oidc/callback"],
    }
    if owner is not None:
        body["owner"] = owner
    made = client.post("/v1/oidc/clients", json=body)
    assert made.status_code == 201, made.text
    return dict(made.json()["client"])


def _record(client: TestClient, client_id: str) -> dict[str, Any]:
    return dict(client.app.state.machine.state.oidc_clients[client_id])  # type: ignore[attr-defined]


def _put(client: TestClient, client_id: str, token: str | None, uris: list[str] = NEW) -> Any:
    headers = {"Authorization": f"Bearer {token}"} if token else None
    return client.put(
        f"/v1/oidc/clients/{client_id}/redirect-uris",
        json={"redirectUris": uris},
        headers=headers,
    )


def test_a_node_on_its_own_moves_its_own_apps_return_address(active_client: TestClient) -> None:
    gpu, _ = enroll(active_client, "gpu-box")
    made = _client(active_client, "app:workbench@gpu-box")
    before = _record(active_client, made["clientId"])
    response = _put(active_client, made["clientId"], gpu.service_token())
    assert response.status_code == 200, response.text
    assert response.json()["redirectUris"] == NEW
    after = _record(active_client, made["clientId"])
    # Same client, same secret, same owner: the app's sign-ins carry on.
    assert after["redirectUris"] == NEW
    assert {k: v for k, v in after.items() if k != "redirectUris"} == {
        k: v for k, v in before.items() if k != "redirectUris"
    }


def test_it_may_move_nothing_but_its_own_apps(active_client: TestClient) -> None:
    gpu, _ = enroll(active_client, "gpu-box")
    enroll(active_client, "attic")
    for owner in (None, "app:workbench@attic", "app:workbench@gpu-box2", "operator"):
        made = _client(active_client, owner)
        response = _put(active_client, made["clientId"], gpu.service_token())
        assert response.status_code == 403, (owner, response.text)
        assert "only for its own apps" in response.json()["detail"]["detail"]
        assert _record(active_client, made["clientId"])["redirectUris"] == made["redirectUris"]


def test_only_an_agent_speaking_for_itself_is_a_node_here(active_client: TestClient) -> None:
    gpu, _ = enroll(active_client, "gpu-box", grants=["gateway"])
    made = _client(active_client, "app:workbench@gpu-box")
    for token in (
        gpu.service_token(sub="gateway"),
        gpu.service_token(aud=("node:gpu-box",)),
        "nonsense",
    ):
        response = _put(active_client, made["clientId"], token)
        assert response.status_code == 401, response.text
    assert _record(active_client, made["clientId"])["redirectUris"] == made["redirectUris"]


def test_the_operator_may_move_any_client(active_client: TestClient) -> None:
    made = _client(active_client, None)
    response = active_client.put(
        f"/v1/oidc/clients/{made['clientId']}/redirect-uris", json={"redirectUris": NEW}
    )
    assert response.status_code == 200, response.text
    assert _record(active_client, made["clientId"])["redirectUris"] == NEW


def test_bad_or_unknown_is_refused_and_the_same_is_not_logged(active_client: TestClient) -> None:
    gpu, _ = enroll(active_client, "gpu-box")
    made = _client(active_client, "app:workbench@gpu-box")
    fragment = _put(active_client, made["clientId"], gpu.service_token(), ["https://x.org/#a"])
    assert fragment.status_code == 422, fragment.text
    assert _put(active_client, "c-nope", gpu.service_token()).status_code == 404
    machine = active_client.app.state.machine  # type: ignore[attr-defined]
    assert _put(active_client, made["clientId"], gpu.service_token()).status_code == 200
    index = machine.state.index
    assert _put(active_client, made["clientId"], gpu.service_token()).status_code == 200
    assert machine.state.index == index
