"""A node acting for an operator who is acting on it (2026-10-01).

Found on the live install: from the console on one machine, installing an
app on another signed the operator out, every time. The console reaches
that machine with a token addressed to it alone (D7), and the install sent
that token on here to mint the app's key and register its sign-in. This root
refuses a token not addressed to it, the 401 travelled back to the browser,
and the console read it as the session ending.

The fix is an actor and a subject: the machine's own `agent` token, and the
operator's token addressed to that machine. What this file pins is that the
pair does what an app install needs and nothing more -- only on the node
named in both, only for things named for that node, never on its own.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from eugene_plexus_control.dependencies import SUBJECT_TOKEN_HEADER

from .conftest import PASSPHRASE, NodeKeys, enroll

APP_KEY = {"name": "app:workbench@gpu-box", "limits": {"writeLogs": True}}
APP_CLIENT = {
    "name": "Workbench",
    "redirectUris": ["http://192.0.2.7:8300/oidc/callback"],
    "owner": "app:workbench@gpu-box",
}


def _session_through(client: TestClient, keys: NodeKeys) -> str:
    response = client.post(
        "/v1/auth/login",
        json={"passphrase": PASSPHRASE},
        headers={"Authorization": f"Bearer {keys.service_token()}"},
    )
    assert response.status_code == 200, response.text
    return str(response.json()["sessionToken"])


def _exchanged(client: TestClient, console: NodeKeys, session: str, node: str) -> str:
    response = client.post(
        "/v1/auth/token",
        json={"subjectToken": session, "audience": f"node:{node}"},
        headers={"Authorization": f"Bearer {console.service_token()}"},
    )
    assert response.status_code == 200, response.text
    return str(response.json()["accessToken"])


def _install(client: TestClient) -> tuple[NodeKeys, NodeKeys, NodeKeys, str, str]:
    """A console (laptop), the machine it acts on (gpu-box), a bystander
    (attic), the operator's session at the console, and the token the
    console's proxy hands gpu-box."""
    laptop, _ = enroll(client, "laptop")
    gpu, _ = enroll(client, "gpu-box")
    attic, _ = enroll(client, "attic")
    session = _session_through(client, laptop)
    return laptop, gpu, attic, session, _exchanged(client, laptop, session, "gpu-box")


def _acting(node: NodeKeys, subject: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {node.service_token()}", SUBJECT_TOKEN_HEADER: subject}


def _keys(client: TestClient) -> list[dict[str, Any]]:
    return list(client.get("/v1/auth/client-keys").json()["keys"])


def test_the_token_the_console_hands_a_machine_does_not_open_this_root_on_its_own(
    active_client: TestClient,
) -> None:
    """What the install did before the fix, and what still must not work: it
    is not addressed here, so passing it on is forwarding past its audience."""
    _, _, _, _, exchanged = _install(active_client)
    response = active_client.post(
        "/v1/auth/client-keys", json=APP_KEY, headers={"Authorization": f"Bearer {exchanged}"}
    )
    assert response.status_code == 401, response.text


def test_the_machine_acting_for_the_operator_makes_its_own_apps_key(
    active_client: TestClient,
) -> None:
    _, gpu, _, _, exchanged = _install(active_client)
    made = active_client.post("/v1/auth/client-keys", json=APP_KEY, headers=_acting(gpu, exchanged))
    assert made.status_code == 201, made.text
    assert made.json()["key"]["name"] == "app:workbench@gpu-box"
    assert made.json()["key"]["id"] in {k["id"] for k in _keys(active_client)}


def test_and_registers_its_own_apps_sign_in(active_client: TestClient) -> None:
    _, gpu, _, _, exchanged = _install(active_client)
    made = active_client.post("/v1/oidc/clients", json=APP_CLIENT, headers=_acting(gpu, exchanged))
    assert made.status_code == 201, made.text
    client_id = made.json()["client"]["clientId"]
    gone = active_client.delete(f"/v1/oidc/clients/{client_id}", headers=_acting(gpu, exchanged))
    assert gone.status_code == 204, gone.text
    listed = active_client.get("/v1/oidc/clients").json()["clients"]
    assert client_id not in {c["clientId"] for c in listed}


def test_it_may_name_nothing_but_its_own_apps(active_client: TestClient) -> None:
    _, gpu, _, _, exchanged = _install(active_client)
    for name in (
        "laptop",
        "app:workbench@attic",
        "app:workbench@gpu-box2",
        "app:@gpu-box",
        "app:x@gpu-box@attic",
    ):
        response = active_client.post(
            "/v1/auth/client-keys", json={"name": name}, headers=_acting(gpu, exchanged)
        )
        assert response.status_code == 403, (name, response.text)
    for owner in (None, "app:workbench@attic", "operator"):
        body = {k: v for k, v in APP_CLIENT.items() if k != "owner"}
        if owner is not None:
            body["owner"] = owner
        response = active_client.post(
            "/v1/oidc/clients", json=body, headers=_acting(gpu, exchanged)
        )
        assert response.status_code == 403, (owner, response.text)
    assert {k["name"] for k in _keys(active_client)} == set()


def test_it_may_remove_nothing_but_its_own_apps(active_client: TestClient) -> None:
    _, gpu, _, _, exchanged = _install(active_client)
    laptop_key = active_client.post("/v1/auth/client-keys", json={"name": "laptop"}).json()["key"]
    other = active_client.post(
        "/v1/oidc/clients", json={**APP_CLIENT, "owner": "app:workbench@attic"}
    ).json()["client"]
    revoked = active_client.delete(
        f"/v1/auth/client-keys/{laptop_key['id']}", headers=_acting(gpu, exchanged)
    )
    assert revoked.status_code == 403, revoked.text
    assert not next(k for k in _keys(active_client) if k["id"] == laptop_key["id"]).get("revokedAt")
    dropped = active_client.delete(
        f"/v1/oidc/clients/{other['clientId']}", headers=_acting(gpu, exchanged)
    )
    assert dropped.status_code == 403, dropped.text

    own = active_client.post(
        "/v1/auth/client-keys", json=APP_KEY, headers=_acting(gpu, exchanged)
    ).json()["key"]
    assert (
        active_client.delete(
            f"/v1/auth/client-keys/{own['id']}", headers=_acting(gpu, exchanged)
        ).status_code
        == 204
    )
    assert next(k for k in _keys(active_client) if k["id"] == own["id"]).get("revokedAt")


def test_only_the_machine_the_token_is_addressed_to_may_present_it(
    active_client: TestClient,
) -> None:
    """A bystander that captured gpu-box's token cannot spend it here."""
    _, _, attic, _, exchanged = _install(active_client)
    for body in (APP_KEY, {"name": "app:workbench@attic"}):
        response = active_client.post(
            "/v1/auth/client-keys", json=body, headers=_acting(attic, exchanged)
        )
        assert response.status_code == 401, response.text


def test_a_machines_own_token_opens_nothing_without_an_operator_behind_it(
    active_client: TestClient,
) -> None:
    _, gpu, _, session, _ = _install(active_client)
    alone = active_client.post(
        "/v1/auth/client-keys",
        json=APP_KEY,
        headers={"Authorization": f"Bearer {gpu.service_token()}"},
    )
    assert alone.status_code == 401, alone.text
    # Nor with a subject addressed to some other machine: the console's own
    # session names the laptop and this root, never gpu-box. Nor with one it
    # minted for itself, addressed to itself: only the root mints a session,
    # so only a session says an operator is behind it.
    for subject in (
        session,
        "nonsense",
        gpu.service_token(),
        gpu.service_token(aud=("node:gpu-box",)),
    ):
        response = active_client.post(
            "/v1/auth/client-keys", json=APP_KEY, headers=_acting(gpu, subject)
        )
        assert response.status_code == 401, (subject[:20], response.text)


def test_the_actor_must_be_an_agent_speaking_for_itself(active_client: TestClient) -> None:
    laptop, _ = enroll(active_client, "laptop")
    gpu, _ = enroll(active_client, "gpu-box", grants=["gateway"])
    session = _session_through(active_client, laptop)
    exchanged = _exchanged(active_client, laptop, session, "gpu-box")
    for actor in (gpu.service_token(sub="gateway"), session, exchanged):
        response = active_client.post(
            "/v1/auth/client-keys",
            json=APP_KEY,
            headers={"Authorization": f"Bearer {actor}", SUBJECT_TOKEN_HEADER: exchanged},
        )
        assert response.status_code == 401, response.text


def test_signing_out_ends_what_the_machine_may_do_for_you(active_client: TestClient) -> None:
    _, gpu, _, session, exchanged = _install(active_client)
    assert (
        active_client.delete(
            "/v1/auth/sessions/current", headers={"Authorization": f"Bearer {session}"}
        ).status_code
        == 204
    )
    response = active_client.post(
        "/v1/auth/client-keys", json=APP_KEY, headers=_acting(gpu, exchanged)
    )
    assert response.status_code == 401, response.text
    assert "signed out" in response.text


def test_the_pair_opens_only_the_routes_an_app_install_needs(active_client: TestClient) -> None:
    """Listing keys, changing a key's limits, people and everything else stay
    an operator session's alone."""
    _, gpu, _, _, exchanged = _install(active_client)
    key = active_client.post("/v1/auth/client-keys", json=APP_KEY, headers=_acting(gpu, exchanged))
    key_id = key.json()["key"]["id"]
    headers = _acting(gpu, exchanged)
    for method, path, body in (
        ("GET", "/v1/auth/client-keys", None),
        ("PUT", f"/v1/auth/client-keys/{key_id}/limits", {"limits": {}}),
        ("GET", "/v1/oidc/clients", None),
        ("GET", "/v1/people", None),
        ("POST", "/v1/nodes/join-token", None),
    ):
        response = active_client.request(method, path, json=body, headers=headers)
        assert response.status_code == 401, (method, path, response.status_code, response.text)
