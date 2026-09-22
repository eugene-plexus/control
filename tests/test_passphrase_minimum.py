"""First-run setup refuses a passphrase shorter than twelve characters.

The passphrase is what every sealed secret in the install is sealed
under, and there is no reset path by design, so the one moment it can
be made strong is the moment it is chosen. Until 2026-09-22 the only
refusal was the empty string: `a` would do.

The rule is for *choosing* a passphrase and nothing else. An install
that already has a short one must still sign in and still unlock --
refusing it at login would be a lockout with no way back, since the
passphrase cannot be changed.
"""

from __future__ import annotations

import base64
from pathlib import Path

from fastapi.testclient import TestClient

from eugene_plexus_control import sealing, security
from eugene_plexus_control.app import create_app
from eugene_plexus_control.routes import auth as auth_routes

from .conftest import settings_for


def test_initialize_refuses_eleven_characters_and_leaves_the_install_fresh(
    uninitialized_client: TestClient,
) -> None:
    response = uninitialized_client.post("/v1/auth/initialize", json={"passphrase": "elevenchars"})

    assert response.status_code == 400, response.text
    body = response.json()["detail"]
    assert "12 characters" in body["detail"]
    assert "elevenchars" not in response.text
    # Refused before anything was minted or sealed: still first-run.
    assert uninitialized_client.get("/v1/auth/status").json()["initialized"] is False


def test_initialize_accepts_exactly_twelve(uninitialized_client: TestClient) -> None:
    response = uninitialized_client.post("/v1/auth/initialize", json={"passphrase": "twelve-chars"})
    assert response.status_code == 204, response.text


def test_the_empty_passphrase_is_still_refused(uninitialized_client: TestClient) -> None:
    """422, not 400: the contract's own `minLength: 1` answers before the
    handler runs. The contract is not where the twelve lives -- the agent
    and the UI change with it, and a spec change is a codegen radius."""
    response = uninitialized_client.post("/v1/auth/initialize", json={"passphrase": ""})
    assert response.status_code == 422
    assert uninitialized_client.get("/v1/auth/status").json()["initialized"] is False


def test_the_minimum_is_the_one_the_agent_and_the_ui_use() -> None:
    """Three components ask for the same passphrase; if their minimums
    disagree, the wizard passes one and the trust root refuses it."""
    assert auth_routes.MIN_PASSPHRASE_LENGTH == 12


def test_an_existing_install_with_a_short_passphrase_still_signs_in(tmp_path: Path) -> None:
    """Seeded the way `initialize` seeds, with a passphrase the new rule
    would refuse -- an install made before the rule existed."""
    short = "hunter2"
    app = create_app(settings_for(tmp_path / "old"))
    with TestClient(app) as client:
        salt = security.generate_master_key_salt()
        master_key = security.derive_master_key(short, salt)
        control_identity = sealing.generate_control_identity()
        recovery = sealing.generate_sealing_keypair()
        app.state.machine.seed_identity(
            {
                "salt": base64.b64encode(salt).decode("ascii"),
                "passphraseVerifier": security.hash_passphrase(short),
                "sealedSigningKey": security.seal_b64(security.generate_signing_key(), master_key),
                "sealedControlKey": security.seal_b64(
                    base64.b64decode(control_identity.private), master_key
                ),
                "controlPublicKey": control_identity.public,
                "sealedRecoveryKey": security.seal_b64(
                    base64.b64decode(recovery.private), master_key
                ),
                "signingKeyId": "1",
            }
        )

        response = client.post("/v1/auth/login", json={"passphrase": short})

    assert response.status_code == 200, response.text
    assert response.json()["sessionToken"]
