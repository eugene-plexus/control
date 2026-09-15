"""The trust root's auth surface.

The property worth stating up front, because it is the one that differs
from every other component: **there is no auth-disabled mode.**
Elsewhere a missing signing key means "running standalone, let
everything through". Here it means the install is not initialized, and
the answer is 503 with a pointer at setup. A trust root that waved
requests through until it was configured would be a trust root with a
window in it, and the window would be open on exactly the machine that
holds the keys.
"""

from __future__ import annotations

import base64
from pathlib import Path

from fastapi.testclient import TestClient

from eugene_plexus_control import security
from eugene_plexus_control.app import create_app

from .conftest import PASSPHRASE, settings_for


def test_a_fresh_install_reports_uninitialized_without_a_token(
    uninitialized_client: TestClient,
) -> None:
    """What the UI asks before it has anything to ask with."""
    response = uninitialized_client.get("/v1/auth/status")
    assert response.status_code == 200
    body = response.json()
    assert body["initialized"] is False
    # S0: a fresh root is also not unlocked, and it says whether this
    # host's keyring could keep it unlocked (memoised False under test).
    assert body["unlocked"] is False
    assert body["keyringAvailable"] is False


def test_an_uninitialized_root_refuses_rather_than_opening_up(
    uninitialized_client: TestClient,
) -> None:
    response = uninitialized_client.get("/v1/nodes")
    assert response.status_code == 503
    assert "initialize" in response.text


def test_initialize_establishes_the_whole_replication_set(tmp_path: Path) -> None:
    """One call produces everything a standby would ever need.

    Asserted field by field because the last time this list was short —
    in the contract rather than the code — the consequence was a standby
    that could not be promoted, and nothing failed until someone tried.
    """
    app = create_app(settings_for(tmp_path / "install"))
    with TestClient(app) as client:
        assert (
            client.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
        )
        identity = app.state.machine.state.identity
        assert identity.salt
        assert identity.passphraseVerifier.startswith("$argon2id$")
        assert identity.sealedSigningKey
        assert identity.sealedControlKey
        assert identity.controlPublicKey
        assert identity.sealedRecoveryKey
        assert identity.signingKeyId == "1"


def test_initialize_is_refused_twice(tmp_path: Path) -> None:
    """There is no reset endpoint, by design: the passphrase is what the
    sealed secrets are sealed under, so a reset would mean discarding
    every credential in the install."""
    app = create_app(settings_for(tmp_path / "install"))
    with TestClient(app) as client:
        assert (
            client.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
        )
        again = client.post("/v1/auth/initialize", json={"passphrase": "something-else"})
        assert again.status_code == 409


def test_a_restart_is_not_a_re_key(tmp_path: Path) -> None:
    """The behavioural change M5 required, asserted rather than assumed.

    Through M4 the signing key was minted per process, so a restart
    invalidated every service token in the install. Harmless with one
    process; an outage with N nodes, because a restart would break every
    component that had not yet been told. The key is now persisted
    sealed and recovered at login.
    """
    directory = tmp_path / "install"
    app = create_app(settings_for(directory))
    with TestClient(app) as client:
        assert (
            client.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
        )
        first_key = app.state.auth_state.signing_key
        assert first_key is not None
        token = client.post("/v1/auth/login", json={"passphrase": PASSPHRASE}).json()[
            "sessionToken"
        ]

    restarted = create_app(settings_for(directory))
    with TestClient(restarted) as client:
        # Locked until someone logs in, and honest about which of the
        # two "no key" situations this is.
        locked = client.get("/v1/nodes")
        assert locked.status_code == 503
        assert "Locked" in locked.text

        client.post("/v1/auth/login", json={"passphrase": PASSPHRASE})
        assert restarted.state.auth_state.signing_key == first_key

        # The token minted before the restart still works, which is the
        # whole point.
        response = client.get("/v1/nodes", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200


def test_a_wrong_passphrase_is_rejected_and_rate_limited(tmp_path: Path) -> None:
    app = create_app(settings_for(tmp_path / "install"))
    with TestClient(app) as client:
        assert (
            client.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
        )
        for _ in range(5):
            assert client.post("/v1/auth/login", json={"passphrase": "wrong"}).status_code == 401
        limited = client.post("/v1/auth/login", json={"passphrase": "wrong"})
        assert limited.status_code == 429
        # And the limit is per source, not a global lockout that a
        # correct passphrase cannot get past — but from this one source
        # even the right passphrase waits, which is the point.
        assert client.post("/v1/auth/login", json={"passphrase": PASSPHRASE}).status_code == 429


def test_a_service_token_can_read_but_not_mutate(active_client: TestClient) -> None:
    """An agent needs the epoch; a compromised peer must not re-key the
    install. That asymmetry is the reason reads and mutations declare
    different levels."""
    auth = active_client.app.state.auth_state  # type: ignore[attr-defined]
    service = security.issue_service_token(signing_key=auth.signing_key, kind="gateway")
    headers = {"Authorization": f"Bearer {service}"}

    assert active_client.get("/v1/control/status", headers=headers).status_code == 200
    assert active_client.get("/v1/nodes", headers=headers).status_code == 200

    assert active_client.post("/v1/nodes/join-token", headers=headers).status_code == 401
    assert active_client.post("/v1/control/rotate-key", headers=headers).status_code == 401
    assert (
        active_client.patch("/v1/config", json={"uiTheme": "dark"}, headers=headers).status_code
        == 401
    )


def test_logout_revokes_only_the_presented_token(active_client: TestClient) -> None:
    token = active_client.headers["Authorization"].removeprefix("Bearer ")
    assert active_client.delete("/v1/auth/sessions/current").status_code == 204
    assert active_client.get("/v1/nodes").status_code == 401

    fresh = active_client.post("/v1/auth/login", json={"passphrase": PASSPHRASE}).json()[
        "sessionToken"
    ]
    assert fresh != token
    assert (
        active_client.get("/v1/nodes", headers={"Authorization": f"Bearer {fresh}"}).status_code
        == 200
    )


def test_the_master_key_is_derived_not_stored(tmp_path: Path) -> None:
    """Nothing on disk lets anyone reach the master key.

    The salt and the verifier are both replicated on purpose, and
    neither is a secret: the salt makes derivation deterministic, the
    verifier proves the operator without holding the key. What is
    absent is what matters.
    """
    directory = tmp_path / "install"
    app = create_app(settings_for(directory))
    with TestClient(app) as client:
        assert (
            client.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
        )
        master = app.state.auth_state.master_key

    on_disk = (directory / "state" / "snapshot.json").read_bytes()
    assert base64.b64encode(master) not in on_disk
    assert master not in on_disk
    assert PASSPHRASE.encode() not in on_disk

    # ...and the same passphrase plus the replicated salt reproduces it,
    # which is what a standby does at promotion.
    identity = app.state.machine.state.identity
    assert security.derive_master_key(PASSPHRASE, base64.b64decode(identity.salt)) == master
