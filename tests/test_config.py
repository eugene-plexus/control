"""The config trio, and the thing that makes it different here.

A `PATCH /v1/config` is a **log entry**. That is the whole point: a
promoted standby has to come up configured the way it would have been
had it been active all along, and config held only in a local file on
the active root is config the standby silently lacks — discovered during
a failover, which is the worst moment to discover anything.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from eugene_plexus_control import config as config_module
from eugene_plexus_control.app import create_app

from .conftest import PASSPHRASE, settings_for


def test_the_schema_describes_every_field_without_a_token(
    uninitialized_client: TestClient,
) -> None:
    """Unauthenticated like every other component's: it describes the
    shape of the form, not its contents."""
    response = uninitialized_client.get("/v1/config/schema")
    assert response.status_code == 200
    body = response.json()
    assert body["component"] == "control"
    keys = {field["key"] for field in body["fields"]}
    assert "standbyUrls" in keys
    assert "securityMode" in keys
    # Not present, and deliberately: the port is the agent's, and the
    # role is an env var because a role a config PATCH could change
    # would be a second writer reachable over HTTP.
    assert "bindPort" not in keys
    assert "role" not in keys


def test_the_keyring_field_says_it_is_host_bound(uninitialized_client: TestClient) -> None:
    """§7's tension, in words an operator reads before choosing rather
    than discovers at promotion time."""
    body = uninitialized_client.get("/v1/config/schema").json()
    security_mode = next(f for f in body["fields"] if f["key"] == "securityMode")
    assert "host-bound" in security_mode["description"]
    assert "standby" in security_mode["description"]


def test_a_patch_becomes_a_log_entry(active_client: TestClient) -> None:
    machine = active_client.app.state.machine  # type: ignore[attr-defined]
    before = machine.state.index

    response = active_client.patch("/v1/config", json={"uiTheme": "dark"})
    assert response.status_code == 200
    assert response.json()["applied"] == ["uiTheme"]

    assert machine.state.index == before + 1
    entries = machine.read_page(before, 10)
    assert entries[-1]["op"] == "patchConfig"
    assert entries[-1]["payload"]["values"] == {"uiTheme": "dark"}


def test_rejected_keys_do_not_discard_the_accepted_ones(active_client: TestClient) -> None:
    """Telling the operator which field was wrong beats refusing all
    five because one was."""
    response = active_client.patch(
        "/v1/config", json={"uiTheme": "dark", "nodePollIntervalSeconds": 99999}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["applied"] == ["uiTheme"]
    assert [r["key"] for r in body["rejected"]] == ["nodePollIntervalSeconds"]
    assert "<= 600" in body["rejected"][0]["message"]


def test_a_rejected_value_never_reaches_the_log(active_client: TestClient) -> None:
    """An entry carrying a value the writer rejected would apply on a
    replica running different validation rules, which is how two roots
    end up with different config from the same log."""
    machine = active_client.app.state.machine  # type: ignore[attr-defined]
    before = machine.state.index
    response = active_client.patch("/v1/config", json={"uiTheme": "chartreuse"})
    assert response.status_code == 200
    assert response.json()["applied"] == []
    assert machine.state.index == before, "nothing valid was patched, so nothing was logged"


def test_url_list_validation_rejects_duplicates_and_junk(active_client: TestClient) -> None:
    good = active_client.patch(
        "/v1/config", json={"standbyUrls": ["http://a:8083", "http://b:8083"]}
    )
    assert good.json()["applied"] == ["standbyUrls"]

    duplicated = active_client.patch(
        "/v1/config", json={"standbyUrls": ["http://a:8083", "http://a:8083"]}
    )
    assert "duplicates" in duplicated.json()["rejected"][0]["message"]

    wrong_type = active_client.patch("/v1/config", json={"standbyUrls": "http://a:8083"})
    assert "expected a list" in wrong_type.json()["rejected"][0]["message"]


def test_config_survives_a_restart_through_the_log(tmp_path: Path) -> None:
    """Not through the YAML file — through the log.

    The file is a boot-time cache, so the assertion deliberately deletes
    it before restarting. If applied config came from the file rather
    than from the log, a compaction would lose it and a standby would
    never have it.
    """
    directory = tmp_path / "install"
    app = create_app(settings_for(directory))
    with TestClient(app) as client:
        assert (
            client.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
        )
        token = client.post("/v1/auth/login", json={"passphrase": PASSPHRASE}).json()["token"]
        client.patch(
            "/v1/config",
            json={"uiTheme": "dark", "joinTokenTtlSeconds": 60},
            headers={"Authorization": f"Bearer {token}"},
        )

    (directory / "control.yaml").unlink()

    restarted = create_app(settings_for(directory))
    with TestClient(restarted):
        values = restarted.state.machine.state.config
        assert values["uiTheme"] == "dark"
        assert values["joinTokenTtlSeconds"] == 60


def test_the_bootstrap_cache_is_written_through(tmp_path: Path) -> None:
    """It exists so the process can find its log level before it has
    replayed anything. Worth asserting it is actually kept current,
    because a stale one means a restart logs at the wrong level and
    nobody notices until they need the logs."""
    directory = tmp_path / "install"
    app = create_app(settings_for(directory))
    with TestClient(app) as client:
        assert (
            client.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
        )
        token = client.post("/v1/auth/login", json={"passphrase": PASSPHRASE}).json()["token"]
        client.patch(
            "/v1/config", json={"logLevel": "DEBUG"}, headers={"Authorization": f"Bearer {token}"}
        )

    on_disk = yaml.safe_load((directory / "control.yaml").read_text(encoding="utf-8"))
    assert on_disk["logLevel"] == "DEBUG"


def test_a_restart_required_field_says_so(active_client: TestClient) -> None:
    response = active_client.patch("/v1/config", json={"logLevel": "DEBUG"})
    body = response.json()
    assert body["requiresRestart"] is True
    assert body["pendingRestart"] == ["logLevel"]


def test_a_broken_config_file_does_not_stop_startup(tmp_path: Path) -> None:
    """Bad config never crashes a component: the config endpoints are
    how a broken config gets repaired, so refusing to boot over one
    locks the operator out of the fix."""
    directory = tmp_path / "install"
    directory.mkdir()
    (directory / "control.yaml").write_text("this: [is: not: valid", encoding="utf-8")

    app = create_app(settings_for(directory))
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/v1/config/schema").status_code == 200


def test_safe_mode_ignores_config_but_not_the_log(tmp_path: Path) -> None:
    """The distinction that matters. Safe mode exists to let an operator
    repair configuration, not to make the install forget itself — a
    control root that came up ignoring its own node registry would hand
    out wrong answers rather than degraded ones."""
    directory = tmp_path / "install"
    app = create_app(settings_for(directory))
    with TestClient(app) as client:
        assert (
            client.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
        )
        token = client.post("/v1/auth/login", json={"passphrase": PASSPHRASE}).json()["token"]
        client.patch(
            "/v1/config", json={"uiTheme": "dark"}, headers={"Authorization": f"Bearer {token}"}
        )
        app.state.machine.append(
            "enrollNode",
            {"name": "gpu-box", "publicKey": "a" * 44, "enrolledAt": "2026-09-09T00:00:00+00:00"},
        )

    settings = settings_for(directory)
    settings.safe_mode = True
    safe = create_app(settings)
    with TestClient(safe) as client:
        health = client.get("/healthz").json()
        assert health["safeMode"] is True
        assert health["status"] == "degraded"
        # The registry is intact; only the config file was skipped.
        assert health["details"]["nodes"] == 1
        assert safe.state.machine.state.config["uiTheme"] == "dark"


def test_effective_config_layers_applied_over_defaults() -> None:
    values = config_module.effective({"uiTheme": "dark"})
    assert values["uiTheme"] == "dark"
    assert values["nodePollIntervalSeconds"] == 15
