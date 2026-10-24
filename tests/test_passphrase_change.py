"""Row 5 (5b): Eugene's passphrase changes, and every sealed value with it.

`specs/docs/design/rows/row-05.md` §3. The root opens every value sealed
under the old master key, seals it under the new one, appends one
`changePassphrase` entry, snapshots and compacts, so nothing on disk opens
with the old key any more. The walk below looks at every string in the
snapshot and the log, so a sealed field added later is caught here too.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_control import keyring_store, passphrase_change, security
from eugene_plexus_control.app import create_app
from eugene_plexus_control.applied import (
    OP_CHANGE_PASSPHRASE,
    OP_PATCH_CONFIG,
    OP_PUT_OIDC_KEY,
    ApplyError,
    apply,
)
from eugene_plexus_control.oidc import new_signing_key, utcnow_iso

from .conftest import PASSPHRASE, login, settings_for, standby_at
from .test_oidc import _client, _tokens

NEW = "a-new-and-longer-passphrase"


def _strings(value: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    """Every string, and every envelope-shaped object, with where it is."""
    if isinstance(value, dict):
        if security.is_envelope(value):
            yield path, value
        for key, item in value.items():
            yield from _strings(item, f"{path}.{key}")
    elif isinstance(value, list):
        for position, item in enumerate(value):
            yield from _strings(item, f"{path}[{position}]")
    elif isinstance(value, str):
        yield path, value


def _opens(value: Any, key: bytes) -> bool:
    try:
        if isinstance(value, dict):
            security.open_envelope(security.Envelope.from_dict(value), key)
        else:
            security.open_b64(value, key)
    except (ValueError, TypeError):
        return False
    return True


def _opening(documents: list[Any], key: bytes) -> list[str]:
    return [path for doc in documents for path, value in _strings(doc) if _opens(value, key)]


def _on_disk(state_dir: Path) -> list[Any]:
    documents: list[Any] = []
    snapshot = state_dir / "snapshot.json"
    if snapshot.exists():
        documents.append(json.loads(snapshot.read_text(encoding="utf-8")))
    log = state_dir / "log.jsonl"
    if log.exists():
        documents += [json.loads(line) for line in log.read_text("utf-8").splitlines() if line]
    return documents


def _with_every_sealed_value(c: TestClient) -> None:
    """A sign-in key, a connected account's secret, the install CA (made at
    initialize) and the three identity keys."""
    _tokens(c, _client(c))
    made = c.post(
        "/v1/auth/providers",
        json={
            "name": "Example SSO",
            "kind": "oidc",
            "issuer": "https://idp.example",
            "clientId": "eugene-test",
            "clientSecret": "upstream-secret",
            "tokenAuthMethod": "client_secret_post",
            "enabled": False,
        },
    )
    assert made.status_code == 200, made.text


def _captured(machine: Any, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Every entry the root writes, before compaction can take it."""
    entries: list[dict[str, Any]] = []
    original = machine._store.append

    def spy(entry: dict[str, Any]) -> None:
        entries.append(json.loads(json.dumps(entry)))
        original(entry)

    monkeypatch.setattr(machine._store, "append", spy)
    return entries


def test_change_reseals_every_sealed_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = create_app(settings_for(tmp_path / "active"))
    state_dir = tmp_path / "active" / "state"
    with TestClient(app) as c:
        assert c.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
        c.headers["Authorization"] = f"Bearer {login(c)}"
        _with_every_sealed_value(c)
        machine = app.state.machine
        state = machine.state
        assert state.identity.sealedInstallCaKey and state.oidc_keys
        assert any(p["sealedSecret"] for p in state.browser_sign_in["providers"].values())
        old_key = app.state.auth_state.master_key
        before = machine.snapshot_document()
        assert _opening(_on_disk(state_dir), old_key), "the walk finds nothing to begin with"
        entries = _captured(machine, monkeypatch)

        answer = c.post("/v1/auth/passphrase", json={"current": PASSPHRASE, "new": NEW})
        assert answer.status_code == 200, answer.text
        body = answer.json()
        assert body["machines"] == [] and body["passphraseFile"] == "none"
        assert PASSPHRASE not in answer.text and NEW not in answer.text

        # The new passphrase unlocks, and the old one gets 401.
        assert c.post("/v1/auth/login", json={"passphrase": PASSPHRASE}).status_code == 401
        assert c.post("/v1/auth/login", json={"passphrase": NEW}).status_code == 200
        new_key = app.state.auth_state.master_key
        assert new_key is not None and new_key != old_key

        # No string in the snapshot or the log opens with the old key, and
        # the values it held open with the new one.
        on_disk = _on_disk(state_dir)
        assert _opening(on_disk, old_key) == []
        assert _opening(on_disk, new_key)
        assert [e["op"] for e in entries] == [OP_CHANGE_PASSPHRASE]
        # Everything before the change is compacted away; the change itself
        # stays, so a standby one entry behind reads it from the log.
        kept = [json.loads(line) for line in (state_dir / "log.jsonl").read_text("utf-8").split()]
        assert [e["op"] for e in kept] == [OP_CHANGE_PASSPHRASE]
        # The keys the passphrase protects are the same keys.
        assert machine.state.identity.rootTokenPublicKey == state.identity.rootTokenPublicKey
        assert [k["kid"] for k in machine.state.oidc_keys] == [k["kid"] for k in state.oidc_keys]
        assert c.get("/v1/oidc/keys").status_code == 200
        assert _tokens(c, _client(c, name="After"), password=NEW)["id_token"]

        # A replica ends with identical canonical bytes, and compacts too.
        replica_dir = tmp_path / "replica"
        replica = standby_at(replica_dir)
        replica.install_snapshot(before)
        for entry in entries:
            replica.accept_replicated(entry)
        assert replica.canonical() == machine.canonical()
        assert _opening(_on_disk(replica_dir), old_key) == []


def test_a_value_that_does_not_open_stops_the_change_and_names_it(
    active_client: TestClient,
) -> None:
    c = active_client
    machine = c.app.state.machine  # type: ignore[attr-defined]
    pem, jwk = new_signing_key()
    stranger = security.seal_b64(pem, bytes(32))
    machine.append(
        OP_PUT_OIDC_KEY,
        {
            "key": {
                "kid": jwk["kid"],
                "sealedKey": stranger,
                "publicJwk": jwk,
                "createdAt": utcnow_iso(),
            }
        },
    )
    salt = machine.state.identity.salt
    answer = c.post("/v1/auth/passphrase", json={"current": PASSPHRASE, "new": NEW})
    assert answer.status_code == 409, answer.text
    assert f"sign-in key {jwk['kid']}" in answer.json()["detail"]["detail"]
    assert machine.state.identity.salt == salt
    assert c.post("/v1/auth/login", json={"passphrase": PASSPHRASE}).status_code == 200


def test_a_sealed_value_the_reseal_does_not_know_stops_the_change(
    active_client: TestClient,
) -> None:
    """The walk is the net under the list: a field sealed under the master key
    that `passphrase_change` was never taught about is found, by where it is."""
    c = active_client
    app = c.app
    machine = app.state.machine  # type: ignore[attr-defined]
    key = app.state.auth_state.master_key  # type: ignore[attr-defined]
    machine.append(OP_PATCH_CONFIG, {"values": {"futureSecret": security.seal_b64(b"x", key)}})
    salt = machine.state.identity.salt
    answer = c.post("/v1/auth/passphrase", json={"current": PASSPHRASE, "new": NEW})
    assert answer.status_code == 409, answer.text
    assert "config.futureSecret" in answer.json()["detail"]["detail"]
    assert machine.state.identity.salt == salt


def test_the_change_needs_the_current_passphrase_a_session_and_a_long_new_one(
    active_client: TestClient,
) -> None:
    c = active_client
    wrong = c.post("/v1/auth/passphrase", json={"current": "not-the-passphrase", "new": NEW})
    assert wrong.status_code == 401
    assert "not-the-passphrase" not in wrong.text and NEW not in wrong.text
    short = c.post("/v1/auth/passphrase", json={"current": PASSPHRASE, "new": "xq7-tiny"})
    assert short.status_code == 422
    assert "xq7-tiny" not in short.text and PASSPHRASE not in short.text
    anonymous = c.post(
        "/v1/auth/passphrase",
        json={"current": PASSPHRASE, "new": NEW},
        headers={"Authorization": ""},
    )
    assert anonymous.status_code == 401
    # The login limiter counts a wrong `current`: four more and it is 429.
    for _ in range(4):
        c.post("/v1/auth/passphrase", json={"current": "still-wrong-passphrase", "new": NEW})
    limited = c.post("/v1/auth/passphrase", json={"current": PASSPHRASE, "new": NEW})
    assert limited.status_code == 429


def test_the_change_waits_for_a_standby_that_cannot_apply_it(active_client: TestClient) -> None:
    from datetime import UTC, datetime

    from .conftest import enroll

    c = active_client
    enroll(c, "b")
    assert c.put("/v1/nodes/b/standby", json={"on": True}).status_code in (200, 202, 204)
    c.app.state.standby_reports["b"] = {  # type: ignore[attr-defined]
        "appliedIndex": 0,
        "at": datetime.now(UTC),
        "seen": 0.0,
        "logVersion": 0,
    }
    answer = c.post("/v1/auth/passphrase", json={"current": PASSPHRASE, "new": NEW})
    assert answer.status_code == 409
    assert (
        answer.json()["detail"]["detail"] == "Update b first; its copy would stop at this change."
    )


def test_apply_refuses_a_change_that_misses_a_sealed_value(active_client: TestClient) -> None:
    c = active_client
    _tokens(c, _client(c))
    state = c.app.state.machine.state  # type: ignore[attr-defined]
    good = passphrase_change.reseal(state, c.app.state.auth_state.master_key, NEW)  # type: ignore[attr-defined]
    entry = {"index": state.index + 1, "epoch": state.epoch, "op": OP_CHANGE_PASSPHRASE}
    assert apply(state, {**entry, "payload": good.payload}).identity.salt == good.payload["salt"]
    missing = json.loads(json.dumps(good.payload))
    missing["sealed"]["oidcKeys"] = {}
    with pytest.raises(ApplyError):
        apply(state, {**entry, "payload": missing})
    extra = json.loads(json.dumps(good.payload))
    extra["sealed"]["oidcKeys"]["another"] = "x"
    with pytest.raises(ApplyError):
        apply(state, {**entry, "payload": extra})


def test_the_keyring_moves_to_the_new_install_id(
    active_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    held: dict[str, bytes] = {}
    monkeypatch.setattr(
        keyring_store, "set_master_key", lambda k, i: held.__setitem__(i, k) or True
    )
    monkeypatch.setattr(keyring_store, "delete_master_key", lambda i: held.pop(i, None) is not None)
    c = active_client
    app = c.app
    assert c.patch("/v1/config", json={"securityMode": "os_keyring"}).status_code == 200
    old_id = keyring_store.install_id_for(app.state.machine.state.identity.salt)  # type: ignore[attr-defined]
    held[old_id] = app.state.auth_state.master_key  # type: ignore[attr-defined]
    assert (
        c.post("/v1/auth/passphrase", json={"current": PASSPHRASE, "new": NEW}).status_code == 200
    )
    new_id = keyring_store.install_id_for(app.state.machine.state.identity.salt)  # type: ignore[attr-defined]
    assert new_id != old_id
    assert held == {new_id: app.state.auth_state.master_key}  # type: ignore[attr-defined]


def test_the_passphrase_file_is_rewritten_or_reported_old(tmp_path: Path) -> None:
    secret = tmp_path / "passphrase"
    secret.write_text(PASSPHRASE, encoding="utf-8", newline="\n")
    app = create_app(settings_for(tmp_path / "root", passphrase_file=secret))
    with TestClient(app) as c:
        assert c.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
        c.headers["Authorization"] = f"Bearer {login(c)}"
        assert c.patch("/v1/config", json={"securityMode": "passphrase_file"}).status_code == 200
        assert passphrase_change.file_state(app) == "matches"
        assert c.get("/v1/auth/status").json()["passphraseFile"] == "matches"
        answer = c.post("/v1/auth/passphrase", json={"current": PASSPHRASE, "new": NEW})
        assert answer.json()["passphraseFile"] == "matches"
        assert secret.read_text(encoding="utf-8") == NEW

        # A file this root cannot write (a container secret) holds the old one.
        target = tmp_path / "read-only-dir"
        target.mkdir()
        fixed = target / "passphrase"
        fixed.write_text(NEW, encoding="utf-8", newline="\n")
        app.state.settings.passphrase_file = fixed

        def refuse(*_: Any, **__: Any) -> None:
            raise PermissionError("read-only file system")

        import eugene_plexus_control.passphrase_file as pf

        original = pf._write
        pf._write = refuse  # type: ignore[assignment]
        try:
            again = c.post("/v1/auth/passphrase", json={"current": NEW, "new": NEW + "-2"})
        finally:
            pf._write = original  # type: ignore[assignment]
        assert again.status_code == 200, again.text
        assert again.json()["passphraseFile"] == "old"
        assert c.get("/v1/auth/status").json()["passphraseFile"] == "old"
        assert fixed.read_text(encoding="utf-8") == NEW


def test_a_crash_before_the_keyring_write_is_cleaned_at_the_next_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    held: dict[str, bytes] = {}
    monkeypatch.setattr(
        keyring_store, "set_master_key", lambda k, i: held.__setitem__(i, k) or True
    )
    monkeypatch.setattr(keyring_store, "get_master_key", lambda i: held.get(i))
    monkeypatch.setattr(keyring_store, "delete_master_key", lambda i: held.pop(i, None) is not None)
    settings = settings_for(tmp_path / "root")
    app = create_app(settings)
    with TestClient(app) as c:
        assert c.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
        c.headers["Authorization"] = f"Bearer {login(c)}"
        assert c.patch("/v1/config", json={"securityMode": "os_keyring"}).status_code == 200
        assert c.post("/v1/auth/login", json={"passphrase": PASSPHRASE}).status_code == 200
        old_id = keyring_store.install_id_for(app.state.machine.state.identity.salt)
        assert old_id in held

        def crash(*_: Any) -> None:
            raise RuntimeError("killed between the entry and the keyring")

        monkeypatch.setattr(passphrase_change, "move_keyring", crash)
        with pytest.raises(RuntimeError):
            c.post("/v1/auth/passphrase", json={"current": PASSPHRASE, "new": NEW})

    caplog.set_level("WARNING")
    with TestClient(create_app(settings)) as c:
        assert held == {}
        assert any(
            "The stored key no longer opens this install (the passphrase changed); sign in."
            in r.getMessage()
            for r in caplog.records
        )
        assert c.get("/v1/auth/status").json()["unlocked"] is False
        assert c.post("/v1/auth/login", json={"passphrase": NEW}).status_code == 200


def test_the_old_master_key_is_not_derivable_from_the_new_salt() -> None:
    salt = base64.b64decode(base64.b64encode(security.generate_master_key_salt()))
    assert security.derive_master_key(PASSPHRASE, salt) != security.derive_master_key(NEW, salt)


async def test_a_standby_follows_the_change_by_entry_or_by_snapshot(tmp_path: Path) -> None:
    """One entry behind, a standby reads the change from the log; further
    behind, it takes the snapshot, whose salt is new and whose identity key
    is the same, and accepts it. Either way it hears the old salt, and the
    root records the log version it applies."""
    import httpx

    from eugene_plexus_control.applied import LOG_VERSION
    from eugene_plexus_control.replication import Follower

    from .conftest import node_keys
    from .test_replication_follower import _LiveServer

    app = create_app(settings_for(tmp_path / "active"))
    with _LiveServer(app) as server:
        async with httpx.AsyncClient(base_url=server.url) as c:
            started = await c.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE})
            assert started.status_code == 204
            signed_in = await c.post("/v1/auth/login", json={"passphrase": PASSPHRASE})
            c.headers["Authorization"] = f"Bearer {signed_in.json()['sessionToken']}"
            spare = node_keys("spare")
            join = (await c.post("/v1/nodes/join-token")).json()["token"]
            assert (await c.post("/v1/nodes/enroll", json=spare.body(join))).status_code == 201
            assert (await c.put("/v1/nodes/spare/standby")).status_code == 202
            machine = app.state.machine
            far_behind = machine.snapshot_document()
            machine.append(OP_PATCH_CONFIG, {"values": {"joinTokenTtlSeconds": 120}})
            one_behind = machine.snapshot_document()
            old_salt = machine.state.identity.salt
            changed = await c.post("/v1/auth/passphrase", json={"current": PASSPHRASE, "new": NEW})
            assert changed.status_code == 200, changed.text

        heard: dict[str, list[str]] = {"near": [], "far": []}
        for name, start in (("near", one_behind), ("far", far_behind)):
            copy = standby_at(tmp_path / name)
            copy.install_snapshot(start)
            follower = Follower(
                copy,
                active_url=server.url,
                interval_seconds=0.01,
                token_provider=lambda: spare.service_token(sub="standby"),
                on_passphrase_changed=heard[name].append,
            )
            try:
                await follower.tick()
            finally:
                await follower.stop()
            assert copy.canonical() == machine.canonical(), name
            assert heard[name] == [old_salt], name
            log_file = tmp_path / name / "log.jsonl"
            kept = log_file.read_text("utf-8").split() if log_file.exists() else []
            assert [json.loads(e)["op"] for e in kept] in ([OP_CHANGE_PASSPHRASE], []), name
        assert app.state.standby_reports["spare"]["logVersion"] == LOG_VERSION


def test_the_standby_forgets_the_old_key_and_its_keyring_entry(
    active_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    deleted: list[str] = []
    monkeypatch.setattr(keyring_store, "delete_master_key", lambda i: deleted.append(i) or True)
    app = active_client.app
    salt = app.state.machine.state.identity.salt  # type: ignore[attr-defined]
    passphrase_change.standby_followed(app, salt)
    assert app.state.auth_state.master_key is None  # type: ignore[attr-defined]
    assert deleted == [keyring_store.install_id_for(salt)]


class _Nodes:
    """The root's view of its machines, and what it sent each one."""

    def __init__(self, probes: dict[str, Any], answers: dict[str, tuple[int, Any]]) -> None:
        self.probes, self.answers = probes, answers
        self.sent: list[tuple[str, str, dict[str, Any]]] = []

    async def probe_all(self, targets: Any, token_for: Any) -> dict[str, Any]:
        return self.probes

    async def post(self, url: str, path: str, token: Any, body: dict[str, Any]) -> Any:
        self.sent.append((url, path, body))
        for name, answer in self.answers.items():
            if name in url:
                return answer
        raise OSError("connection refused")

    async def aclose(self) -> None:
        return None


async def test_the_new_passphrase_goes_only_where_it_is_kept_and_never_in_plain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from eugene_plexus_control import _http
    from eugene_plexus_control.applied import OP_ENROLL_NODE
    from eugene_plexus_control.nodes_client import NodeProbe

    app = create_app(settings_for(tmp_path / "root"))
    app.state.settings.agent_url = "http://127.0.0.1:18079"
    with TestClient(app):
        machine = app.state.machine
        for name in ("root-box", "tls-box", "plain-box", "none-box", "old-box", "off-box"):
            machine.append(
                OP_ENROLL_NODE,
                {"name": name, "url": f"http://{name}:8079/", "enrolledAt": utcnow_iso()},
            )
        app.state.node_name = "root-box"
        probes = {
            "root-box": NodeProbe(name="root-box", reachable=True, own_passphrase=True),
            "tls-box": NodeProbe(name="tls-box", reachable=True, own_passphrase=True),
            "plain-box": NodeProbe(name="plain-box", reachable=True, own_passphrase=True),
            "none-box": NodeProbe(name="none-box", reachable=True, own_passphrase=False),
            "old-box": NodeProbe(name="old-box", reachable=True, own_passphrase=None),
            "off-box": NodeProbe(name="off-box", reachable=False, error="timed out"),
        }
        nodes = _Nodes(probes, {"127.0.0.1": (204, None), "tls-box": (204, None)})
        app.state.nodes_client = nodes
        monkeypatch.setattr(
            _http, "machine_route", lambda url: "https://tls-box:8083" if "tls-box" in url else None
        )
        results = await passphrase_change.tell_machines(app, PASSPHRASE, NEW)

    states = {r["node"]: r["state"] for r in results}
    assert states == {
        "root-box": "changed",
        "tls-box": "changed",
        "plain-box": "plain",
        "old-box": "pending",
        "off-box": "pending",
    }
    assert [url for url, _, _ in nodes.sent] == ["http://127.0.0.1:18079/", "http://tls-box:8079/"]
    assert all(path == "/v1/node/passphrase" for _, path, _ in nodes.sent)
    assert all(body == {"current": PASSPHRASE, "new": NEW} for _, _, body in nodes.sent)
    assert all(NEW not in json.dumps(r) and PASSPHRASE not in json.dumps(r) for r in results)
