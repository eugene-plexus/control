"""Does what we send match `control.yaml`?

Two endpoints deliberately bypass their response model and send bytes
verbatim — `GET /v1/control/snapshot` and `GET /v1/control/log` — because
re-serializing through Pydantic rewrites values in ways that are
semantically identical and byte-different, which is exactly the
difference replay equivalence exists to detect.

That buys fidelity at the cost of the framework's automatic conformance
check, so the check is made explicitly here instead. Skipping this file
is how the two endpoints would quietly drift from the contract they
claim to implement.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from eugene_plexus_control._generated.models import LogPage, Snapshot
from eugene_plexus_control.app import create_app
from eugene_plexus_control.state_machine import StateMachine

from .conftest import PASSPHRASE, settings_for
from .test_replay_equivalence import _write_history, placeholder


def _active(tmp_path: Path) -> tuple[TestClient, dict[str, str]]:
    app = create_app(settings_for(tmp_path / "active"))
    client = TestClient(app)
    client.__enter__()
    assert client.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
    token = client.post("/v1/auth/login", json={"passphrase": PASSPHRASE}).json()["token"]
    machine: StateMachine = app.state.machine
    _write_history(machine)
    # The shared history ends with a revocation, which correctly
    # cascades that node's component and runtime away — so it leaves
    # both collections empty. Re-populate them here, because this file
    # is asserting that a snapshot can *hold* each op's effect and an
    # empty list would pass every check without testing anything.
    machine.append(
        "enrollNode",
        {
            "name": "shed",
            "role": "agent",
            "url": "http://10.0.0.11:8079",
            "publicKey": "c2hlZC1wdWJsaWMta2V5LWJ5dGVzLTMyLWNoYXJz",
            "enrolledAt": "2026-09-09T12:00:00+00:00",
        },
    )
    machine.append(
        "putComponent",
        {
            "node": "shed",
            "name": "right",
            "kind": "inference-driver",
            "url": "http://10.0.0.11:8081",
        },
    )
    machine.append(
        "putRuntime",
        {
            "node": "shed",
            "spec": {
                "name": "gemma",
                "engine": "llama_cpp",
                "modelPath": "/models/gemma-Q4.gguf",
                "modelAlias": "gemma-9b",
            },
        },
    )
    return client, {"Authorization": f"Bearer {token}"}


def test_the_snapshot_bytes_validate_against_the_contract(tmp_path: Path) -> None:
    client, headers = _active(tmp_path)
    try:
        response = client.get("/v1/control/snapshot", headers=headers)
        assert response.status_code == 200
        document = json.loads(response.content)
        Snapshot.model_validate(document)
    finally:
        client.__exit__(None, None, None)


def test_the_snapshot_carries_no_field_the_contract_does_not_declare(tmp_path: Path) -> None:
    """An extra field would be silently dropped by any consumer that
    validates, and the first symptom would be a standby quietly missing
    state. So the canonical form is asserted to be a subset of the
    schema's fields, not merely compatible with it."""
    client, headers = _active(tmp_path)
    try:
        document = json.loads(client.get("/v1/control/snapshot", headers=headers).content)
        declared = set(Snapshot.model_fields)
        extra = set(document) - declared
        assert not extra, (
            f"the snapshot carries {sorted(extra)}, which control.yaml's Snapshot does not "
            "declare. A consumer that validates would drop it, and a standby would be "
            "missing state nobody notices until a promotion."
        )
    finally:
        client.__exit__(None, None, None)


def test_the_snapshot_can_hold_every_ops_effect(tmp_path: Path) -> None:
    """The rule the contract now states: a snapshot **is** the log's
    compacted head, so every `LogOp` has to have somewhere to land.

    This is the assertion that would have caught `config`,
    `signingKeyId` and the runtime `spec` going missing. Each of those
    broke a promotion months later rather than a request now, which is
    why nothing else caught them.
    """
    client, headers = _active(tmp_path)
    try:
        document = json.loads(client.get("/v1/control/snapshot", headers=headers).content)
        # enrollNode / revokeNode — `gpu-büro` was enrolled and then
        # revoked, so its absence is the revocation's effect and
        # `shed`'s presence is the enrollment's.
        assert [n["name"] for n in document["nodes"]] == ["attic", "shed"]
        # putComponent / deleteComponent
        assert [c["name"] for c in document["components"]] == ["right"]
        # putRuntime / deleteRuntime — and the declaration, not just the
        # placement. A name without a spec is not a declaration.
        assert document["runtimes"] and "spec" in document["runtimes"][0]
        assert document["runtimes"][0]["spec"].get("modelPath")
        # patchConfig
        assert document["config"]["uiTheme"] == "light"
        # rotateSigningKey
        assert document["signingKeyId"] == "2"
        assert document["sealedSigningKey"] == placeholder("rotated")
        # promote
        assert document["epoch"] == 2
    finally:
        client.__exit__(None, None, None)


def test_the_log_page_validates_against_the_contract(tmp_path: Path) -> None:
    client, headers = _active(tmp_path)
    try:
        response = client.get("/v1/control/log", params={"after": 0}, headers=headers)
        assert response.status_code == 200
        LogPage.model_validate(json.loads(response.content))
    finally:
        client.__exit__(None, None, None)


def test_the_openapi_document_still_advertises_both_models(tmp_path: Path) -> None:
    """`response_model=None` must not mean "undocumented".

    Both endpoints declare their model through `responses=` instead, so
    the generated OpenAPI is unchanged for a consumer even though the
    framework is no longer serializing through it.
    """
    app = create_app(settings_for(tmp_path / "active"))
    schema = app.openapi()
    snapshot_ref = schema["paths"]["/v1/control/snapshot"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]
    log_ref = schema["paths"]["/v1/control/log"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]
    assert "Snapshot" in json.dumps(snapshot_ref)
    assert "LogPage" in json.dumps(log_ref)
