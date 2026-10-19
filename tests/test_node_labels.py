"""Node labels (docs/design/node-labels.md): a name people see, over a
node's fixed name.

The one failure this slice can create is a label leaking into a key, so
the assertions that matter are that `name` never moves and that two
machines can never be shown under one name.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_control.applied import OP_LABEL_NODE, from_canonical, to_canonical

from .conftest import enroll, mint_join_token, node_keys


@pytest.fixture
def install(active_client: TestClient) -> TestClient:
    for name in ("468e3ed662bf", "Amish_Station"):
        _, joined = enroll(active_client, name)
        assert joined.status_code == 201, joined.text
    return active_client


def _nodes(client: TestClient) -> dict[str, dict[str, Any]]:
    return {n["name"]: n for n in client.get("/v1/nodes").json()["nodes"]}


def test_a_label_is_shown_and_the_name_does_not_move(install: TestClient) -> None:
    answer = install.put("/v1/nodes/468e3ed662bf/label", json={"label": "  The NAS  "})
    assert answer.status_code == 200, answer.text
    assert (answer.json()["name"], answer.json()["label"]) == ("468e3ed662bf", "The NAS")
    nodes = _nodes(install)
    assert nodes["468e3ed662bf"]["label"] == "The NAS"
    assert nodes["Amish_Station"]["label"] is None
    assert "The NAS" not in nodes, "a label is never a key"
    assert install.get("/v1/nodes/468e3ed662bf").json()["label"] == "The NAS"


def test_clearing_and_repeating_append_only_what_changes(install: TestClient) -> None:
    machine = install.app.state.machine  # type: ignore[attr-defined]
    assert (
        install.put("/v1/nodes/Amish_Station/label", json={"label": "Demo Rig"}).status_code == 200
    )
    index = machine.state.index
    assert (
        install.put("/v1/nodes/Amish_Station/label", json={"label": "Demo Rig"}).status_code == 200
    )
    assert machine.state.index == index, "the same label again appends nothing"

    cleared = install.delete("/v1/nodes/Amish_Station/label")
    assert cleared.status_code == 200 and cleared.json()["label"] is None
    assert machine.state.index == index + 1
    assert install.delete("/v1/nodes/Amish_Station/label").status_code == 200
    assert machine.state.index == index + 1, "clearing no label appends nothing"

    own = install.put("/v1/nodes/Amish_Station/label", json={"label": "Amish_Station"})
    assert own.status_code == 200 and own.json()["label"] is None, "its own name is no label"
    assert machine.state.index == index + 1


@pytest.mark.parametrize(
    "label",
    ["amish_station", "AMISH_STATION", "the nas"],
    ids=["another's name", "another's name, cased", "another's label"],
)
def test_two_machines_are_never_shown_under_one_name(install: TestClient, label: str) -> None:
    assert install.put("/v1/nodes/468e3ed662bf/label", json={"label": "The NAS"}).status_code == 200
    target = "Amish_Station" if label == "the nas" else "468e3ed662bf"
    answer = install.put(f"/v1/nodes/{target}/label", json={"label": label})
    assert answer.status_code == 409, answer.text
    assert "already called" in answer.text


@pytest.mark.parametrize("label", ["   ", "Tab\there", "x" * 65])
def test_a_label_that_cannot_be_shown_is_refused(install: TestClient, label: str) -> None:
    machine = install.app.state.machine  # type: ignore[attr-defined]
    index = machine.state.index
    answer = install.put("/v1/nodes/Amish_Station/label", json={"label": label})
    assert answer.status_code in (400, 422), answer.text
    assert machine.state.index == index


def test_labelling_takes_a_session_and_a_known_node(install: TestClient) -> None:
    bare = TestClient(install.app)
    assert bare.put("/v1/nodes/Amish_Station/label", json={"label": "Rig"}).status_code == 401
    assert bare.delete("/v1/nodes/Amish_Station/label").status_code == 401
    assert install.put("/v1/nodes/nobody/label", json={"label": "Rig"}).status_code == 404


def test_enrollment_can_name_the_machine_and_is_refused_before_the_token_is_spent(
    install: TestClient,
) -> None:
    keys, joined = enroll(install, "DESKTOP-4KJ2P9", label="Studio PC")
    assert joined.status_code == 201, joined.text
    assert _nodes(install)["DESKTOP-4KJ2P9"]["label"] == "Studio PC"

    token = mint_join_token(install)
    clash = install.post("/v1/nodes/enroll", json=node_keys("spare").body(token, label="studio pc"))
    assert clash.status_code == 409, clash.text
    retried = install.post("/v1/nodes/enroll", json=node_keys("spare").body(token, label="Spare"))
    assert retried.status_code == 201, "the refused enrollment did not spend the token"
    assert keys.name == "DESKTOP-4KJ2P9"


def test_enrolling_again_keeps_the_label_unless_it_sets_one(install: TestClient) -> None:
    assert (
        install.put("/v1/nodes/Amish_Station/label", json={"label": "Demo Rig"}).status_code == 200
    )
    _, again = enroll(install, "Amish_Station")
    assert again.status_code == 201
    assert _nodes(install)["Amish_Station"]["label"] == "Demo Rig", "a rebuild keeps its label"
    _, renamed = enroll(install, "Amish_Station", label="Workstation")
    assert renamed.status_code == 201
    assert _nodes(install)["Amish_Station"]["label"] == "Workstation"


def test_a_label_survives_the_snapshot(install: TestClient) -> None:
    machine = install.app.state.machine  # type: ignore[attr-defined]
    assert install.put("/v1/nodes/468e3ed662bf/label", json={"label": "The NAS"}).status_code == 200
    restored = from_canonical(to_canonical(machine.state))
    assert restored.nodes["468e3ed662bf"].label == "The NAS"
    assert to_canonical(restored) == to_canonical(machine.state)
    assert any(e["op"] == OP_LABEL_NODE for e in machine.read_page(0, 1000))
