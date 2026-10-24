"""`POST /v1/people/known` (row 53): which people this root still has, for a node.

The install's store deletes what a removed person kept, and learns who was
removed through its own machine's agent, which asks here with its node
token. What is pinned is what a wrong answer would cost: a deleted person is
`unknown`, one turned off and the owner are `known`, nothing is said while the
root is locked, and only a member node's own agent token is answered.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from eugene_plexus_control.applied import OPERATOR_NAME

from .conftest import enroll

PASSWORD = "a-long-enough-password"


def _person(c: TestClient, name: str) -> str:
    made = c.post("/v1/people", json={"name": name, "password": PASSWORD})
    assert made.status_code == 201, made.text
    return str(made.json()["id"])


def _ask(c: TestClient, token: str | None, subjects: list[str]) -> object:
    bare = TestClient(c.app)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return bare.post("/v1/people/known", json={"subjects": subjects}, headers=headers)


def test_a_removed_person_is_unknown_to_a_nodes_agent(active_client: TestClient) -> None:
    keys, enrolled = enroll(active_client, "nas")
    assert enrolled.status_code in (200, 201), enrolled.text
    ann, ben = _person(active_client, "Ann"), _person(active_client, "Ben")
    off = _person(active_client, "Cid")
    assert active_client.patch(f"/v1/people/{off}", json={"disabled": True}).status_code == 200
    assert active_client.delete(f"/v1/people/{ann}").status_code == 204
    answer = _ask(active_client, keys.service_token(), [ann, ben, off, OPERATOR_NAME, ann])
    assert answer.status_code == 200, answer.text  # type: ignore[attr-defined]
    assert answer.json() == {  # type: ignore[attr-defined]
        "known": [ben, off, OPERATOR_NAME],
        "unknown": [ann],
    }


def test_only_a_member_nodes_agent_is_answered(active_client: TestClient) -> None:
    keys, _ = enroll(active_client, "nas")
    ben = _person(active_client, "Ben")
    for token in (
        None,
        keys.service_token(sub="store"),
        keys.service_token(sub="gateway"),
        active_client.headers["Authorization"].split()[1],
    ):
        refused = _ask(active_client, token, [ben])
        assert refused.status_code == 401, refused.text  # type: ignore[attr-defined]


def test_at_most_five_hundred_ids_at_once(active_client: TestClient) -> None:
    keys, _ = enroll(active_client, "nas")
    refused = _ask(active_client, keys.service_token(), [f"id{i}" for i in range(501)])
    assert refused.status_code == 422  # type: ignore[attr-defined]
