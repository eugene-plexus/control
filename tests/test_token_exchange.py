"""RFC 8693 token exchange: how the console reaches another machine.

The console agent presents its own `agent` service token as the actor
and the operator's session as the subject, and gets back a 5-minute
token addressed to one other node. What this file pins is every way
that exchange must refuse, because the exchange is the one place a
session's authority crosses machines (design D7).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from eugene_plexus_control import tokens

from .conftest import PASSPHRASE, NodeKeys, enroll


def _session_through(client: TestClient, keys: NodeKeys) -> str:
    response = client.post(
        "/v1/auth/login",
        json={"passphrase": PASSPHRASE},
        headers={"Authorization": f"Bearer {keys.service_token()}"},
    )
    assert response.status_code == 200, response.text
    return str(response.json()["sessionToken"])


def _exchange(client: TestClient, actor: str, subject: str, audience: str) -> object:
    return client.post(
        "/v1/auth/token",
        json={"subjectToken": subject, "audience": audience},
        headers={"Authorization": f"Bearer {actor}"},
    )


def _view(client: TestClient) -> tokens.TrustBundle:
    machine = client.app.state.machine  # type: ignore[attr-defined]
    return client.app.state.trust.view_for(machine.state)  # type: ignore[attr-defined, no-any-return]


def test_a_console_exchanges_its_session_for_a_token_addressed_to_one_other_node(
    active_client: TestClient,
) -> None:
    console, _ = enroll(active_client, "laptop")
    enroll(active_client, "gpu-box")
    session = _session_through(active_client, console)

    response = _exchange(active_client, console.service_token(), session, "node:gpu-box")
    assert response.status_code == 200, response.text  # type: ignore[attr-defined]
    exchanged = response.json()["accessToken"]  # type: ignore[attr-defined]

    claims = tokens.verify(
        exchanged,
        bundle=_view(active_client),
        recipient="node:gpu-box",
        classes=[tokens.TYP_SESSION],
    )
    assert claims.aud == ("node:gpu-box",)
    assert claims.act == "node:laptop"
    session_jti = tokens.verify(
        session, bundle=_view(active_client), recipient="control", classes=[tokens.TYP_SESSION]
    ).jti
    assert claims.sid == session_jti
    assert claims.exp - claims.iat <= tokens.EXCHANGED_TTL_SECONDS
    # Worthless anywhere else, including at this root.
    for recipient in ("node:laptop", "control"):
        try:
            tokens.verify(
                exchanged,
                bundle=_view(active_client),
                recipient=recipient,
                classes=[tokens.TYP_SESSION],
            )
        except tokens.TokenError:
            continue
        raise AssertionError(f"an exchanged token verified at {recipient}")


def test_a_machine_cannot_exchange_a_session_it_was_not_issued(active_client: TestClient) -> None:
    """A compromised worker that captured some other console's session
    still cannot turn it into access elsewhere: the session names its
    console, and the actor has to be that console."""
    console, _ = enroll(active_client, "laptop")
    thief, _ = enroll(active_client, "gpu-box")
    enroll(active_client, "attic")
    session = _session_through(active_client, console)

    stolen = _exchange(active_client, thief.service_token(), session, "node:attic")
    assert stolen.status_code == 401  # type: ignore[attr-defined]
    assert "cannot exchange" in stolen.text  # type: ignore[attr-defined]


def test_a_direct_login_session_names_no_console_and_cannot_be_exchanged(
    active_client: TestClient,
) -> None:
    console, _ = enroll(active_client, "laptop")
    enroll(active_client, "gpu-box")
    direct = active_client.headers["Authorization"].removeprefix("Bearer ")
    response = _exchange(active_client, console.service_token(), direct, "node:gpu-box")
    assert response.status_code == 401  # type: ignore[attr-defined]


def test_an_exchanged_token_cannot_be_exchanged_again(active_client: TestClient) -> None:
    """It is not addressed to this root, so it is not a subject here."""
    console, _ = enroll(active_client, "laptop")
    target, _ = enroll(active_client, "gpu-box")
    enroll(active_client, "attic")
    session = _session_through(active_client, console)
    first = _exchange(active_client, console.service_token(), session, "node:gpu-box")
    exchanged = first.json()["accessToken"]  # type: ignore[attr-defined]
    again = _exchange(active_client, target.service_token(), exchanged, "node:attic")
    assert again.status_code == 401  # type: ignore[attr-defined]


def test_only_an_agent_speaking_for_itself_may_ask(active_client: TestClient) -> None:
    console, _ = enroll(active_client, "laptop", grants=["gateway"])
    enroll(active_client, "gpu-box")
    session = _session_through(active_client, console)
    for actor in (
        console.service_token(sub="gateway"),
        console.service_token(sub="inference-driver", aud=("node:laptop",)),
        session,
        "nonsense",
    ):
        response = _exchange(active_client, actor, session, "node:gpu-box")
        assert response.status_code == 401, actor[:20]  # type: ignore[attr-defined]


def test_an_audience_that_is_not_an_enrolled_node_is_refused(active_client: TestClient) -> None:
    console, _ = enroll(active_client, "laptop")
    session = _session_through(active_client, console)
    for audience in ("node:nowhere", "control"):
        response = _exchange(active_client, console.service_token(), session, audience)
        assert response.status_code in (400, 422), audience  # type: ignore[attr-defined]


def test_a_signed_out_session_cannot_be_exchanged_and_its_exchanges_die(
    active_client: TestClient,
) -> None:
    console, _ = enroll(active_client, "laptop")
    enroll(active_client, "gpu-box")
    session = _session_through(active_client, console)
    exchanged = _exchange(active_client, console.service_token(), session, "node:gpu-box").json()[  # type: ignore[attr-defined]
        "accessToken"
    ]

    signed_out = active_client.delete(
        "/v1/auth/sessions/current", headers={"Authorization": f"Bearer {session}"}
    )
    assert signed_out.status_code == 204
    again = _exchange(active_client, console.service_token(), session, "node:gpu-box")
    assert again.status_code == 401  # type: ignore[attr-defined]
    published = active_client.app.state.trust.current  # type: ignore[attr-defined]
    try:
        tokens.verify(
            exchanged, bundle=published, recipient="node:gpu-box", classes=[tokens.TYP_SESSION]
        )
    except tokens.TokenError as exc:
        assert "signed out" in str(exc)
    else:  # pragma: no cover - the assertion this test exists for
        raise AssertionError("a token exchanged from a signed-out session still verified")


def test_revoking_the_console_ends_its_sessions_everywhere(active_client: TestClient) -> None:
    """D9: a console machine that is revoked takes every session bound to
    it along, at this root and at every node that sees the new bundle."""
    console, _ = enroll(active_client, "laptop")
    enroll(active_client, "gpu-box")
    session = _session_through(active_client, console)
    exchanged = _exchange(active_client, console.service_token(), session, "node:gpu-box").json()[  # type: ignore[attr-defined]
        "accessToken"
    ]

    assert active_client.delete("/v1/nodes/laptop").status_code == 202
    assert (
        active_client.get("/v1/nodes", headers={"Authorization": f"Bearer {session}"}).status_code
        == 401
    )
    published = active_client.app.state.trust.current  # type: ignore[attr-defined]
    try:
        tokens.verify(
            exchanged, bundle=published, recipient="node:gpu-box", classes=[tokens.TYP_SESSION]
        )
    except tokens.TokenError as exc:
        assert "no longer in the install" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a token exchanged by a revoked console still verified")
