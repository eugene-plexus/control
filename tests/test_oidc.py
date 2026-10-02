"""Eugene as an OpenID Connect provider (C2): the root's half.

The whole flow, through an agent and a real Authlib client, is
`specs/scripts/c2-sign-in-acceptance.py`. What is pinned here is what that
run cannot isolate: which headers the root believes and when, the token
kinds the hub refuses, the page escaping what an app is called, the
limiter, and the rules for people and clients.
"""

from __future__ import annotations

import base64
import hashlib
import re
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_control import oidc, tokens

from .conftest import PASSPHRASE, enroll

CALLBACK = "http://127.0.0.1:9/callback"


def _client(c: TestClient, name: str = "App", uris: list[str] | None = None) -> dict[str, Any]:
    made = c.post("/v1/oidc/clients", json={"name": name, "redirectUris": uris or [CALLBACK]})
    assert made.status_code == 201, made.text
    return made.json()


def _challenge(verifier: str) -> str:
    return (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    )


def _authorize(c: TestClient, client_id: str, **extra: str) -> Any:
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": CALLBACK,
        "scope": "openid profile",
        "state": "the-state",
        "nonce": "the-nonce",
        "code_challenge": _challenge("v" * 50),
        "code_challenge_method": "S256",
        **extra,
    }
    return c.get("/oidc/authorize", params=params, follow_redirects=False)


def _request_id(page: str) -> str:
    found = re.search(r'name="request" value="([^"]+)"', page)
    assert found, page
    return found.group(1)


def _sign_in(
    c: TestClient,
    client_id: str,
    password: str,
    name: str | None = None,
    *,
    scope: str = "openid profile",
    **extra: str,
) -> Any:
    page = _authorize(c, client_id, scope=scope)
    data = {"request": _request_id(page.text), "password": password, **extra}
    if name is not None:
        data["name"] = name
    return c.post("/oidc/authorize", data=data, follow_redirects=False)


def _tokens(
    c: TestClient,
    made: dict[str, Any],
    *,
    name: str | None = None,
    password: str = PASSPHRASE,
    scope: str = "openid profile",
) -> dict[str, Any]:
    answer = _sign_in(c, made["client"]["clientId"], password, name, scope=scope)
    assert answer.status_code == 302, answer.text
    code = parse_qs(urlsplit(answer.headers["location"]).query)["code"][0]
    exchanged = c.post(
        "/oidc/token",
        auth=(made["client"]["clientId"], made["clientSecret"]),
        data={
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": "v" * 50,
            "redirect_uri": CALLBACK,
        },
    )
    assert exchanged.status_code == 200, exchanged.text
    return exchanged.json()


# --------------------------------------------------------------------- #
# the pieces
# --------------------------------------------------------------------- #


def test_the_thumbprint_is_rfc_7638s_own_example() -> None:
    jwk = {
        "kty": "RSA",
        "n": "0vx7agoebGcQSuuPiLJXZptN9nndrQmbXEps2aiAFbWhM78LhWx4cbbfAAtVT86zwu1RK7aPFFxuhDR1L6tSoc_BJECPebWKRXjBZCiFV4n3oknjhMstn64tZ_2W-5JsGY4Hc5n9yBXArwl93lqt7_RN5w6Cf0h4QyQ5v-65YGjQR0_FDW2QvzqY368QQMicAtaSqzs8KJZgnYb9c7d0zgdAZHzu6qMQvRL5hajrn1n91CbOpbISD08qNLyrdkt-bFTWhAI4vMQFh6WeZu0fM4lFd2NcRwr3XPksINHaQ-G_xBniIqbw0Ls1jF44-csFCur-kEgU8awapJzKnqDKgw",
        "e": "AQAB",
        "alg": "RS256",
        "kid": "2011-04-29",
    }
    assert oidc.thumbprint(jwk) == "NzbLsXh8uDCcd-6MNwXF4W_7noWXFZAfHkxZsRGC9Xs"


def test_pkce_s256_matches_and_refuses() -> None:
    verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    assert oidc.pkce_matches(verifier, "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM")
    assert not oidc.pkce_matches(verifier, "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cX")
    assert not oidc.pkce_matches("too-short", _challenge("too-short"))


def test_a_new_key_is_rsa_3072_and_publishes_no_private_part() -> None:
    pem, jwk = oidc.new_signing_key()
    assert b"PRIVATE KEY" in pem
    assert jwk["kty"] == "RSA" and jwk["alg"] == "RS256" and "d" not in jwk
    assert len(base64.urlsafe_b64decode(jwk["n"] + "==")) * 8 == oidc.RSA_BITS


# --------------------------------------------------------------------- #
# the issuer, and which headers are believed
# --------------------------------------------------------------------- #


def test_the_issuer_is_the_roots_own_address_when_nothing_says_otherwise(
    active_client: TestClient,
) -> None:
    meta = active_client.get("/oidc/.well-known/openid-configuration").json()
    assert meta["issuer"] == "http://testserver/oidc"
    assert meta["authorization_endpoint"] == "http://testserver/oidc/authorize"


def test_a_forwarded_host_is_believed_only_beside_an_agents_token(
    active_client: TestClient,
) -> None:
    forwarded = {oidc.FORWARDED_HOST_HEADER: "192.168.1.5:8079"}
    ignored = active_client.get("/oidc/.well-known/openid-configuration", headers=forwarded)
    assert ignored.json()["issuer"] == "http://testserver/oidc"

    # The gateway grant lets this node's gateway token reach the root, so
    # the provider's own check is the only thing that refuses it below.
    keys, enrolled = enroll(active_client, "node-a", grants=["gateway"])
    assert enrolled.status_code == 201, enrolled.text
    token, _ = keys.token.mint(
        typ=tokens.TYP_SERVICE,
        sub=tokens.SUB_AGENT,
        aud=[tokens.RECIPIENT_CONTROL],
        ttl_seconds=300,
    )
    believed = active_client.get(
        "/oidc/.well-known/openid-configuration",
        headers={**forwarded, oidc.NODE_TOKEN_HEADER: token},
    )
    assert believed.json()["issuer"] == "http://192.168.1.5:8079/oidc"

    forged, _ = keys.token.mint(
        typ=tokens.TYP_SERVICE, sub="gateway", aud=[tokens.RECIPIENT_CONTROL], ttl_seconds=300
    )
    app = active_client.app
    claims = tokens.verify(
        forged,
        bundle=app.state.trust.view_for(app.state.machine.state),
        recipient=tokens.RECIPIENT_CONTROL,
        classes=(tokens.TYP_SERVICE,),
    )
    assert claims.sub == "gateway"  # a real service token, and not an agent's
    not_an_agent = active_client.get(
        "/oidc/.well-known/openid-configuration",
        headers={**forwarded, oidc.NODE_TOKEN_HEADER: forged},
    )
    assert not_an_agent.json()["issuer"] == "http://testserver/oidc"


def test_a_set_issuer_wins_over_every_address(active_client: TestClient) -> None:
    assert (
        active_client.patch(
            "/v1/config", json={"oidcIssuer": "https://eugene.example/oidc"}
        ).status_code
        == 200
    )
    meta = active_client.get("/oidc/.well-known/openid-configuration").json()
    assert meta["issuer"] == "https://eugene.example/oidc"


# --------------------------------------------------------------------- #
# the page
# --------------------------------------------------------------------- #


def test_the_page_escapes_what_an_app_is_called_and_cannot_be_framed(
    active_client: TestClient,
) -> None:
    made = _client(active_client, name="<script>alert(1)</script>")
    page = _authorize(active_client, made["client"]["clientId"])
    assert page.status_code == 200
    assert "<script>" not in page.text and "&lt;script&gt;" in page.text
    assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
    assert "http://127.0.0.1:9" in page.headers["content-security-policy"]
    assert page.headers["cache-control"] == "no-store"


def test_an_unknown_app_or_return_address_is_never_followed(active_client: TestClient) -> None:
    made = _client(active_client)
    unknown = _authorize(active_client, "c-nobody")
    elsewhere = _authorize(
        active_client, made["client"]["clientId"], redirect_uri="http://evil.example/cb"
    )
    for answer in (unknown, elsewhere):
        assert answer.status_code == 400 and "location" not in answer.headers


@pytest.mark.parametrize(
    "change",
    [
        {"code_challenge_method": "plain"},
        {"code_challenge": "short"},
        {"response_type": "token"},
        {"scope": "profile"},
        {"nonce": ""},
    ],
)
def test_a_bad_request_goes_back_to_the_app_with_an_error(
    active_client: TestClient, change: dict[str, str]
) -> None:
    made = _client(active_client)
    answer = _authorize(active_client, made["client"]["clientId"], **change)
    assert answer.status_code == 302
    query = parse_qs(urlsplit(answer.headers["location"]).query)
    assert query["error"][0] and query["state"] == ["the-state"]


def test_five_failures_a_minute_and_the_page_says_wait(active_client: TestClient) -> None:
    made = _client(active_client)
    for _ in range(oidc.SIGN_IN_FAILURES):
        assert _sign_in(active_client, made["client"]["clientId"], "wrong").status_code == 200
    limited = _sign_in(active_client, made["client"]["clientId"], PASSPHRASE)
    assert limited.status_code == 429 and "Wait" in limited.text


# --------------------------------------------------------------------- #
# tokens
# --------------------------------------------------------------------- #


def test_sign_in_tokens_open_nothing_in_the_hub(active_client: TestClient) -> None:
    made = _client(active_client)
    issued = _tokens(active_client, made)
    for kind in ("id_token", "access_token", "refresh_token"):
        answer = active_client.get("/v1/nodes", headers={"Authorization": f"Bearer {issued[kind]}"})
        assert answer.status_code == 401, kind


def test_an_id_token_is_not_an_access_token_or_a_refresh_token(active_client: TestClient) -> None:
    made = _client(active_client)
    issued = _tokens(active_client, made)
    as_refresh = active_client.post(
        "/oidc/token",
        auth=(made["client"]["clientId"], made["clientSecret"]),
        data={"grant_type": "refresh_token", "refresh_token": issued["id_token"]},
    )
    assert as_refresh.json()["error"] == "invalid_grant"
    as_access = active_client.get(
        "/oidc/userinfo", headers={"Authorization": f"Bearer {issued['id_token']}"}
    )
    assert as_access.status_code == 401


def test_another_apps_secret_cannot_refresh_this_apps_sign_in(active_client: TestClient) -> None:
    mine, theirs = _client(active_client, "Mine"), _client(active_client, "Theirs")
    issued = _tokens(active_client, mine)
    stolen = active_client.post(
        "/oidc/token",
        auth=(theirs["client"]["clientId"], theirs["clientSecret"]),
        data={"grant_type": "refresh_token", "refresh_token": issued["refresh_token"]},
    )
    assert stolen.json()["error"] == "invalid_grant"


def test_the_sign_in_key_is_replicated_sealed(active_client: TestClient) -> None:
    _tokens(active_client, _client(active_client))
    keys = active_client.app.state.machine.state.oidc_keys
    assert len(keys) == 1
    assert "PRIVATE KEY" not in keys[0]["sealedKey"] and "d" not in keys[0]["publicJwk"]


def test_deleting_an_app_ends_its_sign_ins(active_client: TestClient) -> None:
    made = _client(active_client)
    issued = _tokens(active_client, made)
    assert active_client.delete(f"/v1/oidc/clients/{made['client']['clientId']}").status_code == 204
    gone = active_client.post(
        "/oidc/token",
        auth=(made["client"]["clientId"], made["clientSecret"]),
        data={"grant_type": "refresh_token", "refresh_token": issued["refresh_token"]},
    )
    assert gone.status_code == 401  # the app itself no longer authenticates


# --------------------------------------------------------------------- #
# people and clients
# --------------------------------------------------------------------- #


def test_people_need_the_operator(active_client: TestClient) -> None:
    bare = TestClient(active_client.app)
    assert bare.get("/v1/people").status_code == 401
    assert (
        bare.post("/v1/oidc/clients", json={"name": "x", "redirectUris": [CALLBACK]}).status_code
        == 401
    )


@pytest.mark.parametrize(
    "body,status",
    [
        ({"name": "Operator", "password": "a-long-enough-password"}, 409),
        ({"name": "ada@example", "password": "a-long-enough-password"}, 422),
        ({"name": "Ada", "password": "short"}, 422),
    ],
)
def test_a_person_needs_a_name_of_their_own_and_a_real_password(
    active_client: TestClient, body: dict[str, str], status: int
) -> None:
    assert active_client.post("/v1/people", json=body).status_code == status


def test_two_people_cannot_share_a_name_in_any_case(active_client: TestClient) -> None:
    ok = active_client.post(
        "/v1/people", json={"name": "Ada", "password": "a-long-enough-password"}
    )
    assert ok.status_code == 201
    clash = active_client.post(
        "/v1/people", json={"name": "ADA", "password": "a-long-enough-password"}
    )
    assert clash.status_code == 409


def test_a_persons_apps_must_be_apps(active_client: TestClient) -> None:
    made = active_client.post(
        "/v1/people", json={"name": "Ada", "password": "a-long-enough-password", "apps": ["c-nope"]}
    )
    assert made.status_code == 422


@pytest.mark.parametrize("uri", ["/callback", "ftp://x/cb", "http://127.0.0.1:9/cb#frag"])
def test_a_return_address_is_absolute_http_with_no_fragment(
    active_client: TestClient, uri: str
) -> None:
    made = active_client.post("/v1/oidc/clients", json={"name": "x", "redirectUris": [uri]})
    assert made.status_code == 422


def test_deleting_an_app_takes_it_off_every_persons_list(active_client: TestClient) -> None:
    made = _client(active_client)
    client_id = made["client"]["clientId"]
    person = active_client.post(
        "/v1/people",
        json={"name": "Ada", "password": "a-long-enough-password", "apps": [client_id]},
    ).json()
    active_client.delete(f"/v1/oidc/clients/{client_id}")
    listed = active_client.get("/v1/people").json()["people"]
    assert next(p for p in listed if p["id"] == person["id"])["apps"] == []


# --------------------------------------------------------------------- #
# a person makes their password their own (D10)
# --------------------------------------------------------------------- #

FIRST = "the-owners-first-pick"
MINE = "a-password-of-my-own"


def _person_with(c: TestClient, client_id: str) -> None:
    made = c.post("/v1/people", json={"name": "Ada", "password": FIRST, "apps": [client_id]})
    assert made.status_code == 201, made.text


def test_a_person_changes_their_password_on_the_sign_in_page(active_client: TestClient) -> None:
    made = _client(active_client)
    client_id = made["client"]["clientId"]
    _person_with(active_client, client_id)
    earlier = _tokens(active_client, made, name="Ada", password=FIRST)

    page = _authorize(active_client, client_id)
    assert 'name="new_password"' in page.text
    changed = _sign_in(
        active_client, client_id, FIRST, "Ada", new_password=MINE, new_password_again=MINE
    )
    assert changed.status_code == 302, changed.text

    # The sign-in made before the change ends at its next refresh.
    refreshed = active_client.post(
        "/oidc/token",
        auth=(client_id, made["clientSecret"]),
        data={"grant_type": "refresh_token", "refresh_token": earlier["refresh_token"]},
    )
    assert refreshed.json()["error"] == "invalid_grant"
    assert _sign_in(active_client, client_id, FIRST, "Ada").status_code == 200
    assert _sign_in(active_client, client_id, MINE, "Ada").status_code == 302


@pytest.mark.parametrize(
    "again,said",
    [("a-different-password", "not the same"), ("short", "at least 12")],
)
def test_a_change_that_cannot_be_made_says_why_and_changes_nothing(
    active_client: TestClient, again: str, said: str
) -> None:
    made = _client(active_client)
    client_id = made["client"]["clientId"]
    _person_with(active_client, client_id)
    new = again if said == "at least 12" else MINE
    refused = _sign_in(
        active_client, client_id, FIRST, "Ada", new_password=new, new_password_again=again
    )
    assert refused.status_code == 200 and said in refused.text
    assert _sign_in(active_client, client_id, FIRST, "Ada").status_code == 302


def test_the_owners_passphrase_is_not_changed_on_the_sign_in_page(
    active_client: TestClient,
) -> None:
    made = _client(active_client)
    client_id = made["client"]["clientId"]
    _person_with(active_client, client_id)
    refused = _sign_in(
        active_client,
        client_id,
        PASSPHRASE,
        "operator",
        new_password=MINE,
        new_password_again=MINE,
    )
    assert refused.status_code == 400 and "cannot be changed here" in refused.text
    assert _sign_in(active_client, client_id, PASSPHRASE, "operator").status_code == 302


# --------------------------------------------------------------------- #
# the code, its client and its proof
# --------------------------------------------------------------------- #


def _code(
    c: TestClient, client_id: str, password: str = PASSPHRASE, name: str | None = None
) -> str:
    answer = _sign_in(c, client_id, password, name)
    assert answer.status_code == 302, answer.text
    return parse_qs(urlsplit(answer.headers["location"]).query)["code"][0]


def _trade(c: TestClient, made: dict[str, Any], code: str, **change: str) -> Any:
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "code_verifier": "v" * 50,
        "redirect_uri": CALLBACK,
        **change,
    }
    auth = (made["client"]["clientId"], made["clientSecret"])
    return c.post("/oidc/token", auth=auth, data=data)


def test_the_wrong_client_secret_is_refused(active_client: TestClient) -> None:
    made = _client(active_client)
    code = _code(active_client, made["client"]["clientId"])
    wrong = active_client.post(
        "/oidc/token",
        auth=(made["client"]["clientId"], "not-the-secret"),
        data={
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": "v" * 50,
            "redirect_uri": CALLBACK,
        },
    )
    assert wrong.status_code == 401 and wrong.json()["error"] == "invalid_client"


@pytest.mark.parametrize(
    "change",
    [{"code_verifier": "w" * 50}, {"redirect_uri": "http://127.0.0.1:9/elsewhere"}],
)
def test_a_code_needs_its_own_proof_and_return_address(
    active_client: TestClient, change: dict[str, str]
) -> None:
    made = _client(active_client)
    code = _code(active_client, made["client"]["clientId"])
    assert _trade(active_client, made, code, **change).json()["error"] == "invalid_grant"


def test_a_code_works_once(active_client: TestClient) -> None:
    made = _client(active_client)
    code = _code(active_client, made["client"]["clientId"])
    assert _trade(active_client, made, code).status_code == 200
    assert _trade(active_client, made, code).json()["error"] == "invalid_grant"


def test_a_code_is_only_for_the_app_it_was_issued_to(active_client: TestClient) -> None:
    mine, theirs = _client(active_client, "Mine"), _client(active_client, "Theirs")
    code = _code(active_client, mine["client"]["clientId"])
    stolen = active_client.post(
        "/oidc/token",
        auth=(theirs["client"]["clientId"], theirs["clientSecret"]),
        data={
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": "v" * 50,
            "redirect_uri": CALLBACK,
        },
    )
    assert stolen.json()["error"] == "invalid_grant"


# --------------------------------------------------------------------- #
# who may sign in, and for how long
# --------------------------------------------------------------------- #


def test_a_turned_off_person_is_told_so_and_gets_no_code(active_client: TestClient) -> None:
    made = _client(active_client)
    client_id = made["client"]["clientId"]
    person = active_client.post("/v1/people", json={"name": "Ada", "password": FIRST}).json()
    active_client.patch(f"/v1/people/{person['id']}", json={"disabled": True})
    refused = _sign_in(active_client, client_id, FIRST, "Ada")
    assert refused.status_code == 403 and "turned off" in refused.text
    assert "location" not in refused.headers


def test_a_person_is_refused_an_app_they_are_not_given(active_client: TestClient) -> None:
    given, other = _client(active_client, "Given"), _client(active_client, "Other")
    active_client.post(
        "/v1/people",
        json={"name": "Ada", "password": FIRST, "apps": [given["client"]["clientId"]]},
    )
    refused = _sign_in(active_client, other["client"]["clientId"], FIRST, "Ada")
    assert refused.status_code == 403 and "Other" in refused.text
    assert "location" not in refused.headers
    assert _sign_in(active_client, given["client"]["clientId"], FIRST, "Ada").status_code == 302


def test_an_unknown_name_costs_a_password_check(
    active_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from eugene_plexus_control import security

    made = _client(active_client)
    active_client.post("/v1/people", json={"name": "Ada", "password": FIRST})
    checked: list[str] = []
    real = security.verify_passphrase

    def counting(password: str, verifier: str) -> bool:
        checked.append(verifier)
        return real(password, verifier)

    monkeypatch.setattr(security, "verify_passphrase", counting)
    answer = _sign_in(active_client, made["client"]["clientId"], FIRST, "Nobody")
    assert answer.status_code == 200 and len(checked) == 1


def _refresh(c: TestClient, made: dict[str, Any], token: str) -> Any:
    return c.post(
        "/oidc/token",
        auth=(made["client"]["clientId"], made["clientSecret"]),
        data={"grant_type": "refresh_token", "refresh_token": token},
    )


def test_a_turned_off_persons_sign_in_ends_at_its_next_refresh(active_client: TestClient) -> None:
    made = _client(active_client)
    person = active_client.post("/v1/people", json={"name": "Ada", "password": FIRST}).json()
    issued = _tokens(active_client, made, name="Ada", password=FIRST)
    assert _refresh(active_client, made, issued["refresh_token"]).status_code == 200
    active_client.patch(f"/v1/people/{person['id']}", json={"disabled": True})
    assert _refresh(active_client, made, issued["refresh_token"]).json()["error"] == "invalid_grant"
    userinfo = active_client.get(
        "/oidc/userinfo", headers={"Authorization": f"Bearer {issued['access_token']}"}
    )
    assert userinfo.status_code == 401


def test_a_revoked_sign_in_does_not_refresh(active_client: TestClient) -> None:
    made = _client(active_client)
    issued = _tokens(active_client, made)
    revoked = active_client.post(
        "/oidc/revoke",
        auth=(made["client"]["clientId"], made["clientSecret"]),
        data={"token": issued["refresh_token"]},
    )
    assert revoked.status_code == 200
    assert _refresh(active_client, made, issued["refresh_token"]).json()["error"] == "invalid_grant"


def test_the_log_refuses_a_sign_in_key_with_its_private_part(active_client: TestClient) -> None:
    from eugene_plexus_control.applied import OP_PUT_OIDC_KEY, ApplyError

    _, jwk = oidc.new_signing_key()
    machine = active_client.app.state.machine
    with pytest.raises(ApplyError):
        machine.append(
            OP_PUT_OIDC_KEY,
            {
                "key": {
                    "kid": jwk["kid"],
                    "sealedKey": "c2VhbGVk",
                    "publicJwk": {**jwk, "d": "c2VjcmV0"},
                    "createdAt": oidc.utcnow_iso(),
                }
            },
        )
    assert machine.state.oidc_keys == ()


def test_a_short_password_is_refused_on_both_routes(active_client: TestClient) -> None:
    made = active_client.post("/v1/people", json={"name": "Ada", "password": "short"})
    assert made.status_code == 422 and "password" in made.text
    person = active_client.post("/v1/people", json={"name": "Ada", "password": FIRST}).json()
    reset = active_client.put(f"/v1/people/{person['id']}/password", json={"password": "short"})
    assert reset.status_code == 422 and "password" in reset.text


def _contrast(a: str, b: str) -> float:
    def lum(hex_: str) -> float:
        channels = [int(hex_[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_the_page_text_meets_the_contrast_minimum(scheme: str) -> None:
    light, dark = oidc._STYLE.split("@media (prefers-color-scheme: dark)", 1)
    block = light if scheme == "light" else dark

    def token(name: str) -> str:
        found = re.search(rf"--{name}: (#[0-9a-f]{{6}})", block)
        assert found, name
        return found.group(1)

    for fg, bg in (
        ("on-accent", "accent"),
        ("text", "panel"),
        ("muted", "panel"),
        ("accent", "panel"),
        ("error", "panel"),
    ):
        assert _contrast(token(fg), token(bg)) >= 4.5, (scheme, fg, bg)


# --------------------------------------------------------------------- #
# an email for each person (C4)
# --------------------------------------------------------------------- #


def _id_claims(issued: dict[str, Any]) -> dict[str, Any]:
    import jwt

    claims: dict[str, Any] = jwt.decode(issued["id_token"], options={"verify_signature": False})
    return claims


def _userinfo(c: TestClient, issued: dict[str, Any]) -> dict[str, Any]:
    answer = c.get("/oidc/userinfo", headers={"Authorization": f"Bearer {issued['access_token']}"})
    assert answer.status_code == 200, answer.text
    body: dict[str, Any] = answer.json()
    return body


def test_an_email_reaches_an_app_only_with_the_email_scope_and_never_as_verified(
    active_client: TestClient,
) -> None:
    made = _client(active_client)
    active_client.post(
        "/v1/people", json={"name": "Ada", "password": FIRST, "email": "ada@example.org"}
    )
    asked = _tokens(active_client, made, name="Ada", password=FIRST, scope="openid email profile")
    for claims in (_id_claims(asked), _userinfo(active_client, asked)):
        assert claims["email"] == "ada@example.org"
        assert claims["email_verified"] is False, "Eugene sends no mail and proves nothing"
    assert asked["scope"] == "openid profile email"
    plain = _tokens(active_client, made, name="Ada", password=FIRST)
    for claims in (_id_claims(plain), _userinfo(active_client, plain)):
        assert "email" not in claims and "email_verified" not in claims


def test_without_an_email_nothing_stands_in_for_one(active_client: TestClient) -> None:
    made = _client(active_client)
    active_client.post("/v1/people", json={"name": "Bo", "password": FIRST})
    issued = _tokens(active_client, made, name="Bo", password=FIRST, scope="openid email")
    assert "email" not in _id_claims(issued)
    assert "email" not in _userinfo(active_client, issued)


def test_a_scope_eugene_does_not_know_is_ignored_and_discovery_names_email(
    active_client: TestClient,
) -> None:
    made = _client(active_client)
    issued = _tokens(active_client, made, scope="openid groups email")
    assert issued["scope"] == "openid email"
    found = active_client.get("/oidc/.well-known/openid-configuration").json()
    assert found["scopes_supported"] == ["openid", "profile", "email"]
    assert {"email", "email_verified"} <= set(found["claims_supported"])


def test_two_people_cannot_share_an_email_in_any_case(active_client: TestClient) -> None:
    ada = active_client.post(
        "/v1/people", json={"name": "Ada", "password": FIRST, "email": "Ada@Example.org"}
    ).json()
    taken = active_client.post(
        "/v1/people", json={"name": "Bo", "password": FIRST, "email": "ada@example.ORG"}
    )
    assert taken.status_code == 409 and "email" in taken.json()["detail"]["title"]
    bo = active_client.post(
        "/v1/people", json={"name": "Bo", "password": FIRST, "email": "bo@example.org"}
    ).json()
    moved = active_client.patch(f"/v1/people/{bo['id']}", json={"email": "ADA@example.org"})
    assert moved.status_code == 409
    kept = active_client.patch(f"/v1/people/{ada['id']}", json={"email": "ada@example.org"})
    assert kept.status_code == 200, "a person may keep their own address"
    cleared = active_client.patch(f"/v1/people/{ada['id']}", json={"email": None})
    assert cleared.status_code == 200 and "email" not in cleared.json()
    reused = active_client.patch(f"/v1/people/{bo['id']}", json={"email": "ada@example.org"})
    assert reused.status_code == 200 and reused.json()["email"] == "ada@example.org"


def test_an_email_that_is_not_one_is_refused(active_client: TestClient) -> None:
    refused = active_client.post(
        "/v1/people", json={"name": "Cy", "password": FIRST, "email": "not an address"}
    )
    assert refused.status_code == 422


def test_the_log_carries_an_email_and_refuses_a_bad_one() -> None:
    from eugene_plexus_control import applied

    record = {
        "id": "p1",
        "name": "Ada",
        "email": "ada@example.org",
        "passwordVerifier": "$argon2id$v=19$m=65536,t=3,p=4$x$y",
        "apps": None,
        "disabled": False,
        "createdAt": "2026-10-01T00:00:00+00:00",
        "passwordChangedAt": "2026-10-01T00:00:00+00:00",
    }
    assert applied._person_record(dict(record))["email"] == "ada@example.org"
    with pytest.raises(applied.ApplyError, match="email"):
        applied._person_record({**record, "email": 7})
