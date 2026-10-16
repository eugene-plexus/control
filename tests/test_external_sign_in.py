"""Real root state and signatures; only the upstream provider transport is simulated."""

from __future__ import annotations

import re
import time
from urllib.parse import parse_qs, urlsplit

import httpx
import jwt
import pytest

from eugene_plexus_control import external_sign_in as sso
from eugene_plexus_control import oidc
from eugene_plexus_control._http import set_shared_client
from eugene_plexus_control.applied import from_canonical, to_canonical

from .test_oidc import _authorize, _challenge, _client, _request_id

PASSWORD = "a-private-local-password"
UPSTREAM_SECRET = "upstream-only-test-secret"


def test_managed_sign_in_address_is_canonical_and_explicit_override_survives(active_client):
    c = active_client
    c.app.state.settings.agent_public_origin = "https://eugene.example.org:9443"
    expected = "https://eugene.example.org:9443/oidc"
    reply = c.get("/v1/auth/providers").json()
    assert reply["signInAddress"] == reply["managedSignInAddress"] == expected
    assert c.get("/v1/oidc/clients").json()["issuer"] == expected
    assert (
        c.get(
            "/oidc/.well-known/openid-configuration",
            headers={
                "X-Forwarded-Host": "attacker.example",
                "X-Eugene-Plexus-Forwarded-Host": "evil.example",
            },
        ).json()["issuer"]
        == expected
    )
    custom = "https://existing.example.org/oidc"
    assert c.patch("/v1/config", json={"oidcIssuer": custom}).status_code == 200
    c.app.state.settings.agent_public_origin = "https://new.example.org"
    assert c.get("/oidc/.well-known/openid-configuration").json()["issuer"] == custom
    assert c.get("/v1/auth/providers").json()["signInAddress"] == custom
    assert c.patch("/v1/config", json={"oidcIssuer": None}).status_code == 200
    assert c.get("/v1/auth/providers").json()["signInAddress"] == "https://new.example.org/oidc"


def test_request_headers_alone_cannot_configure_external_sign_in(active_client):
    c = active_client
    reply = c.get("/v1/auth/providers", headers={"X-Forwarded-Host": "evil.example"}).json()
    assert reply["signInAddress"] is None and reply["managedSignInAddress"] is None


def test_managed_address_allows_provider_setup_without_saving_an_override(provider):
    c, configured, _, _ = provider
    c.app.state.settings.agent_public_origin = "https://eugene.example.org"
    assert c.patch("/v1/config", json={"oidcIssuer": None}).status_code == 200
    body = {
        k: configured[k]
        for k in ("name", "kind", "issuer", "clientId", "tokenAuthMethod", "enabled")
    }
    response = c.put(f"/v1/auth/providers/{configured['id']}", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["callbackUrl"] == "https://eugene.example.org/oidc/external/callback"


@pytest.fixture
def provider(active_client):
    c = active_client
    c.base_url = httpx.URL("http://127.0.0.1")
    assert c.patch("/v1/config", json={"oidcIssuer": "http://127.0.0.1/oidc"}).status_code == 200
    pem, key = oidc.new_signing_key()
    context = {"claims": {}, "authorization": {}, "token_calls": 0, "bad_signature": False}

    def respond(request):
        assert not request.headers.get("cookie"), "upstream cookies must not cross visitors"
        if request.url.path.endswith("openid-configuration"):
            return httpx.Response(
                200,
                json={
                    "issuer": "https://idp.example",
                    "authorization_endpoint": "https://idp.example/authorize",
                    "token_endpoint": "https://idp.example/token",
                    "jwks_uri": "https://idp.example/keys",
                    "token_endpoint_auth_methods_supported": [
                        "client_secret_post",
                        "client_secret_basic",
                    ],
                    "id_token_signing_alg_values_supported": ["RS256"],
                },
                headers={"set-cookie": "upstream-private=never-forward-this"},
            )
        if request.url.path == "/keys":
            return httpx.Response(200, json={"keys": [key]})
        assert request.url.path == "/token"
        context["token_calls"] += 1
        form = parse_qs(request.content.decode())
        assert form["client_secret"] == [UPSTREAM_SECRET]
        assert _challenge(form["code_verifier"][0]) == context["authorization"]["code_challenge"][0]
        claims = {
            "iss": "https://idp.example",
            "sub": "external-ada",
            "aud": "eugene-test",
            "exp": int(time.time()) + 300,
            "iat": int(time.time()),
            "nonce": context["authorization"]["nonce"][0],
            "email": "ada@example.org",
            "email_verified": True,
            **context["claims"],
        }
        token = jwt.encode(claims, pem, algorithm="RS256", headers={"kid": key["kid"]})
        if context["bad_signature"]:
            token = token[:-16] + "A" * 16
        return httpx.Response(200, json={"id_token": token})

    transport = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    set_shared_client("external-sign-in", transport)
    made = c.post(
        "/v1/auth/providers",
        json={
            "name": "Example SSO",
            "kind": "oidc",
            "issuer": "https://idp.example",
            "clientId": "eugene-test",
            "clientSecret": UPSTREAM_SECRET,
            "tokenAuthMethod": "client_secret_post",
            "enabled": True,
        },
    )
    assert made.status_code == 200, made.text
    person = c.post(
        "/v1/people", json={"name": "Ada", "email": "ada@example.org", "password": PASSWORD}
    ).json()
    yield c, made.json(), person, context
    set_shared_client("external-sign-in", None)


def hidden(text, name):
    match = re.search(rf"name=['\"]{name}['\"] value=['\"]([^'\"]+)['\"]", text)
    assert match, text
    return match[1]


def invite_start(fixture):
    c, provider, person, context = fixture
    invitation = c.post(
        f"/v1/people/{person['id']}/sign-in/invitations", json={"providerId": provider["id"]}
    )
    assert invitation.status_code == 200, invitation.text
    page = c.get("/oidc/invitation")
    confirmed = c.post(
        "/oidc/invitation",
        data={"binding": hidden(page.text, "binding"), "code": invitation.json()["code"]},
    )
    assert confirmed.status_code == 200, confirmed.text
    response = c.post(
        "/oidc/external/start",
        data={name: hidden(confirmed.text, name) for name in ("invitation", "provider", "binding")},
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text
    context["authorization"] = parse_qs(urlsplit(response.headers["location"]).query)
    return invitation.json(), context["authorization"]["state"][0]


def finish(fixture, state):
    return fixture[0].get(
        "/oidc/external/callback",
        params={"state": state, "code": "upstream-code"},
        follow_redirects=False,
    )


def app_tokens(fixture):
    c, _, _, _ = fixture
    client = _client(c)
    page = _authorize(c, client["client"]["clientId"])
    assert "Continue as Ada" in page.text
    complete = c.post(
        "/oidc/authorize",
        data={"request": _request_id(page.text), "use_session": "yes"},
        follow_redirects=False,
    )
    assert complete.status_code == 302, complete.text
    code = parse_qs(urlsplit(complete.headers["location"]).query)["code"][0]
    answer = c.post(
        "/oidc/token",
        auth=(client["client"]["clientId"], client["clientSecret"]),
        data={
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": "v" * 50,
            "redirect_uri": "http://127.0.0.1:9/callback",
        },
    )
    assert answer.status_code == 200, answer.text
    return client, answer.json()


def test_invitation_links_stable_subject_and_shared_login_without_email_merging(provider):
    c, _config, person, context = provider
    invitation, state = invite_start(provider)
    result = finish(provider, state)
    assert result.status_code == 303, result.text
    assert result.headers["location"].endswith("/oidc/account")
    assert (
        "HttpOnly" in result.headers["set-cookie"]
        and "SameSite=lax" in result.headers["set-cookie"]
    )
    assert "Your sign-in and sessions" in c.get("/oidc/account").text
    client, tokens = app_tokens(provider)
    claims = jwt.decode(tokens["id_token"], options={"verify_signature": False})
    assert claims["sub"] == person["id"]
    assert claims["sub"] != "external-ada"
    records = c.get(f"/v1/people/{person['id']}/sign-in").json()
    assert records["identities"][0]["subject"] == "external-ada"
    assert len(records["sessions"]) == 1 and not records["invitations"]
    assert finish(provider, state).status_code == 400
    assert context["token_calls"] == 1
    # State round-trips exactly; neither the provider secret nor cookie capability is stored raw.
    snapshot = to_canonical(c.app.state.machine.state)
    assert to_canonical(from_canonical(snapshot)) == snapshot
    assert UPSTREAM_SECRET not in str(snapshot)
    assert invitation["code"] not in str(snapshot)
    assert c.cookies.get("ep-signin") not in str(snapshot)
    assert "sealedSecret" not in c.get("/v1/auth/providers").text
    # Revocation reaches already-issued app credentials, not just the browser cookie.
    assert (
        c.delete(
            f"/v1/people/{person['id']}/sign-in/sessions/{records['sessions'][0]['id']}"
        ).status_code
        == 204
    )
    refused = c.post(
        "/oidc/token",
        auth=(client["client"]["clientId"], client["clientSecret"]),
        data={"grant_type": "refresh_token", "refresh_token": tokens["refresh_token"]},
    )
    assert refused.status_code == 400


@pytest.mark.parametrize(
    "claims",
    [
        {"nonce": "wrong"},
        {"iss": "https://attacker.example"},
        {"aud": "another-app"},
        {"azp": "another-app"},
        {"exp": 1},
        {"sub": ""},
    ],
)
def test_invalid_identity_proofs_never_consume_invitation_or_create_session(provider, claims):
    c, _, person, context = provider
    _, state = invite_start(provider)
    context["claims"] = claims
    assert finish(provider, state).status_code == 400
    result = c.get(f"/v1/people/{person['id']}/sign-in").json()
    assert not result["identities"] and not result["sessions"] and len(result["invitations"]) == 1


def test_signature_browser_binding_and_replay_are_checked(provider):
    c, _, person, context = provider
    _, state = invite_start(provider)
    context["bad_signature"] = True
    assert finish(provider, state).status_code == 400
    _, state = invite_start(provider)
    context["bad_signature"] = False
    c.cookies.clear()
    assert finish(provider, state).status_code == 400
    assert context["token_calls"] == 1
    assert not c.get(f"/v1/people/{person['id']}/sign-in").json()["identities"]


def test_uninvited_matching_email_cannot_take_over_existing_account(provider):
    c, config, _, context = provider
    client = _client(c)
    page = _authorize(c, client["client"]["clientId"])
    response = c.post(
        "/oidc/external/start",
        data={"request": _request_id(page.text), "provider": config["id"]},
        follow_redirects=False,
    )
    context["authorization"] = parse_qs(urlsplit(response.headers["location"]).query)
    answer = finish(provider, context["authorization"]["state"][0])
    assert answer.status_code == 400 and "Matching email" in answer.text
    assert len(c.get("/v1/people").json()["people"]) == 1


def test_provider_disable_ends_sessions(provider):
    c, config, person, _ = provider
    _, state = invite_start(provider)
    assert finish(provider, state).status_code == 303
    session_id = c.get(f"/v1/people/{person['id']}/sign-in").json()["sessions"][0]["id"]
    assert sso.valid_session(c.app.state.machine.state, session_id)
    body = {
        k: config[k] for k in ("name", "kind", "issuer", "clientId", "tokenAuthMethod", "enabled")
    }
    body["enabled"] = False
    assert c.put(f"/v1/auth/providers/{config['id']}", json=body).status_code == 200
    assert not sso.valid_session(c.app.state.machine.state, session_id)


def test_invitation_hash_is_not_a_usable_code_and_owner_only_controls_are_enforced(provider):
    c, config, person, _ = provider
    invitation = c.post(
        f"/v1/people/{person['id']}/sign-in/invitations", json={"providerId": config["id"]}
    ).json()
    page = c.get("/oidc/invitation")
    response = c.post(
        "/oidc/external/start",
        data={
            "invitation": sso.digest(invitation["code"]),
            "provider": config["id"],
            "binding": hidden(page.text, "binding"),
        },
        follow_redirects=False,
    )
    assert response.status_code == 400
    c.headers.pop("authorization")
    assert c.get("/v1/auth/providers").status_code == 401
    assert c.get(f"/v1/people/{person['id']}/sign-in").status_code == 401
    assert (
        c.post(
            f"/v1/people/{person['id']}/sign-in/invitations", json={"providerId": config["id"]}
        ).status_code
        == 401
    )


@pytest.mark.parametrize(
    "action", ["unlink", "password", "disabled", "deleted", "provider-deleted"]
)
def test_account_and_connection_changes_end_existing_app_access(provider, action):
    c, config, person, _ = provider
    _, state = invite_start(provider)
    assert finish(provider, state).status_code == 303
    client, tokens = app_tokens(provider)
    records = c.get(f"/v1/people/{person['id']}/sign-in").json()
    path = f"/v1/people/{person['id']}"
    if action == "unlink":
        result = c.delete(path + "/sign-in/identities/" + records["identities"][0]["id"])
    elif action == "password":
        result = c.put(path + "/password", json={"password": PASSWORD + "-changed"})
    elif action == "disabled":
        result = c.patch(path, json={"disabled": True})
    elif action == "deleted":
        result = c.delete(path)
    else:
        result = c.delete("/v1/auth/providers/" + config["id"])
    assert result.is_success, result.text
    assert not sso.valid_session(c.app.state.machine.state, records["sessions"][0]["id"])
    result = c.get("/oidc/userinfo", headers={"Authorization": "Bearer " + tokens["access_token"]})
    assert result.status_code == 401
    refreshed = c.post(
        "/oidc/token",
        auth=(client["client"]["clientId"], client["clientSecret"]),
        data={"grant_type": "refresh_token", "refresh_token": tokens["refresh_token"]},
    )
    assert refreshed.status_code == 400


def test_shared_administration_preserves_named_authority_and_session_revocation(provider):
    c, _, person, _ = provider
    assert (
        c.put(f"/v1/people/{person['id']}/administration", json={"managePeople": True}).status_code
        == 200
    )
    _, state = invite_start(provider)
    assert finish(provider, state).status_code == 303
    page = c.get(
        "/oidc/authorize",
        params={
            "client_id": sso.ADMIN_CLIENT,
            "response_type": "code",
            "scope": "openid profile",
            "redirect_uri": "http://127.0.0.1/administration/",
            "state": "client-state",
            "nonce": "client-nonce",
            "code_challenge": _challenge("v" * 50),
            "code_challenge_method": "S256",
        },
    )
    assert page.status_code == 200, page.text
    response = c.post(
        "/oidc/authorize",
        data={"request": _request_id(page.text), "use_session": "yes"},
        follow_redirects=False,
    )
    assert response.status_code == 302, response.text
    code = parse_qs(urlsplit(response.headers["location"]).query)["code"][0]
    body = {"code": code, "verifier": "v" * 50, "redirectUri": "http://127.0.0.1/administration/"}
    exchanged = c.post("/v1/auth/administration/code", json=body)
    assert exchanged.status_code == 200, exchanged.text
    auth = {"Authorization": "Bearer " + exchanged.json()["sessionToken"]}
    assert c.get("/v1/people", headers=auth).status_code == 200
    assert c.get("/v1/config", headers=auth).status_code == 401
    assert c.get("/v1/auth/providers", headers=auth).status_code == 401
    assert c.post("/v1/auth/administration/code", json=body).status_code == 401
    records = c.get(f"/v1/people/{person['id']}/sign-in").json()
    c.delete(f"/v1/people/{person['id']}/sign-in/sessions/{records['sessions'][0]['id']}")
    assert c.get("/v1/people", headers=auth).status_code == 401


def test_invitations_expire_and_provider_edits_revoke_pending_links(provider):
    c, config, person, _ = provider
    _, state = invite_start(provider)
    body = {
        k: config[k] for k in ("name", "kind", "issuer", "clientId", "tokenAuthMethod", "enabled")
    }
    body["name"] = "Renamed"
    assert c.put("/v1/auth/providers/" + config["id"], json=body).status_code == 200
    assert finish(provider, state).status_code == 400
    assert not c.get(f"/v1/people/{person['id']}/sign-in").json()["invitations"]
    _, state = invite_start(provider)
    c.app.state.external_flows.pending[state].expires = 1
    assert finish(provider, state).status_code == 400


def test_account_actions_require_browser_csrf_and_only_end_own_sessions(provider):
    c, _, person, _ = provider
    _, state = invite_start(provider)
    assert finish(provider, state).status_code == 303
    page = c.get("/oidc/account")
    csrf = hidden(page.text, "csrf")
    assert c.post("/oidc/account", data={"csrf": "wrong", "session": "all"}).status_code == 400
    assert len(c.get(f"/v1/people/{person['id']}/sign-in").json()["sessions"]) == 1
    assert c.post("/oidc/account", data={"csrf": csrf, "session": "all"}).status_code == 200
    assert not c.get(f"/v1/people/{person['id']}/sign-in").json()["sessions"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,bad_key_issuer,bad_tenant",
    [
        ("google", False, False),
        ("microsoft", False, False),
        ("microsoft", True, False),
        ("microsoft", False, True),
    ],
)
async def test_google_and_microsoft_signed_identity_profiles(
    monkeypatch, kind, bad_key_issuer, bad_tenant
):
    from eugene_plexus_control import security

    tenant = "9188040d-6c67-4c5b-b112-36a304b66dad"
    issuer = sso.GOOGLE if kind == "google" else f"https://login.microsoftonline.com/{tenant}/v2.0"
    master = b"x" * 32
    provider = {
        "kind": kind,
        "issuer": issuer,
        "clientId": "client",
        "tokenAuthMethod": "client_secret_basic",
        "sealedSecret": security.seal_b64(b"secret", master),
    }
    sso.validate_provider(provider)
    pem, key = oidc.new_signing_key()
    if kind == "microsoft":
        key["issuer"] = (
            "https://attacker.example"
            if bad_key_issuer
            else "https://login.microsoftonline.com/{tenantid}/v2.0"
        )
    token = jwt.encode(
        {
            "iss": "accounts.google.com" if kind == "google" else issuer,
            "sub": "external-subject",
            "aud": "client",
            "iat": int(time.time()),
            "exp": int(time.time()) + 300,
            "nonce": "expected",
            "tid": "wrong" if bad_tenant else tenant,
        },
        pem,
        algorithm="RS256",
        headers={"kid": key["kid"]},
    )

    async def document(url, **kwargs):
        return {"id_token": token} if url.endswith("token") else {"keys": [key]}

    monkeypatch.setattr(sso, "document", document)
    flow = sso.Flow(
        "id",
        "revision",
        "binding",
        "verifier",
        "expected",
        "https://local/oidc/external/callback",
        {"token_endpoint": issuer + "/token", "jwks_uri": issuer + "/keys"},
        None,
        None,
        time.time() + 600,
    )
    if bad_key_issuer or bad_tenant:
        with pytest.raises(sso.Refused, match="verification"):
            await sso.finish(provider, flow, "code", master)
    else:
        assert await sso.finish(provider, flow, "code", master) == {
            "issuer": issuer,
            "subject": "external-subject",
        }


@pytest.mark.asyncio
async def test_generic_provider_preserves_its_exact_issuer_including_trailing_slash(
    provider, monkeypatch
):
    c, config, _, _ = provider
    original = sso.metadata

    async def document(url, **kwargs):
        assert url == "https://generic.example/.well-known/openid-configuration"
        return {
            "issuer": "https://generic.example/",
            "authorization_endpoint": "https://generic.example/auth",
            "token_endpoint": "https://generic.example/token",
            "jwks_uri": "https://generic.example/keys",
            "token_endpoint_auth_methods_supported": ["client_secret_post"],
        }

    monkeypatch.setattr(sso, "document", document)
    body = {
        k: config[k] for k in ("name", "kind", "issuer", "clientId", "tokenAuthMethod", "enabled")
    }
    body.update(issuer="https://generic.example/", clientSecret=UPSTREAM_SECRET)
    assert (await original(body))["issuer"] == body["issuer"]
    response = c.post("/v1/auth/providers", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["issuer"] == "https://generic.example/"


def test_local_sign_in_reports_capacity_without_server_error(provider, monkeypatch):
    c, _, _, _ = provider
    client = _client(c)
    page = _authorize(c, client["client"]["clientId"])

    def full(*args, **kwargs):
        raise sso.Refused("The shared sign-in limit was reached.")

    monkeypatch.setattr(sso, "new_session", full)
    result = c.post(
        "/oidc/authorize",
        data={"request": _request_id(page.text), "name": "Ada", "password": PASSWORD},
    )
    assert result.status_code == 429
    assert "shared sign-in limit" in result.text
