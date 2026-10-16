"""OIDC relying-party validation and browser-session primitives.

An upstream identity is always (issuer, subject), never an email address.
Only an owner-issued, single-use invitation can connect it to a Eugene person.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import httpx
import jwt
from fastapi import Request, Response

from . import oidc, security
from ._http import shared_internal_client

GOOGLE = "https://accounts.google.com"
SESSION_SECONDS = 8 * 3600
FLOW_SECONDS = 600
ADMIN_CLIENT = "eugene-administration"
ACCOUNT_CLIENT = "eugene-account"


class Refused(ValueError):
    pass


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def safe_url(value: str, *, loopback: bool = False, issuer: bool = False) -> str:
    try:
        parsed = urlsplit(value)
        local = parsed.hostname in ("127.0.0.1", "::1", "localhost")
        if (
            not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.fragment
            or any(c.isspace() or ord(c) < 32 or c == "\\" for c in value)
            or (issuer and parsed.query)
            or (parsed.scheme != "https" and not (loopback and local and parsed.scheme == "http"))
        ):
            raise ValueError
        _ = parsed.port
    except ValueError as exc:
        raise Refused(
            "Use an HTTPS address. HTTP is supported only on loopback for local testing."
        ) from exc
    return value


def validate_provider(record: dict[str, Any]) -> None:
    issuer = safe_url(record["issuer"], loopback=True, issuer=True)
    if record["kind"] == "google" and issuer != GOOGLE:
        raise Refused("Google uses https://accounts.google.com as its issuer.")
    if record["kind"] == "microsoft" and not re.fullmatch(
        r"https://login\.microsoftonline\.com/[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}/v2\.0",
        issuer,
    ):
        raise Refused("Microsoft needs the issuer for a specific tenant ID, ending in /v2.0.")


async def document(
    url: str, *, data: dict[str, str] | None = None, auth: Any = None
) -> dict[str, Any]:
    # Reused pool, bounded responses, no redirects or cookies carried between people/providers.
    client = shared_internal_client("external-sign-in", timeout=15, follow_redirects=False)
    request = client.build_request(
        "POST" if data is not None else "GET",
        url,
        data=data,
        headers={"Accept": "application/json", "Cookie": ""},
    )
    try:
        response = await client.send(request, auth=auth, stream=True)
        try:
            if response.status_code != 200:
                raise Refused(
                    f"The sign-in provider returned HTTP {response.status_code}. "
                    "Check its registration and try again."
                )
            content = bytearray()
            async for chunk in response.aiter_bytes():
                content.extend(chunk)
                if len(content) > 1_048_576:
                    raise Refused("The sign-in provider returned an oversized response.")
            result = json.loads(content)
            if not isinstance(result, dict):
                raise ValueError
            return result
        finally:
            await response.aclose()
    except (httpx.HTTPError, ValueError) as exc:
        if isinstance(exc, Refused):
            raise
        raise Refused(
            "The sign-in provider could not be reached or returned an invalid response."
        ) from exc


async def metadata(provider: dict[str, Any]) -> dict[str, Any]:
    issuer = provider["issuer"]
    result = await document(issuer.rstrip("/") + "/.well-known/openid-configuration")
    if result.get("issuer") != issuer:
        raise Refused("The provider's discovery issuer does not match its configured issuer.")
    for name in ("authorization_endpoint", "token_endpoint", "jwks_uri"):
        value = result.get(name)
        if not isinstance(value, str):
            raise Refused("The provider's discovery document is missing an endpoint.")
        safe_url(value, loopback=urlsplit(issuer).scheme == "http")
    if provider["tokenAuthMethod"] not in result.get(
        "token_endpoint_auth_methods_supported", ["client_secret_basic"]
    ):
        raise Refused("The provider does not support the selected client-secret method.")
    return result


@dataclass
class Flow:
    provider_id: str
    revision: str
    binding: str
    verifier: str
    nonce: str
    callback: str
    metadata: dict[str, Any]
    pending: str | None
    invitation: str | None
    expires: float


@dataclass
class Flows:
    pending: dict[str, Flow] = field(default_factory=dict)

    def add(self, flow: Flow) -> str:
        self.pending = {k: v for k, v in self.pending.items() if v.expires > time.time()}
        if len(self.pending) >= 1000:
            raise Refused("Too many sign-ins are pending. Try again shortly.")
        state = secrets.token_urlsafe(32)
        self.pending[state] = flow
        return state

    def take(self, state: str) -> Flow | None:
        flow = self.pending.pop(state, None)
        return flow if flow and flow.expires > time.time() else None


async def finish(provider: dict[str, Any], flow: Flow, code: str, master: bytes) -> dict[str, Any]:
    secret = security.open_b64(provider["sealedSecret"], master).decode()
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": flow.callback,
        "client_id": provider["clientId"],
        "code_verifier": flow.verifier,
    }
    auth = None
    if provider["tokenAuthMethod"] == "client_secret_post":
        data["client_secret"] = secret
    else:
        from urllib.parse import quote_plus

        auth = httpx.BasicAuth(quote_plus(provider["clientId"]), quote_plus(secret))
    tokens = await document(flow.metadata["token_endpoint"], data=data, auth=auth)
    token = tokens.get("id_token")
    if not isinstance(token, str) or len(token) > 32768:
        raise Refused("The sign-in provider returned no valid ID token.")
    try:
        header = jwt.get_unverified_header(token)
        algorithm = header.get("alg")
        if algorithm not in ("RS256", "ES256") or algorithm not in flow.metadata.get(
            "id_token_signing_alg_values_supported", ["RS256"]
        ):
            raise ValueError("algorithm")
        keys = (await document(flow.metadata["jwks_uri"])).get("keys")
        if not isinstance(keys, list) or len(keys) > 100:
            raise ValueError("keys")
        matches = [
            key
            for key in keys
            if isinstance(key, dict)
            and key.get("kid") == header.get("kid")
            and key.get("use", "sig") == "sig"
            and key.get("alg", algorithm) == algorithm
            and "verify" in key.get("key_ops", ["verify"])
        ]
        if len(matches) != 1:
            raise ValueError("signing key")
        if provider["kind"] == "microsoft":
            key_issuer = matches[0].get("issuer")
            tenant = provider["issuer"].split("/")[3]
            if key_issuer is not None and (
                not isinstance(key_issuer, str)
                or key_issuer.replace("{tenantid}", tenant) != provider["issuer"]
            ):
                raise ValueError("signing key issuer")
        issuer: str | list[str] = (
            [GOOGLE, "accounts.google.com"] if provider["kind"] == "google" else provider["issuer"]
        )
        claims = jwt.decode(
            token,
            jwt.PyJWK.from_dict(matches[0], algorithm=algorithm).key,
            algorithms=[algorithm],
            issuer=issuer,
            audience=provider["clientId"],
            leeway=30,
            options={"require": ["iss", "sub", "aud", "exp", "iat", "nonce"]},
        )
        if not isinstance(claims["sub"], str) or not 1 <= len(claims["sub"]) <= 255:
            raise ValueError("subject")
        if not isinstance(claims["nonce"], str) or not hmac.compare_digest(
            claims["nonce"], flow.nonce
        ):
            raise ValueError("nonce")
        audience = claims["aud"]
        if ((isinstance(audience, list) and len(audience) > 1) or "azp" in claims) and claims.get(
            "azp"
        ) != provider["clientId"]:
            raise ValueError("authorized party")
        if (
            provider["kind"] == "microsoft"
            and claims.get("tid") != provider["issuer"].split("/")[3]
        ):
            raise ValueError("tenant")
        return {"issuer": provider["issuer"], "subject": claims["sub"]}
    except (jwt.PyJWTError, ValueError, TypeError, KeyError) as exc:
        raise Refused(
            "The provider's identity proof failed verification. Start sign-in again."
        ) from exc


def cookie_name(issuer: str, *, flow: bool = False) -> str:
    return (
        ("__Secure-" if issuer.startswith("https:") else "")
        + "ep-signin"
        + ("-flow" if flow else "")
    )


def set_cookie(response: Response, issuer: str, value: str, *, flow: bool = False) -> None:
    response.set_cookie(
        cookie_name(issuer, flow=flow),
        value,
        path="/oidc",
        httponly=True,
        secure=issuer.startswith("https:"),
        samesite="lax",
        max_age=FLOW_SECONDS if flow else SESSION_SECONDS,
    )


def valid_session(
    state: Any, session_id: str, person_id: str | None = None
) -> dict[str, Any] | None:
    record: dict[str, Any] | None = state.browser_sign_in["sessions"].get(session_id)
    if (
        not record
        or record["expiresAt"] <= time.time()
        or (person_id and record["personId"] != person_id)
    ):
        return None
    person = state.people.get(record["personId"])
    if (
        not person
        or person.get("disabled")
        or oidc.iso_to_epoch(person["passwordChangedAt"]) > record["authAt"]
    ):
        return None
    if record["identityId"]:
        identity = state.browser_sign_in["identities"].get(record["identityId"])
        provider = (
            state.browser_sign_in["providers"].get(identity["providerId"]) if identity else None
        )
        if (
            not provider
            or not provider["enabled"]
            or provider["revision"] != record["providerRevision"]
        ):
            return None
    return record


def browser_session(request: Request, issuer: str) -> dict[str, Any] | None:
    machine = request.app.state.machine
    if not machine.is_active or request.app.state.auth_state.master_key is None:
        return None
    raw = request.cookies.get(cookie_name(issuer), "")
    record = valid_session(machine.state, digest(raw)) if raw else None
    return record if record and record["issuer"] == issuer else None


def new_session(
    request: Request, issuer: str, person_id: str, *, identity: dict[str, Any] | None = None
) -> tuple[str, dict[str, Any]]:
    machine = request.app.state.machine
    secret = secrets.token_urlsafe(32)
    moment = time.time()
    if (
        sum(
            record["expiresAt"] > moment
            for record in machine.state.browser_sign_in["sessions"].values()
        )
        >= 2000
    ):
        raise Refused("The shared sign-in limit was reached. Ask the owner to end unused sessions.")
    record = {
        "id": digest(secret),
        "personId": person_id,
        "issuer": issuer,
        "authAt": moment,
        "expiresAt": int(moment) + SESSION_SECONDS,
        "label": (request.headers.get("user-agent") or "Browser")[:200],
        "identityId": identity["id"] if identity else None,
        "providerRevision": machine.state.browser_sign_in["providers"][identity["providerId"]][
            "revision"
        ]
        if identity
        else None,
    }
    machine.append("putBrowserSignIn", {"record": record}, actor=person_id)
    return secret, record


def authorization_url(flow: Flow, provider: dict[str, Any], state: str) -> str:
    return oidc.redirect_with(
        flow.metadata["authorization_endpoint"],
        {
            "response_type": "code",
            "client_id": provider["clientId"],
            "redirect_uri": flow.callback,
            "scope": "openid profile email",
            "state": state,
            "nonce": flow.nonce,
            "code_challenge": oidc.b64url(hashlib.sha256(flow.verifier.encode()).digest()),
            "code_challenge_method": "S256",
            "prompt": "select_account",
        },
    )


def csrf(request: Request, issuer: str) -> str:
    return hmac.new(
        request.cookies.get(cookie_name(issuer), "").encode(), b"account-actions", hashlib.sha256
    ).hexdigest()
