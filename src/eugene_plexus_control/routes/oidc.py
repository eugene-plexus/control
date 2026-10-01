"""The OpenID Connect provider's endpoints (C2).

Design: `specs/docs/design/sign-in-with-eugene.md`; the mechanics are in
`..oidc`. These are OAuth's endpoints, not Eugene's API, so their errors
are OAuth's (`{"error": ...}`) and the sign-in page is HTML.

Reached through an agent's `/oidc/*`, which forwards here with the host,
scheme and caller it served and a token of its own. Those three are
believed only beside that token (`_forwarded`), so a caller who reaches
this port directly cannot choose the issuer or its sign-in limiter bucket.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import hmac
import logging
import secrets
import time
from typing import Any
from urllib.parse import unquote

import jwt
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from .. import oidc, peer, security, tokens
from ..applied import OP_REVOKE_SIGN_IN, OP_SET_PERSON_PASSWORD, OPERATOR_NAME
from ..dependencies import verify_bearer

log = logging.getLogger(__name__)

router = APIRouter(tags=["oidc"])

_NO_STORE = {"Cache-Control": "no-store", "Pragma": "no-cache"}


def _provider(request: Request) -> oidc.Provider:
    provider = getattr(request.app.state, "oidc", None)
    if provider is None:
        provider = oidc.Provider()
        request.app.state.oidc = provider
    return provider


def _forwarded(request: Request) -> tuple[str, str, str] | None:
    """(host, scheme, caller) an agent served, when its token says so."""
    token = request.headers.get(oidc.NODE_TOKEN_HEADER)
    host = request.headers.get(oidc.FORWARDED_HOST_HEADER)
    if not token or not host:
        return None
    try:
        claims = verify_bearer(request, token, classes=(tokens.TYP_SERVICE,))
    except HTTPException:
        return None
    if claims.sub != tokens.SUB_AGENT or claims.issuer_node is None:
        return None
    proto = request.headers.get(oidc.FORWARDED_PROTO_HEADER, "http")
    if proto not in ("http", "https"):
        proto = "http"
    caller = request.headers.get(oidc.FORWARDED_FOR_HEADER) or "unknown"
    return host, proto, caller


def _issuer(request: Request) -> str:
    configured = request.app.state.machine.state.config.get("oidcIssuer")
    if isinstance(configured, str) and configured.strip():
        return configured.strip().rstrip("/")
    forwarded = _forwarded(request)
    if forwarded is not None:
        host, proto, _ = forwarded
        return f"{proto}://{host}{oidc.ISSUER_PATH}"
    return str(request.base_url).rstrip("/") + oidc.ISSUER_PATH


def _caller(request: Request) -> str:
    forwarded = _forwarded(request)
    if forwarded is not None:
        return forwarded[2]
    return (
        peer.peer_of(
            request.client.host if request.client else None,
            request.headers.get(peer.PEER_HEADER),
        )
        or "unknown"
    )


def _oauth_error(status: int, error: str, description: str) -> JSONResponse:
    headers = dict(_NO_STORE)
    if status == 401:
        headers["WWW-Authenticate"] = 'Basic realm="eugene"'
    return JSONResponse(
        {"error": error, "error_description": description}, status_code=status, headers=headers
    )


def _client_auth(request: Request) -> dict[str, Any] | None:
    """`client_secret_basic` (RFC 6749 §2.3.1): the client, or None."""
    header = request.headers.get("authorization") or ""
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "basic":
        return None
    try:
        decoded = base64.b64decode(value.strip(), validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return None
    client_id, sep, secret = decoded.partition(":")
    if not sep:
        return None
    client: dict[str, Any] | None = request.app.state.machine.state.oidc_clients.get(
        unquote(client_id)
    )
    if client is None:
        return None
    if not hmac.compare_digest(oidc.secret_verifier(unquote(secret)), client["secretVerifier"]):
        return None
    return dict(client)


# --------------------------------------------------------------------------- #
# discovery and keys
# --------------------------------------------------------------------------- #


@router.get("/oidc/.well-known/openid-configuration", operation_id="oidcDiscovery")
async def discovery(request: Request) -> JSONResponse:
    issuer = _issuer(request)
    return JSONResponse(
        {
            "issuer": issuer,
            "authorization_endpoint": f"{issuer}/authorize",
            "token_endpoint": f"{issuer}/token",
            "userinfo_endpoint": f"{issuer}/userinfo",
            "revocation_endpoint": f"{issuer}/revoke",
            "jwks_uri": f"{issuer}/jwks",
            "response_types_supported": ["code"],
            "response_modes_supported": ["query"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "subject_types_supported": ["public"],
            "id_token_signing_alg_values_supported": ["RS256"],
            "token_endpoint_auth_methods_supported": ["client_secret_basic"],
            "revocation_endpoint_auth_methods_supported": ["client_secret_basic"],
            "code_challenge_methods_supported": ["S256"],
            "scopes_supported": ["openid", "profile"],
            "claims_supported": [
                "sub",
                "iss",
                "aud",
                "exp",
                "iat",
                "auth_time",
                "nonce",
                "sid",
                "name",
                "preferred_username",
                "eugene_role",
            ],
            "authorization_response_iss_parameter_supported": True,
            "request_parameter_supported": False,
            "request_uri_parameter_supported": False,
        }
    )


@router.get("/oidc/jwks", operation_id="oidcJwks")
async def jwks(request: Request) -> JSONResponse:
    machine = request.app.state.machine
    if not machine.state.oidc_keys and machine.is_active:
        # Make the key before anyone asks for a token signed with it, so
        # a relying party that fetched the JWKS first finds it.
        with contextlib.suppress(oidc.Locked):
            _provider(request).signing_key(machine, request.app.state.auth_state)
    return JSONResponse(oidc.Provider.jwks(machine.state))


# --------------------------------------------------------------------------- #
# the sign-in page
# --------------------------------------------------------------------------- #


def _page(status: int, redirect_uri: str | None = None, **kwargs: Any) -> HTMLResponse:
    return HTMLResponse(
        oidc.page(**kwargs), status_code=status, headers=oidc.page_headers(redirect_uri)
    )


def _hint(state: Any) -> str | None:
    if state.people:
        return f"The owner signs in as {OPERATOR_NAME}, with Eugene's passphrase."
    return None


@router.get("/oidc/authorize", operation_id="oidcAuthorize")
async def authorize(request: Request) -> Response:
    q = request.query_params
    state = request.app.state.machine.state
    client = state.oidc_clients.get(q.get("client_id") or "")
    redirect_uri = q.get("redirect_uri") or ""
    # An unknown app or an unregistered redirect is shown here and never
    # followed: redirecting would make this an open redirector (RFC 9700 §4.11).
    if client is None:
        return _page(
            400,
            title="Unknown app",
            app_name=None,
            error="This app is not registered with Eugene. Ask the owner to add it.",
        )
    if redirect_uri not in client["redirectUris"]:
        return _page(
            400,
            title="Wrong return address",
            app_name=client["name"],
            error="This app asked to send you somewhere it is not registered for, "
            "so Eugene stopped here.",
        )
    issuer = _issuer(request)

    def back(error: str, description: str) -> RedirectResponse:
        params = {"error": error, "error_description": description, "iss": issuer}
        if q.get("state"):
            params["state"] = q["state"]
        return RedirectResponse(oidc.redirect_with(redirect_uri, params), status_code=302)

    if q.get("response_type") != "code":
        return back("unsupported_response_type", "Only the code flow is offered.")
    if "openid" not in (q.get("scope") or "").split():
        return back("invalid_scope", "The openid scope is required.")
    challenge = q.get("code_challenge") or ""
    if q.get("code_challenge_method") != "S256" or not 43 <= len(challenge) <= 128:
        return back("invalid_request", "PKCE with S256 is required.")
    if not q.get("state") or not q.get("nonce"):
        return back("invalid_request", "state and nonce are required.")
    if q.get("request") or q.get("request_uri"):
        return back("request_not_supported", "Request objects are not supported.")
    ident = _provider(request).open_request(
        oidc.PendingRequest(
            client_id=client["clientId"],
            redirect_uri=redirect_uri,
            state=q["state"],
            nonce=q["nonce"],
            challenge=challenge,
            scope="openid profile" if "profile" in q["scope"].split() else "openid",
            issuer=issuer,
            expires=time.time() + oidc.REQUEST_TTL_SECONDS,
        )
    )
    return _page(
        200,
        redirect_uri,
        title=f"Sign in to {client['name']}",
        app_name=client["name"],
        action=f"{issuer}/authorize",
        request_id=ident,
        asks_name=bool(state.people),
        hint=_hint(state),
    )


@router.post("/oidc/authorize", operation_id="oidcSignIn")
async def sign_in(request: Request) -> Response:
    form = await request.form()
    machine = request.app.state.machine
    auth = request.app.state.auth_state
    state = machine.state
    provider = _provider(request)
    ident = str(form.get("request") or "")
    pending = provider.pending(ident)
    if pending is None:
        return _page(
            400,
            title="This sign-in expired",
            app_name=None,
            error="Go back to the app and start again.",
        )
    client = state.oidc_clients.get(pending.client_id)
    if client is None:
        provider.close_request(ident)
        return _page(
            400,
            title="Unknown app",
            app_name=None,
            error="This app is no longer registered with Eugene.",
        )

    password = str(form.get("password") or "")
    name = str(form.get("name") or "").strip()
    asks_name = bool(state.people)

    def again(status: int, error: str) -> HTMLResponse:
        return _page(
            status,
            pending.redirect_uri,
            title=f"Sign in to {client['name']}",
            app_name=client["name"],
            action=f"{pending.issuer}/authorize",
            request_id=ident,
            asks_name=asks_name,
            error=error,
            hint=_hint(state),
        )

    caller = _caller(request)
    buckets = [f"oidc-peer:{caller}"] + ([f"oidc-name:{name.casefold()}"] if asks_name else [])
    for bucket in buckets:
        if auth.is_login_rate_limited(
            bucket, window_seconds=oidc.SIGN_IN_WINDOW_SECONDS, max_in_window=oidc.SIGN_IN_FAILURES
        ):
            return again(
                429,
                f"Too many failed sign-ins. Wait {oidc.SIGN_IN_WINDOW_SECONDS} "
                "seconds and try again.",
            )

    subject: oidc.Subject | None = None
    person: dict[str, Any] | None = None
    owner = not asks_name or name.casefold() == OPERATOR_NAME
    if owner:
        verifier = state.identity.passphraseVerifier
        if verifier and security.verify_passphrase(password, verifier):
            subject = oidc.OWNER
    else:
        person = next(
            (p for p in state.people.values() if p["name"].casefold() == name.casefold()), None
        )
        # The verifier is checked whether or not the name exists, so the
        # time an answer takes does not say which one was wrong.
        verifier = person["passwordVerifier"] if person else _DUMMY_VERIFIER
        if security.verify_passphrase(password, verifier) and person is not None:
            subject = oidc.subject_of(person)
    if subject is None:
        for bucket in buckets:
            auth.record_login_failure(
                bucket,
                window_seconds=oidc.SIGN_IN_WINDOW_SECONDS,
                max_in_window=oidc.SIGN_IN_FAILURES,
            )
        log.warning("failed sign-in to %s from %s", client["name"], caller)
        return again(
            200,
            "That name and password do not match."
            if asks_name
            else "That is not Eugene's passphrase.",
        )
    for bucket in buckets:
        auth.clear_login_failures(bucket)
    if person is not None and person.get("disabled"):
        return again(403, "Signing in is turned off for you. Ask the owner.")
    if person is not None and not oidc.may_use(person, client["clientId"]):
        return _page(
            403,
            pending.redirect_uri,
            title=f"Not for {client['name']}",
            app_name=client["name"],
            error=f"You may not use {client['name']}. The owner can allow it on "
            "Eugene's People page.",
        )
    new_password = str(form.get("new_password") or "")
    if new_password or form.get("new_password_again"):
        if person is None:
            return again(
                400,
                "Eugene's passphrase cannot be changed here. Only people the owner "
                "added change their password on this page.",
            )
        if new_password != str(form.get("new_password_again") or ""):
            return again(200, "The two new passwords are not the same.")
        if len(new_password) < oidc.MIN_PASSWORD:
            return again(200, f"A new password needs at least {oidc.MIN_PASSWORD} characters.")
    try:
        # Made now if it does not exist, so the code can be traded.
        provider.signing_key(machine, auth)
    except oidc.Locked as exc:
        return again(503, str(exc))
    if new_password and person is not None:
        # D10. Every earlier sign-in of this person ends at its next refresh;
        # this one starts after the change, so it is kept.
        machine.append(
            OP_SET_PERSON_PASSWORD,
            {
                "id": person["id"],
                "passwordVerifier": security.hash_passphrase(new_password),
                "passwordChangedAt": oidc.utcnow_iso(),
            },
        )
        log.info("%s changed their password", person["name"])
    provider.close_request(ident)
    code = provider.issue_code(
        oidc.IssuedCode(
            client_id=client["clientId"],
            redirect_uri=pending.redirect_uri,
            challenge=pending.challenge,
            nonce=pending.nonce,
            scope=pending.scope,
            issuer=pending.issuer,
            subject=subject,
            auth_at=time.time(),
            expires=time.time() + oidc.CODE_TTL_SECONDS,
        )
    )
    log.info("%s signed in to %s", subject.preferred_username, client["name"])
    return RedirectResponse(
        oidc.redirect_with(
            pending.redirect_uri, {"code": code, "state": pending.state, "iss": pending.issuer}
        ),
        status_code=302,
        headers={"Cache-Control": "no-store"},
    )


# A real Argon2id verifier of nothing anyone knows, so an unknown name costs
# what a known one does.
_DUMMY_VERIFIER = security.hash_passphrase(secrets.token_urlsafe(24))


# --------------------------------------------------------------------------- #
# tokens
# --------------------------------------------------------------------------- #


@router.post("/oidc/token", operation_id="oidcToken")
async def token(request: Request) -> JSONResponse:
    client = _client_auth(request)
    if client is None:
        return _oauth_error(401, "invalid_client", "Client authentication failed.")
    form = await request.form()
    machine = request.app.state.machine
    auth = request.app.state.auth_state
    provider = _provider(request)
    grant = form.get("grant_type")
    try:
        if grant == "authorization_code":
            issued = provider.take_code(str(form.get("code") or ""))
            if (
                issued is None
                or issued.client_id != client["clientId"]
                or issued.redirect_uri != form.get("redirect_uri")
                or not oidc.pkce_matches(str(form.get("code_verifier") or ""), issued.challenge)
            ):
                return _oauth_error(400, "invalid_grant", "The code is invalid, used or expired.")
            answer = oidc.token_response(
                provider,
                machine,
                auth,
                issuer=issued.issuer,
                client_id=client["clientId"],
                subject=issued.subject,
                scope=issued.scope,
                auth_at=issued.auth_at,
                sid=secrets.token_urlsafe(16),
                nonce=issued.nonce,
            )
            return JSONResponse(answer, headers=_NO_STORE)
        if grant == "refresh_token":
            presented = str(form.get("refresh_token") or "")
            try:
                claims = oidc.Provider.decode(
                    presented, machine.state, typ=oidc.TYP_REFRESH, audience=client["clientId"]
                )
            except jwt.InvalidTokenError:
                return _oauth_error(
                    400, "invalid_grant", "The refresh token is invalid or expired."
                )
            subject = oidc.still_valid(machine.state, claims, client["clientId"])
            if subject is None:
                return _oauth_error(
                    400,
                    "invalid_grant",
                    "This sign-in has ended: signed out, or the person's account "
                    "changed. Sign in again.",
                )
            answer = oidc.token_response(
                provider,
                machine,
                auth,
                issuer=claims["iss"],
                client_id=client["clientId"],
                subject=subject,
                scope=str(claims.get("scope") or "openid"),
                auth_at=float(claims.get("eugene_auth_at") or claims["iat"]),
                sid=claims["sid"],
                nonce=None,
                refresh_token=presented,
            )
            return JSONResponse(answer, headers=_NO_STORE)
    except oidc.Locked as exc:
        return _oauth_error(503, "temporarily_unavailable", str(exc))
    return _oauth_error(400, "unsupported_grant_type", "authorization_code or refresh_token.")


@router.get("/oidc/userinfo", operation_id="oidcUserinfo")
async def userinfo(request: Request) -> JSONResponse:
    header = request.headers.get("authorization") or ""
    scheme, _, value = header.partition(" ")
    machine = request.app.state.machine
    try:
        if scheme.lower() != "bearer":
            raise jwt.InvalidTokenError("no bearer")
        claims = oidc.Provider.decode(
            value.strip(), machine.state, typ=oidc.TYP_ACCESS, audience=None
        )
    except jwt.InvalidTokenError:
        return JSONResponse(
            {"error": "invalid_token"},
            status_code=401,
            headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
        )
    subject = oidc.still_valid(machine.state, claims, str(claims.get("client_id") or ""))
    if subject is None:
        return JSONResponse(
            {"error": "invalid_token"},
            status_code=401,
            headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
        )
    return JSONResponse({"sub": subject.sub, **subject.claims()}, headers=_NO_STORE)


@router.post("/oidc/revoke", operation_id="oidcRevoke")
async def revoke(request: Request) -> Response:
    client = _client_auth(request)
    if client is None:
        return _oauth_error(401, "invalid_client", "Client authentication failed.")
    form = await request.form()
    machine = request.app.state.machine
    try:
        claims = oidc.Provider.decode(
            str(form.get("token") or ""),
            machine.state,
            typ=oidc.TYP_REFRESH,
            audience=client["clientId"],
        )
    except jwt.InvalidTokenError:
        # RFC 7009 §2.2: an invalid token is answered 200; there is nothing to revoke.
        return Response(status_code=200, headers=_NO_STORE)
    if claims["sid"] not in machine.state.revoked_sign_ins:
        machine.append(
            OP_REVOKE_SIGN_IN,
            {"sid": claims["sid"], "exp": int(claims["exp"]), "prunedBefore": oidc.now()},
        )
    return Response(status_code=200, headers=_NO_STORE)
