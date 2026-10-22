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

from .. import external_sign_in as sso
from .. import oidc, peer, permissions, security, tokens
from ..applied import OP_REVOKE_SIGN_IN, OP_SET_PERSON_PASSWORD, OPERATOR_NAME
from ..dependencies import verify_bearer

log = logging.getLogger(__name__)

router = APIRouter(tags=["oidc"])

_NO_STORE = {"Cache-Control": "no-store", "Pragma": "no-cache"}

#: The scopes honoured, in the order a granted scope lists them. Anything
#: else asked for is ignored rather than refused (OIDC Core 3.1.2.1).
GRANTED_SCOPES = ("openid", "profile", "email")


def _granted_scope(asked: str) -> str:
    wanted = set(asked.split())
    return " ".join(s for s in GRANTED_SCOPES if s in wanted)


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
    configured = oidc.configured_issuer(
        request.app.state.machine.state.config, getattr(request.app.state, "settings", None)
    )
    if configured:
        return configured
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


def _token_client(request: Request, form: Any) -> dict[str, Any] | None:
    """Who is at `/oidc/token`: the built-in public client by `client_id` in the
    form with no secret of any kind, else `client_secret_basic`."""
    if form.get("client_id") == oidc.SITE_LINK_CLIENT:
        scheme = (request.headers.get("authorization") or "").partition(" ")[0].lower()
        if scheme == "basic" or form.get("client_secret") is not None:
            return None
        return oidc.find_client(request.app.state.machine.state, oidc.SITE_LINK_CLIENT)
    return _client_auth(request)


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
            "scopes_supported": list(GRANTED_SCOPES),
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
                "email",
                "email_verified",
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
        oidc.page(**kwargs),
        status_code=status,
        headers=oidc.page_headers(redirect_uri, external=bool(kwargs.get("providers"))),
    )


def _hint(state: Any) -> str | None:
    if state.people:
        return f"The owner signs in as {OPERATOR_NAME}, with Eugene's passphrase."
    return None


@router.get("/oidc/authorize", operation_id="oidcAuthorize")
async def authorize(request: Request) -> Response:
    q = request.query_params
    state = request.app.state.machine.state
    client = oidc.find_client(state, q.get("client_id") or "")
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
    if not oidc.redirect_allowed(client, redirect_uri, _issuer(request)):
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
    binding = secrets.token_urlsafe(32)
    ident = _provider(request).open_request(
        oidc.PendingRequest(
            client_id=client["clientId"],
            redirect_uri=redirect_uri,
            state=q["state"],
            nonce=q["nonce"],
            challenge=challenge,
            scope=_granted_scope(q["scope"]),
            issuer=issuer,
            expires=time.time() + oidc.REQUEST_TTL_SECONDS,
            binding=binding,
        )
    )
    session = sso.browser_session(request, issuer)
    if session and q.get("prompt") == "none":
        person = state.people[session["personId"]]
        return complete_sign_in(
            request, ident, oidc.subject_of(person), session["authAt"], session["id"]
        )
    if q.get("prompt") == "none":
        _provider(request).close_request(ident)
        return back("login_required", "Sign in to Eugene first.")
    response = _page(
        200,
        redirect_uri,
        title=f"Sign in to {client['name']}",
        app_name=client["name"],
        action=f"{issuer}/authorize",
        request_id=ident,
        asks_name=bool(state.people),
        continue_name=state.people[session["personId"]]["name"]
        if session and q.get("prompt") not in ("login", "select_account")
        else None,
        hint=_hint(state),
        providers=[p for p in state.browser_sign_in["providers"].values() if p["enabled"]],
    )
    sso.set_cookie(response, issuer, binding, flow=True)
    return response


def complete_sign_in(
    request: Request,
    ident: str | None,
    subject: oidc.Subject,
    auth_at: float,
    browser_session: str | None = None,
) -> Response:
    provider = _provider(request)
    pending = provider.pending(ident or "")
    machine = request.app.state.machine
    if pending is None or pending.issuer != _issuer(request):
        return _page(
            400,
            title="Sign-in expired",
            app_name=None,
            error="Return to the app and sign in again.",
        )
    client = oidc.find_client(machine.state, pending.client_id)
    person = machine.state.people.get(subject.sub)
    allowed = client is not None
    if pending.client_id == sso.ADMIN_CLIENT:
        from ..administration import has_grants

        allowed = person is not None and has_grants(person)
    elif pending.client_id == sso.ACCOUNT_CLIENT:
        allowed = person is not None
    elif pending.client_id == oidc.SITE_LINK_CLIENT:
        allowed = person is not None and permissions.has(
            machine.state, subject.sub, permissions.USE_JOB_SITES
        )
    elif person is not None and client is not None:
        allowed = oidc.may_use(machine.state, person, client["clientId"])
    if person is not None and person.get("disabled"):
        allowed = False
    if browser_session and not sso.valid_session(machine.state, browser_session, subject.sub):
        allowed = False
    if not allowed:
        return _page(
            403,
            title="Access not granted",
            app_name=client["name"] if client else None,
            error="This account may not use this app. Ask the owner to review its access.",
        )
    try:
        provider.signing_key(machine, request.app.state.auth_state)
    except oidc.Locked as exc:
        return _page(503, title="Eugene is locked", app_name=None, error=str(exc))
    provider.close_request(ident or "")
    if pending.client_id == sso.ACCOUNT_CLIENT:
        return RedirectResponse(pending.redirect_uri, status_code=303, headers=_NO_STORE)
    code = provider.issue_code(
        oidc.IssuedCode(
            client_id=pending.client_id,
            redirect_uri=pending.redirect_uri,
            challenge=pending.challenge,
            nonce=pending.nonce,
            scope=pending.scope,
            issuer=pending.issuer,
            subject=subject,
            auth_at=auth_at,
            expires=time.time() + oidc.CODE_TTL_SECONDS,
            browser_session=browser_session,
        )
    )
    return RedirectResponse(
        oidc.redirect_with(
            pending.redirect_uri, {"code": code, "state": pending.state, "iss": pending.issuer}
        ),
        status_code=302,
        headers=_NO_STORE,
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
    if pending is None or (
        pending.binding
        and not hmac.compare_digest(
            pending.binding, request.cookies.get(sso.cookie_name(pending.issuer, flow=True), "")
        )
    ):
        return _page(
            400,
            title="This sign-in expired",
            app_name=None,
            error="Go back to the app and start again.",
        )
    client = oidc.find_client(state, pending.client_id)
    if client is None:
        provider.close_request(ident)
        return _page(
            400,
            title="Unknown app",
            app_name=None,
            error="This app is no longer registered with Eugene.",
        )

    if form.get("use_session"):
        session = sso.browser_session(request, pending.issuer)
        if session is None:
            return _page(
                401, title="Sign-in expired", app_name=None, error="Sign in again to continue."
            )
        return complete_sign_in(
            request,
            ident,
            oidc.subject_of(state.people[session["personId"]]),
            session["authAt"],
            session["id"],
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
            providers=[p for p in state.browser_sign_in["providers"].values() if p["enabled"]],
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
    if client["clientId"] == oidc.SITE_LINK_CLIENT and subject is oidc.OWNER:
        # J37: the link page is for people. Eugene's owner owns no site's links.
        return _page(
            403,
            pending.redirect_uri,
            title=f"Not for {client['name']}",
            app_name=client["name"],
            error=f"You may not use {client['name']}. It links people, and Eugene's owner "
            "is not a person on a machine. Sign in as a person the owner added.",
        )
    if (
        person is not None
        and client["clientId"] == oidc.SITE_LINK_CLIENT
        and not permissions.has(
            request.app.state.machine.state, str(person["id"]), permissions.USE_JOB_SITES
        )
    ):
        # J77: linking at a machine takes `use-job-sites`.
        return _page(
            403,
            pending.redirect_uri,
            title=f"Not for {client['name']}",
            app_name=client["name"],
            error=permissions.refusal(permissions.USE_JOB_SITES),
        )
    if (
        person is not None
        and not client.get("builtin")
        and not oidc.may_use(machine.state, person, client["clientId"])
    ):
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
            actor=subject.sub,
        )
        log.info("%s changed their password", person["name"])
    browser_id = None
    browser_secret = None
    auth_at = time.time()
    if person is not None:
        try:
            browser_secret, session = sso.new_session(request, pending.issuer, person["id"])
        except sso.Refused as exc:
            return again(429, str(exc))
        browser_id, auth_at = session["id"], session["authAt"]
    response = complete_sign_in(request, ident, subject, auth_at, browser_id)
    if browser_secret:
        sso.set_cookie(response, pending.issuer, browser_secret)
    return response


# A real Argon2id verifier of nothing anyone knows, so an unknown name costs
# what a known one does.
_DUMMY_VERIFIER = security.hash_passphrase(secrets.token_urlsafe(24))


# --------------------------------------------------------------------------- #
# tokens
# --------------------------------------------------------------------------- #


@router.post("/oidc/token", operation_id="oidcToken")
async def token(request: Request) -> JSONResponse:
    form = await request.form()
    client = _token_client(request, form)
    if client is None:
        return _oauth_error(401, "invalid_client", "Client authentication failed.")
    machine = request.app.state.machine
    auth = request.app.state.auth_state
    provider = _provider(request)
    grant = form.get("grant_type")
    builtin = bool(client.get("builtin"))
    if builtin and grant != "authorization_code":
        return _oauth_error(
            400, "unsupported_grant_type", "This app may only trade a code for tokens."
        )
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
            code_subject = issued.subject
            if issued.browser_session and not sso.valid_session(
                machine.state, issued.browser_session, code_subject.sub
            ):
                return _oauth_error(400, "invalid_grant", "This sign-in has ended.")
            if not builtin:
                # Group membership can change after approval but before the app
                # exchanges its code. Recheck the current policy at this boundary.
                current = oidc.still_valid(
                    machine.state,
                    {
                        "sub": code_subject.sub,
                        "eugene_auth_at": issued.auth_at,
                        **(
                            {"eugene_browser_session": issued.browser_session}
                            if issued.browser_session
                            else {}
                        ),
                    },
                    client["clientId"],
                )
                if current is None:
                    return _oauth_error(400, "invalid_grant", "This sign-in no longer has access.")
                code_subject = current
            answer = oidc.token_response(
                provider,
                machine,
                auth,
                issuer=issued.issuer,
                client_id=client["clientId"],
                subject=code_subject,
                scope=issued.scope,
                auth_at=issued.auth_at,
                sid=secrets.token_urlsafe(16),
                nonce=issued.nonce,
                with_refresh=not builtin,
                browser_session=issued.browser_session,
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
                browser_session=claims.get("eugene_browser_session"),
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
    scope = str(claims.get("scope") or "openid")
    return JSONResponse(
        {
            "sub": subject.sub,
            **subject.claims(scope),
            **oidc.group_claims(machine.state, subject.sub),
        },
        headers=_NO_STORE,
    )


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
            actor=claims["sub"],
        )
    return Response(status_code=200, headers=_NO_STORE)


#: The most ids one `/oidc/subjects` call may carry; an app asks in batches.
MAX_SUBJECTS = 500


@router.post("/oidc/subjects", operation_id="oidcSubjects")
async def subjects(request: Request) -> JSONResponse:
    """Which of the ids an app sends Eugene still has a person for (row 7).

    An app deletes a person's data on `unknown`, so nothing is said while the root
    cannot vouch for its own list: locked (it has not been unlocked since it started)
    or a standby (it may be behind the active root). A person turned off is `known`;
    so is the owner. Only the ids sent are answered, so nobody can list people."""
    client = _client_auth(request)
    if client is None:
        return _oauth_error(401, "invalid_client", "Client authentication failed.")
    machine = request.app.state.machine
    if not machine.is_active:
        return _oauth_error(
            503,
            "temporarily_unavailable",
            "This control root is a standby; ask the active one.",
        )
    if request.app.state.auth_state.master_key is None:
        return _oauth_error(
            503,
            "temporarily_unavailable",
            "Eugene is locked. Sign in to the console once with the passphrase to unlock it.",
        )
    try:
        body = await request.json()
    except ValueError:
        body = None
    asked = body.get("subjects") if isinstance(body, dict) else None
    if (
        not isinstance(asked, list)
        or not 1 <= len(asked) <= MAX_SUBJECTS
        or not all(isinstance(s, str) and 1 <= len(s) <= 200 for s in asked)
    ):
        return _oauth_error(
            400,
            "invalid_request",
            f"Send subjects: a list of 1 to {MAX_SUBJECTS} ids, each a string.",
        )
    people = machine.state.people
    known: list[str] = []
    unknown: list[str] = []
    for sub in dict.fromkeys(asked):
        (known if sub == OPERATOR_NAME or sub in people else unknown).append(sub)
    return JSONResponse({"known": known, "unknown": unknown}, headers=_NO_STORE)
