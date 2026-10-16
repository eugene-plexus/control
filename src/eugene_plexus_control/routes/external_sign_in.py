"""Provider setup, explicit invitations, and shared sign-in/account pages."""

from __future__ import annotations

import hashlib
import hmac
import html
import secrets
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from .. import external_sign_in as sso
from .. import oidc, security
from .._generated.models import (
    AdministrationCodeRequest,
    AdministrationLoginResponse,
    ExternalProvider,
    ExternalProviderList,
    ExternalProviderWrite,
    PersonSignIn,
    SignInInvitationCreated,
    SignInInvitationRequest,
)
from ..administration import active, has_grants
from ..dependencies import problem, require_operator
from ..trusted_host import require_trusted_host
from . import oidc as local

management = APIRouter(tags=["people"], dependencies=[Depends(require_operator)])
router = APIRouter(tags=["oidc"])


def machine_for(request: Request) -> Any:
    machine = active(request)
    if request.app.state.auth_state.master_key is None:
        raise problem(
            503, "Eugene is locked", "The owner must unlock Eugene before people can sign in."
        )
    return machine


def public_provider(request: Request, record: dict[str, Any]) -> ExternalProvider:
    return ExternalProvider.model_validate(
        {
            **{k: v for k, v in record.items() if k != "sealedSecret"},
            "hasSecret": bool(record["sealedSecret"]),
            "callbackUrl": local._issuer(request) + "/external/callback",
        }
    )


def provider_for(request: Request, ident: str) -> dict[str, Any]:
    record: dict[str, Any] | None = (
        machine_for(request).state.browser_sign_in["providers"].get(ident)
    )
    if record is None or not record["enabled"]:
        raise sso.Refused(
            "This sign-in provider is unavailable. Use your Eugene password or ask the owner."
        )
    return record


@management.get("/v1/auth/providers", response_model=ExternalProviderList)
async def providers(request: Request, response: Response) -> ExternalProviderList:
    response.headers["Cache-Control"] = "no-store"
    return ExternalProviderList(
        signInAddress=oidc.configured_issuer(
            active(request).state.config, request.app.state.settings
        ),
        managedSignInAddress=oidc.managed_issuer(request.app.state.settings),
        providers=[
            public_provider(request, p)
            for p in active(request).state.browser_sign_in["providers"].values()
        ],
    )


async def write_provider(
    request: Request, ident: str, body: ExternalProviderWrite
) -> ExternalProvider:
    machine = machine_for(request)
    before = machine.state.browser_sign_in["providers"].get(ident)
    record = body.model_dump(mode="json", exclude={"clientSecret"})
    record["name"] = record["name"].strip()
    # OIDC compares issuer identifiers exactly, including a provider's trailing slash.
    record["issuer"] = record["issuer"].strip()
    record["clientId"] = record["clientId"].strip()
    if not record["name"] or not record["clientId"]:
        raise problem(422, "Provider details required", "Enter a display name and client ID.")
    if before and any(before[k] != record[k] for k in ("issuer", "clientId", "kind")):
        raise problem(
            409,
            "New provider required",
            "Issuer, provider type and client ID are fixed. Add a new provider to change them.",
        )
    if before is None and len(machine.state.browser_sign_in["providers"]) >= 16:
        raise problem(409, "Provider limit", "An install supports up to 16 sign-in providers.")
    secret = body.clientSecret
    if not secret and before is None:
        raise problem(
            422,
            "Client secret required",
            "Enter the client secret from the provider's registration.",
        )
    record.update(
        id=ident,
        revision=secrets.token_hex(16),
        sealedSecret=security.seal_b64(secret.encode(), request.app.state.auth_state.master_key)
        if secret
        else before["sealedSecret"],
    )
    try:
        sso.validate_provider(record)
        # Disabling a broken provider must never depend on it being reachable.
        if record["enabled"]:
            if not oidc.configured_issuer(machine.state.config, request.app.state.settings):
                raise sso.Refused("Save Eugene's sign-in address before enabling a provider.")
            from urllib.parse import urlsplit

            if urlsplit(local._issuer(request)).path != "/oidc":
                raise sso.Refused("Eugene's sign-in address must use /oidc on its public host.")
            sso.safe_url(local._issuer(request), loopback=True, issuer=True)
            await sso.metadata(record)
    except sso.Refused as exc:
        raise problem(422, "Provider could not be enabled", str(exc)) from exc
    if machine.state.browser_sign_in["providers"].get(ident) != before:
        raise problem(409, "Provider changed", "Reload the provider settings and try again.")
    machine.append("putExternalProvider", {"record": record}, actor="operator")
    return public_provider(request, record)


@management.post("/v1/auth/providers", response_model=ExternalProvider)
async def create_provider(request: Request, body: ExternalProviderWrite) -> ExternalProvider:
    return await write_provider(request, uuid.uuid4().hex, body)


@management.put("/v1/auth/providers/{provider_id}", response_model=ExternalProvider)
async def update_provider(
    request: Request, provider_id: str, body: ExternalProviderWrite
) -> ExternalProvider:
    if provider_id not in active(request).state.browser_sign_in["providers"]:
        raise problem(404, "Provider missing", "This provider was removed.")
    return await write_provider(request, provider_id, body)


@management.delete("/v1/auth/providers/{provider_id}", status_code=204)
async def delete_provider(request: Request, provider_id: str) -> Response:
    machine = machine_for(request)
    if provider_id not in machine.state.browser_sign_in["providers"]:
        raise problem(404, "Provider missing", "This provider was removed.")
    machine.append("deleteExternalProvider", {"id": provider_id}, actor="operator")
    return Response(status_code=204)


def person_for(request: Request, person_id: str) -> dict[str, Any]:
    person = active(request).state.people.get(person_id)
    if person is None:
        raise problem(404, "Account missing", "Choose a current member account.")
    return person


@management.get("/v1/people/{person_id}/sign-in", response_model=PersonSignIn)
async def person_sign_in(request: Request, person_id: str, response: Response) -> PersonSignIn:
    person_for(request, person_id)
    state = active(request).state
    response.headers["Cache-Control"] = "no-store"
    return PersonSignIn.model_validate(
        {
            table: [
                r
                for r in state.browser_sign_in[table].values()
                if r["personId"] == person_id
                and (table != "sessions" or sso.valid_session(state, r["id"]))
                and (table != "invitations" or r["expiresAt"] > time.time())
            ]
            for table in ("identities", "invitations", "sessions")
        }
    )


@management.post(
    "/v1/people/{person_id}/sign-in/invitations", response_model=SignInInvitationCreated
)
async def invite(
    request: Request, person_id: str, body: SignInInvitationRequest, response: Response
) -> SignInInvitationCreated:
    machine = machine_for(request)
    person = person_for(request, person_id)
    if person["disabled"]:
        raise problem(409, "Account disabled", "Enable this account before inviting it.")
    try:
        provider_for(request, body.providerId)
    except sso.Refused as exc:
        raise problem(409, "Provider unavailable", str(exc)) from exc
    if (
        sum(
            item["expiresAt"] > time.time()
            for item in machine.state.browser_sign_in["invitations"].values()
        )
        >= 2000
    ):
        raise problem(409, "Invitation limit", "Revoke unused invitations before creating more.")
    code = secrets.token_urlsafe(32)
    expires = int(time.time()) + 86400
    record = {
        "id": sso.digest(code),
        "personId": person_id,
        "providerId": body.providerId,
        "expiresAt": expires,
        "passwordChangedAt": person["passwordChangedAt"],
    }
    machine.append("putSignInInvitation", {"record": record}, actor="operator")
    response.headers["Cache-Control"] = "no-store"
    return SignInInvitationCreated(
        code=code, expiresAt=expires, url=local._issuer(request) + "/invitation"
    )


def remove_record(request: Request, person_id: str, record_id: str, table: str) -> Response:
    machine = machine_for(request)
    record = machine.state.browser_sign_in[table].get(record_id)
    if not record or record["personId"] != person_id:
        raise problem(404, "Sign-in missing", "This sign-in record no longer exists.")
    op = {
        "identities": "deleteExternalIdentity",
        "sessions": "deleteBrowserSignIn",
        "invitations": "deleteSignInInvitation",
    }[table]
    machine.append(op, {"id": record_id}, actor="operator")
    return Response(status_code=204)


@management.delete("/v1/people/{person_id}/sign-in/identities/{record_id}", status_code=204)
async def remove_identity(request: Request, person_id: str, record_id: str) -> Response:
    return remove_record(request, person_id, record_id, "identities")


@management.delete("/v1/people/{person_id}/sign-in/sessions/{record_id}", status_code=204)
async def remove_session(request: Request, person_id: str, record_id: str) -> Response:
    return remove_record(request, person_id, record_id, "sessions")


@management.delete("/v1/people/{person_id}/sign-in/invitations/{record_id}", status_code=204)
async def remove_invitation(request: Request, person_id: str, record_id: str) -> Response:
    return remove_record(request, person_id, record_id, "invitations")


def flows(request: Request) -> sso.Flows:
    if not hasattr(request.app.state, "external_flows"):
        request.app.state.external_flows = sso.Flows()
    result: sso.Flows = request.app.state.external_flows
    return result


def page(title: str, body: str, *, status: int = 200) -> HTMLResponse:
    e = html.escape
    return HTMLResponse(
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{e(title)} · Eugene</title><style>{oidc._STYLE}"
        "a { color:var(--accent); } ul { padding-left:20px; } "
        ".entry { border-top:1px solid var(--border); margin-top:16px; "
        "padding-top:12px; overflow-wrap:anywhere; }"
        f"</style></head><body><main><h1>{e(title)}</h1>{body}</main></body></html>",
        status_code=status,
        headers=oidc.page_headers(external=True),
    )


def refused(exc: Exception) -> HTMLResponse:
    return page(
        "Sign-in could not finish",
        f"<p role='alert'>{html.escape(str(exc))}</p>"
        "<p><a href='/oidc/account'>Try signing in again</a></p>"
        "<p class='hint'>Your Eugene password and the owner's recovery controls "
        "remain available.</p>",
        status=400,
    )


def invitation_for(request: Request, key: str) -> dict[str, Any]:
    state = machine_for(request).state
    invitation: dict[str, Any] | None = state.browser_sign_in["invitations"].get(key)
    person = state.people.get(invitation["personId"]) if invitation else None
    if (
        not invitation
        or invitation["expiresAt"] <= time.time()
        or not person
        or person["disabled"]
        or invitation["passwordChangedAt"] != person["passwordChangedAt"]
    ):
        raise sso.Refused(
            "This invitation is expired, used or revoked. Ask the owner for a new one."
        )
    return invitation


@router.get("/oidc/invitation")
async def invitation_page(request: Request) -> Response:
    issuer = local._issuer(request)
    binding = secrets.token_urlsafe(32)
    response = page(
        "Connect a sign-in account",
        "<p>Enter the invitation code the owner shared with you. It connects your provider "
        "account to the Eugene account they selected.</p><form method='post'>"
        f"<input type='hidden' name='binding' value='{binding}'>"
        "<label for='code'>Invitation code</label>"
        "<input id='code' name='code' autocomplete='off' required maxlength='128'>"
        "<button>Continue</button></form>",
    )
    sso.set_cookie(response, issuer, binding, flow=True)
    return response


@router.post("/oidc/invitation")
async def invitation_check(request: Request) -> Response:
    form = await request.form()
    issuer = local._issuer(request)
    binding = request.cookies.get(sso.cookie_name(issuer, flow=True), "")
    if not binding or not hmac.compare_digest(binding, str(form.get("binding", ""))):
        return refused(sso.Refused("Start the invitation in this browser again."))
    try:
        code = str(form.get("code", ""))
        invitation = invitation_for(request, sso.digest(code))
        provider = provider_for(request, invitation["providerId"])
        name = machine_for(request).state.people[invitation["personId"]]["name"]
    except sso.Refused as exc:
        return refused(exc)
    e = html.escape
    return page(
        "Confirm account connection",
        f"<p>Connect your {e(provider['name'])} sign-in to Eugene account "
        f"<strong>{e(name)}</strong>. "
        "Continue only if this is your account.</p>"
        f"<form method='post' action='{e(issuer)}/external/start'>"
        f"<input type='hidden' name='invitation' value='{e(code)}'>"
        f"<input type='hidden' name='provider' value='{e(provider['id'])}'>"
        f"<input type='hidden' name='binding' value='{e(binding)}'>"
        f"<button>Connect with {e(provider['name'])}</button></form>",
    )


@router.post("/oidc/external/start")
async def start(request: Request) -> Response:
    form = await request.form()
    issuer = local._issuer(request)
    try:
        sso.safe_url(issuer, loopback=True, issuer=True)
        provider = provider_for(request, str(form.get("provider", "")))
        binding = request.cookies.get(sso.cookie_name(issuer, flow=True), "")
        invitation_code = str(form.get("invitation") or "")
        invitation = sso.digest(invitation_code) if invitation_code else None
        pending_id = str(form.get("request") or "") or None
        if invitation:
            record = invitation_for(request, invitation)
            if (
                record["providerId"] != provider["id"]
                or not binding
                or not hmac.compare_digest(binding, str(form.get("binding", "")))
            ):
                raise sso.Refused("The invitation does not match this browser or provider.")
        else:
            pending = local._provider(request).pending(pending_id or "")
            if (
                not pending
                or pending.issuer != issuer
                or not binding
                or not hmac.compare_digest(pending.binding, binding)
            ):
                raise sso.Refused("This sign-in expired. Start again from the app.")
        metadata = await sso.metadata(provider)
        flow = sso.Flow(
            provider["id"],
            provider["revision"],
            binding,
            secrets.token_urlsafe(48),
            secrets.token_urlsafe(32),
            issuer + "/external/callback",
            metadata,
            pending_id,
            invitation,
            time.time() + sso.FLOW_SECONDS,
        )
        state = flows(request).add(flow)
        return RedirectResponse(
            sso.authorization_url(flow, provider, state),
            status_code=303,
            headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"},
        )
    except sso.Refused as exc:
        return refused(exc)


@router.get("/oidc/external/callback")
async def callback(request: Request) -> Response:
    issuer = local._issuer(request)
    flow = flows(request).take(request.query_params.get("state", ""))
    try:
        if any(
            len(request.query_params.getlist(key)) > 1 for key in ("state", "code", "iss", "error")
        ):
            raise sso.Refused("The provider returned an ambiguous response. Start sign-in again.")
        if (
            not flow
            or flow.callback != issuer + "/external/callback"
            or not hmac.compare_digest(
                flow.binding, request.cookies.get(sso.cookie_name(issuer, flow=True), "")
            )
        ):
            raise sso.Refused(
                "This sign-in was started in another browser, expired, or already used."
            )
        provider = provider_for(request, flow.provider_id)
        if provider["revision"] != flow.revision:
            raise sso.Refused("The provider settings changed. Start sign-in again.")
        if request.query_params.get("error") or not request.query_params.get("code"):
            raise sso.Refused(
                "The provider did not complete sign-in. "
                "You can try again or use your Eugene password."
            )
        if request.query_params.get("iss") not in (None, provider["issuer"]):
            raise sso.Refused("The provider's response issuer does not match.")
        identity = await sso.finish(
            provider, flow, request.query_params["code"], request.app.state.auth_state.master_key
        )
        # Re-read authority after network I/O, including changes made while awaiting.
        machine = machine_for(request)
        if provider_for(request, flow.provider_id)["revision"] != flow.revision:
            raise sso.Refused("The provider settings changed. Start sign-in again.")
        existing = next(
            (
                r
                for r in machine.state.browser_sign_in["identities"].values()
                if (r["issuer"], r["subject"]) == (identity["issuer"], identity["subject"])
            ),
            None,
        )
        if flow.invitation:
            invitation = invitation_for(request, flow.invitation)
            if existing:
                raise sso.Refused(
                    "This provider account is already connected. "
                    "Sign in normally or ask the owner to review its connection."
                )
            identity.update(
                id=uuid.uuid4().hex,
                providerId=provider["id"],
                personId=invitation["personId"],
                createdAt=time.time(),
            )
            machine.append(
                "linkExternalIdentity",
                {"record": identity, "invitationId": flow.invitation},
                actor=identity["personId"],
            )
        elif existing and existing["providerId"] == provider["id"]:
            identity = existing
        else:
            raise sso.Refused(
                "This provider account is not connected to Eugene. "
                "Ask the owner for an invitation. "
                "Matching email addresses do not connect accounts."
            )
        person = machine.state.people.get(identity["personId"])
        if not person or person["disabled"]:
            raise sso.Refused("This Eugene account cannot sign in. Ask the owner.")
        secret, session = sso.new_session(request, issuer, person["id"], identity=identity)
        response = (
            local.complete_sign_in(
                request, flow.pending, oidc.subject_of(person), session["authAt"], session["id"]
            )
            if flow.pending
            else RedirectResponse(issuer + "/account", status_code=303)
        )
        sso.set_cookie(response, issuer, secret)
        return response
    except sso.Refused as exc:
        return refused(exc)


@router.get("/oidc/account")
async def account(request: Request) -> Response:
    issuer = local._issuer(request)
    session = sso.browser_session(request, issuer)
    if session is None:
        # This internal client only sets a browser cookie; it grants no new privileges.
        verifier = secrets.token_urlsafe(48)
        return RedirectResponse(
            oidc.redirect_with(
                issuer + "/authorize",
                {
                    "client_id": sso.ACCOUNT_CLIENT,
                    "response_type": "code",
                    "scope": "openid profile",
                    "redirect_uri": issuer + "/account",
                    "state": secrets.token_urlsafe(24),
                    "nonce": secrets.token_urlsafe(24),
                    "code_challenge": oidc.b64url(hashlib.sha256(verifier.encode()).digest()),
                    "code_challenge_method": "S256",
                },
            ),
            status_code=303,
        )
    state = machine_for(request).state
    person = state.people[session["personId"]]
    e = html.escape
    body = [
        f"<p>Signed in as <strong>{e(person.get('displayName') or person['name'])}</strong>.</p>",
        "<p class='hint'>Your existing chats and access stay with your Eugene account. "
        "For a forgotten password or a lost provider account, "
        "ask the owner to reset your password.</p>",
        "<h2>Connected accounts</h2>",
    ]
    for identity in state.browser_sign_in["identities"].values():
        if identity["personId"] == person["id"]:
            provider = state.browser_sign_in["providers"][identity["providerId"]]
            body.append(f"<p>{e(provider['name'])}</p>")
    body.append(
        "<p class='hint'>The owner can disconnect a provider or issue an invitation "
        "to connect another.</p><h2>Browser sign-ins</h2>"
    )
    for record in state.browser_sign_in["sessions"].values():
        if record["personId"] == person["id"] and sso.valid_session(state, record["id"]):
            current = "This browser" if record["id"] == session["id"] else record["label"]
            body.append(
                f"<form class='entry' method='post'><p>{e(current)}</p>"
                f"<input type='hidden' name='csrf' value='{sso.csrf(request, issuer)}'>"
                f"<input type='hidden' name='session' value='{record['id']}'>"
                "<button>End this sign-in</button></form>"
            )
    body.append(
        "<form method='post'>"
        f"<input type='hidden' name='csrf' value='{sso.csrf(request, issuer)}'>"
        "<input type='hidden' name='session' value='all'>"
        "<button>End all shared sign-ins</button></form>"
        "<p class='hint'>This ends shared Eugene sign-ins. It does not sign you out of Google, "
        "Microsoft or your identity provider. Apps may take up to ten minutes to remove an "
        "already-open page. Older app sessions and direct password-based administration "
        "sessions use their own sign-out controls.</p>"
        f"<p><a href='{e(issuer)}/invitation'>Use an invitation</a></p>"
    )
    if has_grants(person):
        body.append(
            f"<p><a href='{e(issuer.removesuffix('/oidc'))}/administration/'>"
            "Manage people and access</a></p>"
        )
    return page("Your sign-in and sessions", "".join(body))


@router.post("/oidc/account")
async def account_action(request: Request) -> Response:
    issuer = local._issuer(request)
    session = sso.browser_session(request, issuer)
    form = await request.form()
    if not session or not hmac.compare_digest(sso.csrf(request, issuer), str(form.get("csrf", ""))):
        return refused(sso.Refused("This account session expired. Sign in again."))
    machine = machine_for(request)
    chosen = str(form.get("session", ""))
    ended_current = False
    for record in list(machine.state.browser_sign_in["sessions"].values()):
        if record["personId"] == session["personId"] and chosen in ("all", record["id"]):
            machine.append("deleteBrowserSignIn", {"id": record["id"]}, actor=session["personId"])
            ended_current |= record["id"] == session["id"]
    if ended_current:
        response = page(
            "Signed out of Eugene",
            "<p>Your selected Eugene sign-ins have ended.</p>"
            "<p><a href='/oidc/account'>Sign in again</a></p>",
        )
        response.delete_cookie(
            sso.cookie_name(issuer),
            path="/oidc",
            secure=issuer.startswith("https:"),
            httponly=True,
            samesite="lax",
        )
        return response
    return RedirectResponse(issuer + "/account", status_code=303)


@router.post(
    "/v1/auth/administration/code",
    response_model=AdministrationLoginResponse,
    dependencies=[Depends(require_trusted_host)],
)
async def administration_code(
    request: Request, body: AdministrationCodeRequest, response: Response
) -> AdministrationLoginResponse:
    from .administration import issue_session

    machine = machine_for(request)
    code = local._provider(request).take_code(body.code)
    if (
        not code
        or code.client_id != sso.ADMIN_CLIENT
        or code.redirect_uri != body.redirectUri
        or not oidc.pkce_matches(body.verifier, code.challenge)
        or not code.browser_session
    ):
        raise problem(401, "Sign-in expired", "Start administrator sign-in again.")
    session = sso.valid_session(machine.state, code.browser_session, code.subject.sub)
    person = machine.state.people.get(code.subject.sub)
    if not session or not person or not has_grants(person):
        raise problem(
            403,
            "Administration access required",
            "The owner must grant administration access to this account.",
        )
    response.headers["Cache-Control"] = "no-store"
    return issue_session(request, person, browser_session=code.browser_session)
