"""Job Sites: their enrollment, their channel to this root, the console's
membership view, and Workbench's routes (`docs/design/job-sites-own-enrollment.md`).

**A site is its own enrollment** (J19, J23). Its site host joins with a site
invitation (`/v1/sites/enroll`), confirmed at the machine by the person it
names, and from then on polls, claims and answers with its own token, which
this root checks against the replicated registry and never the trust bundle
(`site_tokens`). A site has no address; nothing here ever dials one.

**Membership is not access** (J11, J19). Eugene's owner invites a site,
removes it, and sees that it exists and whether it is online (`/v1/sites`,
`membership`). In dev mode the owner also sees what the site reported and may
give themselves its folders (J13b, J33), which works only if the site's owner
let them in there (J6e). Everything else is the site's owner's, through
Workbench (`/oidc/job-sites`), relayed to the site as management actions it
takes from that owner alone (J6b). A person uses a site's tools through
`/oidc/sites`; the site's own policy decides (J8).
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import Annotated, Any

import jwt
from fastapi import APIRouter, Depends, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials

from .. import capabilities, oidc, site_tokens
from .. import sites as helpers
from .._generated.models import (
    HostedSites,
    JobSiteAccessRequest,
    JobSiteAuditRequest,
    JobSiteFolderCreate,
    JobSiteInviteRequest,
    JobSitePeopleRequest,
    JobSiteRequest,
    JobSiteServerEnable,
    JobSiteSettings,
    SiteCancel,
    SiteEnrollmentRequest,
    SiteFolderOwnerAccess,
    SiteInvitationRequest,
    SiteMcpCall,
    SiteReport,
    SiteResult,
)
from ..applied import (
    OP_ENROLL_SITE,
    OP_REMOVE_SITE,
    OP_SET_SITE_DEV_GRANTS,
    OP_SET_SITE_HOST,
    SiteRecord,
)
from ..dependencies import (
    _bearer,
    _bearer_scheme,
    problem,
    require_node_actor,
    via_public_sites,
)
from ..join_tokens import JoinTokenConsumed, JoinTokenError, JoinTokenStore
from ..state_machine import NotActive, StateMachine
from ..tokens import Claims
from .nodes import (
    JOIN_FAILURE_BUCKET,
    JOIN_FAILURES,
    JOIN_WINDOW_SECONDS,
    _confirm_site_owner,
    _not_active,
    _record_join_failure,
)
from .oidc import _client_auth
from .people import _active


def _private_response(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"


log = logging.getLogger(__name__)
router = APIRouter(tags=["sites"], dependencies=[Depends(_private_response)])
MAX_REQUEST = 40_000
MAX_RESULT = 70_000
INVITES_PER_PERSON = 3
INVITE_SECONDS = 900


def require_site(
    request: Request,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
) -> SiteRecord:
    """A site speaking for itself, with a token its own key signed."""
    machine: StateMachine = request.app.state.machine
    try:
        record = site_tokens.verify(_bearer(request, creds), machine.state.sites)
    except site_tokens.SiteTokenError as exc:
        raise problem(status.HTTP_401_UNAUTHORIZED, "Invalid site token", str(exc)) from None
    request.app.state.site_contacts[record.id] = datetime.now(UTC)
    return record


SiteActor = Annotated[SiteRecord, Depends(require_site)]
NodeActor = Annotated[Claims, Depends(require_node_actor)]


def _contact(request: Request, site: str) -> str | None:
    seen = request.app.state.site_contacts.get(site)
    return seen.isoformat() if seen else None


def _remove(request: Request, site: str) -> None:
    machine: StateMachine = request.app.state.machine
    if site not in machine.state.sites:
        return
    try:
        machine.append(OP_REMOVE_SITE, {"id": site})
    except NotActive as exc:
        raise _not_active(machine) from exc
    helpers.broker(request).forget(site)
    request.app.state.site_contacts.pop(site, None)
    log.warning("site %s removed", site)


def _join_url(request: Request) -> str | None:
    """Where a machine joins: the entry point's nodes name, else the address
    Eugene's owner set (`siteJoinUrl`). A machine on this root's own network
    joins through its LAN address, so no public route is needed (J31)."""
    settings = getattr(request.app.state, "settings", None)
    origin = getattr(settings, "nodes_origin", None)
    if origin:
        return str(origin).rstrip("/")
    configured = request.app.state.machine.state.config.get("siteJoinUrl")
    return str(configured).rstrip("/") if configured else None


# --------------------------------------------------------------------------- #
# A site host: join, poll, claim, answer, leave (J23)
# --------------------------------------------------------------------------- #


@router.post("/v1/sites/enroll", status_code=201)
async def enroll(request: Request, body: SiteEnrollmentRequest) -> dict[str, Any]:
    machine: StateMachine = request.app.state.machine
    auth = request.app.state.auth_state
    store: JoinTokenStore = request.app.state.join_tokens
    if not machine.is_active:
        raise _not_active(machine)
    try:
        if len(base64.b64decode(body.tokenPublicKey, validate=True)) != 32:
            raise ValueError("expected 32 bytes")
    except Exception as exc:
        raise problem(
            400, "Malformed public key", f"tokenPublicKey must be base64 of a 32-byte key ({exc})."
        ) from exc
    state = machine.state
    if any(n.tokenPublicKey == body.tokenPublicKey for n in state.nodes.values()) or any(
        s.tokenPublicKey == body.tokenPublicKey for s in state.sites.values()
    ):
        raise problem(
            400,
            "Token key already enrolled",
            "Another site or node already holds this token key. Each site generates its own.",
        )
    if auth.signing_key is None:
        raise problem(
            503,
            "Locked",
            "This control root has not been unlocked. Sign in to it first; the invitation was "
            "not used.",
        )
    public = via_public_sites(request)
    if public and auth.is_login_rate_limited(
        JOIN_FAILURE_BUCKET, window_seconds=JOIN_WINDOW_SECONDS, max_in_window=JOIN_FAILURES
    ):
        raise problem(
            429,
            "Too many failed joins",
            f"Too many unknown invitations arrived from outside. Wait {JOIN_WINDOW_SECONDS} "
            "seconds and try again.",
        )
    try:
        invitation = store.peek(body.token, node_name=None)
    except JoinTokenConsumed as exc:
        raise problem(409, "Invitation already used", str(exc)) from exc
    except JoinTokenError as exc:
        if public:
            _record_join_failure(request)
        raise problem(401, "Invitation rejected", str(exc)) from exc
    if invitation.owner is None:
        if public:
            _record_join_failure(request)
        raise problem(
            401,
            "Not a site invitation",
            "This join token adds a node, not a job site. Ask for a job site invitation.",
        )
    owner = _confirm_site_owner(request, invitation.owner, body.owner)
    try:
        store.consume(body.token, node_name=None)
    except JoinTokenConsumed as exc:
        raise problem(409, "Invitation already used", str(exc)) from exc
    except JoinTokenError as exc:
        raise problem(401, "Invitation rejected", str(exc)) from exc

    site = helpers.new_site_id()
    label = str(body.label)
    enrolled_at = datetime.now(UTC).isoformat()
    try:
        machine.append(
            OP_ENROLL_SITE,
            {
                "id": site,
                "label": label,
                "owner": owner,
                "tokenPublicKey": body.tokenPublicKey,
                "enrolledAt": enrolled_at,
            },
        )
    except NotActive as exc:
        raise _not_active(machine) from exc
    log.info(
        "site %s (%s) joined, owned by %s", site, label, helpers.person_name(machine.state, owner)
    )
    return {
        "id": site,
        "label": label,
        "owner": owner,
        "ownerName": helpers.person_name(machine.state, owner),
        "controlPublicKey": machine.state.identity.controlPublicKey or "",
        "enrolledAt": enrolled_at,
    }


@router.post("/v1/sites/poll")
async def poll(request: Request, body: SiteReport, site: SiteActor) -> dict[str, Any]:
    _active(request)
    report = body.model_dump(mode="json", exclude_none=True)
    ident = await helpers.broker(request).poll(helpers.binding(site), report)
    return {"operation": ident}


@router.post("/v1/sites/operations/{ident}/claim")
async def claim(request: Request, ident: str, site: SiteActor) -> dict[str, Any]:
    return helpers.broker(request).claim(site.id, ident)


@router.post("/v1/sites/operations/{ident}/result", status_code=204)
async def result(request: Request, ident: str, body: SiteResult, site: SiteActor) -> None:
    value = body.model_dump(mode="json", exclude_none=True)
    if len(json.dumps(value, ensure_ascii=False).encode()) > MAX_RESULT:
        raise problem(413, "Result too large", "A site's answer may be at most 70,000 bytes.")
    helpers.broker(request).finish(site.id, ident, value)


@router.post("/v1/sites/leave", status_code=204)
async def leave(request: Request, site: SiteActor) -> None:
    """Leaving needs no one's permission (remote-nodes.md §3.3)."""
    _active(request)
    _remove(request, site.id)


# --------------------------------------------------------------------------- #
# The console: membership (J19), and dev mode's view (J33)
# --------------------------------------------------------------------------- #


def _dev_view(request: Request, record: SiteRecord) -> dict[str, Any]:
    summary = helpers.broker(request).summary(helpers.binding(record))
    owned = {g["folderId"]: g for g in record.devGrants}
    return {
        "ownerInDevMode": summary.get("ownerInDevMode") if summary else None,
        "folders": [
            {
                "id": folder["id"],
                "name": folder["name"],
                "path": folder["path"],
                "writable": folder["writable"],
                "ownerAccess": (
                    "none"
                    if folder["id"] not in owned
                    else ("write" if owned[folder["id"]]["writable"] else "read")
                ),
            }
            for folder in (summary or {}).get("folders") or []
        ],
        "servers": list((summary or {}).get("servers") or []),
    }


def _console_entry(request: Request, record: SiteRecord) -> dict[str, Any]:
    state = request.app.state.machine.state
    delivery = helpers.broker(request)
    bound = helpers.binding(record)
    status_ = delivery.status(bound)
    report = delivery.report(bound) or {}
    return {
        "id": record.id,
        "label": record.label,
        "owner": record.owner,
        "ownerName": helpers.person_name(state, record.owner),
        "online": bool(status_.get("online")),
        "lastContactAt": _contact(request, record.id),
        "ready": bool(status_.get("ready")),
        "reason": status_.get("reason"),
        "hostVersion": report.get("hostVersion"),
        "hostNode": record.hostNode,
        "enrolledAt": record.enrolledAt,
        "dev": _dev_view(request, record) if helpers.install_mode(state) == helpers.DEV else None,
    }


@router.get("/v1/sites", dependencies=[Depends(capabilities.require(capabilities.MEMBERSHIP))])
async def list_sites(request: Request) -> dict[str, Any]:
    state = request.app.state.machine.state
    return {
        "sites": [
            _console_entry(request, state.sites[site])
            for site in sorted(state.sites, key=lambda s: (state.sites[s].label.casefold(), s))
        ],
        "installMode": helpers.mode_view(state),
        "joinUrl": _join_url(request),
    }


@router.post(
    "/v1/sites/invitations",
    status_code=201,
    dependencies=[Depends(capabilities.require(capabilities.MEMBERSHIP))],
)
async def invite_site(request: Request, body: SiteInvitationRequest) -> dict[str, Any]:
    machine: StateMachine = request.app.state.machine
    if not machine.is_active:
        raise _not_active(machine)
    person = machine.state.people.get(body.owner)
    if person is None or person.get("disabled"):
        raise problem(
            422,
            "Name a person who can sign in",
            "A job site belongs to a person on this install, who confirms the join at the "
            "machine with their own password.",
        )
    ttl = body.ttlSeconds or int(machine.state.config.get("joinTokenTtlSeconds") or 900)
    label = body.label
    minted = request.app.state.join_tokens.mint(
        ttl_seconds=ttl, node_name=label, owner=str(person["id"])
    )
    log.info("minted a job site invitation for %s", person["name"])
    return {
        "id": minted.id,
        "token": minted.token,
        "expiresAt": datetime.fromtimestamp(minted.expires_at, tz=UTC).isoformat(),
        "owner": person["id"],
        "ownerName": person["name"],
        "label": label,
        "joinUrl": _join_url(request),
        "rootKey": machine.state.identity.controlPublicKey or "",
    }


@router.delete(
    "/v1/sites/{site}",
    status_code=204,
    dependencies=[Depends(capabilities.require(capabilities.MEMBERSHIP))],
)
async def remove_site(request: Request, site: str) -> None:
    machine: StateMachine = request.app.state.machine
    if not machine.is_active:
        raise _not_active(machine)
    _remove(request, site)


@router.patch(
    "/v1/sites/{site}/folders/{folder_id}",
    dependencies=[Depends(capabilities.require(capabilities.DEV_ACCESS))],
)
async def set_owner_access(
    request: Request, site: str, folder_id: str, body: SiteFolderOwnerAccess
) -> dict[str, Any]:
    """J13b: Eugene's owner's own access to a site's folder, in dev mode only."""
    machine: StateMachine = request.app.state.machine
    if not machine.is_active:
        raise _not_active(machine)
    if helpers.install_mode(machine.state) != helpers.DEV:
        raise problem(
            409,
            "Production mode",
            "Eugene is in production mode, where its owner gives themselves nothing on a job site.",
        )
    record = machine.state.sites.get(site)
    summary = helpers.broker(request).summary(helpers.binding(record)) if record else None
    folder = next((f for f in (summary or {}).get("folders") or [] if f["id"] == folder_id), None)
    if record is None or folder is None:
        raise problem(404, "No such folder", "That site has not reported that folder.")
    access = str(body.ownerAccess.value)
    if access == "write" and not folder["writable"]:
        raise problem(422, "Read-only folder", "That folder was registered for reading only.")
    grants = [g for g in record.devGrants if g["folderId"] != folder_id]
    if access != "none":
        grants.append({"folderId": folder_id, "writable": access == "write"})
    try:
        machine.append(OP_SET_SITE_DEV_GRANTS, {"id": site, "devGrants": grants})
    except NotActive as exc:
        raise _not_active(machine) from exc
    return _console_entry(request, machine.state.sites[site])


@router.put("/v1/nodes/{name}/hosted-sites", status_code=204)
async def hosted_sites(request: Request, name: str, body: HostedSites, actor: NodeActor) -> None:
    """J32: a node says which sites its agent supervises. Display only."""
    machine: StateMachine = request.app.state.machine
    if actor.issuer_node != name:
        raise problem(403, "Not this node", "A node speaks only for itself here.")
    if not machine.is_active:
        raise _not_active(machine)
    listed = {str(getattr(s, "root", s)) for s in body.sites}
    try:
        for site, record in sorted(machine.state.sites.items()):
            if site in listed and record.hostNode != name:
                machine.append(OP_SET_SITE_HOST, {"id": site, "hostNode": name})
            elif site not in listed and record.hostNode == name:
                machine.append(OP_SET_SITE_HOST, {"id": site, "hostNode": None})
    except NotActive as exc:
        raise _not_active(machine) from exc


# --------------------------------------------------------------------------- #
# Workbench: the servers a person may use, and one MCP request to one
# --------------------------------------------------------------------------- #


def _app_subject(request: Request, token: str) -> str:
    machine = _active(request)
    if request.app.state.auth_state.signing_key is None:
        raise problem(503, "Eugene is locked", "Unlock Eugene before using job sites' tools.")
    client = _client_auth(request)
    if client is None:
        raise problem(401, "App sign-in required", "Workbench's client authentication failed.")
    if not str(client.get("owner") or "").startswith("app:workbench@"):
        raise problem(403, "Workbench required", "This app is not an enrolled Workbench.")
    try:
        claims = oidc.Provider.decode(
            token, machine.state, typ=oidc.TYP_REFRESH, audience=client["clientId"]
        )
    except jwt.InvalidTokenError:
        raise problem(
            401, "Sign-in ended", "Sign into Workbench again to use job sites' tools."
        ) from None
    subject = oidc.still_valid(machine.state, claims, client["clientId"])
    if subject is None:
        raise problem(
            403, "Access withdrawn", "Your sign-in or Workbench access has been withdrawn."
        )
    return subject.sub


@router.post("/oidc/sites/servers")
async def servers(request: Request, body: JobSiteRequest) -> dict[str, Any]:
    subject = _app_subject(request, body.refreshToken)
    state = request.app.state.machine.state
    delivery = helpers.broker(request)
    listed: list[dict[str, Any]] = []
    for site in sorted(state.sites):
        record = state.sites[site]
        bound = helpers.binding(record)
        available, reason = delivery.availability(bound)
        summary = delivery.summary(bound)
        common = {"site": site, "label": record.label, "available": available, "reason": reason}
        if subject == helpers.OPERATOR:
            if not (summary or {}).get("ownerInDevMode"):
                continue
            folders = [
                {"id": g["folderId"], "name": g["name"], "writable": g["writable"]}
                for g in helpers.dev_grants(state, record, summary)
            ]
        else:
            folders = helpers.site_folders_for(summary, subject)
        if folders:
            listed.append(
                {
                    **common,
                    "server": helpers.FILES,
                    "name": "Files",
                    "kind": "files",
                    "folders": folders,
                }
            )
        if subject == helpers.OPERATOR:
            continue
        for server in helpers.site_local_servers_for(summary, subject):
            listed.append(
                {
                    **common,
                    "server": server["id"],
                    "name": server["name"],
                    "kind": "local",
                    "available": available and bool(server.get("available")),
                    "reason": reason or server.get("reason"),
                    "folders": [],
                }
            )
    return {"servers": listed, "installMode": helpers.mode_view(state)}


def _named_tool(request_body: dict[str, Any]) -> str | None:
    if request_body.get("method") != "tools/call":
        return None
    params = request_body.get("params") or {}
    tool = params.get("name") if isinstance(params, dict) else None
    return tool if isinstance(tool, str) else None


def _same_site(machine: StateMachine, bound: dict[str, str]) -> SiteRecord:
    record = machine.state.sites.get(bound["site"])
    if record is None or record.enrolledAt != bound["enrolledAt"]:
        raise problem(409, "Site changed", "This site left or joined again. Choose it again.")
    return record


@router.post("/oidc/sites/mcp")
async def mcp(request: Request, body: SiteMcpCall) -> dict[str, Any]:
    """One MCP request to one server on one site. The root checks the
    Workbench client and the person's sign-in; the site's own policy
    decides (J8)."""
    _app_subject(request, body.refreshToken)
    request_body = body.request.model_dump(mode="json", exclude_none=True, by_alias=True)
    if len(json.dumps(request_body, ensure_ascii=False).encode()) > MAX_REQUEST:
        raise problem(413, "Request too large", "Send a smaller request (40,000 bytes at most).")
    machine = _active(request)
    site, server = (
        str(getattr(body.site, "root", body.site)),
        str(getattr(body.server, "root", body.server)),
    )
    record = machine.state.sites.get(site)
    if record is None:
        raise problem(403, "Not granted", "You have nothing on that site.")
    bound = helpers.binding(record)
    tool = _named_tool(request_body)

    def authorized() -> tuple[str, list[dict[str, Any]]]:
        """Who asks, and the grants their call carries; checked at submit,
        at the claim and again before an answer is released."""
        subject = _app_subject(request, body.refreshToken)
        current = _same_site(machine, bound)
        summary = helpers.broker(request).summary(bound)
        if subject == helpers.OPERATOR:
            if helpers.install_mode(machine.state) != helpers.DEV:
                raise problem(
                    403,
                    "Production mode",
                    "Eugene is in production mode: its owner's own access to a job site does "
                    "not work.",
                )
            grants = helpers.dev_grants(machine.state, current, summary)
            if server != helpers.FILES or not grants:
                raise problem(403, "Not granted", "You have nothing on that site.")
            return subject, grants
        listed = (
            bool(helpers.site_folders_for(summary, subject))
            if server == helpers.FILES
            else any(s["id"] == server for s in helpers.site_local_servers_for(summary, subject))
        )
        if not listed:
            raise problem(403, "Not granted", "You have nothing on that site.")
        return subject, []

    def check() -> dict[str, Any]:
        subject, grants = authorized()
        return helpers.envelope(
            machine.state, bound, subject, server=server, request=request_body, grants=grants
        )

    operation_id = _operation_id(body.refreshToken, body.operationId) if body.operationId else None
    outcome = await helpers.broker(request).submit(
        bound, check, write=tool is not None, operation_id=operation_id
    )
    # Workbench keeps the mode with the result: what production mode hides
    # from the owner's chat reading is decided by it (J13a, J18).
    answer: dict[str, Any] = {
        "status": outcome.get("status", "failed"),
        "installMode": helpers.install_mode(machine.state),
    }
    for key in ("message", "response"):
        if outcome.get(key) is not None:
            answer[key] = outcome[key]
    return answer


def _operation_id(token: str, ident: str) -> str:
    return hashlib.sha256((token + "\0" + ident).encode()).hexdigest()


@router.post("/oidc/sites/cancel", status_code=204)
async def cancel(request: Request, body: SiteCancel) -> None:
    _app_subject(request, body.refreshToken)
    helpers.broker(request).cancel(_operation_id(body.refreshToken, body.operationId))


@router.post("/oidc/install-mode")
async def install_mode(request: Request) -> dict[str, Any]:
    """The mode, for an app to show every person it serves (J18). Any enrolled
    app client: the mode is the same for everyone and is not a secret."""
    if _client_auth(request) is None:
        raise problem(401, "App sign-in required", "The app's client authentication failed.")
    return helpers.mode_view(request.app.state.machine.state)


# --------------------------------------------------------------------------- #
# A site's owner (J9, J11, J6b), through Workbench with their own sign-in
# --------------------------------------------------------------------------- #


def _own_site(request: Request, token: str, site: str) -> tuple[StateMachine, str, SiteRecord]:
    """The signed-in person and their site, or 404.

    One answer for "no such site" and "not yours", so a person cannot learn
    which sites other people own by asking."""
    subject = _app_subject(request, token)
    machine = _active(request)
    record = machine.state.sites.get(site)
    if record is None or record.owner != subject:
        raise problem(404, "No such job site", "You have no such job site.")
    return machine, subject, record


def _person_id(state: Any, name: str) -> str:
    by_name = {p["name"].casefold(): p for p in state.people.values()}
    person = by_name.get(name.strip().casefold())
    if person is None:
        raise problem(404, "No such person", f"Nobody on this install signs in as {name!r}.")
    return str(person["id"])


def _folder_view(state: Any, folder: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": folder["id"],
        "name": folder["name"],
        "path": folder["path"],
        "writable": folder["writable"],
        "people": [
            {
                "person": p["subject"],
                "name": helpers.person_name(state, p["subject"]),
                "writable": p["writable"],
            }
            for p in folder.get("people") or []
        ],
    }


def _server_view(
    state: Any, server: dict[str, Any], people: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "server": server,
        "people": [
            {
                "person": p["subject"],
                "name": helpers.person_name(state, p["subject"]),
                "tools": p["tools"],
            }
            for p in people
        ],
    }


def _site_view(request: Request, record: SiteRecord) -> dict[str, Any]:
    state = request.app.state.machine.state
    delivery = helpers.broker(request)
    bound = helpers.binding(record)
    status_ = delivery.status(bound)
    summary = delivery.summary(bound) or {}
    access = summary.get("access") or []
    return {
        "id": record.id,
        "label": record.label,
        "online": bool(status_.get("online")),
        "ready": bool(status_.get("ready")),
        "reason": status_.get("reason"),
        "account": status_.get("account"),
        "lastContactAt": _contact(request, record.id),
        "hostNode": record.hostNode,
        "folders": [_folder_view(state, f) for f in summary.get("folders") or []],
        "servers": [
            _server_view(
                state,
                server,
                [
                    {"subject": e["subject"], "tools": e["tools"]}
                    for e in access
                    if e["server"] == server["id"]
                ],
            )
            for server in summary.get("servers") or []
            if server["kind"] == "local"
        ],
        "ownerInDevMode": summary.get("ownerInDevMode") if summary else None,
    }


async def _manage(
    request: Request, token: str, site: str, action: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """One management action, relayed to the site, which takes it from the
    owner it recorded at its join and nobody else (J6b). Its refusal comes
    back as 422 in the site's own words."""
    machine, subject, record = _own_site(request, token, site)
    bound = helpers.binding(record)

    def check() -> dict[str, Any]:
        _own_site(request, token, site)
        _same_site(machine, bound)
        return helpers.envelope(
            machine.state, bound, subject, kind="manage", action=action, arguments=arguments
        )

    outcome = await helpers.broker(request).submit(bound, check)
    if outcome.get("status") != "done":
        raise problem(
            422,
            "The site refused it",
            outcome.get("message") or "The site did not take this change.",
        )
    value = outcome.get("result")
    return value if isinstance(value, dict) else {}


@router.post("/oidc/job-sites")
async def my_sites(request: Request, body: JobSiteRequest) -> dict[str, Any]:
    subject = _app_subject(request, body.refreshToken)
    state = request.app.state.machine.state
    return {
        "sites": [
            _site_view(request, state.sites[site])
            for site in sorted(state.sites)
            if state.sites[site].owner == subject
        ],
        "installMode": helpers.mode_view(state),
        "canInvite": subject != helpers.OPERATOR and _join_url(request) is not None,
    }


@router.post("/oidc/job-sites/invite")
async def invite(request: Request, body: JobSiteInviteRequest) -> dict[str, Any]:
    """J9: any signed-in person may add a Job Site for a machine of their own."""
    subject = _app_subject(request, body.refreshToken)
    machine = _active(request)
    if subject == helpers.OPERATOR:
        raise problem(
            403,
            "Job sites belong to people",
            "Eugene's owner signs in with the passphrase, which owns no job sites. Add "
            "yourself as a person on Eugene's People page, sign in to Workbench as them, "
            "and add the machine from there.",
        )
    join_url = _join_url(request)
    if join_url is None:
        raise problem(
            409,
            "No address for machines yet",
            "Eugene's owner has not said what address machines join through. In Eugene's "
            "Settings, set the address machines join through, or give the entry point a "
            "name for machines.",
        )
    store: JoinTokenStore = request.app.state.join_tokens
    if store.outstanding_for(subject) >= INVITES_PER_PERSON:
        raise problem(
            429,
            "Too many open invitations",
            f"You have {INVITES_PER_PERSON} unused job site invitations. Use one, or wait "
            f"{INVITE_SECONDS // 60} minutes for them to expire.",
        )
    label = body.label
    minted = store.mint(ttl_seconds=INVITE_SECONDS, node_name=label, owner=subject)
    log.info("%s minted a job site invitation", helpers.person_name(machine.state, subject))
    return {
        "token": minted.token,
        "expiresAt": datetime.fromtimestamp(minted.expires_at, tz=UTC).isoformat(),
        "label": label,
        "joinUrl": join_url,
        "rootKey": machine.state.identity.controlPublicKey or "",
        "owner": helpers.person_name(machine.state, subject),
    }


@router.post("/oidc/job-sites/{site}/folders", status_code=201)
async def site_add_folder(request: Request, site: str, body: JobSiteFolderCreate) -> dict[str, Any]:
    name, path = body.name.strip(), body.path.strip()
    if not name or not path or any(ord(c) < 32 for c in name + path):
        raise problem(
            422, "Invalid folder", "Use a nonempty folder name and path without control characters."
        )
    folder = await _manage(
        request,
        body.refreshToken,
        site,
        "folder.add",
        {"name": name, "path": path, "writable": bool(body.writable)},
    )
    return _folder_view(request.app.state.machine.state, folder)


@router.post("/oidc/job-sites/{site}/folders/{folder_id}/remove", status_code=204)
async def site_remove_folder(
    request: Request, site: str, folder_id: str, body: JobSiteRequest
) -> None:
    await _manage(request, body.refreshToken, site, "folder.remove", {"id": folder_id})


@router.post("/oidc/job-sites/{site}/folders/{folder_id}/people")
async def site_folder_people(
    request: Request, site: str, folder_id: str, body: JobSitePeopleRequest
) -> dict[str, Any]:
    """J11: the site's owner says who may use a folder, themselves included,
    and who may change files in it; the site keeps the list (J6b, J6g)."""
    _own_site(request, body.refreshToken, site)
    state = request.app.state.machine.state
    people = [
        {"subject": _person_id(state, entry.name), "writable": bool(entry.writable)}
        for entry in body.people
    ]
    folder = await _manage(
        request, body.refreshToken, site, "folder.people", {"id": folder_id, "people": people}
    )
    return _folder_view(request.app.state.machine.state, folder)


@router.post("/oidc/job-sites/{site}/servers/{server}/access")
async def site_server_access(
    request: Request, site: str, server: str, body: JobSiteAccessRequest
) -> dict[str, Any]:
    """Who may use which tools of one local server (J6b)."""
    _own_site(request, body.refreshToken, site)
    state = request.app.state.machine.state
    people = [
        {
            "subject": _person_id(state, entry.name),
            "tools": [t.model_dump(mode="json", exclude_none=True) for t in entry.tools],
        }
        for entry in body.people
    ]
    value = await _manage(
        request, body.refreshToken, site, "access.set", {"server": server, "people": people}
    )
    return _server_view(request.app.state.machine.state, value["server"], value["people"])


@router.post("/oidc/job-sites/{site}/servers/{server}/enabled")
async def site_server_enabled(
    request: Request, site: str, server: str, body: JobSiteServerEnable
) -> dict[str, Any]:
    value = await _manage(
        request,
        body.refreshToken,
        site,
        "server.enable",
        {"server": server, "enabled": body.enabled},
    )
    return _server_view(request.app.state.machine.state, value["server"], value["people"])


@router.post("/oidc/job-sites/{site}/settings")
async def site_settings(request: Request, site: str, body: JobSiteSettings) -> dict[str, Any]:
    """J6e: let Eugene's owner in while Eugene is in dev mode. Off by
    default; dev mode alone opens nothing on a site."""
    value = await _manage(
        request, body.refreshToken, site, "settings.set", {"ownerInDevMode": body.ownerInDevMode}
    )
    _, _, record = _own_site(request, body.refreshToken, site)
    return {**_site_view(request, record), "ownerInDevMode": value.get("ownerInDevMode")}


@router.post("/oidc/job-sites/{site}/audit")
async def site_audit(request: Request, site: str, body: JobSiteAuditRequest) -> dict[str, Any]:
    """The site's own audit log, read from the machine for its owner alone."""
    value = await _manage(
        request, body.refreshToken, site, "audit.read", {"limit": body.limit or 50}
    )
    return {"entries": value.get("entries") or []}


@router.post("/oidc/job-sites/{site}/leave", status_code=204)
async def site_leave(request: Request, site: str, body: JobSiteRequest) -> None:
    """Leaving needs no one's permission (remote-nodes.md §3.3)."""
    _own_site(request, body.refreshToken, site)
    _remove(request, site)
