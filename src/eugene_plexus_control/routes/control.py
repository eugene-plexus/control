"""This root's own state: status, log, snapshot, promote, rotate.

The replication surface. §5 of the M5 design said standbys pull; these
two endpoints are what they pull through, and the direction is
deliberate — it keeps one writer, lets a standby measure its own lag
honestly, and puts the connection in the direction that survives a
firewall between buildings.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Query, Request, Response, status

from .. import root_tls, security, tokens, trust
from .._generated.models import (
    ControlStatus,
    LogPage,
    NodeRole,
    PromoteRequest,
    Reason,
    SignedRootTls,
    SignedTrustBundle,
    Snapshot,
    StandbyStatus,
    TrustChange,
)
from ..applied import OP_ROTATE_SIGNING_KEY, standby_node
from ..auth_state import AuthState
from ..dependencies import (
    problem,
    require_authorized,
    require_operator,
    require_replica,
    verify_with_view,
    via_public_sites,
)
from ..state_machine import AlreadyActive, StateMachine

log = logging.getLogger(__name__)

router = APIRouter(tags=["control"])


@router.get(
    "/v1/control/status",
    response_model=ControlStatus,
    dependencies=[Depends(require_authorized)],
)
async def control_status(request: Request) -> ControlStatus:
    """The one call that answers "is failover safe right now".

    A standby whose `appliedIndex` trails the active root's cannot be
    promoted without losing whatever it has not applied, and this is
    where an operator sees that **before** deciding rather than after.

    Readable by a session and by a member node's `agent` or `gateway`
    service token: an agent needs the current epoch.
    """
    machine: StateMachine = request.app.state.machine
    state = machine.state
    follower = getattr(request.app.state, "follower", None)

    standbys: list[StandbyStatus] | None = None
    if machine.is_active:
        standbys = _standby_statuses(request)

    return ControlStatus(
        role=NodeRole(machine.role),
        epoch=state.epoch,
        appliedIndex=state.index,
        firstAvailableIndex=machine.first_available_index(),
        standbys=standbys,
        activeUrl=(follower.status.active_url if follower is not None else None),
        initialized=state.identity.salt is not None,
    )


@router.get(
    "/v1/control/log",
    response_model=None,
    responses={200: {"model": LogPage}, 409: {"description": "Entries compacted"}},
)
async def read_log(
    request: Request,
    after: int = Query(..., ge=0, description="Entries with index strictly greater than this."),
    limit: int = Query(100, ge=1, le=1000),
    claims: tokens.Claims = Depends(require_replica),
) -> Response:
    """Pull entries after an index. Gapless, oldest first.

    Returns **409** when `after` predates `firstAvailableIndex` — the
    entries were compacted away and the caller must take a snapshot
    instead. That is a distinct answer from an empty page on purpose: an
    empty page means "you are current", and conflating the two would
    have a standby that fell behind compaction sit there believing it
    was up to date.

    **Entries are transmitted verbatim**, not re-serialized through
    `LogPage`. See `read_snapshot` for why; the short version is that a
    round trip through the response model rewrites values in ways that
    are semantically identical and byte-different, and a replica must
    hold what the writer wrote.

    **An operator session or the standby's own token**: the entries
    include the ones that wrote the install's key material, so this is
    the same door as the snapshot and takes the same credentials. The
    standby's `after` is the index it has applied, which is how this
    root knows its lag (warm-standby.md SB4).
    """
    machine: StateMachine = request.app.state.machine
    _heard_from_standby(request, claims, after)
    first_available = machine.first_available_index()
    applied = machine.state.index

    if after < applied and after + 1 < first_available:
        raise problem(
            status.HTTP_409_CONFLICT,
            "Entries compacted",
            f"Entries after index {after} are no longer in the log; the oldest available "
            f"is {first_available}. Take GET /v1/control/snapshot instead — reading around "
            "a gap would produce a state that never existed anywhere.",
        )

    page = {
        "entries": machine.read_page(after, limit),
        "appliedIndex": applied,
        "firstAvailableIndex": first_available,
    }
    return Response(
        content=json.dumps(page, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        media_type="application/json",
    )


@router.get(
    "/v1/control/snapshot",
    response_model=None,
    responses={200: {"model": Snapshot}},
)
async def read_snapshot(
    request: Request, claims: tokens.Claims = Depends(require_replica)
) -> Response:
    """Applied state as of one index, for bootstrapping a standby.

    Carries the replication set: topology, the node registry, config,
    and the sealed key material. Not the passphrase — that exists only
    in the operator's head — and not per-component secrets, which are
    sealed on the hosts that read them and never travel here.

    **The bytes sent are the bytes compared.** This returns
    `canonical_bytes(state)` verbatim rather than validating it through
    the `Snapshot` response model, and that is deliberate rather than
    lazy: Pydantic normalizes on the way out — `http://host:8079` gains
    a trailing slash, `+00:00` becomes `Z` — so a standby bootstrapped
    through the model would differ from the active root in bytes while
    agreeing in meaning. That is precisely the difference the
    replay-equivalence test exists to catch, and it would arrive as a
    false positive from the transport rather than a true one from the
    logic.

    The contract is still honored and still checked:
    `tests/test_contract_shape.py` asserts this document validates
    against `Snapshot`, which is the assertion that actually matters —
    conformance without letting the serializer rewrite the payload.

    **An operator session or the standby's own token**
    (`require_replica`), which is narrower than every other read here
    and deliberately so. What this returns includes `sealedSigningKey`,
    `salt` and `passphraseVerifier`, so any other holder gains an
    offline attack on the operator's passphrase.
    """
    machine: StateMachine = request.app.state.machine
    document = machine.canonical()
    _heard_from_standby(request, claims, machine.state.index)
    return Response(content=document, media_type="application/json")


@router.post(
    "/v1/control/promote",
    response_model=ControlStatus,
    dependencies=[Depends(require_operator)],
)
async def promote(request: Request, body: PromoteRequest) -> ControlStatus:
    """Promote this standby to active. Called **on the standby**.

    The passphrase is required, and that requirement is what resolves
    "does the master key cross a host boundary" in the good direction: a
    standby holds the salt and the verifier but not the key, and
    promotion being manual means the operator is present at the one
    moment it is needed.

    **Refusing to promote is a valid outcome.** A standby that trails
    the active root reports the gap and returns 409, because silently
    promoting a stale root is how an install loses its node registry.
    `force` exists for the operator who has decided that losing the tail
    beats staying down — an explicit choice with a name, not a silent
    default.
    """
    machine: StateMachine = request.app.state.machine
    auth: AuthState = request.app.state.auth_state
    follower = getattr(request.app.state, "follower", None)
    identity = machine.state.identity

    if machine.is_active:
        raise problem(
            status.HTTP_409_CONFLICT,
            "Already active",
            "This control root is already the active one. Promotion is for a standby.",
        )

    if identity.passphraseVerifier is None:
        raise problem(
            status.HTTP_409_CONFLICT,
            "Nothing replicated",
            "This standby holds no passphrase verifier, which means it has never "
            "successfully replicated from an active root. A standby that cannot prove "
            "its state is worse than no standby — check EUGENE_PLEXUS_CONTROL_ACTIVE_URL "
            "and GET /v1/control/status before promoting.",
        )

    passphrase = body.passphrase.get_secret_value()
    if not security.verify_passphrase(passphrase, identity.passphraseVerifier):
        raise problem(
            status.HTTP_401_UNAUTHORIZED,
            "Wrong passphrase",
            "Passphrase did not match the replicated verifier.",
        )

    lag = follower.status.lag_entries if follower is not None else None
    if lag and not body.force:
        raise problem(
            status.HTTP_409_CONFLICT,
            "Standby is behind",
            f"This standby is {lag} entr{'y' if lag == 1 else 'ies'} behind the active "
            f"root (applied {machine.state.index}, active reported "
            f"{follower.status.active_applied_index if follower else 'unknown'}). "
            "Promoting now loses the difference. Re-run with force=true if that is the "
            "trade you want.",
        )

    # Stop replicating before writing. A promoted root that was still
    # pulling from its predecessor would be accepting entries it did not
    # assign, which is the single-writer property gone.
    if follower is not None:
        await follower.stop()
        request.app.state.follower = None

    from .auth import unseal_into

    unseal_into(auth, machine, passphrase)

    node_name = getattr(request.app.state, "node_name", None)
    try:
        machine.promote(node=node_name)
    except AlreadyActive as exc:
        raise problem(
            status.HTTP_409_CONFLICT, "Already active", "This control root is already active."
        ) from exc

    # Announce the epoch: a trust bundle at the new epoch, signed with the
    # identity key promotion deliberately preserved (`sealedControlKey`).
    # Agents record it and refuse the old root's lower one. Best-effort;
    # a node that is down takes it when it next pulls.
    if request.app.state.trust.publish(request.app) is None:
        log.warning(
            "promoted without the control identity key in memory; nodes will learn epoch %d "
            "when this root is next unlocked",
            machine.state.epoch,
        )

    log.warning(
        "PROMOTED to active control root at epoch %d, applied index %d%s",
        machine.state.epoch,
        machine.state.index,
        f" (forced, {lag} entries behind)" if lag and body.force else "",
    )
    return ControlStatus(
        role=NodeRole(machine.role),
        epoch=machine.state.epoch,
        appliedIndex=machine.state.index,
        firstAvailableIndex=machine.first_available_index(),
        standbys=_standby_statuses(request),
        initialized=True,
    )


@router.post(
    "/v1/control/rotate-key",
    response_model=TrustChange,
    status_code=202,
    dependencies=[Depends(require_operator)],
)
async def rotate_signing_key(request: Request) -> TrustChange:
    """Replace this root's token key. Every session and client key ends.

    The emergency lever for "this root's token key may have leaked". A
    new key is generated, sealed into the log, and published in a trust
    bundle that names only it, so everything the old key signed stops
    verifying on every machine at once. **Nothing private is
    distributed**: node keys are untouched and only the bundle moves.
    Until 2026-09-25 this redistributed the one shared key to every node
    and tracked which were stale; there is no shared key any more.

    **This logs you out.** Sign in again afterwards.
    """
    machine: StateMachine = request.app.state.machine
    auth: AuthState = request.app.state.auth_state
    if not machine.is_active:
        raise problem(
            status.HTTP_409_CONFLICT,
            "Not the active root",
            f"This control root is a {machine.role}; rotation belongs to the active root.",
        )
    if auth.master_key is None or auth.control_private_key is None:
        # Checked before minting: a key that could not be sealed, or a
        # bundle that could not be signed, would be a rotation nobody hears of.
        raise problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Locked",
            "The master key or this root's identity key is not in memory, so a new token "
            "key could not be sealed or published. Sign in first.",
        )
    new_key = security.generate_signing_key()
    previous = machine.state.identity.signingKeyId or "0"
    key_id = str(int(previous) + 1) if previous.isdigit() else f"{previous}+1"
    machine.append(
        OP_ROTATE_SIGNING_KEY,
        {
            "signingKeyId": key_id,
            # Sealed by the writer, once. A replica stores this string
            # verbatim: sealing is not deterministic, so recomputing it
            # would break replay equivalence.
            "sealedSigningKey": security.seal_b64(new_key, auth.master_key),
            "rootTokenPublicKey": tokens.public_b64(tokens.load_private(new_key)),
            "reason": "operator",
        },
    )
    auth.set_signing_key(new_key)
    bundle = request.app.state.trust.publish(request.app)
    version = bundle.version if bundle is not None else machine.state.index
    log.warning("token key rotated to generation %s; bundle %d names only it", key_id, version)
    return TrustChange(
        version=version,
        reason=Reason.rotation,
        nodesBehind=trust.nodes_behind(request.app, version),
    )


@router.get("/v1/trust/bundle", response_model=SignedTrustBundle)
async def get_trust_bundle(request: Request) -> SignedTrustBundle:
    """The signed bundle every node pulls. **Public**: the signature is the credential.

    A sealed root serves the last one it signed, which it keeps on disk,
    so a node pulling during an outage keeps what it has rather than
    learning nothing.

    **Except through the public node route** (J3): there it names every
    machine, key and signed-out session to the internet, so a member's own
    `sub: agent` token is required. A bearer sent on any route must verify.
    Checked against the keys applied state names, which needs no private
    key, so a sealed root still answers its members.
    """
    authorization = request.headers.get("authorization")
    if authorization or via_public_sites(request):
        scheme, _, bearer = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not bearer.strip():
            raise problem(
                status.HTTP_401_UNAUTHORIZED,
                "Node token required",
                "Through the public node route the trust bundle is given only to a member "
                "machine presenting its own token.",
            )
        claims = verify_with_view(request, bearer.strip(), classes=(tokens.TYP_SERVICE,))
        if claims.sub != tokens.SUB_AGENT or claims.issuer_node is None:
            raise problem(
                status.HTTP_401_UNAUTHORIZED,
                "Node token required",
                "Only a member machine's own agent token opens the trust bundle here.",
            )
    publisher: trust.TrustPublisher = request.app.state.trust
    if publisher.current is None:
        raise problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "No trust bundle yet",
            "This root has never signed a trust bundle: it is not initialized, or it has not "
            "been unlocked since it was.",
        )
    return SignedTrustBundle(jws=publisher.current.jws)


@router.get("/v1/trust/tls", response_model=SignedRootTls)
async def get_root_tls(request: Request) -> SignedRootTls:
    """The TLS keys this root's nodes name presents, signed by its identity (J7a).

    Public keys only, so public on purpose: a Job Site fetches it before it
    has any credential, over the connection it is checking.
    """
    watcher: root_tls.RootTls = request.app.state.root_tls
    if watcher.origin is None:
        raise problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "No name for machines",
            "This root has no nodes name, so machines outside its network cannot join it. "
            "Its owner sets one in Settings, Container access setup.",
        )
    if not watcher.keys or time.perf_counter() - watcher.checked_at > root_tls.STALE_SECONDS:
        await watcher.probe()
    auth: AuthState = request.app.state.auth_state
    jws = watcher.signed(auth.control_private_key)
    if jws is None:
        raise problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "TLS keys unknown",
            (watcher.error or "This root has not yet read the certificate its nodes name presents")
            + ". No machine can join until it can, because a machine accepts only a key "
            "this root has signed.",
        )
    return SignedRootTls(jws=jws)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


#: A standby heard from within this many follow intervals is reachable.
STANDBY_HEARD_INTERVALS = 3


def _heard_from_standby(request: Request, claims: tokens.Claims, applied: int) -> None:
    """Record where the standby is, from its own pull (warm-standby.md SB4).
    An operator reading the log is not a standby, and records nothing."""
    if claims.is_service and claims.sub == tokens.SUB_STANDBY and claims.issuer_node:
        reports: dict[str, dict[str, Any]] = request.app.state.standby_reports
        reports[claims.issuer_node] = {
            "appliedIndex": applied,
            "at": datetime.now(UTC),
            "seen": time.perf_counter(),
        }


def _standby_statuses(request: Request) -> list[StandbyStatus]:
    """What the active root knows about its standby.

    The node holding the `standby` grant (warm-standby.md), sourced from
    what it last reported when it pulled, not from probing it: the standby
    is the one that can honestly say how far it has applied, and asking it
    would be asking the host that might replace this one to be told its own
    position. `standbyUrls` is no longer read (SB1).
    """
    machine: StateMachine = request.app.state.machine
    name = standby_node(machine.state)
    if name is None:
        return []
    record = machine.state.nodes[name]
    report = getattr(request.app.state, "standby_reports", {}).get(name)
    applied = report.get("appliedIndex") if report else None
    # The install's follow interval: a standby is started with the same
    # setting this root has.
    interval = float(request.app.state.settings.follow_interval_seconds)
    recent = (
        report is not None
        and time.perf_counter() - float(report["seen"]) <= STANDBY_HEARD_INTERVALS * interval
    )
    return [
        StandbyStatus(
            node=name,
            url=record.url,  # type: ignore[arg-type]
            appliedIndex=applied,
            lagEntries=(
                max(0, machine.state.index - int(applied)) if isinstance(applied, int) else None
            ),
            lastContactAt=report["at"] if report else None,
            reachable=recent,
        )
    ]
