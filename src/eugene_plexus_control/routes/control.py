"""This root's own state: status, log, snapshot, promote, rotate.

The replication surface. §5 of the M5 design said standbys pull; these
two endpoints are what they pull through, and the direction is
deliberate — it keeps one writer, lets a standby measure its own lag
honestly, and puts the connection in the direction that survives a
firewall between buildings.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Query, Request, Response, status

from .. import security
from .._generated.models import (
    ControlStatus,
    KeyRotation,
    LogPage,
    NodeRole,
    PromoteRequest,
    Reason,
    Snapshot,
    StandbyStatus,
    State,
)
from ..applied import OP_ROTATE_SIGNING_KEY
from ..auth_state import AuthState
from ..dependencies import problem, require_authorized, require_operator
from ..rotation import (
    REASON_OPERATOR,
    REASON_REVOCATION,
    RotationInFlight,
    RotationProgress,
    RotationTracker,
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

    Readable by any service token as well as the operator: an agent
    needs the current epoch, and a component needs to know whether
    management is available at all.
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
    dependencies=[Depends(require_authorized)],
)
async def read_log(
    request: Request,
    after: int = Query(..., ge=0, description="Entries with index strictly greater than this."),
    limit: int = Query(100, ge=1, le=1000),
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
    """
    machine: StateMachine = request.app.state.machine
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
    dependencies=[Depends(require_authorized)],
)
async def read_snapshot(request: Request) -> Response:
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
    """
    machine: StateMachine = request.app.state.machine
    return Response(content=machine.canonical(), media_type="application/json")


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
    response_model=KeyRotation,
    status_code=202,
    dependencies=[Depends(require_operator)],
)
async def rotate_signing_key(request: Request) -> KeyRotation:
    """Mint a new signing key and redistribute it. Explicit, not a
    side effect of restarting.

    Through M4 tokens were "rotated on each agent restart", which was
    harmless when one process spawned everything and is an outage once
    there are N nodes — a restart that re-keyed the install would break
    every component that had not yet been told.

    **This logs you out, including from this endpoint's own `GET`.** One
    key signs both service tokens and operator sessions, so re-keying
    invalidates every session that was issued under the old one — this
    caller's included. That is correct rather than convenient: the
    reason to rotate is usually that a host should no longer be trusted,
    and a host that held a service token could have held an operator
    session too. Log in again to watch the rotation finish; the progress
    persists until the next rotation starts precisely so that it is
    still there when you get back.
    """
    return await perform_rotation(request, reason=REASON_OPERATOR, revoked_node=None)


@router.get(
    "/v1/control/rotate-key",
    response_model=KeyRotation,
    dependencies=[Depends(require_authorized)],
)
async def get_key_rotation(request: Request) -> KeyRotation:
    """Progress of the current or most recent rotation.

    Persists until the next one starts, so a UI that reconnects after a
    rotation still learns how it ended — including which hosts were left
    holding a superseded key.
    """
    tracker: RotationTracker = request.app.state.rotation
    current = tracker.current
    if current is None:
        raise problem(
            status.HTTP_404_NOT_FOUND,
            "No rotation",
            "No key rotation has run on this control root.",
        )
    return _to_key_rotation(current)


# --------------------------------------------------------------------------- #
# rotation, shared with node revocation
# --------------------------------------------------------------------------- #


async def perform_rotation(
    request: Request, *, reason: str, revoked_node: str | None
) -> KeyRotation:
    """Mint, log, then redistribute. In that order, and it matters.

    The new key is durable in the log **before** any node is told about
    it, so a crash midway leaves an install whose recorded key is the
    new one and whose hosts are a mix — which is recoverable by re-running
    the rotation. The other order would leave nodes holding a key no
    surviving root knows about, which is not.

    Redistribution is best-effort per node and the result names the ones
    it could not reach. Those hold a superseded key and are refused until
    they reconnect and are re-keyed: not a failure to hide, but the
    reason this operation has to be idempotent and resumable.
    """
    machine: StateMachine = request.app.state.machine
    auth: AuthState = request.app.state.auth_state
    tracker: RotationTracker = request.app.state.rotation

    if not machine.is_active:
        raise problem(
            status.HTTP_409_CONFLICT,
            "Not the active root",
            f"This control root is a {machine.role}; rotation belongs to the active root.",
        )
    if auth.master_key is None:
        raise problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Locked",
            "The master key is not available in this process, so a new signing key "
            "cannot be sealed. Log in first.",
        )

    node_names = sorted(machine.state.nodes)
    try:
        tracker.begin(reason=reason, revoked_node=revoked_node, nodes=node_names)
    except RotationInFlight as exc:
        raise problem(status.HTTP_409_CONFLICT, "Rotation in flight", str(exc)) from exc

    try:
        new_key = security.generate_signing_key()
        previous = machine.state.identity.signingKeyId or "0"
        key_id = str(int(previous) + 1) if previous.isdigit() else f"{previous}+1"
        machine.append(
            OP_ROTATE_SIGNING_KEY,
            {
                "signingKeyId": key_id,
                # Sealed by the writer, once. A replica stores this
                # string verbatim and never re-seals it: sealing is not
                # deterministic, so recomputing it would break replay
                # equivalence.
                "sealedSigningKey": security.seal_b64(new_key, auth.master_key),
                "reason": reason,
                "revokedNode": revoked_node,
            },
        )
        auth.set_signing_key(new_key)
        tracker.minted(key_id)
    except Exception as exc:
        tracker.fail(f"{type(exc).__name__}: {exc}")
        raise

    # Redistribution runs in the background: the caller gets a 202 and
    # polls GET, because reaching N hosts in another building is not a
    # request-latency operation. The reference is held on the app rather
    # than dropped — an unreferenced task can be garbage collected
    # mid-flight, which here would abandon a rotation halfway and leave
    # the install in the mixed-key state this whole operation is
    # careful about.
    request.app.state.rotation_task = asyncio.create_task(
        _distribute(request, new_key, key_id), name="control-key-rotation"
    )
    current = tracker.current
    assert current is not None
    return _to_key_rotation(current)


async def _distribute(request: Request, new_key: bytes, key_id: str) -> None:
    """Hand the new key to every remaining node.

    A node that refuses or times out is left pending and named. There is
    deliberately no retry loop here: a rotation that quietly retried for
    an hour would report `distributing` while an operator waited for a
    verdict, and the verdict they need is "these three hosts are stale".
    Re-running the rotation is the retry, and it is idempotent.
    """
    machine: StateMachine = request.app.state.machine
    tracker: RotationTracker = request.app.state.rotation
    auth: AuthState = request.app.state.auth_state
    client = request.app.state.nodes_client

    from .nodes import node_service_token

    token = node_service_token(auth)
    key_b64 = base64.b64encode(new_key).decode("ascii")

    for record in list(machine.state.nodes.values()):
        if not record.url:
            continue
        try:
            await client.post_json(
                record.url,
                "/v1/node/rekey",
                token,
                {"signingKey": key_b64, "signingKeyId": key_id, "epoch": machine.state.epoch},
            )
            tracker.rekeyed(record.name)
        except (httpx.HTTPError, ValueError) as exc:
            log.warning(
                "node %s was not re-keyed (%s); it holds the previous key and will be "
                "refused until it reconnects",
                record.name,
                exc,
            )
    tracker.finish()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _standby_statuses(request: Request) -> list[StandbyStatus]:
    """What the active root knows about its standbys.

    Sourced from what each standby last reported when it pulled, not
    from probing them: the standby is the one that can honestly say how
    far it has applied, and asking it would be asking the host that
    might replace this one to be told its own position.
    """
    reported: dict[str, dict[str, Any]] = getattr(request.app.state, "standby_reports", {})
    machine: StateMachine = request.app.state.machine
    configured = machine.state.config.get("standbyUrls") or []

    urls = list(dict.fromkeys([*configured, *reported.keys()]))
    out: list[StandbyStatus] = []
    for url in urls:
        report = reported.get(url)
        applied = report.get("appliedIndex") if report else None
        out.append(
            StandbyStatus(
                url=url,
                appliedIndex=applied,
                lagEntries=(
                    max(0, machine.state.index - int(applied)) if isinstance(applied, int) else None
                ),
                lastContactAt=(report.get("at") if report else None),
                reachable=bool(report) if report else False,
            )
        )
    return out


def _to_key_rotation(progress: RotationProgress) -> KeyRotation:
    return KeyRotation(
        state=State(progress.state),
        reason=Reason(progress.reason)
        if progress.reason in (REASON_OPERATOR, REASON_REVOCATION)
        else None,
        revokedNode=progress.revoked_node,
        nodesTotal=progress.nodes_total,
        nodesRekeyed=progress.nodes_rekeyed,
        nodesPending=list(progress.nodes_pending),
        error=progress.error,
        startedAt=_parse_iso(progress.started_at),
        finishedAt=_parse_iso(progress.finished_at),
    )


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
