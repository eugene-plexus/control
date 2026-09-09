"""The node registry: list, read, revoke, mint a token, enroll.

Where applied state meets observation, and the two are kept apart on
purpose. `name`, `role`, `publicKey`, `os`, `arch`, `devices` and
`enrolledAt` come from the log. `reachable`, `lastSeenEpoch`,
`lastSeenAt` and `agentVersion` come from the last poll and are layered
on here. A standby cannot have seen what this root saw, so replicating
an observation would manufacture drift out of two hosts honestly
reporting different things.

**A node whose `lastSeenEpoch` trails `GET /v1/control/status`** is the
visible symptom of a host that was partitioned during a promotion. That
window is real, bounded to the partitioned nodes, and self-healing on
reconnect — and it is surfaced right here rather than hidden, because
the price of no quorum is a window you can see.
"""

from __future__ import annotations

import base64
import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Request, status

from .. import sealing, security
from .._generated.models import (
    Enrollment,
    EnrollmentRequest,
    JoinToken,
    JoinTokenRequest,
    KeyRotation,
    Node,
    NodeList,
)
from ..applied import (
    OP_ENROLL_NODE,
    OP_REVOKE_NODE,
    NodeRecord,
)
from ..auth_state import AuthState
from ..dependencies import problem, require_authorized, require_operator
from ..join_tokens import JoinTokenConsumed, JoinTokenError, JoinTokenStore
from ..state_machine import NotActive, StateMachine

log = logging.getLogger(__name__)

router = APIRouter(tags=["nodes"])


@router.get("/v1/nodes", response_model=NodeList, dependencies=[Depends(require_authorized)])
async def list_nodes(request: Request) -> NodeList:
    """Every enrolled host, with its identity and the epoch it has seen.

    Includes this one. A single-host install returns exactly one entry,
    which is the honest answer rather than an empty list.
    """
    machine: StateMachine = request.app.state.machine
    probes = getattr(request.app.state, "node_probes", {})
    return NodeList(
        nodes=[
            _to_node(record, probes.get(record.name))
            for record in sorted(machine.state.nodes.values(), key=lambda r: r.name)
        ]
    )


@router.get("/v1/nodes/{name}", response_model=Node, dependencies=[Depends(require_authorized)])
async def get_node(request: Request, name: str) -> Node:
    machine: StateMachine = request.app.state.machine
    record = machine.state.nodes.get(name)
    if record is None:
        raise problem(
            status.HTTP_404_NOT_FOUND, "No such node", f"No node named {name!r} is enrolled."
        )
    probes = getattr(request.app.state, "node_probes", {})
    return _to_node(record, probes.get(name))


@router.delete(
    "/v1/nodes/{name}",
    response_model=KeyRotation,
    status_code=202,
    dependencies=[Depends(require_operator)],
)
async def revoke_node(request: Request, name: str) -> KeyRotation:
    """Revoke a node's enrollment, **which rotates the signing key.**

    This is a rotation and not a deletion, and the distinction is the
    sharp edge of the whole design: a revoked node still *holds* the
    signing key, so removing its registry entry would not stop it
    authenticating to other components. So the key is re-minted and
    redistributed to every remaining node.

    Two consequences the caller must expect. Nodes that are `down`
    during the rotation hold a superseded key and are refused until they
    reconnect and are re-keyed — which is why this is resumable and
    idempotent by necessity rather than by preference. And the runtimes
    the revoked node was hosting become un-declared: the model files on
    that host's disk are untouched, but nothing in this install will
    route to them.
    """
    machine: StateMachine = request.app.state.machine
    if name not in machine.state.nodes:
        raise problem(
            status.HTTP_404_NOT_FOUND, "No such node", f"No node named {name!r} is enrolled."
        )

    from .control import perform_rotation

    try:
        machine.append(OP_REVOKE_NODE, {"name": name})
    except NotActive as exc:
        raise _not_active(machine) from exc

    log.warning("node %s revoked; rotating the signing key", name)
    return await perform_rotation(request, reason="revocation", revoked_node=name)


@router.post(
    "/v1/nodes/join-token",
    response_model=JoinToken,
    status_code=201,
    dependencies=[Depends(require_operator)],
)
async def mint_join_token(request: Request, body: JoinTokenRequest | None = None) -> JoinToken:
    """Mint a single-use, short-lived, optionally node-scoped token.

    The returned `token` is shown **once** and is not retrievable
    afterwards — it is stored only as a hash, so a lost token is
    re-minted rather than looked up. Minting a second does not
    invalidate the first; expiry and use do.

    Minting is deliberately **not** a log entry: a token is
    active-root-local, unrecoverable by design, and useless if this root
    dies before the node enrolls. The consequence, stated rather than
    discovered: a token does not survive a promotion.
    """
    machine: StateMachine = request.app.state.machine
    if not machine.is_active:
        raise _not_active(machine)

    store: JoinTokenStore = request.app.state.join_tokens
    config_values = machine.state.config
    default_ttl = int(config_values.get("joinTokenTtlSeconds") or 900)
    ttl = int(body.ttlSeconds) if body and body.ttlSeconds is not None else default_ttl
    node_name = body.nodeName if body else None

    minted = store.mint(ttl_seconds=ttl, node_name=node_name)
    log.info(
        "minted a join token%s, valid for %ds",
        f" bound to node {node_name!r}" if node_name else "",
        ttl,
    )
    return JoinToken(
        token=minted.token,
        expiresAt=datetime.fromtimestamp(minted.expires_at, tz=UTC),
        nodeName=minted.node_name,
    )


@router.post("/v1/nodes/enroll", response_model=Enrollment, status_code=201)
async def enroll_node(request: Request, body: EnrollmentRequest) -> Enrollment:
    """Exchange a join token and a public key for membership.

    Called by the **agent**, not the operator, and authenticated by the
    join token rather than a session — the node presenting it has no
    credential yet. This is the one endpoint here that takes no session
    token, which is why it is the one that consumes a single-use
    credential.

    The agent sends only its public half; the private key never leaves
    the node, which is what makes per-node sealing meaningful. In
    exchange it receives the install's signing key — **the thing a
    single-watchdog install could not give it**, because a driver
    spawned on one host used to reject a gateway's token from another.
    """
    machine: StateMachine = request.app.state.machine
    auth: AuthState = request.app.state.auth_state
    store: JoinTokenStore = request.app.state.join_tokens

    if not machine.is_active:
        raise _not_active(machine)

    try:
        base64.b64decode(body.publicKey, validate=True)
    except Exception as exc:
        raise problem(
            status.HTTP_400_BAD_REQUEST,
            "Malformed public key",
            f"publicKey must be base64 ({exc}).",
        ) from exc

    try:
        store.consume(body.token, node_name=body.name)
    except JoinTokenConsumed as exc:
        raise problem(status.HTTP_409_CONFLICT, "Token already used", str(exc)) from exc
    except JoinTokenError as exc:
        raise problem(status.HTTP_401_UNAUTHORIZED, "Join token rejected", str(exc)) from exc

    if auth.signing_key is None:
        raise problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Locked",
            "This control root has not been unlocked, so it cannot hand out the install's "
            "signing key. Log in first.",
        )

    identity = machine.state.identity
    recovery_public = (
        sealing.public_from_private(auth.recovery_private_key)
        if auth.recovery_private_key
        else None
    )
    payload: dict[str, Any] = {
        "name": body.name,
        "role": "agent",
        "publicKey": body.publicKey,
        "agentVersion": body.agentVersion,
        "os": body.os.value if body.os else None,
        "arch": body.arch.value if body.arch else None,
        "devices": [d.model_dump(exclude_none=True, mode="json") for d in body.devices or []],
        # Stamped once, here, into the entry. Replay reads this value
        # rather than the replica's clock — the rule that makes applied
        # state a function of the log and nothing else.
        "enrolledAt": datetime.now(UTC).isoformat(),
    }

    try:
        machine.append(OP_ENROLL_NODE, payload)
    except NotActive as exc:
        raise _not_active(machine) from exc

    log.info("enrolled node %s at epoch %d", body.name, machine.state.epoch)
    return Enrollment(
        name=body.name,
        epoch=machine.state.epoch,
        signingKey=base64.b64encode(auth.signing_key).decode("ascii"),
        controlPublicKey=identity.controlPublicKey,
        recoveryPublicKey=recovery_public,
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _to_node(record: NodeRecord, probe: Any | None) -> Node:
    """Applied state plus this root's observations, kept distinguishable.

    `reachable` defaults to False when nothing has polled yet, and that
    is the honest default: we have not seen this host. False means
    `down`, never `out` — nothing is reassigned on the strength of this
    field.
    """
    return Node.model_validate(
        {
            "name": record.name,
            "url": record.url,
            "role": record.role,
            "reachable": bool(probe.reachable) if probe is not None else False,
            "publicKey": record.publicKey,
            "lastSeenEpoch": probe.epoch if probe is not None else None,
            "agentVersion": (probe.agent_version if probe is not None else None)
            or record.agentVersion,
            "os": record.os,
            "arch": record.arch,
            "devices": [dict(d) for d in record.devices],
            "enrolledAt": record.enrolledAt,
            "lastSeenAt": probe.last_seen_at if probe is not None else None,
        }
    )


def _not_active(machine: StateMachine) -> Any:
    """409 that names where to go instead.

    "Try again" is the wrong advice for a write that reached a standby:
    the caller has to talk to a different host, and the useful response
    says so.
    """
    active = machine.state.nodes
    active_name = next((n.name for n in active.values() if n.role == "control"), None)
    detail = f"This control root is a {machine.role} and does not accept writes. " + (
        f"The registry names {active_name!r} as the active root."
        if active_name
        else "The registry does not name an active root; if this host should be it, "
        "promote it with POST /v1/control/promote."
    )
    return problem(status.HTTP_409_CONFLICT, "Not the active root", detail)


def node_service_token(auth: AuthState) -> str | None:
    """A service token this root presents when calling a node's agent.

    `service:control` rather than an operator token: the control root
    calling an agent is a component doing its job, and using an operator
    credential for it would mean a leaked one could drive the UI
    surface too.
    """
    if auth.signing_key is None:
        return None
    return security.issue_service_token(signing_key=auth.signing_key, kind="control")


def seal_for_node(plaintext: str, node: NodeRecord, recovery_public_key: str | None) -> str:
    """Seal a secret to one node and to the recovery recipient.

    Both, always. Sealing only to the node makes a dead node's secrets
    unrecoverable; sealing only to recovery would put every secret in
    the install behind one key held in one place, which is the blast
    radius this design exists to avoid.
    """
    if node.publicKey is None:
        raise problem(
            status.HTTP_409_CONFLICT,
            "Node has no key",
            f"Node {node.name!r} has no identity public key recorded, so nothing can be "
            "sealed to it. Re-enroll it.",
        )
    recipients = {sealing.node_recipient_label(node.name): node.publicKey}
    if recovery_public_key is not None:
        recipients[sealing.RECIPIENT_RECOVERY] = recovery_public_key
    return sealing.seal_to_recipients(plaintext, recipients)
