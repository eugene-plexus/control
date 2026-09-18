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

from .. import node_address, sealing, security
from .._generated.models import (
    Enrollment,
    EnrollmentRequest,
    JoinToken,
    JoinTokenList,
    JoinTokenRecord,
    JoinTokenRequest,
    KeyRotation,
    Node,
    NodeAddressAck,
    NodeAddressAnnouncement,
    NodeList,
)
from ..applied import (
    OP_ENROLL_NODE,
    OP_REVOKE_NODE,
    OP_UPDATE_NODE,
    NodeRecord,
    normalize_url,
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


# **Declared before `/v1/nodes/{name}`, and that is load-bearing.**
# Starlette matches in declaration order, so a literal segment that could
# be read as a node name has to come first or the parameterised route
# swallows it -- which it did, and `GET /v1/nodes/join-tokens` answered
# `404 No such node named 'join-tokens'` until a test asked for the list.
# `DELETE /v1/nodes/join-tokens/{id}` never collided (two segments), which
# is exactly why the defect was half-invisible.
@router.get(
    "/v1/nodes/join-tokens",
    response_model=JoinTokenList,
    dependencies=[Depends(require_operator)],
)
async def list_join_tokens(request: Request) -> JoinTokenList:
    """What is outstanding. **No tokens in it** — there are none to give.

    Operator-only rather than merely authorized: what a token is bound
    to and when it dies is a picture of who is being invited into this
    install, which is not a service's business.
    """
    store: JoinTokenStore = request.app.state.join_tokens
    return JoinTokenList(
        tokens=[
            JoinTokenRecord(
                id=record.id,
                expiresAt=datetime.fromtimestamp(record.expires_at, tz=UTC),
                nodeName=record.node_name,
                used=record.used,
            )
            for record in store.list()
        ]
    )


@router.delete(
    "/v1/nodes/join-tokens/{id}",
    status_code=204,
    dependencies=[Depends(require_operator)],
)
async def revoke_join_token(request: Request, id: str) -> None:
    """Withdraw a token before it expires.

    **Not gated on `is_active`**, unlike minting. A standby holds no
    join tokens of its own, so this can only ever act on the store of
    the root it is addressed to — and refusing to withdraw a credential
    because of a role check would be the wrong way round: taking one
    away is the safe direction, and an operator who can reach this port
    with an operator token should never be told to go somewhere else
    first.

    404 on an unknown id rather than a silent 204, because "already
    gone" and "you are revoking the wrong thing" want different next
    moves and the operator is looking at a list this root just served.
    """
    store: JoinTokenStore = request.app.state.join_tokens
    if not store.revoke(id):
        raise problem(
            status.HTTP_404_NOT_FOUND,
            "No such join token",
            f"No outstanding join token has id {id!r}. It may have expired, been used and "
            "swept, or already been revoked.",
        )
    log.info("join token %s revoked", id)


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


@router.patch("/v1/nodes/{name}", response_model=NodeAddressAck)
async def announce_node_address(
    request: Request, name: str, body: NodeAddressAnnouncement
) -> NodeAddressAck:
    """A node tells this root its address changed. **No bearer** — the
    credential is the signature.

    The mirror image of `POST /v1/node/rekey` on the agent, and the
    symmetry is the argument: this root proves itself to a node with its
    identity key, a node proves itself to this root with its own, and
    neither uses a bearer because a bearer does not survive the rotation
    that makes both operations necessary.

    A service token would have been the obvious choice and cannot do the
    job twice over — it names a *kind*, not a host, so any agent could
    re-address any node; and it would have been the first mutation here
    authenticated by a service credential, against a rule this module's
    neighbours state explicitly. Operator auth is no use either: the case
    that matters is a host that rebooted at 3am onto a new address, with
    nobody watching.

    Idempotent and deliberately silent when nothing changed: a restart
    that announces the address it already had appends nothing, because
    otherwise every reboot of every node grows the log for no
    information.

    **What the signature does not settle is which address (R2.4).** The
    authentication here was always sound and was the whole of the check:
    a correctly authenticated worker could name any URL at all, with
    `is_loopback_host` as the only filter, so a cloud instance-metadata
    address was a valid answer. Two rules sit after the signature now --
    an address that can never be a node is a 400, and a node putting
    itself on the open internet is a 409 naming re-enrollment as the
    confirmation. See `node_address` for why the classes are built from
    `is_global` rather than `is_private`.
    """
    machine: StateMachine = request.app.state.machine
    record = machine.state.nodes.get(name)
    if record is None:
        raise problem(
            status.HTTP_404_NOT_FOUND, "No such node", f"No node named {name!r} is enrolled."
        )
    if not record.signingPublicKey:
        raise problem(
            status.HTTP_401_UNAUTHORIZED,
            "Node has no signing key",
            f"Node {name!r} enrolled before nodes carried an Ed25519 signing identity, so "
            f"there is nothing here to verify its announcement against. Re-enroll it; its "
            f"model files and runtimes are untouched by that.",
        )

    # **The signed field is read from the raw body, never from `body.url`.**
    # `AnyUrl` appends a trailing slash to an authority-only URL, so
    # `str(body.url)` is not the string the node signed and every
    # announcement would 401 — which is exactly M5's serializer finding
    # ("any byte-identity claim that crosses a serializer is a claim
    # about the serializer") arriving a second time, in a place where it
    # reads as a crypto bug rather than a formatting one. A signature is
    # over bytes on the wire; anything that parses first is a different
    # claim. `format: uri` stays on the schema because the validation and
    # the generated clients are still worth having.
    try:
        raw = await request.json()
    except Exception:
        raw = None
    announced_url = raw.get("url") if isinstance(raw, dict) else None
    if not isinstance(announced_url, str):
        raise problem(
            status.HTTP_400_BAD_REQUEST,
            "Malformed announcement",
            "`url` must be present in the request body as a string.",
        )
    message = sealing.address_message(name=name, sequence=int(body.sequence), url=announced_url)
    if not sealing.verify_address(record.signingPublicKey, message, body.signature):
        log.warning("refused an address announcement for %s: signature did not verify", name)
        raise problem(
            status.HTTP_401_UNAUTHORIZED,
            "Signature rejected",
            f"The announcement is not signed by the identity node {name!r} enrolled with.",
        )

    reason = node_address.rejection(announced_url)
    if reason is not None:
        log.warning(
            "refused an address announcement for %s: %s cannot be a node's address (%s)",
            name,
            announced_url,
            reason,
        )
        raise problem(
            status.HTTP_400_BAD_REQUEST,
            "Address cannot be a node",
            f"{announced_url} cannot be node {name!r}'s address: {reason}. The "
            "announcement was correctly signed; what is refused is the address.",
        )

    normalized = normalize_url(announced_url)
    if normalized == record.url:
        # Nothing to record. Note that the sequence is deliberately NOT
        # advanced here: it lives in applied state, and advancing it
        # would need a log entry, which is exactly what this branch
        # exists to avoid. A replay of this same message stays a no-op.
        return NodeAddressAck(
            name=name,
            url=record.url,  # type: ignore[arg-type]
            sequence=record.advertiseSequence,
            changed=False,
        )

    if node_address.goes_public(record.url, normalized):
        # Signed, sequenced, from the right node -- and still refused,
        # because this is the one transition a homelab node never makes
        # by itself and the one an attacker holding a leaked signing key
        # needs. After it, this root probes the new address every poll
        # and every console proxy hop hands it the operator's bearer.
        log.warning(
            "refused an address announcement for %s: %s -> %s moves it from %s onto the "
            "open internet (%s)",
            name,
            record.url,
            normalized,
            node_address.classify(record.url),
            node_address.classify(normalized),
        )
        raise problem(
            status.HTTP_409_CONFLICT,
            "Address puts this node on the open internet",
            f"Node {name!r} is enrolled at {record.url}, which is "
            f"{node_address.classify(record.url)}, and announced {normalized}, which is "
            f"{node_address.classify(normalized)}. Moving between loopback and the LAN "
            "is ordinary -- that is the Reach switch, and a new LAN address after a "
            "reboot -- but a node may not put itself on the open internet, because "
            "nothing here could tell that apart from a leaked signing key doing it. "
            "Re-enroll the node to confirm the new address; its model files and "
            "runtimes are untouched by that.",
        )

    if int(body.sequence) <= record.advertiseSequence:
        raise problem(
            status.HTTP_409_CONFLICT,
            "Stale announcement",
            f"Sequence {int(body.sequence)} does not advance node {name!r} past "
            f"{record.advertiseSequence}. Either this is a replay, or the node's identity "
            f"file was restored from a backup — in which case re-enroll it rather than "
            f"raising the counter here.",
        )

    try:
        machine.append(
            OP_UPDATE_NODE,
            {"name": name, "url": normalized, "sequence": int(body.sequence)},
        )
    except NotActive as exc:
        raise _not_active(machine) from exc

    log.info(
        "node %s re-advertised: %s -> %s (sequence %d)",
        name,
        record.url,
        normalized,
        int(body.sequence),
    )
    updated = machine.state.nodes[name]
    return NodeAddressAck(
        name=name,
        url=updated.url,  # type: ignore[arg-type]
        sequence=updated.advertiseSequence,
        changed=True,
    )


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
        id=minted.id,
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

    The `url` it sends is recorded as `Node.url`, where this root probes
    it, forwards declarations to it, and where a gateway sends a stop or
    start for a runtime it owns. Recorded verbatim (normalized once, on
    the way into the log, as every URL is): the agent knows which
    interface it used to reach this root, and a root deriving it from the
    request's source address would be right on a flat mesh network and
    wrong behind anything else. M7 — before it, every really-enrolled
    node had no address and the tests were appending a second
    `enrollNode` entry by hand.

    **Verbatim, but not unexamined** (R2.4): the address still has to be
    one a node could be at — http(s), not the bind wildcard, not
    multicast, not link-local. Enrollment is where a node's reachability
    *class* is set, and it is allowed to be any of them, because a join
    token is operator-minted and the operator is therefore present. What
    a node may not do is put *itself* on the open internet later; see
    `announce_node_address`.
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

    announced_url = normalize_url(str(body.url)) if body.url else None
    reason = node_address.rejection(announced_url) if announced_url else None
    if reason is not None:
        # Checked **before** the join token is consumed: a single-use
        # credential spent on a request that was never going to be
        # recorded is an operator minting a second one for no reason.
        raise problem(
            status.HTTP_400_BAD_REQUEST,
            "Address cannot be a node",
            f"{body.url} cannot be this node's address: {reason}.",
        )

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
        "signingPublicKey": body.signingPublicKey,
        "url": announced_url,
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
        signingKeyId=identity.signingKeyId,
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
            "signingPublicKey": record.signingPublicKey,
            "advertiseSequence": record.advertiseSequence,
            "lastSeenEpoch": probe.epoch if probe is not None else None,
            "agentVersion": (probe.agent_version if probe is not None else None)
            or record.agentVersion,
            "os": record.os,
            "arch": record.arch,
            "devices": [dict(d) for d in record.devices],
            "enrolledAt": record.enrolledAt,
            "lastSeenAt": probe.last_seen_at if probe is not None else None,
            # The probe client has always had the reason; until 2026-09-15
            # this view dropped it, and `reachable: false` sent an operator
            # to enrollment and keys when the cause was half a second of
            # clock skew on a worker refusing this root's tokens.
            "lastError": probe.error if probe is not None else None,
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
