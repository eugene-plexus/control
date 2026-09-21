"""Applied state, and the deterministic function that produces it.

**This module is the reason a promotion can be trusted.** Applied state
is a pure function of the log entries applied so far, and
`tests/test_replay_equivalence.py` asserts that a standby's applied
state is byte-identical to the active root's after applying the same
log. That property is what makes promotion safe rather than hopeful, and
it is what would later let consensus be dropped in underneath: the log,
this function, the snapshot and the epoch already exist, so Raft would
be a swap of the transport and the election rather than a redesign.

Four rules hold it up. Each is cheap to keep and expensive to notice the
loss of, because breaking one does not fail a request — it silently
makes two roots disagree about the same install.

**1. `apply` is pure.** No clock, no filesystem, no network, no
randomness. Note that this module imports none of those; that is not an
accident, and `datetime` is deliberately absent even though several
fields hold timestamps.

**2. Every timestamp comes from the payload.** `enrolledAt` is what the
writer stamped when it appended the entry, replayed verbatim years
later. One `datetime.now()` in an apply path and replay equivalence is
quietly false, with nothing failing until a promotion.

**3. Liveness is not applied state.** `reachable`, `lastSeenEpoch`,
`lastSeenAt` and key-rotation progress are one root's *observations* of
an install, not facts about it. They are layered on when a response is
rendered (see `routes/nodes.py`) and are deliberately absent from
`AppliedState`. A standby cannot have seen what the active root saw, so
replicating an observation would produce "drift" that is really just two
hosts honestly reporting different things.

**4. Index is the only ordering.** `LogEntry.at` exists so an operator
can read the log. It is never compared, sorted on, or used to resolve
anything.

## The canonical form is the wire form

`to_canonical` produces **exactly** the `Snapshot` document from
`control.yaml` — same field names, same nesting, same shape — and
`GET /v1/control/snapshot` transmits those bytes verbatim rather than
re-serializing them through a response model.

That is not tidiness. Round-tripping applied state through Pydantic
normalizes values on the way out: `http://host:8079` becomes
`http://host:8079/` and `+00:00` becomes `Z`. Both are correct
serializations and both mean a standby's state would differ from the
active root's in bytes while agreeing in meaning — which is exactly the
difference the replay-equivalence test exists to catch, arriving as a
false positive from the transport instead of a true one from the logic.
So a value is normalized **once, on the way into the log** (see
`normalize_url`), and never again.

A consequence to keep: every field in the canonical form must be a field
`Snapshot` declares. An extra one would be silently dropped by any
consumer that does validate against the schema, and the first symptom
would be a standby quietly missing state.

`apply` raises `ApplyError` on an entry it cannot apply — a gap, a
regressed epoch, a malformed payload, a delete of something absent. A
replica that hits one must re-snapshot rather than guess, and the
alternative (applying what it can and moving on) is the failure mode
where two roots differ and both think they are fine.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

# Op names, matching `LogOp` in control.yaml. Duplicated as constants
# rather than imported from the generated enum so this module stays free
# of anything that could pull in a dependency with an opinion about the
# clock — and so the apply function reads as a closed set.
OP_ENROLL_NODE = "enrollNode"
OP_UPDATE_NODE = "updateNode"
OP_REVOKE_NODE = "revokeNode"
OP_PUT_COMPONENT = "putComponent"
OP_DELETE_COMPONENT = "deleteComponent"
OP_PUT_RUNTIME = "putRuntime"
OP_DELETE_RUNTIME = "deleteRuntime"
OP_PATCH_CONFIG = "patchConfig"
OP_ROTATE_SIGNING_KEY = "rotateSigningKey"
OP_PROMOTE = "promote"
OP_PUT_CLIENT_KEY = "putClientKey"
OP_IMPORT_CLIENT_KEYS = "importClientKeys"
OP_REVOKE_CLIENT_KEY = "revokeClientKey"
OP_SET_CLIENT_KEY_LIMITS = "setClientKeyLimits"
OP_PUT_CLIENT_ADMISSION = "putClientAdmission"

ALL_OPS: frozenset[str] = frozenset(
    {
        OP_ENROLL_NODE,
        OP_UPDATE_NODE,
        OP_REVOKE_NODE,
        OP_PUT_COMPONENT,
        OP_DELETE_COMPONENT,
        OP_PUT_RUNTIME,
        OP_DELETE_RUNTIME,
        OP_PATCH_CONFIG,
        OP_ROTATE_SIGNING_KEY,
        OP_PROMOTE,
        OP_PUT_CLIENT_KEY,
        OP_IMPORT_CLIENT_KEYS,
        OP_REVOKE_CLIENT_KEY,
        OP_SET_CLIENT_KEY_LIMITS,
        OP_PUT_CLIENT_ADMISSION,
    }
)

ROLE_CONTROL = "control"
ROLE_STANDBY = "standby"
ROLE_AGENT = "agent"


class ApplyError(Exception):
    """This entry cannot be applied to this state.

    Always fatal for the replica that raised it: the correct response is
    to re-snapshot, never to skip the entry. Skipping is how two roots
    end up disagreeing while both report healthy."""


def normalize_url(value: str | None) -> str | None:
    """Put a URL into the form a serializer would produce.

    Called on the way *into* the log, once. Pydantic's `AnyUrl` appends a
    trailing slash to an authority-only URL, so a value stored raw would
    differ from the same value after any round trip through the schema —
    a byte-level divergence with no semantic content, arriving as a
    false failure in the one test that must not have any.

    Deliberately minimal: this is not URL validation, which belongs to
    the request model. It only settles the one difference that would
    otherwise show up as replication drift.
    """
    if value is None:
        return None
    trimmed = value.strip()
    if not trimmed:
        return None
    if "://" not in trimmed:
        return trimmed
    scheme, _, rest = trimmed.partition("://")
    if "/" not in rest:
        return f"{scheme}://{rest}/"
    return trimmed


@dataclass(frozen=True)
class NodeRecord:
    """A host, as the log declares it.

    Every field is either operator-supplied or reported by the node at
    enrollment. Nothing here is an observation — see rule 3 above.
    """

    name: str
    role: str = ROLE_AGENT
    url: str | None = None
    publicKey: str | None = None
    signingPublicKey: str | None = None
    """Ed25519, and the only thing it is for is verifying that a
    `PATCH /v1/nodes/{name}` came from this node. `publicKey` is X25519
    and cannot sign. Absent on a node enrolled before M9."""
    advertiseSequence: int = 0
    """Highest announcement sequence accepted. Applied state, so a
    promoted standby refuses the same replays this root would."""
    agentVersion: str | None = None
    os: str | None = None
    arch: str | None = None
    devices: tuple[dict[str, Any], ...] = ()
    enrolledAt: str | None = None
    """From the entry's payload, stamped once by the writer that
    appended it. Never `now()` at apply time."""


@dataclass(frozen=True)
class ComponentRecord:
    """A component and the node it runs on. The node dimension is the
    only thing an agent's own view cannot supply."""

    node: str
    name: str
    kind: str
    url: str | None = None


@dataclass(frozen=True)
class RuntimeRecord:
    """A runtime **declaration** plus its placement.

    `spec` is the agent's `RuntimeSpec`, carried verbatim and left
    opaque: the agent's spec is the one definition of what a runtime
    declaration is, and restating its fields here would create a second
    one that drifts. `name`, `modelAlias` and `engine` are read back out
    of it rather than stored beside it, so they cannot disagree with the
    declaration they came from.
    """

    node: str
    spec: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return str(self.spec.get("name") or "")

    @property
    def model_alias(self) -> str | None:
        value = self.spec.get("modelAlias")
        return str(value) if isinstance(value, str) and value else None

    @property
    def engine(self) -> str | None:
        value = self.spec.get("engine")
        return str(value) if isinstance(value, str) and value else None


@dataclass(frozen=True)
class InstallIdentity:
    """The trust root's own replicated material.

    Established once by `POST /v1/auth/initialize` and thereafter
    mutated only by `rotateSigningKey`. **Initialization is not a log
    op**, and that is correct rather than an omission: it happens
    exactly once, before any standby can exist, so the snapshot always
    carries it forward and it never needs to replicate as an entry.
    `LogOp` stays closed at nine.

    The three sealed values are opaque strings, sealed under the
    passphrase-derived key by whichever root minted them. They are never
    re-sealed by a replica — sealing is non-deterministic, so
    recomputing one would break replay equivalence.

    Note what is **not** here: the recovery recipient's *public* key. It
    is derivable from the private half the moment that is unsealed, and
    a field that can be computed is a field that can disagree with what
    it was computed from.
    """

    salt: str | None = None
    """Base64 Argon2id salt. Without it the same passphrase derives a
    different key, so a standby that lacks it cannot be promoted."""

    passphraseVerifier: str | None = None
    """Argon2id-PHC hash. Lets a standby authenticate the operator at
    promotion without holding the master key."""

    sealedSigningKey: str | None = None
    """The current service-token signing key. Sealed, because a snapshot
    is a file on a second host and this key mints every service token in
    the install."""

    sealedControlKey: str | None = None
    """The control root's identity private key. Replicated because nodes
    check epoch changes against its public half, so a promoted standby
    with a fresh identity would be refused by every node it tried to
    command. Promotion is a change of host, not of identity."""

    controlPublicKey: str | None = None
    """Public half of the above, in the clear so a standby can report
    the identity it would assume without holding the passphrase."""

    sealedRecoveryKey: str | None = None
    """The recovery recipient every secret is additionally sealed to.
    Without it a dead node's secrets are unrecoverable; with it, reading
    them still requires the operator's passphrase."""

    signingKeyId: str | None = None
    """Names the current key generation. Not a secret, and the thing a
    node reports back so a rotation can tell which hosts are stale."""


@dataclass(frozen=True)
class AppliedState:
    """A deterministic function of the entries applied so far.

    Immutable, and `apply` returns a new one. That is not stylistic
    purity: the writer applies to a candidate copy *before* it makes the
    entry durable, so an entry that cannot be applied never reaches the
    log and a failed append leaves state untouched.
    """

    index: int = 0
    epoch: int = 1
    nodes: dict[str, NodeRecord] = field(default_factory=dict)
    components: dict[str, ComponentRecord] = field(default_factory=dict)
    """Keyed `"<node>/<name>"`. A component name is unique per node, not
    per install — two hosts each running a component called `gateway` is
    an ordinary topology."""
    runtimes: dict[str, RuntimeRecord] = field(default_factory=dict)
    """Keyed `"<node>/<name>"`, for the same reason. Note that
    `modelAlias` is the thing that is unique across the *install*: two
    nodes serving one alias is how a replica is expressed, and the
    gateway load balances across them."""
    config: dict[str, Any] = field(default_factory=dict)
    """The control root's own configuration. Replicated because a
    standby has to come up configured the way it would have been had it
    been active all along — a promoted standby with no `standbyUrls`
    silently has no standbys of its own."""
    identity: InstallIdentity = field(default_factory=InstallIdentity)
    client_keys: dict[str, dict[str, Any]] = field(default_factory=dict)
    client_admission: dict[str, Any] = field(default_factory=lambda: {"clock": 0.0, "buckets": {}})
    client_key_imports: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# apply
# --------------------------------------------------------------------------- #


def apply(state: AppliedState, entry: dict[str, Any]) -> AppliedState:
    """Apply one entry, returning new state. Pure; raises `ApplyError`.

    Entries must arrive in index order with no gaps. That is checked
    here rather than trusted, because the one thing worse than a replica
    that stops is a replica that carries on from a hole.
    """
    index = entry.get("index")
    epoch = entry.get("epoch")
    op = entry.get("op")
    payload = entry.get("payload") or {}

    if not isinstance(index, int) or not isinstance(epoch, int):
        raise ApplyError(f"entry index/epoch must be integers, got {index!r}/{epoch!r}")
    if not isinstance(payload, dict):
        raise ApplyError(f"entry {index} payload must be an object, got {type(payload).__name__}")
    if index != state.index + 1:
        raise ApplyError(
            f"out-of-order entry: got index {index}, expected {state.index + 1}. "
            "Re-snapshot; do not skip."
        )
    if epoch < state.epoch:
        raise ApplyError(
            f"entry {index} carries epoch {epoch} below the highest seen ({state.epoch}); "
            "a superseded root is fenced out of the log as well as out of the agents"
        )
    if op not in ALL_OPS:
        raise ApplyError(
            f"entry {index} has unknown op {op!r}. An op absent from LogOp is an op that "
            "would not replicate — add it to the contract first."
        )

    mutated = _HANDLERS[op](state, payload, index)
    # Epoch is carried forward from the entry, never from a local clock
    # or a local guess. `promote` is the only op that raises it, and it
    # does so by stamping a higher epoch on its own entry, which is what
    # lets a standby learn the new generation by replaying.
    return replace(mutated, index=index, epoch=max(state.epoch, epoch))


def apply_all(state: AppliedState, entries: list[dict[str, Any]]) -> AppliedState:
    """Fold `apply` over a sequence. No shortcuts, deliberately: the
    ordering and gap checks are the point."""
    for entry in entries:
        state = apply(state, entry)
    return state


def _require_str(payload: dict[str, Any], key: str, index: int) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ApplyError(f"entry {index}: {key!r} must be a non-empty string, got {value!r}")
    return value


def _optional_str(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) and value else None


def _apply_enroll_node(state: AppliedState, payload: dict[str, Any], index: int) -> AppliedState:
    name = _require_str(payload, "name", index)
    role = _optional_str(payload, "role") or ROLE_AGENT
    if role not in (ROLE_CONTROL, ROLE_STANDBY, ROLE_AGENT):
        raise ApplyError(f"entry {index}: unknown node role {role!r}")

    raw_devices = payload.get("devices") or []
    if not isinstance(raw_devices, list):
        raise ApplyError(f"entry {index}: devices must be a list")

    record = NodeRecord(
        name=name,
        role=role,
        url=_optional_str(payload, "url"),
        publicKey=_optional_str(payload, "publicKey"),
        signingPublicKey=_optional_str(payload, "signingPublicKey"),
        agentVersion=_optional_str(payload, "agentVersion"),
        os=_optional_str(payload, "os"),
        arch=_optional_str(payload, "arch"),
        devices=tuple(d for d in raw_devices if isinstance(d, dict)),
        enrolledAt=_optional_str(payload, "enrolledAt"),
    )
    # Re-enrollment overwrites rather than conflicting. A host that was
    # rebuilt presents a new keypair under the same operator-chosen name,
    # and refusing that would make "reinstall the OS on the GPU box" an
    # operation with no path through the API.
    return replace(state, nodes={**state.nodes, name: record})


def _apply_update_node(state: AppliedState, payload: dict[str, Any], index: int) -> AppliedState:
    """A node has moved. Only the address and the sequence change.

    Deliberately narrow: this op exists because `Node.url` is applied
    state a node must be able to change with no operator present, and
    widening it into a general node patch would let the one credential
    that can reach it — a signature over `{name, sequence, url}` —
    cover fields it does not name.

    The sequence is checked *here* as well as in the route, because
    apply is the only thing a standby runs. A replay that got into the
    log would otherwise be re-applied on every replica and on every
    replay, which is the thing the counter exists to stop.
    """
    name = _require_str(payload, "name", index)
    record = state.nodes.get(name)
    if record is None:
        raise ApplyError(f"entry {index}: cannot update unknown node {name!r}")
    sequence = payload.get("sequence")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
        raise ApplyError(f"entry {index}: 'sequence' must be a positive integer, got {sequence!r}")
    if sequence <= record.advertiseSequence:
        raise ApplyError(
            f"entry {index}: sequence {sequence} does not advance node {name!r} past "
            f"{record.advertiseSequence}"
        )
    updated = replace(record, url=_optional_str(payload, "url"), advertiseSequence=sequence)
    return replace(state, nodes={**state.nodes, name: updated})


def _apply_revoke_node(state: AppliedState, payload: dict[str, Any], index: int) -> AppliedState:
    name = _require_str(payload, "name", index)
    if name not in state.nodes:
        raise ApplyError(f"entry {index}: cannot revoke unknown node {name!r}")
    # The revoked node's declarations go with it. The model files on that
    # host's disk are untouched — but nothing in this install will route
    # to them, which is the honest consequence of removing a host.
    return replace(
        state,
        nodes={k: v for k, v in state.nodes.items() if k != name},
        components={k: v for k, v in state.components.items() if v.node != name},
        runtimes={k: v for k, v in state.runtimes.items() if v.node != name},
    )


def _apply_put_component(state: AppliedState, payload: dict[str, Any], index: int) -> AppliedState:
    node = _require_str(payload, "node", index)
    name = _require_str(payload, "name", index)
    kind = _require_str(payload, "kind", index)
    if node not in state.nodes:
        raise ApplyError(f"entry {index}: component {name!r} placed on unknown node {node!r}")
    record = ComponentRecord(node=node, name=name, kind=kind, url=_optional_str(payload, "url"))
    return replace(state, components={**state.components, f"{node}/{name}": record})


def _apply_delete_component(
    state: AppliedState, payload: dict[str, Any], index: int
) -> AppliedState:
    node = _require_str(payload, "node", index)
    name = _require_str(payload, "name", index)
    key = f"{node}/{name}"
    if key not in state.components:
        raise ApplyError(f"entry {index}: cannot delete unknown component {key!r}")
    return replace(state, components={k: v for k, v in state.components.items() if k != key})


def _apply_put_runtime(state: AppliedState, payload: dict[str, Any], index: int) -> AppliedState:
    node = _require_str(payload, "node", index)
    if node not in state.nodes:
        raise ApplyError(f"entry {index}: runtime placed on unknown node {node!r}")
    spec = payload.get("spec")
    if not isinstance(spec, dict):
        raise ApplyError(f"entry {index}: runtime spec must be an object")
    name = spec.get("name")
    if not isinstance(name, str) or not name:
        raise ApplyError(
            f"entry {index}: the runtime spec has no `name`, so there is nothing to key it by"
        )
    record = RuntimeRecord(node=node, spec=spec)
    return replace(state, runtimes={**state.runtimes, f"{node}/{name}": record})


def _apply_delete_runtime(state: AppliedState, payload: dict[str, Any], index: int) -> AppliedState:
    node = _require_str(payload, "node", index)
    name = _require_str(payload, "name", index)
    key = f"{node}/{name}"
    if key not in state.runtimes:
        raise ApplyError(f"entry {index}: cannot delete unknown runtime {key!r}")
    return replace(state, runtimes={k: v for k, v in state.runtimes.items() if k != key})


def _apply_patch_config(state: AppliedState, payload: dict[str, Any], index: int) -> AppliedState:
    values = payload.get("values")
    if not isinstance(values, dict):
        raise ApplyError(f"entry {index}: patchConfig payload needs a `values` object")
    # Validation happened at the writer, before the entry existed. Apply
    # is not the place to re-litigate a value: an entry already in the
    # log has to apply the same way on every replica, including one
    # running a build whose validation rules have since changed.
    return replace(state, config={**state.config, **values})


def _apply_rotate_signing_key(
    state: AppliedState, payload: dict[str, Any], index: int
) -> AppliedState:
    identity = replace(
        state.identity,
        sealedSigningKey=_require_str(payload, "sealedSigningKey", index),
        signingKeyId=_require_str(payload, "signingKeyId", index),
    )
    return replace(state, identity=identity)


def _apply_promote(state: AppliedState, payload: dict[str, Any], index: int) -> AppliedState:
    """Record which node became the active root.

    The epoch bump itself is carried by the entry's own `epoch` field
    and applied in `apply`, so a replica learns the new generation by
    replaying rather than by being told out of band.
    """
    node = _optional_str(payload, "node")
    if node is None:
        # A promotion whose node is not named is still a valid epoch
        # bump — a standby may be promoted before anything has enrolled
        # it as a node. Fencing must not depend on the registry being
        # complete.
        return state
    nodes = dict(state.nodes)
    for key, record in nodes.items():
        if record.role == ROLE_CONTROL and key != node:
            nodes[key] = replace(record, role=ROLE_STANDBY)
    if node in nodes:
        nodes[node] = replace(nodes[node], role=ROLE_CONTROL)
    return replace(state, nodes=nodes)


def _key_record(raw: Any) -> dict[str, Any]:
    from datetime import datetime

    if not isinstance(raw, dict):
        raise ApplyError("client-key record must be an object")
    for key in ("id", "name", "tail", "createdAt", "expiresAt"):
        if not isinstance(raw.get(key), str) or (not raw[key] and key != "tail"):
            raise ApplyError(f"invalid client-key {key}")
    for key in ("createdAt", "expiresAt", "revokedAt"):
        if raw.get(key) is not None:
            try:
                parsed = datetime.fromisoformat(raw[key])
                if parsed.utcoffset() is None:
                    raise ValueError("timezone missing")
            except (ValueError, TypeError) as exc:
                raise ApplyError(f"invalid client-key {key}") from exc
    # A log entry/snapshot must never retain a token or an unexpected secret field.
    allowed = {
        "id",
        "name",
        "tail",
        "createdAt",
        "expiresAt",
        "revokedAt",
        "originNode",
        "migrated",
        "lastUsedAt",
        "limits",
    }
    if raw.keys() - allowed:
        raise ApplyError("unexpected client-key property")
    from .client_admission import validate_limits

    try:
        validate_limits(raw.get("limits"))
    except ValueError as exc:
        raise ApplyError("invalid client-key limits") from exc
    return dict(raw)


def _key_records(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, list):
        raise ApplyError("client-key records must be a list")
    records = {}
    for item in raw:
        record = _key_record(item)
        if record["id"] in records:
            raise ApplyError("duplicate client-key identifier")
        records[record["id"]] = record
    return records


def _apply_put_client_key(state: AppliedState, payload: dict[str, Any], index: int) -> AppliedState:
    record = _key_record(payload.get("key"))
    if record["id"] in state.client_keys:
        raise ApplyError("client-key identifier already exists")
    return replace(state, client_keys={**state.client_keys, record["id"]: record})


def _apply_import_client_keys(
    state: AppliedState, payload: dict[str, Any], index: int
) -> AppliedState:
    node = payload.get("node")
    if node not in state.nodes or not isinstance(payload.get("digest"), str):
        raise ApplyError("invalid client-key import origin")
    imported = _key_records(payload.get("keys"))
    keys = dict(state.client_keys)
    for record in imported.values():
        record = dict(record)
        record.update(originNode=node, migrated=True)
        previous = keys.get(record["id"])
        if previous is not None:
            if previous.get("originNode") != node or any(
                previous.get(k) != record.get(k) for k in ("tail", "createdAt", "expiresAt")
            ):
                raise ApplyError("client-key import conflicts with an existing record")
            if "limits" in previous:
                record["limits"] = previous["limits"]
            else:
                record.pop("limits", None)
            if previous.get("revokedAt") is not None:
                record["revokedAt"] = previous["revokedAt"]
        keys[record["id"]] = record
    return replace(
        state,
        client_keys=keys,
        client_key_imports={**state.client_key_imports, node: payload["digest"]},
    )


def _apply_set_client_key_limits(
    state: AppliedState, payload: dict[str, Any], index: int
) -> AppliedState:
    key_id = payload.get("id")
    if key_id not in state.client_keys or not isinstance(payload.get("limits"), dict):
        raise ApplyError("unknown key or missing limits")
    record = _key_record({**state.client_keys[key_id], "limits": payload["limits"]})
    return replace(state, client_keys={**state.client_keys, key_id: record})


def _apply_put_client_admission(
    state: AppliedState, payload: dict[str, Any], index: int
) -> AppliedState:
    from .client_admission import clean_buckets, validate_ledger

    try:
        now = payload["clock"]
        if now < state.client_admission["clock"]:
            raise ValueError("admission clock regressed")
        buckets = clean_buckets(state.client_admission["buckets"], now)
        buckets[payload["keyId"]] = payload["bucket"]
        ledger = validate_ledger({"clock": now, "buckets": buckets})
    except (ValueError, TypeError, KeyError) as exc:
        raise ApplyError("invalid admission update") from exc
    return replace(state, client_admission=ledger)


def _apply_revoke_client_key(
    state: AppliedState, payload: dict[str, Any], index: int
) -> AppliedState:
    key_id = payload.get("id")
    if key_id not in state.client_keys:
        raise ApplyError("unknown client-key identifier")
    previous = state.client_keys[key_id]
    record = _key_record(
        {**previous, "revokedAt": previous.get("revokedAt") or payload.get("revokedAt")}
    )
    if record.get("revokedAt") is None:
        raise ApplyError("revocation timestamp required")
    return replace(state, client_keys={**state.client_keys, key_id: record})


_Handler = Callable[["AppliedState", dict[str, Any], int], "AppliedState"]

_HANDLERS: dict[str, _Handler] = {
    OP_ENROLL_NODE: _apply_enroll_node,
    OP_UPDATE_NODE: _apply_update_node,
    OP_REVOKE_NODE: _apply_revoke_node,
    OP_PUT_COMPONENT: _apply_put_component,
    OP_DELETE_COMPONENT: _apply_delete_component,
    OP_PUT_RUNTIME: _apply_put_runtime,
    OP_DELETE_RUNTIME: _apply_delete_runtime,
    OP_PATCH_CONFIG: _apply_patch_config,
    OP_ROTATE_SIGNING_KEY: _apply_rotate_signing_key,
    OP_PROMOTE: _apply_promote,
    OP_PUT_CLIENT_KEY: _apply_put_client_key,
    OP_IMPORT_CLIENT_KEYS: _apply_import_client_keys,
    OP_REVOKE_CLIENT_KEY: _apply_revoke_client_key,
    OP_SET_CLIENT_KEY_LIMITS: _apply_set_client_key_limits,
    OP_PUT_CLIENT_ADMISSION: _apply_put_client_admission,
}


# --------------------------------------------------------------------------- #
# Canonical form — the `Snapshot` document, and what "byte-identical" means
# --------------------------------------------------------------------------- #


def to_canonical(state: AppliedState) -> dict[str, Any]:
    """Applied state as the `Snapshot` document from `control.yaml`.

    The rules that make a comparison of two of these meaningful:

    * **Every field is present**, `None` included, so "set then cleared"
      is distinguishable from "never set" and two states cannot compare
      equal by both omitting different things.
    * **Collections are sorted by key.** Insertion order is a property
      of the order entries happened to arrive on one host, and
      `sort_keys` in the serializer sorts object keys, not the arrays we
      build from dicts.
    * **No observations.** `reachable` is emitted as `false` because the
      schema requires it and a snapshot carries no observations — a
      reader must not read it as a report about the host.
    * **Exactly the fields `Snapshot` declares**, so nothing is silently
      dropped by a consumer that validates.
    """
    identity = state.identity
    return {
        "index": state.index,
        "epoch": state.epoch,
        "nodes": [
            {
                "name": n.name,
                "role": n.role,
                "url": n.url,
                "reachable": False,
                "publicKey": n.publicKey,
                "signingPublicKey": n.signingPublicKey,
                "advertiseSequence": n.advertiseSequence,
                "agentVersion": n.agentVersion,
                "os": n.os,
                "arch": n.arch,
                "devices": [dict(d) for d in n.devices],
                "enrolledAt": n.enrolledAt,
            }
            for n in sorted(state.nodes.values(), key=lambda r: r.name)
        ],
        "components": [
            {"node": c.node, "name": c.name, "kind": c.kind, "url": c.url}
            for c in sorted(state.components.values(), key=lambda r: (r.node, r.name))
        ],
        "runtimes": [
            {"node": r.node, "spec": r.spec}
            for r in sorted(state.runtimes.values(), key=lambda r: (r.node, r.name))
        ],
        "config": dict(state.config),
        "clientKeys": [state.client_keys[key] for key in sorted(state.client_keys)],
        "clientAdmission": state.client_admission,
        "clientKeyImports": dict(state.client_key_imports),
        "salt": identity.salt,
        "passphraseVerifier": identity.passphraseVerifier,
        "sealedSigningKey": identity.sealedSigningKey,
        "sealedControlKey": identity.sealedControlKey,
        "controlPublicKey": identity.controlPublicKey,
        "sealedRecoveryKey": identity.sealedRecoveryKey,
        "signingKeyId": identity.signingKeyId,
    }


def canonical_bytes(state: AppliedState) -> bytes:
    """The bytes the replay-equivalence test compares, and the bytes
    `GET /v1/control/snapshot` sends.

    One serializer, one form, both ends. `sort_keys` plus the tightest
    separators, so nothing about formatting can differ between two
    hosts, and UTF-8 without escaping so a node name in any script
    round-trips as itself.
    """
    return json.dumps(
        to_canonical(state),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def from_canonical(raw: dict[str, Any]) -> AppliedState:
    """Rebuild state from the `Snapshot` document. Inverse of
    `to_canonical`, and the path a standby takes when it bootstraps.

    Round-trip exactness is load-bearing: a standby that snapshots and
    then tails must reach the same bytes as the root that never
    snapshotted, so anything this drops is a divergence waiting for a
    promotion to reveal it.
    """
    from .client_admission import validate_ledger

    runtimes: dict[str, RuntimeRecord] = {}
    for entry in raw.get("runtimes") or []:
        spec = entry.get("spec") or {}
        node = str(entry.get("node") or "")
        name = str(spec.get("name") or "")
        runtimes[f"{node}/{name}"] = RuntimeRecord(node=node, spec=spec)

    return AppliedState(
        index=int(raw.get("index") or 0),
        epoch=int(raw.get("epoch") or 1),
        nodes={
            str(n["name"]): NodeRecord(
                name=str(n["name"]),
                role=str(n.get("role") or ROLE_AGENT),
                url=n.get("url"),
                publicKey=n.get("publicKey"),
                signingPublicKey=n.get("signingPublicKey"),
                advertiseSequence=int(n.get("advertiseSequence") or 0),
                agentVersion=n.get("agentVersion"),
                os=n.get("os"),
                arch=n.get("arch"),
                devices=tuple(n.get("devices") or ()),
                enrolledAt=n.get("enrolledAt"),
            )
            for n in raw.get("nodes") or []
        },
        components={
            f"{c['node']}/{c['name']}": ComponentRecord(
                node=str(c["node"]),
                name=str(c["name"]),
                kind=str(c["kind"]),
                url=c.get("url"),
            )
            for c in raw.get("components") or []
        },
        runtimes=runtimes,
        config=dict(raw.get("config") or {}),
        client_admission=validate_ledger(raw.get("clientAdmission")),
        client_keys=_key_records(raw.get("clientKeys", [])),
        client_key_imports=dict(raw.get("clientKeyImports") or {}),
        identity=InstallIdentity(
            salt=raw.get("salt"),
            passphraseVerifier=raw.get("passphraseVerifier"),
            sealedSigningKey=raw.get("sealedSigningKey"),
            sealedControlKey=raw.get("sealedControlKey"),
            controlPublicKey=raw.get("controlPublicKey"),
            sealedRecoveryKey=raw.get("sealedRecoveryKey"),
            signingKeyId=raw.get("signingKeyId"),
        ),
    )
