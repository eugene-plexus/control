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
from functools import partial
from typing import Any

from . import sign_in_state

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
OP_REVOKE_CLIENT_KEY = "revokeClientKey"
OP_SET_CLIENT_KEY_LIMITS = "setClientKeyLimits"
OP_PUT_CLIENT_ADMISSION = "putClientAdmission"
OP_REVOKE_SESSION = "revokeSession"
# Signing in with Eugene (C2, docs/design/sign-in-with-eugene.md).
OP_PUT_PERSON = "putPerson"
OP_PUT_PERMISSION_GROUP = "putPermissionGroup"
OP_DELETE_PERMISSION_GROUP = "deletePermissionGroup"
# **Retired in v0.2 and refused since v0.3.1.** `putNodeHelper`, an `enrollNode`
# with the grant `files` (a Job Site that joined as a node), a person's
# `helperGrants` and a snapshot's `nodeHelpers` are what v0.1 wrote. v0.3.0
# converted them once at start; replaying one now raises `ApplyError` through
# `_retired`, which names it and the way out.
OP_SET_PERSON_PASSWORD = "setPersonPassword"
OP_DELETE_PERSON = "deletePerson"
OP_PUT_OIDC_CLIENT = "putOidcClient"
OP_DELETE_OIDC_CLIENT = "deleteOidcClient"
OP_PUT_OIDC_KEY = "putOidcKey"
OP_REVOKE_SIGN_IN = "revokeSignIn"
OP_SET_OIDC_CLIENT_REDIRECTS = "setOidcClientRedirects"
OP_ENROLL_SITE = "enrollSite"
OP_REMOVE_SITE = "removeSite"
OP_SET_SITE_HOST = "setSiteHost"
OP_SET_SITE_DEV_GRANTS = "setSiteDevGrants"
OP_SET_STANDBY = "setStandby"
OP_LABEL_NODE = "labelNode"

#: The grant that makes a node the install's warm standby (warm-standby.md).
GRANT_STANDBY = "standby"

#: The owner's name on the sign-in page, which no person may take.
OPERATOR_NAME = "operator"
#: Provider keys kept: the current one and the one before it, so tokens
#: signed just before a rotation still verify until they expire.
OIDC_KEYS_KEPT = 2

ALL_OPS: frozenset[str] = frozenset(
    set(sign_in_state.OPS)
    | {
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
        OP_REVOKE_CLIENT_KEY,
        OP_SET_CLIENT_KEY_LIMITS,
        OP_PUT_CLIENT_ADMISSION,
        OP_REVOKE_SESSION,
        OP_PUT_PERSON,
        OP_PUT_PERMISSION_GROUP,
        OP_DELETE_PERMISSION_GROUP,
        OP_SET_PERSON_PASSWORD,
        OP_DELETE_PERSON,
        OP_PUT_OIDC_CLIENT,
        OP_DELETE_OIDC_CLIENT,
        OP_PUT_OIDC_KEY,
        OP_REVOKE_SIGN_IN,
        OP_SET_OIDC_CLIENT_REDIRECTS,
        OP_ENROLL_SITE,
        OP_REMOVE_SITE,
        OP_SET_SITE_HOST,
        OP_SET_SITE_DEV_GRANTS,
        OP_SET_STANDBY,
        OP_LABEL_NODE,
    }
)

ROLE_CONTROL = "control"
ROLE_STANDBY = "standby"
ROLE_AGENT = "agent"


def _retired(what: str) -> RetiredData:
    """The refusal for what v0.1 wrote and v0.2 retired: named, with the way out.

    Nothing here converts it any more; v0.3.0 did, once, on the active root that held
    it. A standby was never converted: its own records are set aside at start and it
    follows the root again (`app.py`)."""
    return RetiredData(
        f"{what} was retired in v0.2; start Eugene v0.3.0 once on the active root to "
        "convert it, then update."
    )


class ApplyError(Exception):
    """This entry cannot be applied to this state.

    Always fatal for the replica that raised it: the correct response is
    to re-snapshot, never to skip the entry. Skipping is how two roots
    end up disagreeing while both report healthy."""


class RetiredData(ApplyError):
    """What v0.1 wrote and v0.2 retired, met on replay (`_retired`)."""


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
    label: str | None = None
    """The name people see (`labelNode`, node-labels.md). Display only:
    never a key, never signed. `None` shows `name`."""
    url: str | None = None
    publicKey: str | None = None
    signingPublicKey: str | None = None
    """Ed25519, and the only thing it is for is verifying that a
    `PATCH /v1/nodes/{name}` came from this node. `publicKey` is X25519
    and cannot sign. Absent on a node enrolled before M9."""
    advertiseSequence: int = 0
    """Highest announcement sequence accepted. Applied state, so a
    promoted standby refuses the same replays this root would."""
    tokenPublicKey: str | None = None
    """Base64 of the node's raw Ed25519 **token** public key: what its
    tokens are signed with, and what the trust bundle lists as
    `node:<name>`. The private half was generated on the node and has
    never been anywhere else (2026-09-25)."""
    grants: tuple[str, ...] = ("node",)
    """What the node's key may issue. `gateway` comes from the join token
    that enrolled it, which the operator minted; never from the node.
    A Job Site is never a node (J19): an `enrollNode` that grants `files`
    is refused as retired."""
    owner: str | None = None
    """The person a node was enrolled for, when the enrollment said."""
    agentVersion: str | None = None
    os: str | None = None
    arch: str | None = None
    devices: tuple[dict[str, Any], ...] = ()
    enrolledAt: str | None = None
    """From the entry's payload, stamped once by the writer that
    appended it. Never `now()` at apply time."""


@dataclass(frozen=True)
class SiteRecord:
    """A Job Site (J19): its own enrollment, held by its site host, never a
    node. Every field comes from a log entry's payload."""

    id: str
    label: str
    owner: str
    """The person who confirmed its join at the machine (rule 1 of
    remote-nodes.md §3.3). Only they decide who may use it."""
    tokenPublicKey: str
    """Base64 of the site's raw Ed25519 token key. Checked by this root
    alone; never in the trust bundle nodes receive."""
    enrolledAt: str
    hostNode: str | None = None
    """The node whose agent says it supervises this site (J32). Display
    only: nothing is authorized by it."""
    devGrants: tuple[dict[str, Any], ...] = ()
    """Eugene's owner's own dev-mode grants here (J13b), `{folderId,
    writable}`, sorted by folder."""


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
    """This root's token key: what signs sessions, exchanged tokens and
    client keys. Sealed, because a snapshot is a file on a second host.
    It never leaves a root in any other form."""

    rootTokenPublicKey: str | None = None
    """Base64 of the raw public half of the above, in the clear, so the
    trust bundle can name it without unsealing anything."""

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
    access_audit: list[dict[str, Any]] = field(default_factory=list)
    browser_sign_in: dict[str, Any] = field(
        default_factory=lambda: {
            k: {} for k in ("providers", "identities", "invitations", "sessions")
        }
    )
    revoked_sessions: dict[str, int] = field(default_factory=dict)
    """Signed-out sessions, `jti -> exp`, as `revokeSession` wrote them.
    Replicated because the trust bundle every machine checks sessions
    against is built from applied state: a sign-out a standby had not
    seen would be a session a promotion brought back."""
    people: dict[str, dict[str, Any]] = field(default_factory=dict)
    permission_groups: dict[str, dict[str, Any]] = field(default_factory=dict)
    """People who may sign in to apps (C2), by id, as `SnapshotPerson`:
    their Argon2id verifiers included, for the reason `passphraseVerifier`
    is replicated -- a promoted standby signs them in."""
    sites: dict[str, SiteRecord] = field(default_factory=dict)
    """The Job Site registry (J19), by id."""
    oidc_clients: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Apps that sign in with Eugene, by `clientId`, as `SnapshotOidcClient`."""
    oidc_keys: tuple[dict[str, Any], ...] = ()
    """The provider's RSA keys, newest first, each sealed under the master
    key as `sealedSigningKey` is. At most `OIDC_KEYS_KEPT`."""
    revoked_sign_ins: dict[str, int] = field(default_factory=dict)
    """Sign-ins revoked at `/oidc/revoke`, `sid -> exp`, as `revokeSession`
    keeps sessions."""


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
    if op == "putNodeHelper":
        raise _retired(f"entry {index} (putNodeHelper, an operator-managed node folder)")
    if op not in ALL_OPS:
        raise ApplyError(
            f"entry {index} has unknown op {op!r}. An op absent from LogOp is an op that "
            "would not replicate — add it to the contract first."
        )

    mutated = _HANDLERS[op](state, payload, index)
    from .access_audit import changed

    mutated = replace(mutated, access_audit=changed(state.access_audit, entry))
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

    # A machine that enrolls again under its name keeps its label unless
    # the enrollment sets one: rebuilding the GPU box should not rename it.
    previous = state.nodes.get(name)
    if "label" in payload:
        label = _label_value(payload.get("label"), index)
        refusal = label_refusal(state, name, label)
        if refusal is not None:
            raise ApplyError(f"entry {index}: {refusal}")
    else:
        label = previous.label if previous is not None else None
    record = NodeRecord(
        name=name,
        role=role,
        label=None if label == name else label,
        url=_optional_str(payload, "url"),
        publicKey=_optional_str(payload, "publicKey"),
        signingPublicKey=_optional_str(payload, "signingPublicKey"),
        tokenPublicKey=_optional_str(payload, "tokenPublicKey"),
        grants=_grants(payload.get("grants"), index),
        owner=_optional_str(payload, "owner"),
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


_NODE_GRANTS = frozenset({"node", "gateway", "files", GRANT_STANDBY})


def _grants(raw: Any, index: int) -> tuple[str, ...]:
    """A node's grants, sorted so replay is byte-stable: always including
    `node`. `files` marked a Job Site that joined as a node (J19), which v0.2
    retired: it is refused, naming what it was."""
    if raw is None:
        return ("node",)
    if not isinstance(raw, list) or not all(isinstance(g, str) for g in raw):
        raise ApplyError(f"entry {index}: grants must be a list of strings")
    unknown = set(raw) - _NODE_GRANTS
    if unknown:
        raise ApplyError(f"entry {index}: a node cannot hold {sorted(unknown)}")
    if "files" in raw:
        raise _retired(
            f"entry {index} (an enrollNode with the grant `files`, a Job Site as a node)"
        )
    return tuple(sorted(set(raw) | {"node"}))


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


#: `NodeLabel.maxLength` in control.yaml.
LABEL_MAX = 64


def clean_label(raw: str) -> str:
    """A label as it is stored: trimmed, nothing else changed."""
    return raw.strip()


def label_refusal(state: AppliedState, name: str, label: str | None) -> str | None:
    """Why `label` cannot be node `name`'s label, or None if it can.

    The route asks before appending and apply asks again, because apply is
    all a standby runs. **A label may not match another node's name or
    label, ignoring case** (node-labels.md §1): two machines shown under one
    name is a console that sends a click to the wrong one."""
    if label is None:
        return None
    if not label or label != label.strip():
        return "a label is trimmed and not empty"
    if len(label) > LABEL_MAX:
        return f"a label is at most {LABEL_MAX} characters"
    if any(ord(c) < 0x20 or 0x7F <= ord(c) < 0xA0 for c in label):
        return "a label cannot hold a control character"
    folded = label.casefold()
    for other in state.nodes.values():
        if other.name == name:
            continue
        if other.name.casefold() == folded or (other.label or "").casefold() == folded:
            return f"another machine is already called {label!r} ({other.name!r})"
    return None


def _label_value(raw: Any, index: int) -> str | None:
    if raw is None or isinstance(raw, str):
        return raw
    raise ApplyError(f"entry {index}: 'label' must be a string or null, got {raw!r}")


def _apply_label_node(state: AppliedState, payload: dict[str, Any], index: int) -> AppliedState:
    """Set or clear one node's label. Nothing else about the node changes."""
    name = _require_str(payload, "name", index)
    record = state.nodes.get(name)
    if record is None:
        raise ApplyError(f"entry {index}: cannot label unknown node {name!r}")
    label = _label_value(payload.get("label"), index)
    refusal = label_refusal(state, name, label)
    if refusal is not None:
        raise ApplyError(f"entry {index}: {refusal}")
    updated = replace(record, label=None if label == name else label)
    return replace(state, nodes={**state.nodes, name: updated})


def _apply_revoke_node(state: AppliedState, payload: dict[str, Any], index: int) -> AppliedState:
    name = _require_str(payload, "name", index)
    if name not in state.nodes:
        raise ApplyError(f"entry {index}: cannot revoke unknown node {name!r}")
    return _without_node(state, name)


def _without_node(state: AppliedState, name: str) -> AppliedState:
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
    """Replace the signing key and, for a revocation, remove the node.

    **The two are one entry since 2026-09-25.** A revocation used to be
    a `revokeNode` entry followed by this one, and the route wrote the
    first before it knew whether the second could be written: a rotation
    already in flight (409), a locked root or a missing control identity
    (503) left the node deleted, the key it holds still valid everywhere,
    and a retry answering 404. One entry cannot half-happen.

    A `revokedNode` that is not enrolled is not an error, because every
    log written before the change carries the `revokeNode` entry first
    and names the same node here again.
    """
    identity = replace(
        state.identity,
        sealedSigningKey=_require_str(payload, "sealedSigningKey", index),
        signingKeyId=_require_str(payload, "signingKeyId", index),
        rootTokenPublicKey=_optional_str(payload, "rootTokenPublicKey"),
    )
    state = replace(state, identity=identity)
    revoked = payload.get("revokedNode")
    if isinstance(revoked, str) and revoked in state.nodes:
        state = _without_node(state, revoked)
    return state


def _apply_revoke_session(state: AppliedState, payload: dict[str, Any], index: int) -> AppliedState:
    """Record a sign-out, and forget the ones that have expired.

    `prunedBefore` is the writer's clock, stamped into the payload once,
    like every timestamp here (rule 2). A sign-out whose session expired
    before it has nothing left to refuse.
    """
    jti = _require_str(payload, "jti", index)
    exp = payload.get("exp")
    pruned = payload.get("prunedBefore", 0)
    if not isinstance(exp, int) or isinstance(exp, bool):
        raise ApplyError(f"entry {index}: 'exp' must be an integer")
    if not isinstance(pruned, int) or isinstance(pruned, bool):
        raise ApplyError(f"entry {index}: 'prunedBefore' must be an integer")
    kept = {k: v for k, v in state.revoked_sessions.items() if v >= pruned}
    kept[jti] = exp
    return replace(state, revoked_sessions=kept)


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
        # The promoted machine is the root now and follows nobody, so its
        # standby grant goes (warm-standby.md §3.1).
        promoted = nodes[node]
        nodes[node] = replace(
            promoted,
            role=ROLE_CONTROL,
            grants=tuple(g for g in promoted.grants if g != GRANT_STANDBY),
        )
    return replace(state, nodes=nodes)


def standby_node(state: AppliedState) -> str | None:
    """The node that holds the standby grant, or None. At most one does."""
    return next((n for n, r in state.nodes.items() if GRANT_STANDBY in r.grants), None)


def _apply_set_standby(state: AppliedState, payload: dict[str, Any], index: int) -> AppliedState:
    """Give one node the standby grant, or take it away (warm-standby.md SB1).

    The route has refused everything this refuses before appending, so on
    the writer these never fire; on a replica they mean the log is not one
    this root wrote."""
    name = _require_str(payload, "node", index)
    on = payload.get("on")
    if not isinstance(on, bool):
        raise ApplyError(f"entry {index}: setStandby needs on: true or false")
    record = state.nodes.get(name)
    if record is None:
        raise ApplyError(f"entry {index}: cannot make unknown node {name!r} the standby")
    if on:
        holder = standby_node(state)
        if holder not in (None, name):
            raise ApplyError(f"entry {index}: {holder!r} is already the standby")
        grants = tuple(sorted({*record.grants, GRANT_STANDBY}))
    else:
        grants = tuple(g for g in record.grants if g != GRANT_STANDBY)
    return replace(state, nodes={**state.nodes, name: replace(record, grants=grants)})


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


# --------------------------------------------------------------------------- #
# signing in with Eugene (C2)
# --------------------------------------------------------------------------- #


def _timestamp(raw: dict[str, Any], key: str, what: str, *, optional: bool = False) -> None:
    from datetime import datetime

    value = raw.get(key)
    if value is None and optional:
        return
    if not isinstance(value, str):
        raise ApplyError(f"invalid {what} {key}")
    try:
        if datetime.fromisoformat(value).utcoffset() is None:
            raise ValueError("timezone missing")
    except ValueError as exc:
        raise ApplyError(f"invalid {what} {key}") from exc


#: `PersonPermission` (J77): what a person may do with job sites.
PERSON_PERMISSIONS = frozenset({"add-job-sites", "use-job-sites"})


def _string_list(raw: Any, label: str, *, limit: int, item_limit: int | None = None) -> None:
    if (
        not isinstance(raw, list)
        or len(raw) > limit
        or any(
            not isinstance(v, str) or not v or (item_limit is not None and len(v) > item_limit)
            for v in raw
        )
        or len(set(raw)) != len(raw)
    ):
        raise ApplyError(f"invalid {label}")


def _permission_group_record(raw: Any) -> dict[str, Any]:
    required = {"id", "name", "description", "apps", "permissions", "createdAt", "updatedAt"}
    if not isinstance(raw, dict) or set(raw) != required:
        raise ApplyError("invalid permission group properties")
    for key, limit in (("id", 64), ("name", 80), ("description", 500)):
        value = raw[key]
        if (
            not isinstance(value, str)
            or len(value) > limit
            or (key != "description" and not value.strip())
        ):
            raise ApplyError(f"invalid permission group {key}")
    if raw["name"] != raw["name"].strip():
        raise ApplyError("permission group name must be trimmed")
    if raw["apps"] is not None:
        _string_list(raw["apps"], "permission group apps", limit=256)
    _string_list(raw["permissions"], "permission group permissions", limit=16)
    if any(p not in PERSON_PERMISSIONS for p in raw["permissions"]):
        raise ApplyError("unknown permission group permission")
    _timestamp(raw, "createdAt", "permission group")
    _timestamp(raw, "updatedAt", "permission group")
    return dict(raw)


def _check_group_memberships(groups: dict[str, dict[str, Any]], person: dict[str, Any]) -> None:
    if any(g not in groups for g in person.get("groups") or []):
        raise ApplyError("unknown permission group membership")
    if any(g not in groups for g in person.get("administration", {}).get("membershipGroups", [])):
        raise ApplyError("unknown administration membership group")


def _apply_put_permission_group(
    state: AppliedState, payload: dict[str, Any], index: int
) -> AppliedState:
    record = _permission_group_record(payload.get("group"))
    if any(
        g["id"] != record["id"] and g["name"].casefold() == record["name"].casefold()
        for g in state.permission_groups.values()
    ):
        raise ApplyError(f"entry {index}: permission group name is taken")
    if any(a not in state.oidc_clients for a in record["apps"] or []):
        raise ApplyError(f"entry {index}: unknown permission group app")
    return replace(state, permission_groups={**state.permission_groups, record["id"]: record})


def _apply_delete_permission_group(
    state: AppliedState, payload: dict[str, Any], index: int
) -> AppliedState:
    group_id = payload.get("id")
    if not isinstance(group_id, str) or group_id not in state.permission_groups:
        raise ApplyError(f"entry {index}: unknown permission group")
    people = {
        key: {**person, "groups": [g for g in person["groups"] if g != group_id]}
        if person.get("groups") is not None
        else person
        for key, person in state.people.items()
    }
    for key, person in people.items():
        grants = person.get("administration", {})
        if group_id in grants.get("membershipGroups", []):
            people[key] = {
                **person,
                "administration": {
                    **grants,
                    "membershipGroups": [g for g in grants["membershipGroups"] if g != group_id],
                },
                "administrationVersion": person.get("administrationVersion", 0) + 1,
            }
    return replace(
        state,
        people=people,
        permission_groups={k: v for k, v in state.permission_groups.items() if k != group_id},
    )


def _person_record(raw: Any) -> dict[str, Any]:
    """One `SnapshotPerson`, checked as a client-key record is: exactly the
    fields the schema has, so nothing else can ride into the log."""
    if not isinstance(raw, dict):
        raise ApplyError("person record must be an object")
    allowed = {
        "id",
        "name",
        "displayName",
        "email",
        "passwordVerifier",
        "apps",
        "permissions",
        "groups",
        "disabled",
        "createdAt",
        "passwordChangedAt",
        "administration",
        "administrationVersion",
    }
    if "helperGrants" in raw:
        raise _retired("a person record with folder grants (helperGrants)")
    if raw.keys() - allowed:
        raise ApplyError("unexpected person property")
    grants = raw.get("administration", {})
    if not isinstance(grants, dict) or grants.keys() - {
        "managePeople",
        "manageGroups",
        "membershipGroups",
    }:
        raise ApplyError("invalid administration grants")
    for permission in ("managePeople", "manageGroups"):
        if type(grants.get(permission, False)) is not bool:
            raise ApplyError("invalid administration permission")
    _string_list(
        grants.get("membershipGroups", []), "administration scope", limit=64, item_limit=64
    )
    version = raw.get("administrationVersion", 0)
    if type(version) is not int or version < 0:
        raise ApplyError("invalid administration version")
    for key in ("id", "name", "passwordVerifier"):
        if not isinstance(raw.get(key), str) or not raw[key]:
            raise ApplyError(f"invalid person {key}")
    if not raw["passwordVerifier"].startswith("$argon2id$"):
        raise ApplyError("a person's verifier must be Argon2id")
    if raw["name"].casefold() == OPERATOR_NAME:
        raise ApplyError(f"{OPERATOR_NAME!r} is the owner's name")
    if raw.get("displayName") is not None and not isinstance(raw["displayName"], str):
        raise ApplyError("invalid person displayName")
    email = raw.get("email")
    if email is not None and (not isinstance(email, str) or "@" not in email):
        raise ApplyError("invalid person email")
    apps = raw.get("apps")
    if apps is not None and (
        not isinstance(apps, list) or any(not isinstance(a, str) or not a for a in apps)
    ):
        raise ApplyError("invalid person apps")
    permissions = raw.get("permissions")
    if permissions is not None and (
        not isinstance(permissions, list)
        or len(permissions) > 16
        or any(not isinstance(p, str) or p not in PERSON_PERMISSIONS for p in permissions)
        or len(set(permissions)) != len(permissions)
    ):
        raise ApplyError("invalid person permissions")
    groups = raw.get("groups")
    if groups is not None:
        _string_list(groups, "person groups", limit=64, item_limit=64)
    if not isinstance(raw.get("disabled"), bool):
        raise ApplyError("invalid person disabled")
    _timestamp(raw, "createdAt", "person")
    _timestamp(raw, "passwordChangedAt", "person")
    return dict(raw)


def _apply_put_person(state: AppliedState, payload: dict[str, Any], index: int) -> AppliedState:
    """Add a person or replace one, by id. A name is unique, case-folded."""
    record = _person_record(payload.get("person"))
    _check_group_memberships(state.permission_groups, record)
    folded = record["name"].casefold()
    for other in state.people.values():
        if other["id"] != record["id"] and other["name"].casefold() == folded:
            raise ApplyError(f"entry {index}: the name {record['name']!r} is taken")
    return replace(state, people={**state.people, record["id"]: record})


def _apply_set_person_password(
    state: AppliedState, payload: dict[str, Any], index: int
) -> AppliedState:
    person_id = payload.get("id")
    if person_id not in state.people:
        raise ApplyError(f"entry {index}: unknown person")
    record = _person_record(
        {
            **state.people[person_id],
            "passwordVerifier": payload.get("passwordVerifier"),
            "passwordChangedAt": payload.get("passwordChangedAt"),
            "administrationVersion": state.people[person_id].get("administrationVersion", 0) + 1,
        }
    )
    return replace(state, people={**state.people, person_id: record})


def _apply_delete_person(state: AppliedState, payload: dict[str, Any], index: int) -> AppliedState:
    person_id = payload.get("id")
    if person_id not in state.people:
        raise ApplyError(f"entry {index}: unknown person")
    # A site has one owner, and only they decide who may use it: their
    # sites go with them.
    return replace(
        state,
        people={k: v for k, v in state.people.items() if k != person_id},
        browser_sign_in=sign_in_state.without_person(state.browser_sign_in, person_id),
        sites={k: v for k, v in state.sites.items() if v.owner != person_id},
    )


def _oidc_client_record(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ApplyError("sign-in client record must be an object")
    if raw.keys() - {"clientId", "name", "secretVerifier", "redirectUris", "owner", "createdAt"}:
        raise ApplyError("unexpected sign-in client property")
    for key in ("clientId", "name", "secretVerifier"):
        if not isinstance(raw.get(key), str) or not raw[key]:
            raise ApplyError(f"invalid sign-in client {key}")
    uris = raw.get("redirectUris")
    if not isinstance(uris, list) or not uris or any(not isinstance(u, str) or not u for u in uris):
        raise ApplyError("invalid sign-in client redirectUris")
    if raw.get("owner") is not None and not isinstance(raw["owner"], str):
        raise ApplyError("invalid sign-in client owner")
    _timestamp(raw, "createdAt", "sign-in client")
    return dict(raw)


def _apply_put_oidc_client(
    state: AppliedState, payload: dict[str, Any], index: int
) -> AppliedState:
    record = _oidc_client_record(payload.get("client"))
    if record["clientId"] in state.oidc_clients:
        raise ApplyError(f"entry {index}: that client id already exists")
    return replace(state, oidc_clients={**state.oidc_clients, record["clientId"]: record})


def _apply_set_oidc_client_redirects(
    state: AppliedState, payload: dict[str, Any], index: int
) -> AppliedState:
    """One client's redirect URIs, replaced; its id, secret and owner kept."""
    client_id = payload.get("clientId")
    if not isinstance(client_id, str) or client_id not in state.oidc_clients:
        raise ApplyError(f"entry {index}: unknown sign-in client")
    record = state.oidc_clients[client_id]
    updated = _oidc_client_record({**record, "redirectUris": payload.get("redirectUris")})
    return replace(state, oidc_clients={**state.oidc_clients, client_id: updated})


def _apply_delete_oidc_client(
    state: AppliedState, payload: dict[str, Any], index: int
) -> AppliedState:
    client_id = payload.get("clientId")
    if client_id not in state.oidc_clients:
        raise ApplyError(f"entry {index}: unknown sign-in client")
    # A person's list of apps keeps no id of an app that is gone.
    people = {
        key: (
            {**person, "apps": [a for a in person["apps"] if a != client_id]}
            if isinstance(person.get("apps"), list)
            else person
        )
        for key, person in state.people.items()
    }
    clients = {k: v for k, v in state.oidc_clients.items() if k != client_id}
    groups = {
        key: {**group, "apps": [a for a in group["apps"] if a != client_id]}
        if group["apps"] is not None
        else group
        for key, group in state.permission_groups.items()
    }
    return replace(state, oidc_clients=clients, people=people, permission_groups=groups)


def _oidc_key_record(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.keys() - {"kid", "sealedKey", "publicJwk", "createdAt"}:
        raise ApplyError("invalid sign-in key record")
    for key in ("kid", "sealedKey"):
        if not isinstance(raw.get(key), str) or not raw[key]:
            raise ApplyError(f"invalid sign-in key {key}")
    jwk = raw.get("publicJwk")
    if not isinstance(jwk, dict) or jwk.get("kty") != "RSA" or "d" in jwk:
        raise ApplyError("a sign-in key's public JWK must be RSA, and public")
    _timestamp(raw, "createdAt", "sign-in key")
    return dict(raw)


def _apply_put_oidc_key(state: AppliedState, payload: dict[str, Any], index: int) -> AppliedState:
    record = _oidc_key_record(payload.get("key"))
    if any(k["kid"] == record["kid"] for k in state.oidc_keys):
        raise ApplyError(f"entry {index}: that sign-in key is already here")
    return replace(state, oidc_keys=(record, *state.oidc_keys)[:OIDC_KEYS_KEPT])


def _apply_revoke_sign_in(state: AppliedState, payload: dict[str, Any], index: int) -> AppliedState:
    """As `revokeSession`, for a sign-in at `/oidc/revoke`."""
    sid = _require_str(payload, "sid", index)
    exp = payload.get("exp")
    pruned = payload.get("prunedBefore", 0)
    if not isinstance(exp, int) or isinstance(exp, bool):
        raise ApplyError(f"entry {index}: 'exp' must be an integer")
    if not isinstance(pruned, int) or isinstance(pruned, bool):
        raise ApplyError(f"entry {index}: 'prunedBefore' must be an integer")
    kept = {k: v for k, v in state.revoked_sign_ins.items() if v >= pruned}
    kept[sid] = exp
    return replace(state, revoked_sign_ins=kept)


SITE_ID_ALPHABET = frozenset("abcdefghijklmnopqrstuvwxyz234567")


def valid_site_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 28
        and value.startswith("s-")
        and set(value[2:]) <= SITE_ID_ALPHABET
    )


def _site_id(payload: dict[str, Any], index: int) -> str:
    value = payload.get("id")
    if not valid_site_id(value):
        raise ApplyError(f"entry {index}: invalid site id {value!r}")
    return str(value)


def _dev_grants(raw: Any, index: int) -> tuple[dict[str, Any], ...]:
    if not isinstance(raw, list) or len(raw) > 64:
        raise ApplyError(f"entry {index}: devGrants must be a list of at most 64")
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for grant in raw:
        if (
            not isinstance(grant, dict)
            or set(grant) != {"folderId", "writable"}
            or not isinstance(grant["folderId"], str)
            or not grant["folderId"]
            or len(grant["folderId"]) > 64
            or type(grant["writable"]) is not bool
            or grant["folderId"] in seen
        ):
            raise ApplyError(f"entry {index}: invalid dev grant")
        seen.add(grant["folderId"])
        out.append({"folderId": grant["folderId"], "writable": grant["writable"]})
    return tuple(sorted(out, key=lambda g: g["folderId"]))


def _apply_enroll_site(state: AppliedState, payload: dict[str, Any], index: int) -> AppliedState:
    site = _site_id(payload, index)
    if site in state.sites:
        raise ApplyError(f"entry {index}: site {site} is enrolled already")
    owner = _require_str(payload, "owner", index)
    if owner not in state.people:
        raise ApplyError(f"entry {index}: a site's owner must be a person on this install")
    record = SiteRecord(
        id=site,
        label=_require_str(payload, "label", index),
        owner=owner,
        tokenPublicKey=_require_str(payload, "tokenPublicKey", index),
        enrolledAt=_require_str(payload, "enrolledAt", index),
    )
    return replace(state, sites={**state.sites, site: record})


def _apply_remove_site(state: AppliedState, payload: dict[str, Any], index: int) -> AppliedState:
    site = _site_id(payload, index)
    if site not in state.sites:
        raise ApplyError(f"entry {index}: cannot remove unknown site {site}")
    return replace(state, sites={k: v for k, v in state.sites.items() if k != site})


def _apply_set_site_host(state: AppliedState, payload: dict[str, Any], index: int) -> AppliedState:
    site = _site_id(payload, index)
    record = state.sites.get(site)
    if record is None:
        raise ApplyError(f"entry {index}: unknown site {site}")
    host = payload.get("hostNode")
    if host is not None and (not isinstance(host, str) or not host):
        raise ApplyError(f"entry {index}: hostNode must be a node name or null")
    return replace(state, sites={**state.sites, site: replace(record, hostNode=host)})


def _apply_set_site_dev_grants(
    state: AppliedState, payload: dict[str, Any], index: int
) -> AppliedState:
    site = _site_id(payload, index)
    record = state.sites.get(site)
    if record is None:
        raise ApplyError(f"entry {index}: unknown site {site}")
    grants = _dev_grants(payload.get("devGrants"), index)
    return replace(state, sites={**state.sites, site: replace(record, devGrants=grants)})


def _site_canonical(site: SiteRecord) -> dict[str, Any]:
    return {
        "id": site.id,
        "label": site.label,
        "owner": site.owner,
        "tokenPublicKey": site.tokenPublicKey,
        "enrolledAt": site.enrolledAt,
        "hostNode": site.hostNode,
        "devGrants": [dict(g) for g in site.devGrants],
    }


def _site_from_canonical(raw: Any) -> SiteRecord:
    if not isinstance(raw, dict) or not valid_site_id(raw.get("id")):
        raise ApplyError("invalid site in snapshot")
    return SiteRecord(
        id=str(raw["id"]),
        label=str(raw["label"]),
        owner=str(raw["owner"]),
        tokenPublicKey=str(raw["tokenPublicKey"]),
        enrolledAt=str(raw["enrolledAt"]),
        hostNode=raw.get("hostNode"),
        devGrants=_dev_grants(raw.get("devGrants") or [], 0),
    )


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
    OP_REVOKE_CLIENT_KEY: _apply_revoke_client_key,
    OP_SET_CLIENT_KEY_LIMITS: _apply_set_client_key_limits,
    OP_PUT_CLIENT_ADMISSION: _apply_put_client_admission,
    OP_REVOKE_SESSION: _apply_revoke_session,
    OP_PUT_PERSON: _apply_put_person,
    OP_PUT_PERMISSION_GROUP: _apply_put_permission_group,
    OP_DELETE_PERMISSION_GROUP: _apply_delete_permission_group,
    OP_SET_PERSON_PASSWORD: _apply_set_person_password,
    OP_DELETE_PERSON: _apply_delete_person,
    OP_PUT_OIDC_CLIENT: _apply_put_oidc_client,
    OP_DELETE_OIDC_CLIENT: _apply_delete_oidc_client,
    OP_PUT_OIDC_KEY: _apply_put_oidc_key,
    OP_REVOKE_SIGN_IN: _apply_revoke_sign_in,
    OP_SET_OIDC_CLIENT_REDIRECTS: _apply_set_oidc_client_redirects,
    OP_ENROLL_SITE: _apply_enroll_site,
    OP_REMOVE_SITE: _apply_remove_site,
    OP_SET_SITE_HOST: _apply_set_site_host,
    OP_SET_SITE_DEV_GRANTS: _apply_set_site_dev_grants,
    OP_SET_STANDBY: _apply_set_standby,
    OP_LABEL_NODE: _apply_label_node,
    **{op: partial(sign_in_state.apply, op=op) for op in sign_in_state.OPS},
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
                "label": n.label,
                "url": n.url,
                "reachable": False,
                "publicKey": n.publicKey,
                "signingPublicKey": n.signingPublicKey,
                "tokenPublicKey": n.tokenPublicKey,
                "grants": list(n.grants),
                "owner": n.owner,
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
        "accessAudit": state.access_audit,
        "browserSignIn": state.browser_sign_in,
        "salt": identity.salt,
        "passphraseVerifier": identity.passphraseVerifier,
        "sealedSigningKey": identity.sealedSigningKey,
        "rootTokenPublicKey": identity.rootTokenPublicKey,
        "revokedSessions": [
            {"jti": jti, "exp": exp} for jti, exp in sorted(state.revoked_sessions.items())
        ],
        "sealedControlKey": identity.sealedControlKey,
        "controlPublicKey": identity.controlPublicKey,
        "sealedRecoveryKey": identity.sealedRecoveryKey,
        "signingKeyId": identity.signingKeyId,
        "people": [state.people[key] for key in sorted(state.people)],
        "permissionGroups": [
            state.permission_groups[key] for key in sorted(state.permission_groups)
        ],
        "sites": [_site_canonical(state.sites[key]) for key in sorted(state.sites)],
        "oidcClients": [state.oidc_clients[key] for key in sorted(state.oidc_clients)],
        "oidcKeys": [dict(k) for k in state.oidc_keys],
        "revokedSignIns": [
            {"jti": sid, "exp": exp} for sid, exp in sorted(state.revoked_sign_ins.items())
        ],
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
    from .access_audit import validate as validate_audit
    from .client_admission import validate_ledger

    if raw.get("nodeHelpers"):
        raise _retired("a snapshot with node folders (nodeHelpers)")
    for node in raw.get("nodes") or []:
        if isinstance(node, dict) and "files" in (node.get("grants") or ()):
            raise _retired(f"a snapshot node {node.get('name')!r} with the grant `files`")

    runtimes: dict[str, RuntimeRecord] = {}
    for entry in raw.get("runtimes") or []:
        spec = entry.get("spec") or {}
        node = str(entry.get("node") or "")
        name = str(spec.get("name") or "")
        runtimes[f"{node}/{name}"] = RuntimeRecord(node=node, spec=spec)

    groups: dict[str, dict[str, Any]] = {}
    for item in raw.get("permissionGroups") or []:
        group = _permission_group_record(item)
        if group["id"] in groups or any(
            g["name"].casefold() == group["name"].casefold() for g in groups.values()
        ):
            raise ApplyError("duplicate permission group")
        groups[group["id"]] = group
    people = {(p := _person_record(item))["id"]: p for item in raw.get("people") or []}
    for person in people.values():
        _check_group_memberships(groups, person)
    return AppliedState(
        permission_groups=groups,
        index=int(raw.get("index") or 0),
        epoch=int(raw.get("epoch") or 1),
        nodes={
            str(n["name"]): NodeRecord(
                name=str(n["name"]),
                role=str(n.get("role") or ROLE_AGENT),
                label=n.get("label"),
                url=n.get("url"),
                publicKey=n.get("publicKey"),
                signingPublicKey=n.get("signingPublicKey"),
                tokenPublicKey=n.get("tokenPublicKey"),
                grants=tuple(n.get("grants") or ("node",)),
                owner=n.get("owner"),
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
        access_audit=validate_audit(raw.get("accessAudit", [])),
        client_keys=_key_records(raw.get("clientKeys", [])),
        revoked_sessions={str(r["jti"]): int(r["exp"]) for r in raw.get("revokedSessions") or []},
        people=people,
        browser_sign_in=sign_in_state.validate(raw.get("browserSignIn", {}), people),
        sites={(s := _site_from_canonical(item)).id: s for item in raw.get("sites") or []},
        oidc_clients={
            (c := _oidc_client_record(item))["clientId"]: c for item in raw.get("oidcClients") or []
        },
        oidc_keys=tuple(_oidc_key_record(item) for item in raw.get("oidcKeys") or []),
        revoked_sign_ins={str(r["jti"]): int(r["exp"]) for r in raw.get("revokedSignIns") or []},
        identity=InstallIdentity(
            salt=raw.get("salt"),
            passphraseVerifier=raw.get("passphraseVerifier"),
            sealedSigningKey=raw.get("sealedSigningKey"),
            rootTokenPublicKey=raw.get("rootTokenPublicKey"),
            sealedControlKey=raw.get("sealedControlKey"),
            controlPublicKey=raw.get("controlPublicKey"),
            sealedRecoveryKey=raw.get("sealedRecoveryKey"),
            signingKeyId=raw.get("signingKeyId"),
        ),
    )
