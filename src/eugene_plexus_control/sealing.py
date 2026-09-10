"""Per-node sealing, with a recovery recipient.

The model this replaces threaded one install-wide master key to every
spawned child. Across hosts that means one compromised GPU box in
another building yields **every secret in the install**, and "GPUs in
different physical buildings, internet-connected only" is a stated
deployment target. So a secret is sealed to the node whose component
will read it — topology already says which node that is — and to nothing
else.

Except one thing. Sealing only to the node makes a dead node's secrets
unrecoverable, so every secret is sealed to **two** recipients: the
target node, and a recovery key held by the control root that is itself
sealed under the passphrase-derived key. Normal operation never decrypts
a secret here.

| Compromise | Yields |
|---|---|
| One node | that node's secrets only |
| This process | nothing directly — the recovery key is sealed under the passphrase |
| This process **and** the passphrase | everything. Unavoidable, and what the passphrase *is* |

Anonymous sealed boxes (X25519 + XSalsa20-Poly1305) rather than
authenticated boxes: the recipient needs to know a secret arrived and
decrypt it, and does not need to know which of the control root's keys
sent it, because the transport is already authenticated and inside a
mesh VPN. An authenticated box would add a second key to distribute for
no property we use.

A sealed value is opaque to everything except the two recipients, which
is why it can be carried in the log and the snapshot as a string.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

import nacl.exceptions
import nacl.public
import nacl.signing

SEAL_ALG = "sealedbox-x25519-xsalsa20poly1305"

RECIPIENT_RECOVERY = "recovery"
RECIPIENT_NODE_PREFIX = "node:"


class SealError(Exception):
    """A key was malformed, or a ciphertext did not open."""


@dataclass(frozen=True)
class Keypair:
    """Base64 halves of an X25519 keypair. Strings rather than bytes
    because every consumer stores or transmits them."""

    private: str
    public: str


def generate_sealing_keypair() -> Keypair:
    """A recipient keypair. Used for the recovery recipient here, and
    generated independently by each agent for its own node identity —
    whose private half never leaves that host, which is the entire point
    of per-node sealing."""
    private = nacl.public.PrivateKey.generate()
    return Keypair(
        private=base64.b64encode(bytes(private)).decode("ascii"),
        public=base64.b64encode(bytes(private.public_key)).decode("ascii"),
    )


def generate_control_identity() -> Keypair:
    """The control root's Ed25519 identity.

    Signing rather than sealing, because its job is different: nodes
    record the public half at enrollment as `controlPublicKey` and check
    that later epoch changes come from a root they recognise rather than
    from anything that can reach their port.

    Replicated (private half sealed) so that promotion is a change of
    host rather than a change of identity — a promoted standby with a
    fresh keypair would be refused by every node it tried to command,
    which is the fencing mechanism firing at exactly the wrong target.
    """
    signing = nacl.signing.SigningKey.generate()
    return Keypair(
        private=base64.b64encode(bytes(signing)).decode("ascii"),
        public=base64.b64encode(bytes(signing.verify_key)).decode("ascii"),
    )


def _load_public(public_b64: str) -> nacl.public.PublicKey:
    try:
        raw = base64.b64decode(public_b64, validate=True)
    except Exception as exc:
        raise SealError(f"public key is not valid base64: {exc}") from exc
    if len(raw) != 32:
        raise SealError(f"public key must be 32 bytes, got {len(raw)}")
    return nacl.public.PublicKey(raw)


def _load_private(private_b64: str) -> nacl.public.PrivateKey:
    try:
        raw = base64.b64decode(private_b64, validate=True)
    except Exception as exc:
        raise SealError(f"private key is not valid base64: {exc}") from exc
    if len(raw) != 32:
        raise SealError(f"private key must be 32 bytes, got {len(raw)}")
    return nacl.public.PrivateKey(raw)


def seal_to_recipients(plaintext: str, recipients: dict[str, str]) -> str:
    """Seal one plaintext to several public keys, one ciphertext each.

    Returns a single opaque string so a sealed secret is one value
    wherever it is stored. Recipient labels are inside it rather than
    positional, because a secret re-sealed after a node was rebuilt has
    to say which node it now belongs to.

    Not deterministic — a sealed box carries a fresh ephemeral key — so a
    sealed value must be **written once and replicated verbatim**. A
    replica that re-sealed the same plaintext would produce different
    bytes and break replay equivalence, which is why key material
    travels as a payload the writer stamped rather than as something a
    replica recomputes.
    """
    if not recipients:
        raise SealError("a secret with no recipients is a secret nobody can read")
    sealed: dict[str, str] = {}
    for label, public_b64 in recipients.items():
        box = nacl.public.SealedBox(_load_public(public_b64))
        sealed[label] = base64.b64encode(box.encrypt(plaintext.encode("utf-8"))).decode("ascii")
    document: dict[str, Any] = {"alg": SEAL_ALG, "recipients": sealed}
    return base64.b64encode(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")


def open_as_recipient(sealed: str, *, label: str, private_b64: str) -> str:
    """Open a sealed secret as one named recipient.

    A node calls this with its own label and private key; the control
    root calls it with `recovery` during an explicit recovery, having
    first unsealed the recovery key under the passphrase.
    """
    document = _parse(sealed)
    ciphertext_b64 = document["recipients"].get(label)
    if ciphertext_b64 is None:
        raise SealError(
            f"this secret was not sealed to {label!r}; it names {sorted(document['recipients'])}"
        )
    try:
        ciphertext = base64.b64decode(ciphertext_b64, validate=True)
    except Exception as exc:
        raise SealError(f"ciphertext for {label!r} is not valid base64: {exc}") from exc
    box = nacl.public.SealedBox(_load_private(private_b64))
    try:
        return box.decrypt(ciphertext).decode("utf-8")
    except nacl.exceptions.CryptoError as exc:
        raise SealError(f"could not open the secret as {label!r}: {exc}") from exc


def recipients_of(sealed: str) -> list[str]:
    """Who can read this. Lets an operator see whether a secret is still
    reachable after a node was revoked, without holding any key."""
    return sorted(_parse(sealed)["recipients"])


def public_from_private(private_b64: str) -> str:
    """Derive an X25519 public key from its private half.

    The recovery recipient's public key is **not** replicated for this
    reason: a field that can be computed is a field that can disagree
    with what it was computed from. A root that holds the private half —
    which means a root the operator has unlocked — can always produce
    it, and a root that does not hold it has no business handing it out.
    """
    return base64.b64encode(bytes(_load_private(private_b64).public_key)).decode("ascii")


def rekey_message(*, signing_key: str, signing_key_id: str, epoch: int) -> bytes:
    """The canonical bytes a re-key is signed over — `RekeyRequest.signature`
    in `agent.yaml`, byte for byte: the JSON object with keys sorted and no
    whitespace. Three fields, one serializer, stated in the contract so the
    agent implements it from the same sentence."""
    return json.dumps(
        {"epoch": epoch, "signingKey": signing_key, "signingKeyId": signing_key_id},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sign_rekey(control_private_key: str, message: bytes) -> str:
    """Detached Ed25519 signature by the control identity, base64.

    Why a signature and not a bearer: a rotation invalidates every service
    token in the install, including any this root could present, and a
    re-run of an interrupted rotation cannot know which key each node
    still holds. The identity key does not rotate, which is what makes it
    the one credential that survives the operation — and it is what nodes
    recorded `controlPublicKey` at enrollment for."""
    try:
        seed = base64.b64decode(control_private_key, validate=True)
        signing = nacl.signing.SigningKey(seed)
    except Exception as exc:
        raise SealError(f"control identity key is malformed: {exc}") from exc
    return base64.b64encode(signing.sign(message).signature).decode("ascii")


def verify_rekey(control_public_key: str, message: bytes, signature: str) -> bool:
    """The agent's side, kept here so the fake agents in this repo's tests
    verify exactly what a real one does."""
    try:
        verify_key = nacl.signing.VerifyKey(base64.b64decode(control_public_key, validate=True))
        verify_key.verify(message, base64.b64decode(signature, validate=True))
    except Exception:
        return False
    return True


def node_recipient_label(node_name: str) -> str:
    return f"{RECIPIENT_NODE_PREFIX}{node_name}"


def _parse(sealed: str) -> dict[str, Any]:
    try:
        document = json.loads(base64.b64decode(sealed, validate=True).decode("utf-8"))
    except Exception as exc:
        raise SealError(f"sealed value is malformed: {exc}") from exc
    if not isinstance(document, dict):
        raise SealError("sealed value is not an object")
    if document.get("alg") != SEAL_ALG:
        raise SealError(f"unsupported seal alg: {document.get('alg')!r}")
    recipients = document.get("recipients")
    if not isinstance(recipients, dict):
        raise SealError("sealed value has no recipients object")
    return {"alg": SEAL_ALG, "recipients": recipients}
