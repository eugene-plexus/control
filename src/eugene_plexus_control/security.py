"""The trust root's secrets: the passphrase, the master key, sealing, join tokens.

Tokens are minted and verified in `tokens` (per-node token keys,
2026-09-25). The token key is independent of the master encryption key.
"""

from __future__ import annotations

import base64
import logging
import secrets
from dataclasses import dataclass
from typing import Any

import argon2
import argon2.low_level
import nacl.exceptions
import nacl.secret
import nacl.utils
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

log = logging.getLogger(__name__)

# Argon2id parameters — 2026 OWASP recommendations for interactive
# logins. This runs on login and at promotion, so expensive is correct.
_ARGON2_TIME_COST = 3
_ARGON2_MEMORY_COST = 65_536  # KiB
_ARGON2_PARALLELISM = 4
_ARGON2_HASH_LEN = 32  # bytes — drives the secretbox key length


ENVELOPE_ALG = "secretbox-xsalsa20poly1305"


# --------------------------------------------------------------------------- #
# Passphrase
# --------------------------------------------------------------------------- #


_password_hasher = argon2.PasswordHasher(
    time_cost=_ARGON2_TIME_COST,
    memory_cost=_ARGON2_MEMORY_COST,
    parallelism=_ARGON2_PARALLELISM,
    hash_len=_ARGON2_HASH_LEN,
)


def hash_passphrase(passphrase: str) -> str:
    """Argon2id-PHC hash suitable for storage and for replication.

    This is the `passphraseVerifier` in the snapshot: it lets a standby
    authenticate the operator at promotion without ever holding the
    master key."""
    if not passphrase:
        raise ValueError("passphrase must not be empty")
    return _password_hasher.hash(passphrase)


def verify_passphrase(passphrase: str, stored_hash: str) -> bool:
    """Constant-time verification. Returns True on match."""
    if not passphrase or not stored_hash:
        return False
    try:
        _password_hasher.verify(stored_hash, passphrase)
        return True
    except argon2.exceptions.VerifyMismatchError:
        return False
    except argon2.exceptions.InvalidHashError:
        log.warning("stored passphrase verifier is malformed; treating as no-match")
        return False


# --------------------------------------------------------------------------- #
# Master key
# --------------------------------------------------------------------------- #


def generate_master_key_salt() -> bytes:
    """16 random bytes, per install, replicated in the snapshot.

    Replicated because without it the same passphrase derives a
    different key, and a standby that cannot derive the master key
    cannot be promoted."""
    return secrets.token_bytes(16)


def derive_master_key(passphrase: str, salt: bytes) -> bytes:
    """Deterministic 32-byte key from passphrase + salt via Argon2id."""
    if not passphrase:
        raise ValueError("passphrase must not be empty")
    if len(salt) < 8:
        raise ValueError("salt must be at least 8 bytes")
    return argon2.low_level.hash_secret_raw(
        secret=passphrase.encode("utf-8"),
        salt=salt,
        time_cost=_ARGON2_TIME_COST,
        memory_cost=_ARGON2_MEMORY_COST,
        parallelism=_ARGON2_PARALLELISM,
        hash_len=_ARGON2_HASH_LEN,
        type=argon2.low_level.Type.ID,
    )


# --------------------------------------------------------------------------- #
# At-rest envelope (libsodium secretbox)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Envelope:
    """Canonical shape of an at-rest encrypted secret. Wire-identical to
    `MasterKeyEnvelope` in common.yaml."""

    alg: str
    nonce: str  # base64
    ciphertext: str  # base64

    def to_dict(self) -> dict[str, str]:
        return {"alg": self.alg, "nonce": self.nonce, "ciphertext": self.ciphertext}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Envelope:
        alg = raw.get("alg")
        nonce = raw.get("nonce")
        ciphertext = raw.get("ciphertext")
        if alg != ENVELOPE_ALG:
            raise ValueError(f"unsupported envelope alg: {alg!r}")
        if not isinstance(nonce, str) or not isinstance(ciphertext, str):
            raise ValueError("envelope nonce/ciphertext must be base64 strings")
        return cls(alg=alg, nonce=nonce, ciphertext=ciphertext)


def is_envelope(value: Any) -> bool:
    """Structural check — does this dict look like an Envelope?"""
    return (
        isinstance(value, dict)
        and value.get("alg") == ENVELOPE_ALG
        and "nonce" in value
        and "ciphertext" in value
    )


def seal(plaintext: str, master_key: bytes) -> Envelope:
    """Encrypt a plaintext value. Fresh 24-byte nonce per call.

    Note for anything that ends up in the log or a snapshot: sealing is
    **not** deterministic, so a sealed value must be written once and
    then replicated as an opaque string. Re-sealing the same plaintext
    on two roots would produce two different ciphertexts and break
    replay equivalence — which is why key material travels as a payload
    the writer stamped, never as something a replica recomputes."""
    if len(master_key) != 32:
        raise ValueError("master key must be 32 bytes")
    box = nacl.secret.SecretBox(master_key)
    nonce = nacl.utils.random(nacl.secret.SecretBox.NONCE_SIZE)
    ciphertext = box.encrypt(plaintext.encode("utf-8"), nonce).ciphertext
    return Envelope(
        alg=ENVELOPE_ALG,
        nonce=base64.b64encode(nonce).decode("ascii"),
        ciphertext=base64.b64encode(ciphertext).decode("ascii"),
    )


def open_envelope(envelope: Envelope, master_key: bytes) -> str:
    """Decrypt back to plaintext. Raises ValueError on bad key or
    tampered ciphertext."""
    if len(master_key) != 32:
        raise ValueError("master key must be 32 bytes")
    try:
        nonce = base64.b64decode(envelope.nonce, validate=True)
        ciphertext = base64.b64decode(envelope.ciphertext, validate=True)
    except Exception as e:
        raise ValueError(f"envelope decoding failed: {e}") from e
    box = nacl.secret.SecretBox(master_key)
    try:
        return box.decrypt(ciphertext, nonce).decode("utf-8")
    except nacl.exceptions.CryptoError as e:
        raise ValueError(f"envelope decryption failed: {e}") from e


def seal_b64(raw: bytes, master_key: bytes) -> str:
    """Seal raw key material into a single opaque string.

    Used for the three things the snapshot carries sealed — the signing
    key, the control root's identity private key, and the recovery
    recipient's private key. One string rather than an envelope dict
    because these travel as `Snapshot` fields typed as strings."""
    envelope = seal(base64.b64encode(raw).decode("ascii"), master_key)
    return base64.b64encode(f"{envelope.nonce}.{envelope.ciphertext}".encode("ascii")).decode(
        "ascii"
    )


def open_b64(sealed: str, master_key: bytes) -> bytes:
    """Inverse of `seal_b64`."""
    try:
        joined = base64.b64decode(sealed, validate=True).decode("ascii")
        nonce, ciphertext = joined.split(".", 1)
    except Exception as e:
        raise ValueError(f"sealed value is malformed: {e}") from e
    plaintext = open_envelope(
        Envelope(alg=ENVELOPE_ALG, nonce=nonce, ciphertext=ciphertext), master_key
    )
    return base64.b64decode(plaintext, validate=True)


# --------------------------------------------------------------------------- #
# This root's token key
# --------------------------------------------------------------------------- #


def generate_signing_key() -> bytes:
    """This root's token key: Ed25519, PKCS8 PEM, sealed into the log.

    Signing lives in `tokens`; this only makes the key, in the one format
    `seal_b64` carries.
    """
    return Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


# --------------------------------------------------------------------------- #
# Join tokens
# --------------------------------------------------------------------------- #


def generate_join_token() -> str:
    """A single-use enrollment credential, shown once.

    Deliberately opaque random rather than a JWT: it is stored only as a
    hash, so there is nothing to verify a signature against and nothing
    to look up. A lost token is re-minted, never recovered."""
    return secrets.token_urlsafe(32)


def hash_join_token(token: str) -> str:
    """What actually gets stored. `sha256` and not Argon2id on purpose:
    the token is 256 bits of machine-generated entropy, so there is no
    dictionary to slow an attacker down against, and minting has to stay
    fast enough to be a UI action."""
    import hashlib

    return hashlib.sha256(token.encode("utf-8")).hexdigest()
