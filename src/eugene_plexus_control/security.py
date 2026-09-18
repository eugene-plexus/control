"""Trust-root primitives: passphrase, master key, tokens, envelopes.

This moved here from the agent, which held it only because it used to be
the single process in the install. The control root is the trust root
now, and there is exactly one active.

What this module owns:

  * **Passphrase verification** via Argon2id-PHC. Parameters and salt are
    embedded in the stored hash, so verifying needs no out-of-band
    context.

  * **Master-key derivation** via Argon2id raw mode, from the passphrase
    plus a separately stored 16-byte salt. Deterministic, so the same
    passphrase always derives the same key — which is what lets a
    standby derive it at promotion from a salt it replicated rather than
    from a key it was shipped. The master key never lands on disk in
    plaintext.

  * **Session and service token issuance** via JWT-HS256. HS256 because
    the signing key is shared with components that must validate
    independently; asymmetric would force every component to hold a
    public key or make a round trip, neither of which simplifies
    anything.

  * **At-rest envelopes** via libsodium secretbox, wire-identical to
    `MasterKeyEnvelope` in common.yaml so an envelope written by one
    component opens in another given the same key.

The one behavioural change from the agent's version: **a restart is no
longer a re-key.** Through M4 service tokens were "rotated on each
watchdog restart", which was harmless when one process spawned
everything and is an outage once there are N nodes — a restart that
re-keys the install breaks every component that has not yet been told.
Rotation is now an explicit operation with its own log entry
(`rotateSigningKey`), and the key is persisted sealed under the master
key so a restart keeps it.
"""

from __future__ import annotations

import base64
import logging
import secrets
import time
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

import argon2
import argon2.low_level
import jwt
import nacl.exceptions
import nacl.secret
import nacl.utils

log = logging.getLogger(__name__)

# Argon2id parameters — 2026 OWASP recommendations for interactive
# logins. This runs on login and at promotion, so expensive is correct.
_ARGON2_TIME_COST = 3
_ARGON2_MEMORY_COST = 65_536  # KiB
_ARGON2_PARALLELISM = 4
_ARGON2_HASH_LEN = 32  # bytes — drives the secretbox key length

_JWT_ALG = "HS256"
_DEFAULT_SESSION_TTL_SECONDS = 14 * 24 * 3600
_DEFAULT_SERVICE_TTL_SECONDS = 365 * 24 * 3600

AUDIENCE_OPERATOR = "operator"
SERVICE_AUDIENCE_PREFIX = "service:"

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
# JWT session + service tokens
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TokenPayload:
    """Decoded JWT claims. `iat` / `exp` are unix seconds."""

    sub: str
    aud: str
    iat: int
    exp: int


def generate_signing_key() -> bytes:
    """32 random bytes, used as the HMAC key for JWT signing.

    One per install, not one per process. Every node's components verify
    against this, which is precisely what a single-watchdog install could
    not do: a driver spawned on one host rejected a gateway's token from
    another because the keys were unrelated."""
    return secrets.token_bytes(32)


def issue_operator_token(
    *,
    signing_key: bytes,
    ttl_seconds: int = _DEFAULT_SESSION_TTL_SECONDS,
    now: int | None = None,
) -> tuple[str, int]:
    """Issue a UI session token. Returns `(token, exp_unix_seconds)`.

    The `jti` is not decoration. Without it, two logins in the same
    second produce byte-identical tokens — same claims, deterministic
    HS256 — so logging out and logging straight back in hands the
    operator the token they just revoked, and they are locked out until
    the clock ticks over. A random session id makes revocation mean
    "this session" rather than "every session issued this second".
    """
    issued_at = now if now is not None else int(time.time())
    expires_at = issued_at + ttl_seconds
    claims = {
        "sub": "operator",
        "aud": AUDIENCE_OPERATOR,
        "iat": issued_at,
        "exp": expires_at,
        "jti": secrets.token_urlsafe(12),
    }
    return jwt.encode(claims, signing_key, algorithm=_JWT_ALG), expires_at


def issue_service_token(
    *,
    signing_key: bytes,
    kind: str,
    ttl_seconds: int = _DEFAULT_SERVICE_TTL_SECONDS,
    now: int | None = None,
) -> str:
    """Issue a service token for one component kind.

    Encoded as `aud: "service:<kind>"` so a component can additionally
    check the audience matches on inbound calls — a leaked driver token
    cannot be used against the gateway."""
    issued_at = now if now is not None else int(time.time())
    claims = {
        "sub": kind,
        "aud": f"{SERVICE_AUDIENCE_PREFIX}{kind}",
        "iat": issued_at,
        "exp": issued_at + ttl_seconds,
    }
    return jwt.encode(claims, signing_key, algorithm=_JWT_ALG)


CLOCK_SKEW_LEEWAY_SECONDS = 300
"""How far apart two hosts' clocks may drift before a token is refused.

Five minutes: Kerberos's `MaxClockSkew`, and the window Entra and most
OAuth validators apply to `iat`, `nbf` and `exp`. **It was zero until
2026-09-15.** On the live two-machine install the control root's clock
ran half a second ahead of a worker whose Windows Time service had
stopped, and every token the root minted in the first half of each
second was refused by that worker as "not yet valid (iat)" a few
milliseconds later. Long-lived tokens (the gateway's, the operator's
session) passed, every health check said ok, and the root listed the
node `down` with no reason -- so it read as a key or enrollment fault
and was neither. A skew large enough to matter for security is a
broken clock; a broken clock is *reported* (`_note_clock_skew`), not
enforced by refusing traffic between two healthy hosts.
"""

_SKEW_WARN_AFTER_SECONDS = 2.0
_SKEW_WARN_INTERVAL_SECONDS = 60.0
_last_skew_warning = 0.0


def _note_clock_skew(iat: int, *, now: float | None = None) -> None:
    """Warn, at most once a minute, when a token was issued in this host's future.

    Accepted within `CLOCK_SKEW_LEEWAY_SECONDS`, so nothing breaks. Logged
    so a wrong clock on either host is visible long before the skew grows
    past the leeway and starts refusing traffic.
    """
    global _last_skew_warning
    current = time.time() if now is None else now
    ahead = iat - current
    if ahead <= _SKEW_WARN_AFTER_SECONDS:
        return
    if current - _last_skew_warning < _SKEW_WARN_INTERVAL_SECONDS:
        return
    _last_skew_warning = current
    log.warning(
        "accepted a token issued %.1f s in this host's future: the issuer's clock or "
        "this host's is wrong (tolerated up to %d s, then tokens are refused)",
        ahead,
        CLOCK_SKEW_LEEWAY_SECONDS,
    )


def decode_token(
    *,
    token: str,
    signing_key: bytes,
    accept_operator: bool = True,
    accept_any_service: bool = True,
    accept_service_kinds: Collection[str] | None = None,
) -> TokenPayload:
    """Verify signature and expiry, then check the audience class.

    Raises `jwt.InvalidTokenError` or a subclass on any failure, so the
    caller can collapse every rejection path into one branch.

    `accept_service_kinds` names the exact `service:<kind>` audiences
    that are acceptable, for the endpoints where "any component of this
    install" is too wide a door. It exists because the replication
    surface hands out the sealed signing key, the Argon2id salt and the
    passphrase verifier, and every `service:*` holder could read it --
    including the ones that hold no master key and so have something to
    gain from an offline attack on the verifier. Set it *with*
    `accept_any_service=False`; passing both means "any service, and
    also these", which is the wider of the two and almost never what a
    caller wants.
    """
    if not (accept_operator or accept_any_service or accept_service_kinds):
        raise ValueError("must accept at least one audience class")

    options: Any = {
        "require": ["sub", "aud", "iat", "exp"],
        # The caller decides which audiences are acceptable ("operator
        # OR any service:*"), so PyJWT's own single-audience check is
        # the wrong shape. `aud` is still required present, above.
        "verify_aud": False,
    }
    claims = jwt.decode(
        token,
        key=signing_key,
        algorithms=[_JWT_ALG],
        options=options,
        leeway=CLOCK_SKEW_LEEWAY_SECONDS,
    )
    _note_clock_skew(int(claims["iat"]))

    aud = str(claims["aud"])
    is_operator = accept_operator and aud == AUDIENCE_OPERATOR
    is_service = accept_any_service and aud.startswith(SERVICE_AUDIENCE_PREFIX)
    if not is_service and accept_service_kinds and aud.startswith(SERVICE_AUDIENCE_PREFIX):
        is_service = aud.removeprefix(SERVICE_AUDIENCE_PREFIX) in set(accept_service_kinds)
    if not (is_operator or is_service):
        raise jwt.InvalidAudienceError(f"audience {aud!r} not accepted")

    return TokenPayload(
        sub=str(claims["sub"]),
        aud=aud,
        iat=int(claims["iat"]),
        exp=int(claims["exp"]),
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
