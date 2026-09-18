"""Process-wide auth posture for the trust root.

Unlike every other component, this one is not handed a signing key at
spawn. It **mints** the signing key and derives the master key from the
operator's passphrase, and the agent that supervises it receives the key
rather than supplying it.

Three things live here, and the distinction between them is the whole
security model:

  * **The signing key** — in memory, and persisted sealed under the
    master key so a restart is not a re-key. Through M4 tokens were
    "rotated on each watchdog restart"; with N nodes that is an outage,
    because a restart re-keying the install breaks every component that
    has not yet been told.

  * **The master key** — memory only, derived from passphrase + salt,
    never on disk in plaintext. Re-derivable, which is why it is not in
    the replication set: a standby needs the *salt* and the *verifier*,
    not the key, and promotion is manual so the operator is present to
    supply the passphrase at the one moment it is needed. That is what
    resolves "does the master key cross a host boundary" in the good
    direction.

  * **Login rate limiting and session revocation** — per process, not
    replicated. A failed login on one host is not a fact about the
    install, and a standby that inherited a lockout would be a denial of
    service with extra steps.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class AuthState:
    """Mutable auth posture. One per process.

    Mutable, unlike the verify-only `AuthState` in the other components,
    because this is where the key is *born*: it changes on
    initialization, on login, and on an explicit rotation.
    """

    signing_key: bytes | None = None
    """Current service-token signing key. None before initialization, or
    on a standby that has not been promoted — a standby holds it sealed
    and cannot open it without the passphrase."""

    master_key: bytes | None = None
    """Derived from the passphrase at login. Memory only, always."""

    control_private_key: str | None = None
    """Base64 Ed25519 private half of this root's identity, once
    unsealed. Needed to attest an epoch change to a node."""

    recovery_private_key: str | None = None
    """Base64 X25519 private half of the recovery recipient, once
    unsealed. Held only during an explicit recovery or re-seal; normal
    operation never needs it and never opens a component secret."""

    _revoked: set[str] = field(default_factory=set)
    _failures: dict[str, list[float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # ----- posture ----------------------------------------------------

    @property
    def auth_disabled(self) -> bool:
        """True before initialization.

        Not a dev-mode escape hatch: it means no passphrase has been set
        yet, so the only reachable endpoints are `/healthz`,
        `GET /v1/auth/status` and `POST /v1/auth/initialize`. A trust
        root that let everything through until configured would be a
        trust root with a window."""
        return self.signing_key is None

    def has_master_key(self) -> bool:
        return self.master_key is not None

    def set_signing_key(self, key: bytes) -> None:
        if len(key) != 32:
            raise ValueError("signing key must be 32 bytes")
        self.signing_key = key

    def set_master_key(self, key: bytes) -> None:
        if len(key) != 32:
            raise ValueError("master key must be 32 bytes")
        self.master_key = key

    def forget_master_key(self) -> None:
        """Drop the derived key. Used when a standby is demoted, so a
        host that is no longer the writer stops holding the one thing
        that would let it act like one."""
        self.master_key = None
        self.control_private_key = None
        self.recovery_private_key = None

    # ----- sessions ---------------------------------------------------

    def revoke(self, token: str) -> None:
        with self._lock:
            self._revoked.add(token)

    def is_revoked(self, token: str) -> bool:
        with self._lock:
            return token in self._revoked

    # ----- login rate limiting ----------------------------------------

    def is_login_rate_limited(
        self, source: str, *, window_seconds: int, max_in_window: int
    ) -> bool:
        now = time.perf_counter()
        with self._lock:
            recent = [t for t in self._failures.get(source, []) if now - t < window_seconds]
            self._failures[source] = recent
            return len(recent) >= max_in_window

    def record_login_failure(self, source: str, *, window_seconds: int, max_in_window: int) -> None:
        now = time.perf_counter()
        with self._lock:
            recent = [t for t in self._failures.get(source, []) if now - t < window_seconds]
            recent.append(now)
            # Bounded so a source hammering the endpoint cannot grow
            # this list without limit — the cap is the decision
            # threshold, and anything beyond it changes nothing.
            self._failures[source] = recent[-(max_in_window + 1) :]

    def clear_login_failures(self, source: str) -> None:
        with self._lock:
            self._failures.pop(source, None)
