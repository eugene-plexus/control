"""Join tokens: the one credential a new node has.

A token is deliberately narrow — **single-use, short-lived, and
optionally scoped to one node name** — because it is the only thing
standing between "can reach the control root's port" and "is a member of
this install". Binding it to a name turns a leaked token into a token
that can only do the thing it was minted for.

**Minting is not a log op**, and that is a decision rather than an
oversight. A token is active-root-local: it is stored only as a hash, so
there is nothing to look up and nothing to recover, and if the root that
minted it dies before the node enrolls then the token was never usable
anyway. `enrollNode` is the mutation that matters and it *is* in the log.
That keeps `LogOp` closed at nine, which is the property that makes "an
operation absent from the list is an operation that would not replicate"
a true statement rather than an aspiration.

Consequence worth stating: **a token does not survive a promotion.** An
operator who minted one, lost the host, and promoted a standby has to
mint another. That is the right trade — the alternative is replicating a
membership credential to every standby.

Expiry is checked against the wall clock here, which is fine precisely
because none of this is replicated state: no other host has to agree
with this one about when a token died.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from . import security


class JoinTokenError(Exception):
    """The token is unknown, expired, already used, or bound elsewhere."""


class JoinTokenConsumed(JoinTokenError):
    """This token already enrolled a node.

    Distinct from "unknown" so the route can answer 409 rather than 401:
    a replayable join token is a stealable join token, and the caller
    deserves to know the difference between "wrong credential" and "that
    one is spent"."""


@dataclass(frozen=True)
class MintedToken:
    """What the operator sees, exactly once."""

    token: str
    expires_at: float
    node_name: str | None


@dataclass(frozen=True)
class _Record:
    token_hash: str
    expires_at: float
    node_name: str | None
    used: bool = False


class JoinTokenStore:
    """In-memory, hashed, single-use tokens.

    Bounded by expiry sweeps rather than by a cap: minting is
    operator-only, so the population is limited by a human rather than
    by an attacker, and a cap would mean refusing a legitimate mint
    because of tokens nobody used.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, _Record] = {}

    def mint(self, *, ttl_seconds: int, node_name: str | None) -> MintedToken:
        token = security.generate_join_token()
        token_hash = security.hash_join_token(token)
        expires_at = time.time() + ttl_seconds
        with self._lock:
            self._sweep_locked()
            self._records[token_hash] = _Record(
                token_hash=token_hash, expires_at=expires_at, node_name=node_name
            )
        # Returned once and never stored in recoverable form. A lost
        # token is re-minted, not looked up.
        return MintedToken(token=token, expires_at=expires_at, node_name=node_name)

    def consume(self, token: str, *, node_name: str) -> None:
        """Spend a token for one node name. Raises on any problem.

        Marks the token used *before* the caller does anything with the
        enrollment, so a crash between the two spends the token rather
        than leaving it replayable. Losing an enrollment attempt is
        cheap; a token that can be presented twice is not.
        """
        token_hash = security.hash_join_token(token)
        now = time.time()
        with self._lock:
            self._sweep_locked()
            record = self._records.get(token_hash)
            if record is None:
                raise JoinTokenError("join token is unknown or has expired")
            if record.used:
                raise JoinTokenConsumed("this join token has already enrolled a node")
            if record.expires_at <= now:
                del self._records[token_hash]
                raise JoinTokenError("join token has expired; mint another")
            if record.node_name is not None and record.node_name != node_name:
                raise JoinTokenError(
                    f"this join token is bound to node {record.node_name!r} and cannot "
                    f"enroll {node_name!r}"
                )
            self._records[token_hash] = _Record(
                token_hash=record.token_hash,
                expires_at=record.expires_at,
                node_name=record.node_name,
                used=True,
            )

    def _sweep_locked(self) -> None:
        now = time.time()
        # Expiry only — a **used** token is kept until it expires, so
        # replaying one answers 409 "already spent" rather than 401
        # "unknown". The contract distinguishes those and the difference
        # is worth something to whoever is holding the token: one means
        # "you have the wrong credential", the other means "that
        # enrollment already happened, go look at the node list".
        # Bounded because the TTL is short and minting is operator-only.
        self._records = {h: r for h, r in self._records.items() if r.expires_at > now}
