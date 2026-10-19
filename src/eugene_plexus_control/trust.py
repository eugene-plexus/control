"""The trust bundle, as this root builds, signs, keeps and pushes it.

Per-node token keys (2026-09-25; `specs/docs/design/per-node-token-keys.md`).
Every machine in the install trusts exactly the keys this bundle lists,
and what each may issue. It is built from applied state, so a standby
promoted at the same index builds the same keys, and it is signed with
this root's identity key, the one every node pinned at enrollment.

Three views of one thing, kept apart on purpose:

* **The verification view** (`view`) is what this root checks tokens
  against. It needs no private key, so a sealed root still knows which
  keys exist, and it is rebuilt whenever the log moves.
* **The signed bundle** (`TrustPublisher.current`) is what nodes are
  given. Signing needs the identity key, so only an unlocked root makes
  one. The last one is kept on disk so a sealed root still serves it.
* **The push** is best-effort. Every agent also pulls, which is how a
  node that was down during a revocation catches up by itself.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from . import tokens
from .applied import AppliedState
from .auth_state import AuthState

log = logging.getLogger(__name__)

BUNDLE_FILE = "trust_bundle.json"


def trust_keys(state: AppliedState) -> list[tokens.TrustKey]:
    """Every key the install trusts: this root's, and each enrolled node's."""
    keys: list[tokens.TrustKey] = []
    root_public = state.identity.rootTokenPublicKey
    if root_public:
        try:
            public = tokens.load_public(root_public)
        except ValueError as exc:
            # Fail closed: no root key listed means no session verifies,
            # which is loud, rather than a guess about which key it was.
            log.error("the root token public key in applied state is unreadable: %s", exc)
            return []
        keys.append(
            tokens.TrustKey(
                kid=tokens.thumbprint(public),
                issuer=tokens.ISSUER_CONTROL,
                public=public,
                grants=frozenset({tokens.GRANT_AUTHORITY}),
            )
        )
    seen = {k.kid for k in keys}
    for record in sorted(state.nodes.values(), key=lambda r: r.name):
        if not record.tokenPublicKey:
            continue
        try:
            public = tokens.load_public(record.tokenPublicKey)
        except ValueError as exc:
            log.warning("node %s has an unreadable token key and is left out: %s", record.name, exc)
            continue
        kid = tokens.thumbprint(public)
        if kid in seen:
            # Two records naming one key would let one node's tokens be
            # read as the other's. Refusing both is the only safe answer.
            log.error("node %s reuses a token key already listed; it is left out", record.name)
            continue
        seen.add(kid)
        keys.append(
            tokens.TrustKey(
                kid=kid,
                issuer=tokens.node_recipient(record.name),
                public=public,
                grants=frozenset(record.grants) | {tokens.GRANT_NODE},
            )
        )
    return keys


def _revoked(state: AppliedState, now: int) -> list[tuple[str, int]]:
    return [
        (jti, exp)
        for jti, exp in state.revoked_sessions.items()
        if exp + tokens.LEEWAY_SECONDS >= now
    ]


def view(state: AppliedState, *, now: int | None = None) -> tokens.TrustBundle:
    """The keys and sign-outs this root checks tokens against. Unsigned."""
    issued = int(time.time()) if now is None else now
    keys = trust_keys(state)
    return tokens.TrustBundle(
        version=state.index,
        epoch=state.epoch,
        iat=issued,
        authority=state.identity.controlPublicKey or "",
        keys={k.kid: k for k in keys},
        revoked_sessions=frozenset(jti for jti, _ in _revoked(state, issued)),
        jws="",
        payload={},
    )


class TrustPublisher:
    """Signs the bundle, keeps the last one, and pushes it to every node."""

    def __init__(self, state_dir: Path) -> None:
        self._path = state_dir / BUNDLE_FILE
        self.current: tokens.TrustBundle | None = None
        self._view_cache: tuple[int, int, tokens.TrustBundle] | None = None
        self._push_task: asyncio.Task[dict[str, bool]] | None = None

    # ----- the verification view ---------------------------------------

    def view_for(self, state: AppliedState) -> tokens.TrustBundle:
        """`view`, cached per log index and per second of sign-out pruning."""
        now = int(time.time())
        cached = self._view_cache
        if cached is not None and cached[0] == state.index and now - cached[1] < 60:
            return cached[2]
        built = view(state, now=now)
        self._view_cache = (state.index, now, built)
        return built

    # ----- signing and keeping -----------------------------------------

    def load(self, authority: str | None) -> None:
        """The last bundle signed before a restart, so a sealed root can serve it."""
        if not authority:
            return
        try:
            document = json.loads(self._path.read_text(encoding="utf-8"))
            self.current = tokens.parse_bundle(str(document["jws"]), authority=authority)
        except FileNotFoundError:
            return
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log.warning("the kept trust bundle at %s is unreadable: %s", self._path, exc)

    def sign(self, state: AppliedState, auth: AuthState) -> tokens.TrustBundle | None:
        """Sign a bundle for the current state, or None while locked."""
        if auth.control_private_key is None:
            return None
        authority = tokens.load_private(auth.control_private_key)
        bundle = tokens.build_bundle(
            authority=authority,
            version=state.index,
            epoch=state.epoch,
            keys=trust_keys(state),
            revoked_sessions=_revoked(state, int(time.time())),
        )
        self.current = bundle
        self._view_cache = None
        try:
            tokens.write_bundle_file(self._path, bundle)
        except OSError as exc:
            log.warning("could not keep the trust bundle at %s: %s", self._path, exc)
        return bundle

    # ----- pushing -------------------------------------------------------

    def publish(self, app: Any) -> tokens.TrustBundle | None:
        """Sign for the current state and push it to every node, in the background."""
        bundle = self.sign(app.state.machine.state, app.state.auth_state)
        if bundle is None:
            log.warning(
                "the trust bundle changed but this root is locked, so it cannot be signed; "
                "nodes keep the last one until someone signs in"
            )
            return None
        previous = self._push_task
        if previous is not None and not previous.done():
            previous.cancel()
        self._push_task = asyncio.create_task(push(app, bundle), name="control-trust-push")
        return bundle

    async def wait(self) -> None:
        """For tests: let the current push finish."""
        task = self._push_task
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await task


async def push(app: Any, bundle: tokens.TrustBundle) -> dict[str, bool]:
    """POST the bundle to every node with an address. Returns who took it.

    **No bearer.** The body is the credential: a JWS signed by the key
    every node pinned. A node that is down takes the bundle when it next
    pulls, so nothing here retries.
    """
    machine = app.state.machine
    client = app.state.nodes_client
    results: dict[str, bool] = {}

    async def one(name: str, url: str) -> None:
        try:
            await client.post_json(url, "/v1/node/trust-bundle", None, {"jws": bundle.jws})
            results[name] = True
        except (httpx.HTTPError, ValueError) as exc:
            log.info("node %s did not take trust bundle %d yet: %s", name, bundle.version, exc)
            results[name] = False

    targets = [(r.name, r.url) for r in machine.state.nodes.values() if r.url]
    await asyncio.gather(*(one(name, str(url)) for name, url in targets))
    acks: dict[str, int] = getattr(app.state, "trust_acks", {})
    for name, took in results.items():
        if took:
            acks[name] = max(acks.get(name, 0), bundle.version)
    app.state.trust_acks = acks
    return results


def nodes_behind(app: Any, version: int) -> list[str]:
    """Nodes that have not said they hold `version`, by push ack or by probe."""
    machine = app.state.machine
    acks: dict[str, int] = getattr(app.state, "trust_acks", {})
    probes = getattr(app.state, "node_probes", {})
    behind = []
    for name in sorted(machine.state.nodes):
        probe = probes.get(name)
        held = max(acks.get(name, 0), getattr(probe, "trust_bundle_version", None) or 0)
        if held < version:
            behind.append(name)
    return behind
