"""FastAPI app factory.

Startup order matters and reads oddly, so it is spelled out:

1. **The log is opened and replayed first**, before anything else. Even
   in safe mode. Applied state is the install, not a preference — a
   control root that came up ignoring its own node registry would hand
   out wrong answers rather than degraded ones, and safe mode exists to
   let an operator repair *config*, not to make the install forget
   itself.
2. Auth state is built empty. This component is the trust root, so it
   receives no key from the agent that spawns it — it holds the signing
   key sealed in applied state and opens it at login.
3. A standby starts its follower. An active root starts polling nodes.

Nothing here spawns a process. That is the split: supervision is the
agent's job, there is one per host, and on the host that holds the
control root `control` is itself just another local component in that
agent's topology.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI

from . import __version__, keyring_store, passphrase_file, security
from . import config as config_module
from .applied import normalize_url
from .auth_state import AuthState
from .join_tokens import JoinTokenStore
from .log_store import LogStore
from .nodes_client import NodesClient
from .replication import Follower
from .rotation import RotationTracker
from .routes import admin as admin_routes
from .routes import auth as auth_routes
from .routes import client_keys as client_key_routes
from .routes import config as config_routes
from .routes import control as control_routes
from .routes import health as health_routes
from .routes import nodes as node_routes
from .routes import topology as topology_routes
from .settings import Settings, load_settings
from .state_machine import ROLE_ACTIVE, ROLE_STANDBY, StateMachine

log = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings

    bootstrap = config_module.BootstrapConfig(settings.config_file)
    if settings.safe_mode:
        log.warning(
            "starting in SAFE MODE (EUGENE_PLEXUS_CONTROL_SAFE_MODE=1); ignoring %s and "
            "running on default config. The log is still replayed — safe mode is for "
            "repairing configuration, not for forgetting the install.",
            settings.config_file,
        )
    else:
        try:
            bootstrap.load()
        except Exception as exc:
            # Degraded, not dead. The config endpoints are how an
            # operator repairs a broken config, so refusing to start
            # over one locks them out of the fix.
            log.error(
                "config file %s could not be loaded (%s); running on defaults until the "
                "log supplies better. Fix it via PATCH /v1/config.",
                settings.config_file,
                exc,
            )
    app.state.bootstrap_config = bootstrap
    app.state.safe_mode = settings.safe_mode

    role = ROLE_STANDBY if settings.role.strip().lower() == "standby" else ROLE_ACTIVE
    machine = StateMachine(LogStore(settings.state_dir), role=role)
    machine.load()
    app.state.machine = machine

    if not hasattr(app.state, "auth_state"):
        app.state.auth_state = AuthState()

    # Auto-unlock, when the operator opted into one of the two ways.
    # Without it a trust root comes back from a restart holding its keys
    # sealed and locked - correct, and also an install that cannot
    # recover from a power cut without a person present, which is what
    # these fields promise.
    #
    # Safe mode skips both deliberately: safe mode exists to get a broken
    # config back to an endpoint, and a hanging keyring backend or a
    # missing secret mount is one more thing between the operator and
    # that.
    if (
        not settings.safe_mode
        and machine.state.identity.salt is not None
        and not app.state.auth_state.has_master_key()
    ):
        mode = config_module.effective(machine.state.config)["securityMode"]
        if mode == "os_keyring":
            _auto_unlock(app, machine)
        elif mode == "passphrase_file":
            _auto_unlock_from_file(app, machine, settings.passphrase_file)

    app.state.join_tokens = JoinTokenStore()
    app.state.rotation = RotationTracker()
    app.state.node_probes = {}
    app.state.standby_reports = {}
    app.state.nodes_client = NodesClient(
        timeout_provider=lambda: config_module.effective(machine.state.config)[
            "nodeRequestTimeoutSeconds"
        ]
    )

    # Which node this control root runs on, so `POST /v1/runtimes` can
    # default `node` and a promotion can name itself in the registry.
    # An env-free default of None is correct on a host that has not
    # enrolled: the operator names the target explicitly until it has.
    app.state.node_name = _local_node_name(machine)

    app.state.follower = None
    app.state.node_poller = None

    if role == ROLE_STANDBY:
        follower = Follower(
            machine,
            active_url=normalize_url(settings.active_url),
            interval_seconds=settings.follow_interval_seconds,
            token_provider=lambda: _service_token(app),
        )
        follower.start()
        app.state.follower = follower
        log.info("running as a STANDBY control root, following %s", settings.active_url)
    else:
        app.state.node_poller = asyncio.create_task(_poll_nodes(app), name="control-node-poller")
        log.info(
            "running as the ACTIVE control root at epoch %d, applied index %d",
            machine.state.epoch,
            machine.state.index,
        )

    try:
        yield
    finally:
        if app.state.follower is not None:
            await app.state.follower.stop()
        if app.state.node_poller is not None:
            app.state.node_poller.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await app.state.node_poller
        await app.state.nodes_client.aclose()


def _auto_unlock(app: FastAPI, machine: StateMachine) -> None:
    """Recover the master key from the OS keyring and open the sealed values.

    Never fatal. A trust root that refused to start because a keyring
    backend was locked would be worse than one that asks for the
    passphrase - the passphrase path is exactly the behaviour with this
    field unset, so every failure here degrades to it.

    A stored key that does not open the sealed values is reported and
    then discarded. The keyring entry outlives the install directory, so
    a state wipe and re-initialize under a different passphrase - any
    fresh test install - leaves an entry that derives to nothing. A
    secret whose failure mode is "auto-unlock appeared to work" is worse
    than no secret.
    """
    install_id = auth_routes.install_id_of(machine)
    stored = keyring_store.get_master_key(install_id)
    if stored is None:
        log.warning(
            "securityMode is os_keyring but no key was retrievable from the OS keyring; "
            "this root is locked until POST /v1/auth/login. Logging in stores the key, "
            "so the next restart should auto-unlock."
        )
        return
    try:
        auth_routes.unseal_with_master_key(app.state.auth_state, machine, stored)
    except Exception as exc:
        # Includes the HTTPException that `unseal_with_master_key`
        # raises for an unreadable signing key. At startup that is not
        # an HTTP concern, it is a locked root.
        app.state.auth_state.forget_master_key()
        keyring_store.delete_master_key(install_id)
        log.error(
            "the key stored in the OS keyring did not open this install's sealed values "
            "(%s). The stored key has been discarded; log in with the passphrase. This is "
            "what a passphrase rotation since the key was stored looks like.",
            exc,
        )
        return
    log.info("master key recovered from the OS keyring; this root is unlocked")


def _auto_unlock_from_file(app: FastAPI, machine: StateMachine, path: Path | None) -> None:
    """Open the sealed values with a passphrase read from a mounted file.

    The sibling of `_auto_unlock`, for hosts with no OS keyring — which
    is every container, and why this exists at all. Same contract: never
    fatal, every failure degrades to the passphrase prompt, which is the
    behaviour with the field unset.

    **Two ways it deliberately differs from the keyring path.**

    It verifies the passphrase before deriving. The keyring holds a
    derived key and can only find out it is wrong by failing to open
    something; here the install's own verifier can say so directly, so a
    wrong passphrase is reported as a wrong passphrase instead of as a
    sealed value that would not open. Those call for different actions
    and an operator reading a log at 3am should not have to guess which.

    And **it never deletes the file.** `keyring_store` discards a stored
    key that does not open the install, correctly: the agent wrote that
    entry, it outlives the install directory, and a stale one fails as
    "auto-unlock appeared to work". This file is the operator's, placed
    by a secret mount or by hand, and quite possibly read-only. Deleting
    it would destroy configuration we did not create in order to report
    a mistake we can simply describe.
    """
    passphrase = passphrase_file.read_passphrase(path)
    if passphrase is None:
        # read_passphrase logged which of the four things went wrong.
        return

    verifier = machine.state.identity.passphraseVerifier
    if verifier is not None and not security.verify_passphrase(passphrase, verifier):
        log.error(
            "the passphrase in %s is not this install's passphrase; the root stays locked. "
            "The file is left alone. This is what pointing a new container at an older "
            "install's secret looks like.",
            path,
        )
        return

    try:
        auth_routes.unseal_into(app.state.auth_state, machine, passphrase)
    except Exception as exc:
        app.state.auth_state.forget_master_key()
        log.error(
            "the passphrase in %s verified but did not open this install's sealed values "
            "(%s); the root stays locked and the file is left alone.",
            path,
            exc,
        )
        return
    log.info("master key derived from %s; this root is unlocked", path)


_LOCKED_POLL_SLEEP_SECONDS = 1.0
"""How long the node poller waits between checks while this root is
sealed. Not the poll interval: see the locked branch below."""


async def _poll_nodes(app: FastAPI) -> None:
    """Ask each node whether it is reachable and which epoch it has seen.

    **Reporting only.** Nothing here reassigns anything, marks a node
    `out`, or promotes a standby: a node that stops answering is `down`,
    and the whole point of refusing `mon_osd_down_out_interval` is that
    no timer gets to make that call.

    The interval is re-read from applied config on every pass, so a
    `PATCH /v1/config` takes effect without a restart.
    """
    machine: StateMachine = app.state.machine
    client: NodesClient = app.state.nodes_client
    announced_locked = False

    while True:
        try:
            values = config_module.effective(machine.state.config)
            targets = [
                (record.name, normalize_url(record.url)) for record in machine.state.nodes.values()
            ]

            # **A locked root does not poll.** It cannot mint a service
            # token without the install's signing key, and probing anyway
            # is worse than not probing: every node answers 401, once per
            # interval, forever. The operator then reads
            # `"GET /v1/node HTTP/1.1" 401 Unauthorized` on a worker and
            # goes looking at enrollment and tokens, which are fine --
            # while the actual cause, a sealed trust root on the other
            # machine, says nothing anywhere. Reported exactly that way
            # on 2026-09-12.
            #
            # The probes are useless while locked in any case: every
            # surface that would report them answers 503.
            token = _service_token(app)
            if token is None:
                # Two reasons for no signing key, and they are not the
                # same situation -- `dependencies.py` takes care to
                # separate them because "run first-run setup" is advice
                # to wipe an install that already exists. An install that
                # has never been set up has no nodes to poll either, so
                # it is not worth a word.
                if machine.state.identity.salt is not None and not announced_locked:
                    log.warning(
                        "not polling %d node(s): this root is locked, so it cannot present a "
                        "service token and they would all answer 401. POST /v1/auth/login, "
                        "or set securityMode so it unlocks without a person present.",
                        len(targets),
                    )
                    announced_locked = True
                # **A short sleep, not the poll interval.** While locked
                # this loop does nothing but re-read a local variable, and
                # the interval is what the operator waits AFTER unlocking
                # before any node has been observed at all -- during which
                # `/v1/nodes` reports every one of them with no probe,
                # which the console has to render as "not checked yet".
                # Reported on 2026-09-17: update the container, unlock,
                # and every node reads as down for the next fifteen
                # seconds. A second of no-op iterations costs nothing and
                # covers every unlock route there is, including ones added
                # later -- an event set by the login handler would have to
                # be remembered by each of them.
                await asyncio.sleep(
                    min(_LOCKED_POLL_SLEEP_SECONDS, float(values["nodePollIntervalSeconds"]))
                )
                continue
            if announced_locked:
                log.info("unlocked; resuming node polling")
                announced_locked = False

            if targets:
                app.state.node_probes = await client.probe_all(targets, token)
            await asyncio.sleep(float(values["nodePollIntervalSeconds"]))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Never let the poller die. Losing it would freeze every
            # node's `reachable` at whatever it last was, which is worse
            # than reporting nothing: a stale `true` reads as healthy.
            log.warning("node poll failed: %s", exc)
            await asyncio.sleep(5.0)


def _service_token(app: FastAPI) -> str | None:
    from .routes.nodes import node_service_token

    return node_service_token(app.state.auth_state)


def _local_node_name(machine: StateMachine) -> str | None:
    for record in machine.state.nodes.values():
        if record.role in ("control", "standby"):
            return record.name
    return None


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a FastAPI app with every router mounted."""
    settings = settings or load_settings()

    app = FastAPI(
        title="Eugene Plexus — control",
        description=(
            "The install's trust root, node registry, install-wide topology and "
            "replicated control-state log. Exactly one active. Spawns nothing."
        ),
        version=__version__,
        lifespan=_lifespan,
    )
    app.state.settings = settings

    # Health and the auth surface stay outside the bearer requirement:
    # a supervisor probes health without credentials, and the UI asks
    # `/v1/auth/status` before it has a token. Every other router
    # declares its own level per route rather than at the mount, so the
    # bar is visible next to the handler — nodes and control each mix
    # operator-only mutations with service-readable reads, and a
    # per-router level would have let a service token revoke a host.
    app.include_router(health_routes.router)
    app.include_router(auth_routes.router)
    app.include_router(client_key_routes.router)
    app.include_router(node_routes.router)
    app.include_router(control_routes.router)
    app.include_router(topology_routes.router)
    app.include_router(config_routes.router)
    app.include_router(admin_routes.router)

    return app
