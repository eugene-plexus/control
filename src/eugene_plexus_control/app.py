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

from fastapi import FastAPI

from . import __version__
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

    while True:
        try:
            values = config_module.effective(machine.state.config)
            targets = [
                (record.name, normalize_url(record.url)) for record in machine.state.nodes.values()
            ]
            if targets:
                app.state.node_probes = await client.probe_all(targets, _service_token(app))
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
    app.include_router(node_routes.router)
    app.include_router(control_routes.router)
    app.include_router(topology_routes.router)
    app.include_router(config_routes.router)
    app.include_router(admin_routes.router)

    return app
