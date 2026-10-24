"""FastAPI app factory.

Startup order matters and reads oddly, so it is spelled out:

1. **The log is opened and replayed first**, before anything else. Even
   in safe mode. Applied state is the install, not a preference — a
   control root that came up ignoring its own node registry would hand
   out wrong answers rather than degraded ones, and safe mode exists to
   let an operator repair *config*, not to make the install forget
   itself. What v0.1 wrote and v0.2 retired is no longer converted: replaying
   it raises `ApplyError`, which names it and the way out (`applied._retired`).
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
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from . import __version__, keyring_store, passphrase_file, security
from . import config as config_module
from . import machine_list as machine_list_module
from .admission_worker import AdmissionWorker
from .applied import OP_PATCH_CONFIG, RetiredData, normalize_url
from .auth_state import AuthState
from .join_tokens import JoinTokenStore
from .log_store import LogStore
from .machine_list import MachineListPublisher
from .nodes_client import NodesClient
from .replication import Follower, StandbyToken
from .root_tls import RootTls
from .routes import admin as admin_routes
from .routes import administration as administration_routes
from .routes import auth as auth_routes
from .routes import client_keys as client_key_routes
from .routes import config as config_routes
from .routes import control as control_routes
from .routes import external_sign_in as external_sign_in_routes
from .routes import health as health_routes
from .routes import nodes as node_routes
from .routes import oidc as oidc_routes
from .routes import people as people_routes
from .routes import topology as topology_routes
from .routes.nodes import service_token_for
from .settings import Settings, load_settings
from .state_machine import ROLE_ACTIVE, ROLE_STANDBY, StateMachine
from .trust import TrustPublisher

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
    store = LogStore(settings.state_dir)
    machine = StateMachine(store, role=role)
    try:
        machine.load()
    except RetiredData as exc:
        # v0.3.0 converted what v0.1 left on the active root, never on a standby: a
        # standby that followed an old root can still hold it in its own records. They
        # are the root's to hold, so they are set aside, kept, and the standby starts
        # empty and bootstraps from the root's snapshot as a new standby does. An active
        # root has no one to take them from: that refusal stands, naming the way out.
        if role != ROLE_STANDBY:
            raise
        kept = _set_aside(store)
        log.warning(
            "%s This standby set its records aside (%s) and follows the root again.", exc, kept
        )
        store = LogStore(settings.state_dir)
        machine = StateMachine(store, role=role)
        machine.load()
    if role == ROLE_STANDBY and _promoted_here(machine, settings):
        # Row 4: this copy was promoted, and its agent, which started it as
        # a standby, has not heard yet (a crash before the new root's bundle
        # reached it). Its own log is the evidence: the promote entry names
        # this node, at an epoch above the one the agent holds.
        log.warning(
            "this copy was promoted to the root at epoch %d (its log says so; its agent "
            "holds epoch %s): it starts as the root, not as a standby",
            machine.state.epoch,
            settings.follow_epoch,
        )
        machine.force_role(ROLE_ACTIVE)
        role = ROLE_ACTIVE
    app.state.machine = machine
    app.state.admission_worker = AdmissionWorker()

    if not hasattr(app.state, "auth_state"):
        app.state.auth_state = AuthState()
    # Row 4: every token this root signs carries the epoch it is at.
    app.state.auth_state.epoch_source = lambda: app.state.machine.state.epoch

    # The last bundle this root signed, so a sealed root still serves one.
    app.state.trust = TrustPublisher(Path(settings.state_dir))
    app.state.trust.load(machine.state.identity.controlPublicKey)
    app.state.trust_acks = {}

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
    # Row 5: a passphrase change that stopped between its entry and the
    # keyring leaves the old entry behind; it opens nothing now.
    from . import passphrase_change

    if not settings.safe_mode:
        passphrase_change.finish_interrupted(app)
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
    app.state.node_probes = {}
    app.state.site_contacts = {}
    app.state.root_tls = RootTls(
        Path(settings.state_dir), settings.nodes_origin, settings.nodes_probe
    )
    app.state.root_tls_task = None
    app.state.machine_list = MachineListPublisher(Path(settings.state_dir))
    app.state.standby_reports = {}
    # Row 4: when each node behind on the epoch was last sent it again.
    app.state.epoch_pushes = {}
    app.state.machine_list_push = None
    app.state.nodes_client = NodesClient(
        timeout_provider=lambda: config_module.effective(machine.state.config)[
            "nodeRequestTimeoutSeconds"
        ],
        local_agent_url=settings.agent_url,
        local_agent_public_origin=settings.agent_public_origin,
    )

    # Which node this control root runs on, so `POST /v1/runtimes` can
    # default `node` and a promotion can name itself in the registry.
    # An env-free default of None is correct on a host that has not
    # enrolled: the operator names the target explicitly until it has.
    app.state.node_name = settings.node_name or _local_node_name(machine)

    app.state.follower = None
    app.state.node_poller = None
    # Row 5: whether the passphrase file holds this install's passphrase,
    # at start and every hour; and the sign-in key's schedule.
    app.state.passphrase_file_state = None
    if not settings.safe_mode and machine.state.identity.salt is not None:
        await asyncio.to_thread(passphrase_change.recheck_file, app)
    app.state.hourly = asyncio.create_task(_hourly(app), name="control-hourly")

    if role == ROLE_STANDBY:
        follower = Follower(
            machine,
            active_url=normalize_url(settings.active_url),
            interval_seconds=settings.follow_interval_seconds,
            # The standby's own token, from the agent that started it
            # because its node holds the standby grant (warm-standby.md).
            # A standby started by hand has none, and is refused.
            token_provider=(
                StandbyToken(agent_url=settings.agent_url, spawn_token=settings.service_token)
                if settings.agent_url and settings.service_token
                else None
            ),
            on_passphrase_changed=lambda old: passphrase_change.standby_followed(app, old),
        )
        follower.start()
        app.state.follower = follower
        log.info("running as a STANDBY control root, following %s", settings.active_url)
    else:
        start_root_tasks(app)
        log.info(
            "running as the ACTIVE control root at epoch %d, applied index %d",
            machine.state.epoch,
            machine.state.index,
        )

    # An auto-unlocked root signs a fresh bundle for its state, and
    # pushes it once the node client exists.
    if role == ROLE_ACTIVE:
        machine_list_module.install_table(machine.state, app.state.node_name)
    if app.state.auth_state.control_private_key is not None and role == ROLE_ACTIVE:
        app.state.trust.publish(app)

    try:
        yield
    finally:
        if app.state.follower is not None:
            await app.state.follower.stop()
        if app.state.node_poller is not None:
            app.state.node_poller.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await app.state.node_poller
        if app.state.root_tls_task is not None:
            app.state.root_tls_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await app.state.root_tls_task
        app.state.hourly.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await app.state.hourly
        if app.state.machine_list_push is not None:
            app.state.machine_list_push.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await app.state.machine_list_push
        await app.state.nodes_client.aclose()
        await app.state.admission_worker.close()


def _promoted_here(machine: StateMachine, settings: Settings) -> bool:
    """A copy started as a standby whose own log says this node is the
    root, at an epoch above the one its agent holds (row 4).

    The promote entry is written by this copy alone, after the passphrase,
    and only it makes a node's record `control`; a later promotion of
    another machine records this one as a standby again. The epoch test
    keeps a stale copy (one that was the root, and was replaced while it
    was off) a standby of the root that replaced it."""
    from .state_machine import active_node_name

    name = settings.node_name
    if not name or settings.follow_epoch is None:
        return False
    return active_node_name(machine.state) == name and machine.state.epoch > int(
        settings.follow_epoch
    )


async def _hourly(app: FastAPI) -> None:
    """Row 5, once an hour on every root: the passphrase file checked again,
    and, on the active root, the sign-in key changed when it is due. The
    first check of the key is an hour after start, so a restart loop never
    changes it twice."""
    from . import oidc_keys, passphrase_change

    while True:
        await asyncio.sleep(oidc_keys.CHECK_EVERY_SECONDS)
        try:
            await asyncio.to_thread(passphrase_change.recheck_file, app)
            await asyncio.to_thread(oidc_keys.tick, app)
        except Exception as exc:
            log.warning("the hourly check failed: %s", exc)


def _set_aside(store: LogStore) -> list[str]:
    """Rename a standby's log and snapshot to the first free `.retired-v0.1` name.

    Renamed, never deleted or written over: they may be the only copy of what they
    held. What is left is an empty state directory, which a standby fills from the
    root."""
    kept = []
    for path in (store.log_path, store.snapshot_path):
        if not path.exists():
            continue
        for number in range(100):
            target = path.with_name(path.name + ".retired-v0.1" + (f".{number}" if number else ""))
            if not target.exists():
                path.rename(target)
                kept.append(target.name)
                break
        else:
            raise OSError(f"100 earlier copies of {path.name} are already kept beside it")
    return kept


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
        # ...and the sign-in key unsealed from it (C2).
        if getattr(app.state, "oidc", None) is not None:
            app.state.oidc.forget()
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
        # ...and the sign-in key unsealed from it (C2).
        if getattr(app.state, "oidc", None) is not None:
            app.state.oidc.forget()
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

BEHIND_PUSH_SECONDS = 60.0
"""Row 4: a node seen reachable and behind on this root's epoch, or on the
address its root came with, is sent the bundle and the machine list again
at most this often."""


def start_root_tasks(app: FastAPI) -> None:
    """What an active root runs beside its routes: the node poller and, with
    a nodes name, the watch on its TLS keys. At start, and when a standby
    is promoted (row 4): the promoted copy is not restarted, so it starts
    them here. Each at most once."""
    if getattr(app.state, "node_poller", None) is None:
        app.state.node_poller = asyncio.create_task(_poll_nodes(app), name="control-node-poller")
    settings: Settings = app.state.settings
    if settings.nodes_origin and getattr(app.state, "root_tls_task", None) is None:
        app.state.root_tls_task = asyncio.create_task(
            app.state.root_tls.run(), name="control-root-tls"
        )


async def push_to_nodes_behind(
    app: FastAPI, probes: dict[str, Any], *, now: float | None = None
) -> list[str]:
    """Send the bundle and the machine list again to each reachable node
    behind this root (row 4). Returns the nodes sent to.

    Behind means it reported a lower `epoch` (it missed a promotion's
    bundle), or a lower `controlUrlEpoch` while this root's list names an
    address to move to (it holds the epoch but still follows the root it
    had). Each node at most once in `BEHIND_PUSH_SECONDS`, so a node that
    refuses is not asked on every poll. A node that is down is reached by
    its own pulls, or by asking the machine that held the standby grant."""
    from . import trust as trust_module

    machine: StateMachine = app.state.machine
    epoch = machine.state.epoch
    control = machine_list_module.control_block(
        machine.state,
        node=getattr(app.state, "node_name", None),
        door_file=getattr(app.state.settings, "machine_door_file", None),
        nodes_origin=getattr(app.state.settings, "nodes_origin", None),
    )
    behind = sorted(
        name
        for name, probe in probes.items()
        if getattr(probe, "reachable", False)
        and (
            (isinstance(probe.epoch, int) and probe.epoch < epoch)
            or (
                control["url"] is not None
                and isinstance(probe.control_url_epoch, int)
                and probe.control_url_epoch < epoch
            )
        )
    )
    clock = time.perf_counter() if now is None else now
    pushed: dict[str, float] = app.state.epoch_pushes
    due = [
        name for name in behind if name not in pushed or clock - pushed[name] >= BEHIND_PUSH_SECONDS
    ]
    if not due:
        return []
    for name in due:
        pushed[name] = clock
    bundle = app.state.trust.current
    await trust_module.push_pairs(
        app,
        bundle if bundle is not None and bundle.epoch == epoch else None,
        machine_list_module.signed(app),
        names=due,
    )
    log.info(
        "sent the bundle and machine list of epoch %d again to %s: behind this root",
        epoch,
        ", ".join(due),
    )
    return due


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
            if app.state.auth_state.signing_key is None:
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

            # The root's own machine table, from its own records (LT1): every
            # poll, so a change replicated or configured is never far behind.
            machine_list_module.install_table(machine.state, app.state.node_name)
            if targets:
                app.state.node_probes = await client.probe_all(
                    targets, lambda node: service_token_for(app.state.auth_state, node)
                )
                _note_list_capable(app, app.state.node_probes)
                await push_to_nodes_behind(app, app.state.node_probes)
            _require_when_ready(app, machine)
            await asyncio.sleep(float(values["nodePollIntervalSeconds"]))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Never let the poller die. Losing it would freeze every
            # node's `reachable` at whatever it last was, which is worse
            # than reporting nothing: a stale `true` reads as healthy.
            log.warning("node poll failed: %s", exc)
            await asyncio.sleep(5.0)


def _note_list_capable(app: FastAPI, probes: dict[str, Any]) -> None:
    """Row 3: remember every node seen reporting a machine list version, so
    a change of the root's own key knows which agents recover by themselves
    after missing it, even while they are off."""
    seen = {name for name, probe in probes.items() if probe.list_version is not None}
    if seen:
        app.state.machine_list.note_list_capable(seen)


def _require_when_ready(app: FastAPI, machine: StateMachine) -> None:
    """LT-C2: an automatic install requires encryption once every machine can.

    Once every enrolled machine has a machine key on record and this root's
    last poll reached it over TLS to that key, one `patchConfig` makes it
    required, and it stays so: a machine on an older version that joins
    later is refused until it is updated, rather than turning encryption
    off for everyone. An expert sets `off` to stop this.
    """
    from .machine_list import AUTOMATIC, ENCRYPTION_KEY, REQUIRED, link

    if (machine.state.config.get(ENCRYPTION_KEY) or AUTOMATIC) != AUTOMATIC:
        return
    probes = app.state.node_probes
    nodes = list(machine.state.nodes.values())
    if not nodes:
        return
    for record in nodes:
        state = link(record, probes.get(record.name))
        own = state["state"] == "unknown" and state["detail"] == "This root's own machine."
        if state["state"] != "encrypted" and not own:
            return
    machine.append(OP_PATCH_CONFIG, {"values": {ENCRYPTION_KEY: REQUIRED}})
    app.state.bootstrap_config.write_through(machine.state.config)
    log.warning(
        "every machine in this install has a machine key and answers on its machine port: "
        "encryption between machines is now required (machineEncryption). Set it to `off` "
        "to go back."
    )


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
    # Signing in with Eugene (C2): the provider's endpoints, and people.
    app.include_router(oidc_routes.router)
    from .routes import oidc_keys as oidc_key_routes

    app.include_router(oidc_key_routes.router)
    app.include_router(external_sign_in_routes.router)
    app.include_router(external_sign_in_routes.management)
    app.include_router(people_routes.router)
    app.include_router(administration_routes.router)
    app.include_router(people_routes.app_clients)
    from .routes import sites as site_routes

    app.include_router(site_routes.router)
    app.add_exception_handler(site_routes.HeldAtTheMachine, site_routes.held_response)

    return app
