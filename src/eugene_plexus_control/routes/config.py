"""The config trio — where a PATCH is a log entry.

`GET` and `GET /schema` are ordinary. `PATCH` is not: it validates,
appends a `patchConfig` entry, and only then reports success, because a
promoted standby has to come up configured the way it would have been
had it been active all along.

The write-through to `control.yaml` happens after the entry is durable
and is best-effort. If it fails the PATCH still succeeded — the log is
the authority and the file is a boot-time cache — and refusing the
operator's change because a cache could not be updated would be the
wrong way round.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials

from .. import config as config_module
from .. import keyring_store, tokens
from .._generated.models import (
    ConfigDocument,
    ConfigFieldStatus,
    ConfigFieldStatusLevel,
    ConfigSchema,
    ConfigUpdateRequest,
    ConfigUpdateResult,
)
from ..applied import OP_PATCH_CONFIG
from ..auth_state import AuthState
from ..dependencies import problem, require_authorized, require_operator
from ..state_machine import NotActive, StateMachine
from .auth import install_id_of

log = logging.getLogger(__name__)

router = APIRouter(tags=["config"])


@router.get("/v1/config", response_model=ConfigDocument, dependencies=[Depends(require_authorized)])
async def get_config(request: Request) -> ConfigDocument:
    """Current effective values — applied state over built-in defaults.

    Readable with a service token because the UI proxies it and an agent
    may want to know the node poll interval it is being held to. Nothing
    here is sensitive: this component holds keys, but it holds them in
    applied state under the master key, not as config values.
    """
    machine: StateMachine = request.app.state.machine
    return config_module.as_document(machine.state.config)


@router.get("/v1/config/schema", response_model=ConfigSchema)
async def get_config_schema(request: Request) -> ConfigSchema:
    """Unauthenticated, like every other component's: it describes the
    shape of the form, not its contents.

    **What the values are doing is contents**, so it is added only for a
    caller who may read them (settings never lie, 2026-09-30): a restart
    that is still pending and what runs meanwhile, and a security mode this
    host cannot carry out. An anonymous caller gets the shape alone.
    """
    schema = config_module.as_schema()
    fields = [
        f.model_copy(update={"unsetMeans": config_module.UNSET_MEANS[f.key]})
        if f.key in config_module.UNSET_MEANS
        else f
        for f in schema.fields
    ]
    if _may_read(request):
        machine: StateMachine = request.app.state.machine
        values = machine.state.config
        pending = config_module.requires_restart(list(values), values)
        running = config_module.running()
        mode = config_module.effective(values)["securityMode"]
        keyring: bool | None = None
        if mode == "os_keyring":
            # Memoised per process; a locked Secret Service can block on a
            # prompt nobody answers, so the status route's deadline.
            try:
                keyring = await asyncio.wait_for(
                    asyncio.to_thread(keyring_store.probe_sync), timeout=3.0
                )
            except TimeoutError:
                keyring = None
        status = _security_mode_status(request, mode, keyring)
        decorated = []
        for f in fields:
            update: dict[str, object] = {}
            if f.key in pending:
                update["pendingRestart"] = True
                update["inEffect"] = running.get(f.key)
            if f.key == "securityMode" and status is not None:
                update["status"] = status
            decorated.append(f.model_copy(update=update) if update else f)
        fields = decorated
    schema.fields = fields
    return schema


def _may_read(request: Request) -> bool:
    """Whether the caller could read `GET /v1/config` -- the same check."""
    scheme, _, token = (request.headers.get("authorization") or "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return False
    try:
        require_authorized(
            request, HTTPAuthorizationCredentials(scheme="Bearer", credentials=token.strip())
        )
    except HTTPException:
        return False
    return True


def _security_mode_status(
    request: Request, mode: object, keyring: bool | None
) -> ConfigFieldStatus | None:
    settings = getattr(request.app.state, "settings", None)
    if mode == "passphrase_file" and getattr(settings, "passphrase_file", None) is None:
        return ConfigFieldStatus(
            level=ConfigFieldStatusLevel.warning,
            text=(
                "This root has no passphrase file configured "
                "(EUGENE_PLEXUS_CONTROL_PASSPHRASE_FILE), so it comes back locked after "
                "every restart, as Prompt on startup does."
            ),
        )
    if mode == "os_keyring" and keyring is False:
        return ConfigFieldStatus(
            level=ConfigFieldStatusLevel.warning,
            text=(
                "This host's keyring refused a test entry, so it cannot keep the key: the "
                "root comes back locked after every restart, as Prompt on startup does."
            ),
        )
    return None


def _check_mode_change(
    request: Request, machine: StateMachine, accepted: dict[str, object]
) -> None:
    """Switching the install mode is the `install-mode` capability (J15), and
    is dated in the same entry so every person can be told (J18)."""
    from .. import capabilities, sites
    from ..dependencies import verify_bearer

    _, _, token = (request.headers.get("authorization") or "").partition(" ")
    claims = verify_bearer(request, token.strip(), classes=(tokens.TYP_SESSION,))
    if capabilities.INSTALL_MODE not in capabilities.held_by(claims):
        raise problem(
            status.HTTP_403_FORBIDDEN,
            "Not permitted",
            "Switching the install mode needs the 'install-mode' capability.",
        )
    if accepted["installMode"] != sites.install_mode(machine.state):
        accepted[config_module.MODE_CHANGED_AT] = datetime.now(UTC).isoformat()
        log.warning("install mode switched to %s; every person is told", accepted["installMode"])


@router.patch(
    "/v1/config", response_model=ConfigUpdateResult, dependencies=[Depends(require_operator)]
)
async def patch_config(request: Request, body: ConfigUpdateRequest) -> ConfigUpdateResult:
    """Partial update. Per-key validation, then one log entry.

    Rejections are reported per key and the accepted keys still apply —
    a PATCH of five fields where one is a typo should not silently
    discard the other four, and telling the operator which one was wrong
    is more useful than refusing all of them.

    Only the accepted keys enter the log. An entry carrying a value the
    writer rejected would apply on a replica running different
    validation rules, which is how two roots end up with different
    config from the same log.
    """
    machine: StateMachine = request.app.state.machine
    bootstrap: config_module.BootstrapConfig = request.app.state.bootstrap_config

    accepted, rejected = config_module.validate_patch(body)
    if "installMode" in accepted:
        _check_mode_change(request, machine, accepted)

    # Snapshot before applying, so a securityMode transition is visible.
    # The keyring side-effects live at this layer rather than in the
    # state machine: the machine is a replicated log and an OS secret
    # store is emphatically local, so an entry that wrote one would mean
    # a standby storing the active root's key on replay.
    prior_mode = config_module.effective(machine.state.config)["securityMode"]

    if accepted:
        try:
            machine.append(OP_PATCH_CONFIG, {"values": accepted})
        except NotActive as exc:
            from .nodes import _not_active

            raise _not_active(machine) from exc
        bootstrap.write_through(machine.state.config)

    new_mode = config_module.effective(machine.state.config)["securityMode"]
    if prior_mode == "os_keyring" and new_mode != "os_keyring":
        # The operator moved to the stronger boundary, so the stored
        # auto-unlock secret has to go - otherwise the root would still
        # come back unlocked, contradicting the mode it now reports.
        if keyring_store.delete_master_key(install_id_of(machine)):
            log.info(
                "securityMode changed from os_keyring to %s; discarded the stored master key",
                new_mode,
            )
    elif prior_mode != "os_keyring" and new_mode == "os_keyring":
        auth: AuthState = request.app.state.auth_state
        if auth.master_key is not None and keyring_store.set_master_key(
            auth.master_key, install_id_of(machine)
        ):
            log.info("securityMode changed to os_keyring; stored the master key for auto-unlock")
        else:
            # Flipping it while locked is legitimate - a standby is
            # normally locked - so this is a note, not a failure. The
            # next login stores the key.
            log.info(
                "securityMode changed to os_keyring but this root holds no master key yet; "
                "the next login will store it"
            )

    pending = config_module.requires_restart(list(accepted), machine.state.config)
    return ConfigUpdateResult(
        applied=sorted(accepted),
        rejected=rejected,
        requiresRestart=bool(pending),
        pendingRestart=pending,
    )
