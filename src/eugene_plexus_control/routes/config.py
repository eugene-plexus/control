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

import logging

from fastapi import APIRouter, Depends, Request

from .. import config as config_module
from .._generated.models import (
    ConfigDocument,
    ConfigSchema,
    ConfigUpdateRequest,
    ConfigUpdateResult,
)
from ..applied import OP_PATCH_CONFIG
from ..dependencies import require_authorized, require_operator
from ..state_machine import NotActive, StateMachine

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
async def get_config_schema() -> ConfigSchema:
    """Unauthenticated, like every other component's: it describes the
    shape of the form, not its contents."""
    return config_module.as_schema()


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

    if accepted:
        try:
            machine.append(OP_PATCH_CONFIG, {"values": accepted})
        except NotActive as exc:
            from .nodes import _not_active

            raise _not_active(machine) from exc
        bootstrap.write_through(machine.state.config)

    pending = config_module.requires_restart(list(accepted))
    return ConfigUpdateResult(
        applied=sorted(accepted),
        rejected=rejected,
        requiresRestart=bool(pending),
        pendingRestart=pending,
    )
