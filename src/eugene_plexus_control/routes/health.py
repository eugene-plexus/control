"""GET /healthz — liveness, and honest about being a standby."""

from __future__ import annotations

from fastapi import APIRouter, Request

from .. import __version__
from .._generated.models import Health, Status
from ..state_machine import StateMachine

router = APIRouter(tags=["meta"])


@router.get("/healthz", response_model=Health)
async def healthz(request: Request) -> Health:
    """Healthy on a standby as well as on the active root.

    A standby that is up and replicating is doing its job, so reporting
    it as unhealthy would train an operator to ignore the field on the
    one host whose health matters most on the day of a failover. Role
    and lag are `GET /v1/control/status`; this answers only whether the
    process is alive.

    Two things do make it `degraded`, and both are conditions an
    operator has to act on rather than transient states:

    * **Not initialized.** No passphrase, so nothing but setup works.
      Reported as degraded rather than as an error because a fresh
      install is a legitimate state and the wizard is the fix.
    * **A torn log tail was discarded at boot**, meaning this process
      was killed uncleanly. Nothing was lost that a caller had been
      promised, but an operator should learn it from the dashboard
      rather than from the logs of a process that has since restarted.

    Safe mode is reported too, and note what it does *not* skip: the
    log. Applied state is the install, not a preference — a control root
    that came up ignoring its own node registry would hand out wrong
    answers rather than degraded ones.
    """
    machine: StateMachine = request.app.state.machine
    safe_mode = bool(getattr(request.app.state, "safe_mode", False))
    state = machine.state
    initialized = state.identity.salt is not None

    details: dict[str, object] = {
        "role": machine.role,
        "epoch": state.epoch,
        "appliedIndex": state.index,
        "nodes": len(state.nodes),
        "initialized": initialized,
    }

    follower = getattr(request.app.state, "follower", None)
    if follower is not None:
        details["activeReachable"] = follower.status.reachable
        if follower.status.lag_entries is not None:
            details["lagEntries"] = follower.status.lag_entries

    dropped = machine.dropped_tail_lines
    if dropped:
        details["discardedTornTailLines"] = dropped

    degraded = safe_mode or not initialized or bool(dropped)
    return Health(
        status=Status.degraded if degraded else Status.ok,
        version=__version__,
        component="control",
        safeMode=safe_mode,
        details=details,
    )
