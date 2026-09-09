"""POST /v1/admin/restart — schedule a process exit so the agent relaunches us."""

from __future__ import annotations

import asyncio
import logging
import os

from fastapi import APIRouter, Depends

from .._generated.models import RestartResult
from ..dependencies import require_operator

router = APIRouter(tags=["meta"])

# Long enough for the 202 body to flush over a slow LAN, short enough
# that the operator isn't left watching a "restarting…" dialog.
_EXIT_DELAY_MS = 500


@router.post(
    "/v1/admin/restart",
    response_model=RestartResult,
    status_code=202,
    dependencies=[Depends(require_operator)],
)
async def restart() -> RestartResult:
    """Restarting the control root is safe by construction.

    Every acknowledged mutation is already durable in the log, and the
    process rebuilds applied state from the snapshot plus the entries
    after it — the same code path a standby uses to bootstrap. What is
    lost is the derived master key, so the operator will have to log in
    again unless `securityMode` is `os_keyring`.

    What is **not** lost, and used to be: the signing key. Through M4 a
    restart re-keyed the install. It is now persisted sealed, so
    restarting this component does not invalidate every service token on
    every host.
    """
    log = logging.getLogger(__name__)
    log.warning("restart requested via /v1/admin/restart; exiting in %dms", _EXIT_DELAY_MS)

    loop = asyncio.get_event_loop()
    loop.call_later(_EXIT_DELAY_MS / 1000.0, lambda: os._exit(0))

    return RestartResult(
        scheduled=True,
        delayMs=_EXIT_DELAY_MS,
        message=(
            f"Process exiting in {_EXIT_DELAY_MS}ms. The local agent is expected to "
            "relaunch it; running standalone, relaunch manually. Inference is unaffected "
            "either way — the data path does not pass through here."
        ),
    )
