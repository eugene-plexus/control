"""Alerts, SLOs and the alert webhook (row 38).

`specs/docs/design/rows/row-38.md`, `alerts.py`. The owner sets the rules,
the SLO targets and the webhook (`/v1/alerting`). The owner, and a person
the owner gave an operations power, read what fired and how each SLO stands
(`/v1/alerts`): a person with a power sees the alerts about the machines
their power reaches and about models, never an SLO set on a client key.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request, Response, status
from pydantic import ValidationError

from .. import alerts, security
from .._generated.models import AlertingSettings, AlertingUpdate, AlertsView, AlertWebhookStatus
from ..administration import (
    Administrator,
    has_operations,
    operations_reach,
    require_administrator,
)
from ..applied import OP_SET_ALERTING
from ..dependencies import problem, require_operator
from ..state_machine import SAVED_AGAIN, NotActive, SealedUnderOldKey, StateMachine

log = logging.getLogger(__name__)

router = APIRouter(tags=["alerts"])

#: `setAlerting` needs a standby that applies this log version (row 5's rule).
ALERTING_LOG_VERSION = 2


def _service(request: Request) -> alerts.AlertService:
    service: alerts.AlertService | None = getattr(request.app.state, "alerts", None)
    if service is None:
        raise problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Alerts are not running",
            "This control root started without its alert evaluator (safe mode).",
        )
    return service


def _operations_view(
    service: alerts.AlertService,
    view: dict[str, Any],
    person: dict[str, Any],
    targets: list[dict[str, Any]],
) -> dict[str, Any]:
    """What a person with an operations power reads (row 38 review S1, S6).

    - An alert about a machine, and a read that failed on one, only inside
      their reach.
    - What the gateway counted (failure rates, SLOs on a model, their burn
      alerts, a failed read of the gateway) only when the gateway's machine is
      inside their reach.
    - An SLO alert only when its SLO is one set now, and not on a client key:
      decided from the targets in effect, never from the last SLO read, which
      is empty when that read failed.
    """
    gateway = service._cache.gateway_node
    sees_gateway = (
        operations_reach(person, gateway)
        if gateway
        else person.get("administration", {}).get("operationsMachines") is None
    )
    model_slos = {s["id"] for s in targets if not s.get("clientKeyId")}

    def shown(alert: dict[str, Any]) -> bool:
        machine = alert.get("machine")
        if machine is not None:
            return operations_reach(person, machine)
        if alert.get("kind") == "slo-burn" or alert.get("sloId"):
            return sees_gateway and alert.get("sloId") in model_slos
        return sees_gateway

    problems = [
        text
        for machine, text in service.problems
        if (operations_reach(person, machine) if machine else sees_gateway)
    ]
    return {
        **view,
        "problems": problems,
        "firing": [a for a in view["firing"] if shown(a)],
        "recent": [a for a in view["recent"] if shown(a)],
        "slos": [s for s in view["slos"] if sees_gateway and s.get("id") in model_slos]
        if sees_gateway
        else [],
    }


@router.get("/v1/alerts", response_model=AlertsView, response_model_exclude_none=False)
async def get_alerts(
    request: Request,
    response: Response,
    admin: Annotated[Administrator, Depends(require_administrator)],
) -> AlertsView:
    view = _service(request).view()
    response.headers["Cache-Control"] = "no-store"
    if admin.owner:
        return AlertsView.model_validate(view)
    machine: StateMachine = request.app.state.machine
    person = machine.state.people.get(admin.claims.sub)
    if person is None or not has_operations(person):
        raise problem(
            status.HTTP_403_FORBIDDEN,
            "No operations power",
            "Only the owner, and people the owner gave an operations power, read alerts.",
        )
    _, targets = alerts.effective(machine.state.alerting)
    return AlertsView.model_validate(_operations_view(_service(request), view, person, targets))


@router.get(
    "/v1/alerting", response_model=AlertingSettings, dependencies=[Depends(require_operator)]
)
async def get_alerting(request: Request) -> AlertingSettings:
    machine: StateMachine = request.app.state.machine
    return AlertingSettings.model_validate(alerts.settings_view(machine.state.alerting))


@router.put(
    "/v1/alerting", response_model=AlertingSettings, dependencies=[Depends(require_operator)]
)
async def set_alerting(request: Request, body: dict[str, Any]) -> AlertingSettings:
    from .control import standby_short_of

    try:
        update = AlertingUpdate.model_validate(body)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first.get("loc", ()))
        raise problem(
            status.HTTP_400_BAD_REQUEST,
            "Alert settings refused",
            f"{where}: {first.get('msg')}" if where else str(first.get("msg")),
        ) from exc
    rules = [alerts._rule(r.model_dump(mode="json")) for r in update.rules]
    slos = [alerts._slo(s.model_dump(mode="json")) for s in update.slos]
    refusal = alerts.check_update(rules, slos)
    if refusal is not None:
        raise problem(status.HTTP_400_BAD_REQUEST, "Alert settings refused", refusal)

    machine: StateMachine = request.app.state.machine
    if not machine.is_active:
        raise problem(
            status.HTTP_409_CONFLICT,
            "Not the active root",
            "This control root is a standby; change alerts on the active root.",
        )
    behind = standby_short_of(request.app, ALERTING_LOG_VERSION)
    if behind is not None:
        raise problem(
            status.HTTP_409_CONFLICT,
            "Update the standby first",
            f"The standby on {behind} is older than this root and could not copy alert "
            "settings. Update it, then save again.",
        )
    master, generation = request.app.state.auth_state.sealing_key()
    if master is None:
        raise problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Locked",
            "Eugene is locked. Sign in once with the passphrase to unlock it.",
        )
    held = machine.state.alerting or {}
    sealed, host = held.get("sealedWebhookUrl"), held.get("webhookHost")
    if "webhookUrl" in update.model_fields_set:
        url = (update.webhookUrl or "").strip()
        if not url:
            sealed = host = None
        else:
            bad = alerts.check_webhook_url(url)
            if bad is not None:
                raise problem(status.HTTP_400_BAD_REQUEST, "Alert settings refused", bad)
            sealed = security.seal_b64(url.encode("utf-8"), master)
            host = alerts.webhook_host(url)
    try:
        machine.append(
            OP_SET_ALERTING,
            {"rules": rules, "slos": slos, "sealedWebhookUrl": sealed, "webhookHost": host},
            sealed_under=generation,
        )
    except NotActive as exc:
        raise problem(status.HTTP_409_CONFLICT, "Not the active root", str(exc)) from exc
    except SealedUnderOldKey as exc:
        raise problem(status.HTTP_409_CONFLICT, "Saved again", SAVED_AGAIN) from exc
    log.info(
        "alert settings saved: %d rule(s), %d SLO(s), webhook %s",
        len(rules),
        len(slos),
        f"to {host}" if sealed else "off",
    )
    return AlertingSettings.model_validate(alerts.settings_view(machine.state.alerting))


@router.post(
    "/v1/alerting/test",
    response_model=AlertWebhookStatus,
    dependencies=[Depends(require_operator)],
)
async def test_alert_webhook(request: Request) -> AlertWebhookStatus:
    service = _service(request)
    try:
        return AlertWebhookStatus.model_validate(await service.test())
    except LookupError as exc:
        raise problem(
            status.HTTP_409_CONFLICT, "No webhook", "Set a webhook address first, then test it."
        ) from exc
