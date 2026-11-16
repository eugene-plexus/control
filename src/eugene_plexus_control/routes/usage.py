"""Row 40: usage per key, person and group, and the owner's price for each model.

`usage.py` counts and decides; this module answers. The owner sees every key,
person and group; a directory administrator (row 39) sees people and groups but
no keys; a signed-in person sees their own through `GET /oidc/usage`, with the
access token an app such as Workbench holds for them.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Annotated, Any

import jwt
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from .. import oidc
from .. import usage as counts
from .._generated.models import ModelPriceList, ModelPriceWrite, UsageReport
from ..administration import Administrator, require_directory_reader
from ..applied import OP_SET_MODEL_PRICE
from ..dependencies import problem, require_operator
from ..state_machine import StateMachine

router = APIRouter(tags=["usage"])

_NO_STORE = {"Cache-Control": "no-store", "Pragma": "no-cache"}


def _active(request: Request) -> StateMachine:
    machine: StateMachine = request.app.state.machine
    if not machine.is_active:
        raise problem(503, "Not the active root", "Usage is kept at the active control root.")
    return machine


def standby_can_hold(app: Any, values: dict[str, Any] | None) -> None:
    """Refuse a write that sets a usage allowance while the standby's copy
    would stop at it (log version 3, row 40): update the standby first."""
    if not values or not any(values.get(k) is not None for k in counts.GROUP_ALLOWANCES):
        return
    _standby_current(app)


def _standby_current(app: Any) -> None:
    from .control import standby_short_of

    behind = standby_short_of(app, counts.LOG_VERSION_NEEDED)
    if behind is not None:
        raise problem(
            409,
            "Standby too old",
            f"Update {behind} first; its copy would stop at this change.",
        )


def _period(
    day: str | None, month: str | None, first: str | None, last: str | None
) -> tuple[str, str, str | None, str | None]:
    now = datetime.now(UTC)
    try:
        chosen_day = date.fromisoformat(day).isoformat() if day else counts.day_of(now)
    except ValueError:
        raise problem(422, "Invalid day", "Give the day as YYYY-MM-DD.") from None
    if month is not None and not counts._valid_month(month):
        raise problem(422, "Invalid month", "Give the month as YYYY-MM.")
    chosen_month = month or counts.month_of(now)
    if (first is None) != (last is None):
        raise problem(422, "Invalid range", "Give both the first and the last day, or neither.")
    if first is not None and last is not None:
        try:
            start, end = date.fromisoformat(first), date.fromisoformat(last)
        except ValueError:
            raise problem(422, "Invalid range", "Give each day as YYYY-MM-DD.") from None
        if end < start or end - start > timedelta(days=counts.DAYS_KEPT - 1):
            raise problem(
                422,
                "Invalid range",
                f"The last day must be on or after the first, at most {counts.DAYS_KEPT} days.",
            )
        first, last = start.isoformat(), end.isoformat()
    return chosen_day, chosen_month, first, last


def _visible(state: Any, admin: Administrator) -> list[str]:
    """The subjects this caller may see (row 39's directory scope)."""
    every = counts.known_subjects(state)
    if admin.owner:
        return every
    grants = admin.session.grants
    if grants.managePeople or grants.manageGroups:
        return [s for s in every if not s.startswith("key:")]
    groups = set(grants.model_dump()["membershipGroups"] or [])
    members = {p["id"] for p in state.people.values() if groups & set(p.get("groups") or [])}
    return [
        s
        for s in every
        if (s.startswith("group:") and s[6:] in groups)
        or (s.startswith("person:") and s[7:] in members)
    ]


@router.get("/v1/usage", response_model=UsageReport, response_model_exclude_none=True)
async def get_usage(
    request: Request,
    admin: Annotated[Administrator, Depends(require_directory_reader)],
    day: str | None = None,
    month: str | None = None,
) -> UsageReport:
    machine = _active(request)
    # `from` is a Python keyword; read the range by its names.
    first = request.query_params.get("from")
    last = request.query_params.get("to")
    chosen_day, chosen_month, first, last = _period(day, month, first, last)
    state = machine.state
    return UsageReport.model_validate(
        counts.report(
            state,
            _visible(state, admin),
            day=chosen_day,
            month=chosen_month,
            first=first,
            last=last,
        )
    )


@router.get("/oidc/usage", operation_id="getOwnUsage")
async def get_own_usage(request: Request) -> JSONResponse:
    """The signed-in person's own usage, by their access token from this root."""
    machine = _active(request)
    header = request.headers.get("authorization") or ""
    scheme, _, value = header.partition(" ")
    refused = JSONResponse(
        {"error": "invalid_token"},
        status_code=401,
        headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
    )
    state = machine.state
    try:
        if scheme.lower() != "bearer":
            raise jwt.InvalidTokenError("no bearer")
        claims = oidc.Provider.decode(value.strip(), state, typ=oidc.TYP_ACCESS, audience=None)
    except jwt.InvalidTokenError:
        return refused
    subject = oidc.still_valid(state, claims, str(claims.get("client_id") or ""))
    if subject is None:
        return refused
    person = subject.sub
    groups = [
        g["id"]
        for g in counts.groups_of(state, person)
        if any((g.get("policies") or {}).get(k) is not None for k in counts.GROUP_ALLOWANCES)
    ]
    chosen_day, chosen_month, _, _ = _period(None, None, None, None)
    body = counts.report(
        state,
        [f"person:{person}", *(f"group:{g}" for g in groups)],
        day=chosen_day,
        month=chosen_month,
        # A group's totals only: what each other member used, model by model, is theirs.
        breakdown=lambda subject: subject.startswith("person:"),
    )
    return JSONResponse(
        UsageReport.model_validate(body).model_dump(mode="json", exclude_none=True),
        headers=_NO_STORE,
    )


def _price_list(machine: StateMachine) -> ModelPriceList:
    prices = machine.state.model_prices
    return ModelPriceList.model_validate(
        {"prices": [{"model": model, **prices[model]} for model in sorted(prices)]}
    )


@router.get(
    "/v1/usage/prices",
    response_model=ModelPriceList,
    dependencies=[Depends(require_directory_reader)],
)
async def get_prices(request: Request) -> ModelPriceList:
    return _price_list(_active(request))


@router.put(
    "/v1/usage/prices",
    response_model=ModelPriceList,
    dependencies=[Depends(require_operator)],
)
async def set_price(request: Request, body: ModelPriceWrite) -> ModelPriceList:
    machine = _active(request)
    model = body.model.strip()
    if not model or model != body.model:
        raise problem(422, "Invalid model", "Name the model by its id, with no spaces around it.")
    _standby_current(request.app)
    price = body.price.model_dump(mode="json") if body.price is not None else None
    try:
        machine.append(OP_SET_MODEL_PRICE, {"model": model, "price": price}, actor="operator")
    except ValueError as exc:
        raise problem(422, "Invalid price", str(exc)) from exc
    return _price_list(machine)
