"""The sign-in key, for the owner: when it changes, and changing it (row 5, 5a).

The schedule and the rules are `oidc_keys`; the provider's own endpoints,
which apps call, are `routes/oidc.py`.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, Request, status

from .. import oidc, oidc_keys, tokens
from .._generated.models import OidcKeyChangeRequest, OidcKeyList
from ..dependencies import problem, require_operator

router = APIRouter(tags=["oidc"])

Operator = Annotated[tokens.Claims, Depends(require_operator)]


def _owner(claims: tokens.Claims) -> None:
    if claims.sub != tokens.SUB_OPERATOR:
        raise problem(status.HTTP_403_FORBIDDEN, "Owner required", "Only Eugene's owner may.")


@router.get("/v1/oidc/keys", response_model=OidcKeyList, response_model_exclude_none=True)
async def list_keys(request: Request, claims: Operator) -> OidcKeyList:
    """The keys, newest first, and when the next change is due."""
    _owner(claims)
    return OidcKeyList.model_validate(oidc_keys.key_list(request.app.state.machine.state))


@router.post("/v1/oidc/keys/change", response_model=OidcKeyList, response_model_exclude_none=True)
async def change_key(
    request: Request,
    body: OidcKeyChangeRequest,
    claims: Operator,
) -> OidcKeyList:
    """A new sign-in key now; with `endSignIns`, every other key retired."""
    _owner(claims)
    if getattr(request.app.state, "oidc", None) is None:
        request.app.state.oidc = oidc.Provider()
    try:
        await asyncio.to_thread(oidc_keys.change, request.app, end_sign_ins=bool(body.endSignIns))
    except oidc.Locked as exc:
        raise problem(status.HTTP_503_SERVICE_UNAVAILABLE, "Locked", str(exc)) from exc
    except oidc_keys.TooSoon as exc:
        raise problem(status.HTTP_409_CONFLICT, "Changed too recently", str(exc)) from exc
    except oidc_keys.StandbyBehind as exc:
        raise problem(status.HTTP_409_CONFLICT, "Standby too old", str(exc)) from exc
    return OidcKeyList.model_validate(oidc_keys.key_list(request.app.state.machine.state))
