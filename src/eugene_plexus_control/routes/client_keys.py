"""Durable install-wide client credentials, owned by the active control root."""

from __future__ import annotations

import secrets
import time
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from .. import tokens
from .._generated.models import (
    ClientAdmissionRequest,
    ClientAdmissionResult,
    ClientKey,
    ClientKeyCreated,
    ClientKeyCreateRequest,
    ClientKeyLimits,
    ClientKeyList,
    ClientKeyPolicy,
    ClientKeyUpdateRequest,
)
from ..applied import (
    OP_PUT_CLIENT_ADMISSION,
    OP_PUT_CLIENT_KEY,
    OP_REVOKE_CLIENT_KEY,
    OP_SET_CLIENT_KEY_LIMITS,
    ApplyError,
)
from ..client_admission import AdmissionClock, AdmissionRefusal, decide, validate_limits
from ..dependencies import problem, require_key_policy, require_operator
from ..state_machine import StateMachine

router = APIRouter(tags=["client keys"])


def active(request: Request) -> StateMachine:
    machine: StateMachine = request.app.state.machine
    if not machine.is_active:
        raise problem(503, "Client-key authority unavailable", "Contact the active control root.")
    return machine


def authority(machine: StateMachine) -> str:
    return "control:" + str(machine.state.identity.controlPublicKey)


def records(machine: StateMachine) -> list[ClientKey]:
    now = datetime.now(UTC)
    return sorted(
        (
            key
            for raw in machine.state.client_keys.values()
            if (key := ClientKey.model_validate(raw)).expiresAt > now
        ),
        key=lambda key: key.createdAt,
        reverse=True,
    )


def listing(machine: StateMachine) -> ClientKeyList:
    return ClientKeyList.model_validate(
        dict(
            keys=records(machine),
            authority=authority(machine),
            revision=machine.state.index,
            scope="install",
        )
    )


@router.get(
    "/v1/auth/client-keys",
    response_model=ClientKeyList,
    response_model_exclude_none=True,
    dependencies=[Depends(require_operator)],
)
async def list_keys(request: Request) -> ClientKeyList:
    return listing(active(request))


@router.post(
    "/v1/auth/client-keys",
    response_model=ClientKeyCreated,
    status_code=201,
    response_model_exclude_none=True,
    dependencies=[Depends(require_operator)],
)
async def create_key(request: Request, body: ClientKeyCreateRequest) -> ClientKeyCreated:
    machine = active(request)
    name = body.name.strip()
    if not name:
        raise problem(422, "Name required", "Give the key a name.")
    issued = int(time.time())
    expires = issued + (body.ttlDays or 365) * 86400
    key_id = secrets.token_hex(16)
    signer = request.app.state.auth_state.signer()
    if signer is None:
        raise problem(503, "Locked", "This root's token key is not in memory. Sign in first.")
    token, _ = signer.mint(
        typ=tokens.TYP_CLIENT,
        sub=name,
        aud=[tokens.RECIPIENT_GATEWAY],
        ttl_seconds=expires - issued,
        now=issued,
        jti=key_id,
    )
    try:
        limits = validate_limits((body.limits or ClientKeyLimits()).model_dump(exclude_none=True))
    except ValueError as exc:
        raise problem(422, "Invalid key limits", str(exc)) from exc
    key = ClientKey(
        id=key_id,
        name=name,
        tail=token[-6:],
        createdAt=datetime.fromtimestamp(issued, UTC),
        expiresAt=datetime.fromtimestamp(expires, UTC),
        limits=ClientKeyLimits.model_validate(limits),
    )
    machine.append(OP_PUT_CLIENT_KEY, {"key": key.model_dump(mode="json", exclude_none=True)})
    return ClientKeyCreated(key=key, token=token)


@router.get(
    "/v1/auth/client-keys/policy",
    response_model=ClientKeyPolicy,
    response_model_exclude_none=True,
    dependencies=[Depends(require_key_policy)],
)
async def policy(request: Request, response: Response) -> ClientKeyPolicy:
    machine = active(request)
    response.headers["Cache-Control"] = "no-store"
    return ClientKeyPolicy.model_validate(
        dict(
            authority=authority(machine),
            revision=machine.state.index,
            generatedAt=time.time(),
            keys=[
                key.model_dump(
                    mode="json", include={"id", "expiresAt", "revokedAt"}, exclude_none=True
                )
                for key in records(machine)
            ],
        )
    )


@router.delete(
    "/v1/auth/client-keys/{id}", status_code=204, dependencies=[Depends(require_operator)]
)
async def revoke_key(request: Request, id: str) -> None:
    machine = active(request)
    record = machine.state.client_keys.get(id)
    if record is None:
        raise problem(404, "No such key", "This key is not in the install's registry.")
    if not record.get("revokedAt"):
        machine.append(OP_REVOKE_CLIENT_KEY, {"id": id, "revokedAt": datetime.now(UTC).isoformat()})


@router.put(
    "/v1/auth/client-keys/{key_id}/limits",
    response_model=ClientKey,
    response_model_exclude_none=True,
    dependencies=[Depends(require_operator)],
)
async def set_limits(request: Request, key_id: str, body: ClientKeyUpdateRequest) -> ClientKey:
    machine = active(request)
    if key_id not in machine.state.client_keys:
        raise problem(404, "No such key", "This key is not in the registry.")
    try:
        limits = validate_limits(body.limits.model_dump(exclude_none=True))
        machine.append(OP_SET_CLIENT_KEY_LIMITS, {"id": key_id, "limits": limits})
    except ValueError as exc:
        raise problem(422, "Invalid limits", str(exc)) from exc
    return ClientKey.model_validate(machine.state.client_keys[key_id])


@router.post(
    "/v1/auth/client-keys/admission",
    response_model=ClientAdmissionResult,
    response_model_exclude_none=True,
    dependencies=[Depends(require_key_policy)],
)
async def client_admission(request: Request, body: ClientAdmissionRequest) -> ClientAdmissionResult:
    machine = active(request)
    try:
        ledger = machine.state.client_admission
        clock = getattr(request.app.state, "admission_clock", None)
        if clock is None:
            clock = AdmissionClock(ledger["clock"])
            request.app.state.admission_clock = clock
        result, candidate = decide(
            ledger,
            machine.state.client_keys.get(body.keyId),
            key_id=body.keyId,
            action=body.action.value,
            request_id=body.requestId,
            model=body.model,
            now=clock.now(ledger["clock"]),
        )
        if candidate is not None:
            machine.append(
                OP_PUT_CLIENT_ADMISSION,
                {
                    "clock": candidate["clock"],
                    "keyId": body.keyId,
                    "bucket": candidate["buckets"].get(body.keyId, {}),
                },
            )
        return ClientAdmissionResult.model_validate(result)
    except AdmissionRefusal as exc:
        raise HTTPException(
            exc.status,
            detail=exc.detail,
            headers={"Retry-After": str(exc.retry)} if exc.retry else None,
        ) from exc
    except (OSError, ValueError, ApplyError) as exc:
        raise problem(
            503, "Admission unavailable", "The authority could not commit admission state."
        ) from exc
