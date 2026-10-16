"""Bounded, replicated access-change receipts; never copy mutation payloads."""

from itertools import pairwise
from typing import Any

from ._generated.models import AccessAuditEvent

LIMIT = 1000
TARGETS = {
    "putPerson": ("person", "id"),
    "deletePerson": ("id",),
    "setPersonPassword": ("id",),
    "putPermissionGroup": ("group", "id"),
    "deletePermissionGroup": ("id",),
    "putOidcClient": ("client", "clientId"),
    "deleteOidcClient": ("clientId",),
    "setOidcClientRedirects": ("clientId",),
    "putClientKey": ("key", "id"),
    "revokeClientKey": ("id",),
    "setClientKeyLimits": ("id",),
    "revokeSignIn": ("sid",),
    "revokeSession": ("jti",),
}


def validate(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > LIMIT:
        raise ValueError("invalid access audit")
    events = [AccessAuditEvent.model_validate(e).model_dump(mode="json") for e in value]
    if any(a["index"] >= b["index"] for a, b in pairwise(events)):
        raise ValueError("access audit is out of order")
    return events


def changed(events: list[dict[str, Any]], entry: dict[str, Any]) -> list[dict[str, Any]]:
    path = TARGETS.get(entry["op"])
    if path is None:
        return events
    target: Any = entry.get("payload", {})
    for name in path:
        target = target.get(name) if isinstance(target, dict) else None
    event = AccessAuditEvent(
        index=entry["index"],
        at=str(entry.get("at") or ""),
        actor=entry.get("actor") or "unattributed",
        action=entry["op"],
        target=str(target or "unknown"),
    )
    return [*events[-(LIMIT - 1) :], event.model_dump(mode="json")]
