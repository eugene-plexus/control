"""A signed person proof narrows an app key; it never replaces that key."""

from __future__ import annotations

import math
from typing import Any

import jwt

from . import oidc
from .client_admission import AdmissionRefusal


def check(state: Any, key_id: str, token: str | None) -> dict[str, str]:
    key = state.client_keys.get(key_id)
    name = str((key or {}).get("name") or "")
    if not token:
        if name.startswith("app:workbench@"):
            raise AdmissionRefusal(401, "Workbench requires a signed-in person's token.")
        return {}
    try:
        claims = oidc.Provider.decode(token, state, typ=oidc.TYP_ACCESS, audience=None)
        client_id = claims.get("client_id")
        client = state.oidc_clients.get(client_id) if isinstance(client_id, str) else None
        if (
            not key
            or not name.startswith("app:")
            or client is None
            or client.get("owner") != name
            or claims.get("aud") != client_id
            or not isinstance(claims.get("sid"), str)
            or not claims["sid"]
            or type(claims.get("eugene_auth_at")) not in (int, float)
            or not math.isfinite(claims["eugene_auth_at"])
        ):
            raise jwt.InvalidTokenError("person proof is not bound to this app")
    except (jwt.InvalidTokenError, ValueError, TypeError, KeyError):
        raise AdmissionRefusal(
            401, "The person's token is invalid or belongs to another app."
        ) from None
    assert isinstance(client_id, str)  # checked above, after signature verification
    subject = oidc.still_valid(state, claims, client_id)
    if subject is None:
        raise AdmissionRefusal(403, "This person's sign-in or access to the app has ended.")
    return {"personId": subject.sub, "appClientId": client_id}
