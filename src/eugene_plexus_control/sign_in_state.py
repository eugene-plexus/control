"""Deterministic, replicated external identities and browser sign-ins (U4)."""

from dataclasses import replace
from typing import Any

from ._generated.models import BrowserSignInState

OPS = {
    "putExternalProvider": "providers",
    "deleteExternalProvider": "providers",
    "putSignInInvitation": "invitations",
    "deleteSignInInvitation": "invitations",
    "linkExternalIdentity": "identities",
    "deleteExternalIdentity": "identities",
    "putBrowserSignIn": "sessions",
    "deleteBrowserSignIn": "sessions",
}


def validate(raw: Any, people: dict[str, Any]) -> dict[str, Any]:
    from .applied import ApplyError

    try:
        result = BrowserSignInState.model_validate(raw).model_dump(mode="json")
        for table, records in result.items():
            if not isinstance(records, dict):
                raise ValueError("sign-in collections must be objects")
            for key, record in records.items():
                if key != record["id"]:
                    raise ValueError("record id does not match its key")
                if table != "providers" and record["personId"] not in people:
                    raise ValueError("unknown account")
                if (
                    table in ("identities", "invitations")
                    and record["providerId"] not in result["providers"]
                ):
                    raise ValueError("unknown provider")
        identities = list(result["identities"].values())
        if len({(r["issuer"], r["subject"]) for r in identities}) != len(identities):
            raise ValueError("external identity already connected")
        for record in identities:
            if record["issuer"] != result["providers"][record["providerId"]]["issuer"]:
                raise ValueError("external identity issuer mismatch")
        for record in result["sessions"].values():
            identity_id = record["identityId"]
            if identity_id is not None:
                identity = result["identities"].get(identity_id)
                if identity is None or identity["personId"] != record["personId"]:
                    raise ValueError("browser session identity mismatch")
        return result
    except (ValueError, TypeError, KeyError) as exc:
        raise ApplyError(f"invalid browser sign-in state: {exc}") from exc


def without_person(raw: dict[str, Any], person_id: str) -> dict[str, Any]:
    return {
        table: {k: v for k, v in records.items() if v.get("personId") != person_id}
        for table, records in raw.items()
    }


def apply(state: Any, payload: dict[str, Any], index: int, op: str) -> Any:
    from .applied import ApplyError

    data = {table: dict(records) for table, records in state.browser_sign_in.items()}
    for table in ("providers", "identities", "invitations", "sessions"):
        data.setdefault(table, {})
    table = OPS[op]
    if op.startswith("delete"):
        key = payload.get("id")
        if key not in data[table]:
            raise ApplyError(f"entry {index}: unknown sign-in record")
        del data[table][key]
        if table == "providers":
            for name in ("identities", "invitations"):
                data[name] = {k: v for k, v in data[name].items() if v["providerId"] != key}
        if table in ("providers", "identities"):
            data["sessions"] = {
                k: v
                for k, v in data["sessions"].items()
                if v["identityId"] is None or v["identityId"] in data["identities"]
            }
    else:
        record = payload.get("record")
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise ApplyError("sign-in record required")
        if op == "linkExternalIdentity":
            invitation = data["invitations"].get(payload.get("invitationId"))
            if invitation is None or any(
                invitation[k] != record[k] for k in ("personId", "providerId")
            ):
                raise ApplyError("invitation already used or does not match")
            del data["invitations"][invitation["id"]]
        if op == "putBrowserSignIn":
            # The writer supplies time; replay never reads the clock.
            data[table] = {
                k: v for k, v in data[table].items() if v["expiresAt"] > record["authAt"]
            }
            if len(data[table]) >= 2000:
                raise ApplyError("too many active sign-ins")
        if op == "putSignInInvitation":
            # Invitations always last 24 hours. Derive the writer's time for pure replay.
            data[table] = {
                k: v for k, v in data[table].items() if v["expiresAt"] > record["expiresAt"] - 86400
            }
            if len(data[table]) >= 2000:
                raise ApplyError("too many active invitations")
        if op == "putExternalProvider":
            # Changes require fresh invitations as well as fresh browser authentication.
            data["invitations"] = {
                k: v for k, v in data["invitations"].items() if v["providerId"] != record["id"]
            }
        data[table][record["id"]] = record
    return replace(state, browser_sign_in=validate(data, state.people))
