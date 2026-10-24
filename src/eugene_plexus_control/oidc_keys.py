"""The sign-in key's schedule: changed by itself, or on demand (row 5, 5a).

`specs/docs/design/rows/row-05.md` §2. The provider's RSA key (`oidc.py`)
was made at first need and never changed. Here:

* **A routine change** appends `putOidcKey`, which a standby on any version
  applies. The log keeps two keys (`applied.OIDC_KEYS_KEPT`), so the key it
  replaces keeps verifying, and apps stay signed in, until the last refresh
  token it signed expires. A second change would drop that key, so a routine
  change waits until the newest key is 30 days and 10 minutes old.
* **After a suspected leak** (*End every app sign-in now*) it appends
  `putOidcKey` and then `retireOidcKeys`, keeping only the new key: every
  token the old one signed stops verifying at once. Refused while the
  standby applies a log version older than `retireOidcKeys`.
* **The timer.** The active, unlocked root checks once an hour and changes
  the key when it is `signInKeyDays` old. A locked root says so once and
  waits.

Kept out of `oidc.py` so the provider's flow and the key's schedule change
apart.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from . import config as config_module
from . import oidc, security
from .applied import LOG_VERSION, OP_PUT_OIDC_KEY, OP_RETIRE_OIDC_KEYS

log = logging.getLogger(__name__)

#: `retireOidcKeys` came with log version 1 (row 5).
RETIRE_LOG_VERSION = 1
assert RETIRE_LOG_VERSION <= LOG_VERSION

#: A refresh token lives this long (`oidc.REFRESH_TOKEN_TTL_SECONDS`), so the
#: key that signed it must still be held this long after it stopped signing.
REFRESH_LIFETIME = timedelta(seconds=oidc.REFRESH_TOKEN_TTL_SECONDS)
#: A routine change waits until the newest key is this old: the refresh
#: lifetime, and ten minutes for clocks and a sign-in in flight.
ROUTINE_MIN_AGE = REFRESH_LIFETIME + timedelta(minutes=10)
#: How often the active root checks whether the key is due (`app._hourly`).
CHECK_EVERY_SECONDS = 3600.0


def now() -> datetime:
    """The schedule's clock. Tests move it; tokens keep the real one."""
    return datetime.now(UTC)


def day(moment: datetime) -> str:
    """`12 Oct`: the day, in words a person reads."""
    return f"{moment.day} {moment:%b}"


class TooSoon(Exception):
    """A routine change would drop a key whose refresh tokens may be live."""

    def __init__(self, changed: datetime, due: datetime) -> None:
        super().__init__(
            f"The sign-in key changed on {day(changed)}. Changing it again before "
            f"{day(due)} ends some app sign-ins. Choose End every app sign-in now to do it "
            "anyway."
        )


class StandbyBehind(Exception):
    """The standby's copy would stop at `retireOidcKeys`."""

    def __init__(self, node: str) -> None:
        super().__init__(f"Update {node} first; its copy would stop at this change.")


def _made(key: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(str(key["createdAt"]))


def due_at(state: Any) -> datetime | None:
    """When the newest key is `signInKeyDays` old; None for never or no key."""
    days = int(config_module.effective(state.config).get("signInKeyDays") or 0)
    if days <= 0 or not state.oidc_keys:
        return None
    return _made(state.oidc_keys[0]) + timedelta(days=days)


def key_list(state: Any) -> dict[str, Any]:
    """`OidcKeyList`: the keys, newest first, without key material."""
    keys: list[dict[str, Any]] = []
    newer: datetime | None = None
    for key in state.oidc_keys:
        made = _made(key)
        item: dict[str, Any] = {"kid": key["kid"], "createdAt": made, "signs": newer is None}
        if newer is not None:
            item["verifiesUntil"] = newer + REFRESH_LIFETIME
        keys.append(item)
        newer = made
    answer: dict[str, Any] = {"keys": keys}
    due = due_at(state)
    if due is not None:
        answer["nextChangeAt"] = due
    return answer


def change(app: Any, *, end_sign_ins: bool) -> None:
    """Make a new key and, after a leak, retire the others. Synchronous (an
    RSA key takes a moment to make), so the route runs it in a thread.

    Raises `oidc.Locked` on a standby or a locked root, `TooSoon` for a
    routine change that would drop a live key, and `StandbyBehind`."""
    from .routes.control import standby_short_of

    machine = app.state.machine
    auth = app.state.auth_state
    if not machine.is_active:
        raise oidc.Locked("this control root is a standby; change the key on the active one")
    master = auth.master_key
    if master is None:
        raise oidc.Locked(
            "Eugene is locked. Sign in to the console once with the passphrase to unlock it."
        )
    keys = machine.state.oidc_keys
    if keys and not end_sign_ins:
        changed = _made(keys[0])
        if now() - changed < ROUTINE_MIN_AGE:
            raise TooSoon(changed, changed + ROUTINE_MIN_AGE)
    if end_sign_ins:
        behind = standby_short_of(app, RETIRE_LOG_VERSION)
        if behind is not None:
            raise StandbyBehind(behind)
    pem, jwk = oidc.new_signing_key()
    machine.append(
        OP_PUT_OIDC_KEY,
        {
            "key": {
                "kid": jwk["kid"],
                "sealedKey": security.seal_b64(pem, master),
                "publicJwk": jwk,
                "createdAt": now().isoformat(),
            }
        },
    )
    if end_sign_ins:
        machine.append(OP_RETIRE_OIDC_KEYS, {"keep": [jwk["kid"]]})
    provider = getattr(app.state, "oidc", None)
    if provider is not None:
        provider.keep_only({k["kid"] for k in machine.state.oidc_keys})
    log.warning(
        "the sign-in key changed (%s)%s",
        jwk["kid"],
        "; every other key was retired, so every app signs in again"
        if end_sign_ins
        else "; the one before keeps verifying the sign-ins it made",
    )


def tick(app: Any) -> bool:
    """One check of the schedule. True when it changed the key."""
    machine = app.state.machine
    if not machine.is_active:
        return False
    due = due_at(machine.state)
    if due is None or now() < due:
        app.state.sign_in_key_wait_said = False
        return False
    if app.state.auth_state.master_key is None:
        if not getattr(app.state, "sign_in_key_wait_said", False):
            log.info(
                "the sign-in key was due on %s, and this root is locked; it changes the key "
                "once someone signs in",
                day(due),
            )
            app.state.sign_in_key_wait_said = True
        return False
    app.state.sign_in_key_wait_said = False
    change(app, end_sign_ins=False)
    return True
