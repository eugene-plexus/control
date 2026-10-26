"""Changing Eugene's passphrase on the control root (row 5, 5b).

`specs/docs/design/rows/row-05.md` §3. The master key is Argon2id(passphrase,
salt), and everything the root keeps sealed is sealed under it. A change
makes a new salt and a new master key, opens every sealed value with the old
key and seals it under the new one, and writes it all as one
`changePassphrase` entry. The keys themselves stay the same: the token key,
the identity key, the recovery key, the install CA and the sign-in keys are
re-wrapped, not replaced (R5-RESEAL, Troy 2026-10-21).

**The two nets.** `applied.sealed_inventory` is the list of sealed values,
and `changePassphrase` applies only when it names exactly that list. Under
it, `still_open` walks every string of the state the change would write and
refuses the change if one still opens with the old key: a sealed field that
was added later and never put on the list is found here, by where it is, and
never silently left sealed under a key nobody can derive.

**In order** (the route in `routes/auth.py` checks the session, `current`
and the standby first):

1. `reseal`, then `still_open` on the state the entry would produce.
2. The entry; the state machine writes a snapshot and compacts the log.
3. The keyring (`os_keyring`): the new key under the new install id, then
   the old entry deleted. A marker file covers a crash in between.
4. The passphrase file (`passphrase_file`): rewritten, or reported `old`.
5. `tell_machines`: each agent that keeps its own passphrase, over TLS only.

Neither passphrase is ever logged, put in a problem detail, or kept.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import config as config_module
from . import keyring_store, passphrase_file, security
from .applied import (
    OP_CHANGE_PASSPHRASE,
    SEALED_IDENTITY,
    AppliedState,
    ApplyError,
    apply,
    sealed_inventory,
    to_canonical,
)
from .state_machine import SealedUnderOldKey

log = logging.getLogger(__name__)

#: In the root's state directory while a change has written its entry and
#: has not yet moved the keyring entry: the old install id. Not a secret.
KEYRING_MARKER = "passphrase-change-keyring.json"

#: What each identity value is called when it does not open.
_IDENTITY_NAMES = {
    "signingKey": "token key",
    "controlKey": "identity key",
    "recoveryKey": "recovery key",
    "installCaKey": "install CA key",
}

_CHANGE_LOCK = threading.Lock()


class CannotOpen(Exception):
    """A sealed value did not open with the current master key. Nothing changed."""

    def __init__(self, what: str) -> None:
        super().__init__(what)
        self.what = what


class StillOpens(Exception):
    """The state the change would write still holds values sealed under the
    old key, at these places. Nothing changed."""

    def __init__(self, paths: list[str]) -> None:
        super().__init__(", ".join(paths))
        self.paths = paths


class Moved(Exception):
    """The state changed while the change was being prepared. Nothing changed."""


@dataclass(frozen=True)
class Resealed:
    """The `changePassphrase` payload, and the master key it was sealed under."""

    payload: dict[str, Any]
    master_key: bytes


@dataclass(frozen=True)
class Changed:
    at: datetime
    passphrase_file: str
    cleanup: str | None = None
    """Why the copies sealed under the old key are still on disk, when they
    are (`StateMachine.compact_pending`, retried later); else None."""


# --------------------------------------------------------------------------- #
# re-sealing
# --------------------------------------------------------------------------- #


def reseal(state: AppliedState, old_key: bytes, new_passphrase: str) -> Resealed:
    """Every sealed value, opened with `old_key` and sealed under the key the
    new passphrase derives with a fresh salt. Raises `CannotOpen`, naming the
    first value that does not open."""
    salt = security.generate_master_key_salt()
    new_key = security.derive_master_key(new_passphrase, salt)
    inventory = sealed_inventory(state)

    def again(sealed: str, what: str) -> str:
        try:
            raw = security.open_b64(sealed, old_key)
        except ValueError as exc:
            raise CannotOpen(what) from exc
        return security.seal_b64(raw, new_key)

    providers = state.browser_sign_in.get("providers") or {}
    out: dict[str, Any] = {
        name: again(inventory[name], _IDENTITY_NAMES[name])
        for name in SEALED_IDENTITY
        if name in inventory
    }
    out["oidcKeys"] = {
        kid: again(sealed, f"sign-in key {kid}") for kid, sealed in inventory["oidcKeys"].items()
    }
    out["externalProviders"] = {
        ident: again(sealed, f"connected account {providers[ident].get('name') or ident}")
        for ident, sealed in inventory["externalProviders"].items()
    }
    return Resealed(
        payload={
            "salt": base64.b64encode(salt).decode("ascii"),
            "passphraseVerifier": security.hash_passphrase(new_passphrase),
            "sealed": out,
        },
        master_key=new_key,
    )


def _places(value: Any, path: str = "") -> Any:
    """Every string, and every envelope-shaped object, with where it is."""
    if isinstance(value, dict):
        if security.is_envelope(value):
            yield path, value
        for key, item in value.items():
            yield from _places(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for position, item in enumerate(value):
            yield from _places(item, f"{path}[{position}]")
    elif isinstance(value, str):
        yield path, value


def _opens(value: Any, key: bytes) -> bool:
    try:
        if isinstance(value, dict):
            security.open_envelope(security.Envelope.from_dict(value), key)
        else:
            security.open_b64(value, key)
    except (ValueError, TypeError):
        return False
    return True


def still_open(state: AppliedState, old_key: bytes) -> list[str]:
    """Where the canonical form of `state` holds a value that opens with
    `old_key`: empty when the re-seal reached everything."""
    return [path for path, value in _places(to_canonical(state)) if _opens(value, old_key)]


# --------------------------------------------------------------------------- #
# the change
# --------------------------------------------------------------------------- #


def install_id(salt: str | None) -> str:
    return keyring_store.install_id_for(salt) if salt else ""


def _marker(app: Any) -> Path:
    return Path(app.state.settings.state_dir) / KEYRING_MARKER


def change(app: Any, new: str, *, actor: str | None = None) -> Changed:
    """Re-seal, append, and move the keyring and the file. Synchronous: two
    Argon2id runs, so the route calls it in a thread. The caller has checked
    the session, `current` and the standby."""
    machine = app.state.machine
    auth = app.state.auth_state
    with _CHANGE_LOCK:
        old_key, old_salt = auth.sealing_key()
        if old_key is None:
            raise CannotOpen("master key (this root is locked)")
        state = machine.state
        if old_salt != state.identity.salt:
            raise Moved()
        resealed = reseal(state, old_key, new)
        candidate = apply(
            state,
            {
                "index": state.index + 1,
                "epoch": state.epoch,
                "op": OP_CHANGE_PASSPHRASE,
                "payload": resealed.payload,
            },
        )
        missed = still_open(candidate, old_key)
        if missed:
            raise StillOpens(missed)
        if machine.state is not state:
            raise Moved()
        keyring = config_module.effective(state.config)["securityMode"] == "os_keyring"
        if keyring:
            note_stale_keyring(Path(app.state.settings.state_dir), old_salt)
        # The entry and the switch, with unlocks held off: an unlock that
        # read the old salt cannot publish its key between the two.
        with auth.publishing():
            try:
                machine.append(
                    OP_CHANGE_PASSPHRASE, resealed.payload, actor=actor, sealed_under=old_salt
                )
            except (ApplyError, SealedUnderOldKey) as exc:
                # Something sealed was added or removed since `reseal` read
                # the state: the entry no longer names exactly what is held.
                _marker(app).unlink(missing_ok=True)
                raise Moved() from exc
            # The entry is durable, so the new key is the one in effect now,
            # before anything that can fail: a sealed write after this point
            # seals under it, and one sealed before is refused by the log.
            auth.set_master_key(resealed.master_key, salt=resealed.payload["salt"])
        cleanup = machine.compact_pending()
        if keyring:
            move_keyring(app, old_salt)
        file = write_file(app, new)
        log.warning(
            "Eugene's passphrase changed: every sealed value was sealed again under the new key%s",
            "" if cleanup else ", and the old copies were removed",
        )
        return Changed(at=datetime.now(UTC), passphrase_file=file, cleanup=cleanup)


def _private_write(path: Path, text: str) -> None:
    from ._private_files import write_private_text

    path.parent.mkdir(parents=True, exist_ok=True)
    write_private_text(path, text)


def move_keyring(app: Any, old_salt: str | None) -> None:
    """The new key under the new install id, then the old entry deleted, then
    the marker. A crash part-way is finished by `finish_interrupted`."""
    machine = app.state.machine
    key = app.state.auth_state.master_key
    new_id = install_id(machine.state.identity.salt)
    if key is not None and keyring_store.set_master_key(key, new_id):
        log.info("the new master key is in the OS keyring for auto-unlock")
    else:
        log.warning(
            "the new master key could not be stored in the OS keyring; this root asks for the "
            "passphrase after its next restart"
        )
    old_id = install_id(old_salt)
    if old_id and old_id != new_id:
        keyring_store.delete_master_key(old_id)
    _marker(app).unlink(missing_ok=True)


def note_stale_keyring(state_dir: Path, old_salt: str | None) -> None:
    """Before a passphrase change is taken (written by the root, or applied
    by a standby): which keyring entry it makes stale. A kill before the
    cleanup leaves this, and `finish_interrupted` finishes it at the next
    start."""
    if old_salt:
        _private_write(
            Path(state_dir) / KEYRING_MARKER, json.dumps({"oldInstallId": install_id(old_salt)})
        )


def finish_interrupted(app: Any) -> None:
    """At start: a change that was taken and stopped before its keyring entry
    was cleaned up (on the root, before it moved; on a standby, before it was
    deleted). The old entry opens nothing now; it is deleted, and a root on
    the keyring says why it is locked."""
    marker = _marker(app)
    try:
        old_id = json.loads(marker.read_text(encoding="utf-8")).get("oldInstallId")
    except FileNotFoundError:
        return
    except (OSError, ValueError, AttributeError) as exc:
        log.warning("could not read %s (%s); removing it", marker, exc)
        marker.unlink(missing_ok=True)
        return
    current = install_id(app.state.machine.state.identity.salt)
    if isinstance(old_id, str) and old_id and old_id != current:
        held = keyring_store.delete_master_key(old_id)
        if held and keyring_store.get_master_key(current) is None:
            log.warning(
                "The stored key no longer opens this install (the passphrase changed); sign in."
            )
    marker.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# the passphrase file
# --------------------------------------------------------------------------- #


def _file_path(app: Any) -> Path | None:
    machine = app.state.machine
    if config_module.effective(machine.state.config)["securityMode"] != "passphrase_file":
        return None
    path = getattr(app.state.settings, "passphrase_file", None)
    return Path(path) if path else None


def write_file(app: Any, new: str) -> str:
    """After a change: the new passphrase into the file this root unlocks
    from. `old` when it cannot be written (a container secret), `none` when
    this root does not unlock from a file."""
    path = _file_path(app)
    if path is None:
        state = "none"
    elif passphrase_file.write_passphrase(path, new):
        state = "matches"
    else:
        state = "old"
    app.state.passphrase_file_state = state
    app.state.passphrase_file_checked = _file_inputs(app)
    return state


def _file_inputs(app: Any) -> tuple[Any, ...]:
    """What the file's state depends on, besides the file: the mode, the path
    and the verifier. A change to any of them is checked again at once."""
    return (_file_path(app), app.state.machine.state.identity.passphraseVerifier)


def file_state(app: Any) -> str | None:
    """Whether the file this root unlocks from holds this install's
    passphrase: `matches`, `old`, `none` (no file in use), or None when it
    cannot be read (`passphrase_file` logged why). One Argon2id run."""
    path = _file_path(app)
    if path is None:
        return "none"
    passphrase = passphrase_file.read_passphrase(path)
    if passphrase is None:
        return None
    verifier = app.state.machine.state.identity.passphraseVerifier
    if verifier is None:
        return None
    return "matches" if security.verify_passphrase(passphrase, verifier) else "old"


def recheck_file(app: Any) -> str | None:
    """At start and every hour: remember the file's state, and say once when
    it holds another passphrase."""
    state = file_state(app)
    before = getattr(app.state, "passphrase_file_state", None)
    app.state.passphrase_file_state = state
    app.state.passphrase_file_checked = _file_inputs(app)
    if state == "old" and before != "old":
        log.warning(
            "the passphrase file %s holds another passphrase than this install's: after a "
            "restart this root stays locked until someone signs in. Put the new passphrase "
            "in it.",
            _file_path(app),
        )
    return state


def current_file_state(app: Any) -> str | None:
    """`AuthStatus.passphraseFile`: the last check, made again only when the
    mode, the path or the passphrase changed since (the file's own content
    is checked every hour). Cheap unless something changed."""
    if getattr(app.state, "passphrase_file_checked", None) != _file_inputs(app):
        recheck_file(app)
    state: str | None = getattr(app.state, "passphrase_file_state", None)
    return state


# --------------------------------------------------------------------------- #
# the machines that keep their own passphrase
# --------------------------------------------------------------------------- #


def _on_this_machine(url: str) -> bool:
    """A loopback address never leaves this machine (two agents on one
    host), so a passphrase sent there crosses no network."""
    import ipaddress
    from urllib.parse import urlsplit

    host = (urlsplit(url).hostname or "").strip("[]").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _problem_type(body: Any) -> str:
    if isinstance(body, dict):
        for holder in (body, body.get("detail")):
            if isinstance(holder, dict) and isinstance(holder.get("type"), str):
                return str(holder["type"])
    return ""


def _problem_detail(body: Any) -> str:
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, dict):
            detail = detail.get("detail")
        if isinstance(detail, str) and detail:
            return detail[:300]
    return ""


async def tell_machines(app: Any, current: str, new: str) -> list[dict[str, Any]]:
    """Send `{current, new}` to each machine whose agent keeps its own
    passphrase (row 5 §3 step 6). Over its machine port's pinned TLS, or on
    loopback to this machine's own agent; never in plain HTTP. Each machine
    is `changed`, `pending` (it takes it at its next sign-in) or `plain`."""
    from ._http import machine_route
    from .applied import normalize_url
    from .routes.nodes import service_token_for

    machine = app.state.machine
    client = app.state.nodes_client
    auth = app.state.auth_state
    own = getattr(app.state, "node_name", None)
    records = sorted(machine.state.nodes.values(), key=lambda r: r.name)
    targets = [(r.name, normalize_url(r.url)) for r in records]
    probes = await client.probe_all(targets, lambda node: service_token_for(auth, node))
    results: list[dict[str, Any]] = []
    for name, url in targets:
        probe = probes.get(name)
        if probe is None or not probe.reachable:
            why = probe.error if probe is not None and probe.error else "it did not answer"
            results.append({"node": name, "state": "pending", "detail": f"Not reached ({why})."})
            continue
        if probe.own_passphrase is False:
            continue
        if probe.own_passphrase is None:
            results.append(
                {
                    "node": name,
                    "state": "pending",
                    "detail": "It runs an agent older than this change. Update it; if it keeps "
                    "its own passphrase, it takes the new one at its next sign-in.",
                }
            )
            continue
        if name == own:
            # This machine's own agent, on loopback, where the agent takes a
            # passphrase without TLS: never its LAN address.
            url = normalize_url(getattr(app.state.settings, "agent_url", None))
        if url is None:
            results.append({"node": name, "state": "pending", "detail": "No address on record."})
            continue
        if name != own and not _on_this_machine(url) and machine_route(url) is None:
            results.append(
                {
                    "node": name,
                    "state": "plain",
                    "detail": "It is reached in plain HTTP, so nothing was sent to it.",
                }
            )
            continue
        try:
            status, body = await client.post(
                url,
                "/v1/node/passphrase",
                service_token_for(auth, name),
                {"current": current, "new": new},
            )
        except Exception as exc:
            results.append(
                {
                    "node": name,
                    "state": "pending",
                    "detail": f"Not reached ({type(exc).__name__}).",
                }
            )
            continue
        kind = _problem_type(body)
        if status in (200, 204):
            took: dict[str, Any] = {"node": name, "state": "changed"}
            if isinstance(body, dict) and body.get("passphraseFile") == "old":
                cause = str(body.get("detail") or "it did not say why")[:200]
                took["detail"] = f"this machine could not save its passphrase file: {cause}"
            results.append(took)
        elif status == 409 and kind.endswith("no-own-passphrase"):
            continue
        elif status == 409 and kind.endswith("passphrase-differs"):
            results.append(
                {
                    "node": name,
                    "state": "pending",
                    "detail": "Its own passphrase was another one; it takes the new one at its "
                    "next sign-in.",
                }
            )
        elif status == 403:
            results.append(
                {
                    "node": name,
                    "state": "plain",
                    "detail": "It refused the passphrase over a connection that was not TLS.",
                }
            )
        else:
            detail = _problem_detail(body) or f"It answered {status}."
            results.append({"node": name, "state": "pending", "detail": detail})
    for item in results:
        log.info(
            "passphrase change on %s: %s%s",
            item["node"],
            item["state"],
            f" ({item['detail']})" if item.get("detail") else "",
        )
    return results


# --------------------------------------------------------------------------- #
# a standby that followed a change
# --------------------------------------------------------------------------- #


def standby_followed(app: Any, old_salt: str) -> None:
    """A standby's copy took a passphrase change (an entry or a snapshot): the
    master key it may hold opens nothing now, and its keyring entry is under
    the old install id. Both go; promotion takes the new passphrase."""
    auth = app.state.auth_state
    had = auth.master_key is not None
    auth.forget_master_key()
    provider = getattr(app.state, "oidc", None)
    if provider is not None:
        provider.forget()
    keyring_store.delete_master_key(install_id(old_salt))
    _marker(app).unlink(missing_ok=True)
    recheck_file(app)
    log.warning(
        "the install's passphrase changed; this standby %sholds no key now, and a promotion "
        "takes the new passphrase",
        "forgot its old master key and " if had else "",
    )
