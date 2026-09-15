"""OS-managed storage for the trust root's master key.

Used only when `securityMode == "os_keyring"`. Same shape as the agent's
module of this name, and deliberately not shared code — components share
schemas, not code, and the two differ in the one way that matters below.

**The service name must not be the agent's.** An agent and a control
root normally run on the same host, and they hold *different* master
keys: the agent derives its own, and the trust root mints and derives
its own precisely so the agent cannot read the install's secrets. Both
storing under `eugene-plexus-agent`/`master-key` would have each
overwrite the other, and the symptom would be an auto-unlock that opens
nothing — a sealed value that will not open, blamed on the passphrase.

**And the entry is scoped per install (S0 of the hobbyist UX plan,
2026-09-15).** The same overwrite happens between two *installs* of the
control root on one machine — the live one and a `.dev-install`, the
live one and every acceptance run — and with `os_keyring` becoming the
desktop default, a second install's wizard would silently replace the
first install's stored key. So the username carries a fingerprint of
this install's master-key salt: unique per install, minted before the
master key, stored beside it. A legacy single-slot entry is read once,
moved to its scoped name and deleted, so a root that predates this keeps
auto-unlocking across the upgrade.

What the OS boundary actually is, per platform:

  * Windows: Credential Manager, per Windows user
  * macOS: Keychain, per macOS user, may prompt on first access
  * Linux: Secret Service, per session, needs a running daemon

**What this trades away, stated plainly because it is the trust root.**
With auto-unlock on, the install's signing key and the recovery
recipient are reachable by anyone who can run code as this OS user,
without knowing the passphrase. That is a real reduction: for a driver
it exposes one backend's credential, here it exposes the whole install's
trust. It buys the thing the field promises — an install that comes back
from a power cut without a person present. Since S0 the wizard defaults
to it on a desktop, where "anyone who can run code as this user" is the
one person who set the passphrase; a server or a container keeps
`prompt_on_startup` or takes `passphrase_file`.

Every call is wrapped broadly: a missing or locked backend must never
crash the root. The fallback is always the passphrase prompt, which is
the behaviour with the field unset.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import secrets
import threading

import keyring
import keyring.errors

log = logging.getLogger(__name__)

# NOT "eugene-plexus-agent" — see the module docstring. The two processes
# co-exist on a host and hold different keys.
SERVICE = "eugene-plexus-control"

# The pre-scoping username. Read as a fallback and migrated; never
# written to again.
LEGACY_USERNAME = "master-key"

# Kept for readers of the old name. Nothing in this module writes to it.
USERNAME = LEGACY_USERNAME


def install_id_for(master_salt_b64: str) -> str:
    """A short, stable fingerprint of one install, from its master-key
    salt (`identity.salt`, base64). See the agent's twin for why twelve
    hex characters is enough."""
    raw = base64.b64decode(master_salt_b64)
    return hashlib.sha256(raw).hexdigest()[:12]


def username_for(install_id: str) -> str:
    return f"{LEGACY_USERNAME}:{install_id}"


def _read(username: str) -> bytes | None:
    try:
        encoded = keyring.get_password(SERVICE, username)
    except keyring.errors.KeyringError as e:
        log.warning("keyring read failed (%s); falling back to the passphrase prompt", e)
        return None
    except Exception as e:
        # Headless backends raise things that are not KeyringError — a
        # RuntimeError out of the dbus probe, for one.
        log.warning("keyring read raised unexpected %s (%s)", type(e).__name__, e)
        return None
    if not encoded:
        return None
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as e:
        log.warning("stored keyring value is not valid base64 (%s); ignoring", e)
        return None
    if len(raw) != 32:
        log.warning("stored keyring value is %d bytes, expected 32; ignoring", len(raw))
        return None
    return raw


def _write(username: str, master_key: bytes) -> bool:
    encoded = base64.b64encode(master_key).decode("ascii")
    try:
        keyring.set_password(SERVICE, username, encoded)
        return True
    except keyring.errors.KeyringError as e:
        log.warning("keyring write failed (%s); master key NOT persisted", e)
        return False
    except Exception as e:
        log.warning("keyring write raised unexpected %s (%s)", type(e).__name__, e)
        return False


def _delete(username: str) -> bool:
    try:
        keyring.delete_password(SERVICE, username)
        return True
    except keyring.errors.PasswordDeleteError:
        # Nothing stored. Not an error here.
        return False
    except keyring.errors.KeyringError as e:
        log.warning("keyring delete failed (%s)", e)
        return False
    except Exception as e:
        log.warning("keyring delete raised unexpected %s (%s)", type(e).__name__, e)
        return False


def get_master_key(install_id: str) -> bytes | None:
    """This install's stored master key, or None if absent, unreadable,
    or the backend is unavailable. Never raises.

    Reads the scoped entry first; failing that, the legacy single-slot
    entry, which is then moved to the scoped name — written first,
    deleted second, so a failure between the two leaves a duplicate
    rather than nothing. Whether the key opens this install's sealed
    values is still the caller's to find out, and `_auto_unlock` still
    discards one that does not.
    """
    scoped = username_for(install_id)
    found = _read(scoped)
    if found is not None:
        return found
    legacy = _read(LEGACY_USERNAME)
    if legacy is None:
        return None
    if _write(scoped, legacy):
        _delete(LEGACY_USERNAME)
        log.info("moved the stored master key to its per-install keyring entry")
    return legacy


def set_master_key(master_key: bytes, install_id: str) -> bool:
    """Persist the master key under this install's entry. True on success.

    Best-effort by design: a failure means the operator does not get
    auto-unlock next boot, not that they cannot finish unlocking now.
    """
    if len(master_key) != 32:
        raise ValueError("master key must be 32 bytes")
    return _write(username_for(install_id), master_key)


def delete_master_key(install_id: str) -> bool:
    """Remove this install's stored key. True if something was deleted.

    Called when the operator leaves `os_keyring`, and whenever a stored
    key turns out not to open this install's sealed values.

    That second case is not hypothetical: the keyring entry outlives the
    install directory. Wipe the state and re-initialize with a different
    passphrase - which is what any fresh test install is - and the old
    entry is still there, deriving to a key that opens nothing. There is
    no passphrase-change endpoint today, so this is the way a stale
    secret actually arises. Deleting beats keeping a value whose failure
    mode is "auto-unlock appeared to work". The legacy slot is cleared
    too, for a root that switched modes before it ever migrated.
    """
    scoped = _delete(username_for(install_id))
    legacy = _delete(LEGACY_USERNAME)
    return scoped or legacy


# --------------------------------------------------------------------------- #
# Availability probe
# --------------------------------------------------------------------------- #

_PROBE_LOCK = threading.Lock()
_probe_result: bool | None = None
_probe_done = False


def probe_sync() -> bool:
    """Whether this host's keyring can hold a secret for us: write, read
    back and delete a throwaway entry. Measured once per process, not
    assumed from the platform — a container reports False here and
    `passphrase_file` is its answer. Synchronous; the status route runs
    it in a thread with a deadline."""
    global _probe_result, _probe_done
    with _PROBE_LOCK:
        if _probe_done:
            return bool(_probe_result)
        username = f"probe:{secrets.token_hex(6)}"
        payload = secrets.token_bytes(32)
        ok = False
        try:
            if _write(username, payload):
                ok = _read(username) == payload
        finally:
            _delete(username)
        _probe_result = ok
        _probe_done = True
        if not ok:
            log.info(
                "this host's OS keyring did not accept a probe entry; securityMode "
                "os_keyring would not auto-unlock here (passphrase_file is the unattended path)"
            )
        return ok


def reset_probe_cache() -> None:
    """Tests only: forget the memoised probe result."""
    global _probe_result, _probe_done
    with _PROBE_LOCK:
        _probe_result = None
        _probe_done = False
