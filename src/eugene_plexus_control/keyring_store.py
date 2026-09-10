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
from a power cut without a person present — which is why it is offered
at all, and why it is off by default.

Every call is wrapped broadly: a missing or locked backend must never
crash the root. The fallback is always the passphrase prompt, which is
the behaviour with the field unset.
"""

from __future__ import annotations

import base64
import logging

import keyring
import keyring.errors

log = logging.getLogger(__name__)

# NOT "eugene-plexus-agent" — see the module docstring. The two processes
# co-exist on a host and hold different keys.
SERVICE = "eugene-plexus-control"

# Single-operator, like the agent's. Multi-operator would key per
# operator here.
USERNAME = "master-key"


def get_master_key() -> bytes | None:
    """The stored master key, or None if absent, unreadable, or the
    backend is unavailable. Never raises."""
    try:
        encoded = keyring.get_password(SERVICE, USERNAME)
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


def set_master_key(master_key: bytes) -> bool:
    """Persist the master key. True on success.

    Best-effort by design: a failure means the operator does not get
    auto-unlock next boot, not that they cannot finish unlocking now.
    """
    if len(master_key) != 32:
        raise ValueError("master key must be 32 bytes")
    encoded = base64.b64encode(master_key).decode("ascii")
    try:
        keyring.set_password(SERVICE, USERNAME, encoded)
        return True
    except keyring.errors.KeyringError as e:
        log.warning("keyring write failed (%s); master key NOT persisted", e)
        return False
    except Exception as e:
        log.warning("keyring write raised unexpected %s (%s)", type(e).__name__, e)
        return False


def delete_master_key() -> bool:
    """Remove the stored key. True if something was deleted.

    Called when the operator leaves `os_keyring`, and whenever a stored
    key turns out not to open this install's sealed values.

    That second case is not hypothetical: the keyring entry outlives the
    install directory. Wipe the state and re-initialize with a different
    passphrase - which is what any fresh test install is - and the old
    entry is still there, deriving to a key that opens nothing. There is
    no passphrase-change endpoint today, so this is the way a stale
    secret actually arises. Deleting beats keeping a value whose failure
    mode is "auto-unlock appeared to work".
    """
    try:
        keyring.delete_password(SERVICE, USERNAME)
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
