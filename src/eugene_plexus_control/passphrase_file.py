"""File-supplied passphrase for the trust root's unattended unlock.

Used only when `securityMode == "passphrase_file"`. The sibling of
`keyring_store`, and it exists because that module cannot work
everywhere: the OS keyring is Credential Manager, Keychain or a Secret
Service daemon, and **a container has none of them**. A control plane in
Docker therefore came back from every restart initialized-but-locked,
routing nothing, until a person opened a browser and typed the
passphrase. Found on a real UnRAID install, 2026-09-12, on the first
restart that install had ever had.

**Why a file and not an environment variable.** The obvious container
idiom is `EUGENE_PLEXUS_CONTROL_PASSPHRASE=...`, and it is the wrong one
here for two reasons that are specific rather than general:

  * The agent spawns every component with `os.environ.copy()` and
    filters nothing, so a passphrase in the environment is copied into
    the gateway, the library and every inference-driver — processes that
    front third-party backends and hold their own credentials.
  * `docker inspect` exposes environment values to anything that can
    reach the Docker socket, and the NAS UIs people actually run this on
    display them in a text box on screen.

A mounted file avoids both. It is also the established idiom for exactly
this (`POSTGRES_PASSWORD_FILE` and friends), so Docker secrets, Compose
`secrets:` and Kubernetes secret volumes all deliver it without anything
new being invented: each lands as a file, on tmpfs, mode 0400.

**What this trades away, stated plainly because it is the trust root.**
With the file in place, anyone who can read it can unlock the install —
without knowing anything else. That is the same reduction
`keyring_store` documents, moved from "can run code as this OS user" to
"can read this path". It buys what the field promises, an install that
comes back from a power cut without a person present, which is why it is
offered and why it stays off by default.

**One way it is better than the keyring, and the config field says so:**
`os_keyring` is host-bound, so a standby control root on another host
cannot inherit auto-unlock and asks for the passphrase at promotion. A
file is not host-bound. Mount the same secret on the standby and a
failover needs no human either.

**The passphrase, not the derived key.** `keyring_store` holds the
32 raw bytes of the master key, which it can because login puts them
there. Nothing can put them in a file without a new endpoint on the
trust root that emits key material, which is the surface least worth
adding. So this path takes the passphrase and runs the same Argon2id
derivation login does — one derivation per boot, and the derived key
never touches disk.

Every call degrades rather than raises: a missing, unreadable or wrong
passphrase leaves the root locked and asking, which is the behaviour
with the field unset.
"""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path

log = logging.getLogger(__name__)

# A passphrase is a passphrase, not a file full of them. Generous enough
# for a long diceware phrase and small enough that pointing this at a
# disk image is refused rather than read into memory.
MAX_BYTES = 4096


def read_passphrase(path: Path | None) -> str | None:
    """The passphrase held in `path`, or None if it cannot be had.

    Never raises. Every `None` is logged with the reason, because the
    visible symptom of all of them is identical — a locked root — and
    "which of the four things went wrong" is not something an operator
    can determine from the outside.
    """
    if path is None:
        log.warning(
            "securityMode is passphrase_file but EUGENE_PLEXUS_CONTROL_PASSPHRASE_FILE "
            "is not set; this root is locked until POST /v1/auth/login"
        )
        return None

    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        log.warning(
            "passphrase file %s does not exist; this root is locked until "
            "POST /v1/auth/login. In a container, check the secret is actually mounted "
            "at that path rather than declared.",
            path,
        )
        return None
    except OSError as e:
        # Permission, a directory, a dangling mount. All the same to us.
        log.warning("passphrase file %s could not be read (%s); this root stays locked", path, e)
        return None

    if len(raw) > MAX_BYTES:
        log.warning(
            "passphrase file %s is %d bytes, over the %d-byte limit; refusing to treat it "
            "as a passphrase",
            path,
            len(raw),
            MAX_BYTES,
        )
        return None

    _warn_if_widely_readable(path)

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        log.warning(
            "passphrase file %s is not valid UTF-8 (%s); this root stays locked. A file "
            "written by a Windows editor may carry a byte-order mark.",
            path,
            e,
        )
        return None

    passphrase = _strip_one_trailing_newline(text)
    if not passphrase:
        log.warning("passphrase file %s is empty; this root stays locked", path)
        return None
    return passphrase


def _strip_one_trailing_newline(text: str) -> str:
    """Remove exactly one trailing newline, and nothing else.

    `echo secret > file` appends one, `printf` does not, and Docker and
    Kubernetes secrets preserve whatever was given — so the same
    passphrase reaches this function two ways and only one of them can be
    right. One newline is stripped because a human writing the file with
    a shell or an editor gets it without asking.

    Nothing else is stripped. `.strip()` would be the obvious call and is
    wrong: a passphrase may legitimately begin or end with a space, and
    silently mangling it would present as "the passphrase in my file is
    not accepted" with no way to tell why.
    """
    if text.endswith("\r\n"):
        return text[:-2]
    if text.endswith("\n"):
        return text[:-1]
    return text


def _warn_if_widely_readable(path: Path) -> None:
    """Say so when the file is readable beyond its owner.

    A warning and not a refusal, by `easy-default-expert-override`'s
    corollary: an eager refusal can be wrong and an explanation of a real
    state cannot. Container secret mounts already arrive at 0400, bind
    mounts from a NAS share routinely do not, and refusing to start over
    a permission bit would lock an operator out of an install that works.

    Silent on Windows, where the POSIX bits are not the access control
    and reading them would produce a confident wrong answer.
    """
    if os.name == "nt":
        return
    try:
        mode = path.stat().st_mode
    except OSError:
        return
    if mode & (stat.S_IRGRP | stat.S_IROTH):
        log.warning(
            "passphrase file %s is readable by group or other (mode %o). Anyone who can "
            "read it can unlock this install. chmod 400 it.",
            path,
            stat.S_IMODE(mode),
        )


__all__ = ["MAX_BYTES", "read_passphrase"]
