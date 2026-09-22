"""Files only this account can read, written so a crash leaves the old one.

**What this component keeps on disk is the install's most secret
material.** The replicated log and the snapshot carry the sealed signing
key, the sealed control and recovery keys, the Argon2id salt and the
passphrase verifier -- and a salt plus a verifier is an offline attack on
the operator's passphrase for anyone who can read them. Until 2026-09-22
the log was created by `open("a")` and the config cache by `open("w")`,
so both landed at the process umask: `0644` on a stock Linux, readable by
every account on the host. The snapshot was already private, by accident
rather than design -- `tempfile` creates its files `0600` and
`os.replace` keeps the mode.

So every write of a file this component owns goes through here, and
both halves are the point:

  * **Private from the first byte.** `os.open(..., 0o600)` sets the mode
    at creation. Creating the file and `chmod`-ing it afterwards leaves a
    window in which another account can open it, and an open descriptor
    outlives the `chmod`.
  * **Replaced, not rewritten.** A temp file beside the target, flushed
    and fsync'd, then `os.replace`. `open("w")` truncates before it
    writes, so a kill between the two leaves an empty file where the
    config was. The temp is created `O_EXCL`, so a name somebody placed
    there in advance is refused rather than written through.

A **symlinked** target is written through: the real path is resolved
first and the temp lives beside *it*, so an operator who linked the file
somewhere keeps the link. `os.replace` on the link itself would swap the
link for a regular file.

On Windows the mode argument sets only the read-only bit; access there
is the ACL inherited from the directory, which for a per-user install is
the user's own profile. The code is the same on both, which is what lets
a test on this box assert the mode was *asked for*.

Deliberately duplicated in each component rather than shared: components
share schemas, not code (see CLAUDE.md, "no shared `core` library").
"""

from __future__ import annotations

import contextlib
import os
import uuid
from pathlib import Path
from typing import TextIO

__all__ = ["PRIVATE_MODE", "open_private_append", "write_private_text"]

PRIVATE_MODE = 0o600
"""Owner read and write, nobody else anything."""

# Without it a Windows descriptor is in the C runtime's text mode and
# every "\n" we write becomes "\r\n". Zero, and so a no-op, elsewhere.
_O_BINARY = getattr(os, "O_BINARY", 0)


def write_private_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Replace `path` with `text`, readable by this account only.

    Raises what the filesystem raises, after removing the temp file: a
    caller that treats the write as best-effort catches `OSError`, one
    that must not lose it lets it propagate. Either way no `.tmp` is
    left beside the target to be mistaken for something.
    """
    target = Path(os.path.realpath(path))
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    data = text.encode(encoding)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_BINARY, PRIVATE_MODE)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def open_private_append(path: Path, *, encoding: str = "utf-8", newline: str = "\n") -> TextIO:
    """Open `path` for appending, creating it `0600` if it is new.

    For the append-only log, where a replace-per-write would rewrite the
    whole file on every entry. An **existing** file is also narrowed to
    `0600`: an install whose log was created before this module existed
    is exactly as readable as it was, and the next append is the first
    chance to fix that. Best-effort -- a file owned by another account
    cannot be narrowed by this one, and refusing to append over it would
    lose an entry to tidy a permission.
    """
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | _O_BINARY, PRIVATE_MODE)
    try:
        if hasattr(os, "fchmod"):
            with contextlib.suppress(OSError):
                os.fchmod(fd, PRIVATE_MODE)
        return os.fdopen(fd, "a", encoding=encoding, newline=newline)
    except BaseException:
        os.close(fd)
        raise
