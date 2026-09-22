"""Which names first-run setup and sign-in answer to.

**The attack is DNS rebinding, and it matters here more than anywhere
else in the install** because `POST /v1/auth/initialize` is first-come:
until an operator has chosen a passphrase, whoever calls it owns the
trust root. A page on `evil.example.com` that re-resolves its own name
to `127.0.0.1` after loading is, to the browser, same-origin with this
machine's loopback, so the same-origin policy that would stop it reading
another site stops nothing here. Login is the other unauthenticated
door, and since 2026-09-13 it is also how a sealed root is unlocked.

**What rebinding cannot change is the name.** The request still says
`Host: evil.example.com`, because that is the name the page was loaded
from. So these two routes -- and only these -- refuse a `Host` that is
none of:

  * an IP literal (a browser at an address typed that address);
  * `localhost`, or anything under `.localhost`;
  * a dotless name (`tower`, and the test transport's `testserver`),
    which only a local resolver or a search domain can answer;
  * a name under a suffix that public DNS cannot delegate to an
    attacker: `.local` (mDNS), `.lan`, `.home.arpa` (RFC 8375),
    `.internal`, and `.ts.net`, whose records Tailscale serves;
  * this machine's own hostname or FQDN;
  * a name listed in `EUGENE_PLEXUS_CONTROL_ALLOWED_HOSTS`, or anything
    at all when that list is `*`.

A request with **no** `Host` is allowed: HTTP/1.0 clients send none,
and a browser always sends one, so a missing header is not the attack.

**Why not every route.** Agents and standbys dial this root by the
`controlUrl` an operator typed, which may be any DNS name they run, and
every one of those routes wants a bearer token or a signature a
rebinding page does not have. Gating them would break a working
multi-host install to protect doors that are already shut.

**Why an environment variable is acceptable for the allowlist.** In
normal use these calls do not arrive by name at all: the browser talks
to a node agent, and the agent's proxy dials this root by address and
drops the browser's `Host`. The list is for an operator who points a
browser, or a script, straight at this port by a name of their own --
an expert override, set once beside the rest of the bootstrap.

A FastAPI dependency on the two routes rather than a middleware: the
level is then visible next to each handler, the way every other auth
requirement in this component is declared.
"""

from __future__ import annotations

import functools
import ipaddress
import socket

from fastapi import Request, status

from .dependencies import problem

__all__ = ["host_name", "is_trusted", "parse_allowed", "require_trusted_host"]

ALLOWED_HOSTS_ENV = "EUGENE_PLEXUS_CONTROL_ALLOWED_HOSTS"

_LOCAL_SUFFIXES = (".localhost", ".local", ".lan", ".home.arpa", ".internal", ".ts.net")

_ANY = "*"


def host_name(header: str) -> str:
    """The name in a `Host` header: port stripped, brackets off an IPv6
    literal, lower-cased, and one trailing dot removed.

    The trailing dot matters: `evil.example.com.` is the same name as
    `evil.example.com` to a resolver, so it must be the same name here.
    """
    value = header.strip()
    if value.startswith("["):
        end = value.find("]")
        name = value[1:end] if end != -1 else value[1:]
    elif value.count(":") == 1:
        name = value.split(":", 1)[0]
    else:
        # No port, or a bare IPv6 literal, which is not a valid Host but
        # is unambiguous: every colon is part of the address.
        name = value
    name = name.lower()
    return name[:-1] if name.endswith(".") else name


def parse_allowed(value: str | None) -> frozenset[str]:
    """`EUGENE_PLEXUS_CONTROL_ALLOWED_HOSTS` as a set of names.

    Comma-separated; each entry goes through `host_name`, so an operator
    who writes `nas.example.org:8083` gets the name they meant rather
    than an entry that can never match."""
    if not value:
        return frozenset()
    names = set()
    for entry in value.split(","):
        entry = entry.strip()
        if entry == _ANY:
            names.add(_ANY)
        elif entry:
            names.add(host_name(entry))
    return frozenset(names)


@functools.cache
def _machine_names() -> frozenset[str]:
    """This machine's hostname and FQDN, lower-cased.

    Cached for the life of the process, and asked only when every cheap
    rule has already said no: `getfqdn` can wait on a reverse lookup, and
    a dotless hostname -- the common case -- never gets this far.
    """
    names = set()
    for lookup in (socket.gethostname, socket.getfqdn):
        try:
            name = lookup()
        except OSError:
            continue
        if name:
            names.add(host_name(name))
    return frozenset(names)


def is_trusted(name: str, allowed: frozenset[str]) -> bool:
    """Could a browser pointed at this name be pointed at this machine
    by someone other than an attacker who controls the name's DNS?"""
    if not name or _ANY in allowed or name in allowed:
        return True
    try:
        ipaddress.ip_address(name)
    except ValueError:
        pass
    else:
        return True
    if name == "localhost" or "." not in name or name.endswith(_LOCAL_SUFFIXES):
        return True
    return name in _machine_names()


def require_trusted_host(request: Request) -> None:
    """Refuse a request whose `Host` names somewhere this machine is not.

    403 rather than 400 or 421: the request is well-formed and reached
    the right process; what is refused is the caller, by what it said
    about where it thinks it is."""
    header = request.headers.get("host")
    if not header:
        return
    name = host_name(header)
    settings = request.app.state.settings
    if is_trusted(name, parse_allowed(getattr(settings, "allowed_hosts", None))):
        return
    shown = name[:253]
    raise problem(
        status.HTTP_403_FORBIDDEN,
        "Unrecognized address",
        f"This request was addressed to {shown!r}, which is not an IP address, a local "
        "network name or this machine's own name, so first-run setup and sign-in will not "
        "answer it. That is what stops a web page somewhere else from renaming itself to "
        "reach this machine. If you reach this install through a name of your own, add it "
        f"to {ALLOWED_HOSTS_ENV} (comma-separated) on the machine running the control root "
        "and restart it -- or open the install through its agent, which needs no list.",
    )
