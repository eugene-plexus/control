"""What address a node is allowed to name for itself.

`PATCH /v1/nodes/{name}` is signed rather than bearer-authenticated, and
that part is sound: a node proves itself with its own Ed25519 identity,
the sequence stops a replay, and the signature covers the name so one
node's announcement cannot be aimed at another's record. What was
missing is the other half — **a correctly authenticated worker was
trusted to name an arbitrary URL.** `is_loopback_host` was the only
filter, so `http://169.254.169.254/` passed, and after that the root
probes it on a loop and the console proxy dials it carrying the
operator's own bearer.

Two rules, and they answer different questions.

**Some addresses are never a node.** A scheme that is not http(s), the
bind wildcard, a multicast group, a link-local address (which is where
every cloud provider's instance-metadata service lives) — none of these
is an install of anything, so they are refused outright with the reason,
at enrollment as well as at announcement.

**A node may not put itself on the open internet.** Classes are
`loopback`, `private` and `public`, where *private* is anything not
reachable from the open internet — RFC 1918, unique-local IPv6, and the
100.64.0.0/10 carrier range, which is where a tailnet lives and which
Python's `is_private` does **not** cover.

The refused transition is **exactly** *was not public, announces
public*, and it is deliberately not the more obvious "may not widen".
Writing it as a rank comparison is what the reproduction check caught
first: `loopback` → `private` is a widening on any ordering, and it is
also what the S5 Reach switch does on a *click*, on Home, on the
commonest install there is. private → private is DHCP after a reboot.
Both have to be invisible to this rule or the rule is a rule against the
product. Going public is the one transition a homelab node never makes
by itself and the one that redirects the operator's credential onto the
open internet. It is refused, and the remedy is re-enrollment — an
operator-minted join token, which is the confirmation this rule is
asking for.

A **hostname** is counted public, because the answer belongs to whoever
resolves it and this root cannot know it is not the attacker's. A node
that is genuinely reached by name enrolls by name and stays level
forever; one that was enrolled by IP and now wants a name re-enrolls
once, with the operator present.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

CLASS_LOOPBACK = "loopback"
CLASS_PRIVATE = "private"
CLASS_PUBLIC = "public"

_ALLOWED_SCHEMES = ("http", "https")


def host_of(url: str | None) -> str | None:
    """The host part, brackets already stripped from an IPv6 literal."""
    if not url:
        return None
    try:
        return urlsplit(url.strip()).hostname
    except ValueError:
        return None


def _ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def rejection(url: str | None) -> str | None:
    """Why this URL can never be a node's address, or None if it can be.

    Deliberately not "is this host up" and not "do I like this host" —
    only the addresses that are categorically not an install."""
    if not url or not url.strip():
        return "the address is empty"
    try:
        parts = urlsplit(url.strip())
    except ValueError as exc:
        return f"the address does not parse as a URL ({exc})"
    if parts.scheme.lower() not in _ALLOWED_SCHEMES:
        return (
            f"the scheme is {parts.scheme or '(none)'!r}; a node is reached over "
            "http or https and nothing else"
        )
    host = parts.hostname
    if not host:
        return "the address names no host"

    address = _ip(host)
    if address is None:
        return None
    if address.is_unspecified:
        return (
            f"{host} is the bind wildcard, not an address anything can dial. Set "
            "`advertiseUrl` on that node to the address other hosts use for it"
        )
    if address.is_multicast:
        return f"{host} is a multicast group, not a host"
    if address.is_link_local:
        return (
            f"{host} is link-local, which is not routable between hosts and is where "
            "cloud instance-metadata services listen. A node reached over a link-local "
            "address is a node nothing in this install can reach"
        )
    if address.is_reserved and not address.is_loopback:
        return f"{host} is in a reserved range and cannot be a host on this network"
    return None


def classify(url: str | None) -> str | None:
    """`loopback`, `private` or `public` — or None when there is no host.

    *private* is **not** Python's `is_private`: that answers False for
    100.64.0.0/10, which is where a tailnet address lives, so a node on
    a tailnet would read as public and the Reach switch would be refused
    on exactly the deployment this product is for. `is_global` asks the
    question the rule is actually about — is this reachable from the
    open internet — so the classes are built from its negation.
    """
    host = host_of(url)
    if not host:
        return None
    address = _ip(host)
    if address is None:
        lowered = host.lower()
        if lowered == "localhost" or lowered.endswith(".localhost"):
            return CLASS_LOOPBACK
        return CLASS_PUBLIC
    if address.is_loopback:
        return CLASS_LOOPBACK
    if address.is_global:
        return CLASS_PUBLIC
    return CLASS_PRIVATE


def goes_public(previous: str | None, announced: str | None) -> bool:
    """True when a node that was not on the open internet says it now is.

    Not "is this further away than before". `loopback` -> `private` is
    further away and is the Reach switch, which is a button on Home;
    treating it as a widening refuses the commonest legitimate
    announcement this install makes. The only transition worth a refusal
    is the one that hands a credential to the internet.

    False when there is no previous address — a node that has never had
    one is not moving anywhere, and enrollment is operator-witnessed.
    """
    before = classify(previous)
    after = classify(announced)
    if before is None or after is None:
        return False
    return after == CLASS_PUBLIC and before != CLASS_PUBLIC


__all__ = [
    "CLASS_LOOPBACK",
    "CLASS_PRIVATE",
    "CLASS_PUBLIC",
    "classify",
    "goes_public",
    "host_of",
    "rejection",
]
