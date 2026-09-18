"""Who a login actually came from, when it reached the trust root through a proxy.

**The browser never talks to this component directly.** It loads the UI
from some node's agent, and every call it makes goes to that agent's
`/api/proxy/control/...`, which reaches this process over loopback. So
by the time the login handler asks `request.client.host`, the answer is
`127.0.0.1` for every caller on earth — the one address it can never
usefully be.

Two separate defects come out of that, and only one of them needs an
attacker.

**Every proxied login shares one bucket.** The limiter in
`routes/auth.py` is per source, five failures in sixty seconds. With
every proxied caller keyed `127.0.0.1`, five mistyped passphrases from
one person lock *every* browser out of the trust root for a minute — and
the root's login is also what unlocks a sealed install, so the screen
this happens on is the one somebody reaches when nothing else works.
Nobody has to do anything wrong for it.

**And the key was attacker-supplied.** uvicorn ships
`ProxyHeadersMiddleware` on by default, trusting `X-Forwarded-For` from
`127.0.0.1` — which the proxy's peer always is. The agent forwarded the
caller's headers verbatim, so anyone who could reach the login page
could pick their own limiter bucket, a fresh one per attempt, and the
limiter counted to one forever.

## The shape of the fix, and why it is a header we own

The forwarding headers are stripped on the way through the agent's
proxy, uvicorn is told to trust none of them (`forwarded_allow_ips=[]`
in every entrypoint), and that proxy sets `PEER_HEADER` itself from the
peer it can actually see.

`PEER_HEADER` is believable for exactly the reason the agent's
`install_proxy.HOP_HEADER` is: **it is in that proxy's stripped set**, so
a caller's copy never survives the hop and the only value arriving here
is one an agent of this install just wrote. That is also why it is read
**only when the immediate peer is loopback**. Off-host the TCP peer is
better evidence than any header, and a request that crossed a network to
get here did not come through a proxy of ours.

**This is a copy of the agent's module, deliberately.** Components share
schemas, not code; `security.py` already lives in five copies for the
same reason. The trust root proxies nothing — it only ever *reads* the
header, under the same rule — so its copy is the reading half.

It confers no authority. It picks a rate-limit bucket, and nothing else.
"""

from __future__ import annotations

import ipaddress

__all__ = ["FORWARDING_HEADERS", "PEER_HEADER", "is_loopback", "peer_of"]

# Written by an agent's proxy from the peer it saw, and stripped by that
# same proxy on the way in, so a caller cannot supply one.
PEER_HEADER = "x-eugene-plexus-peer"

# The headers that claim to say where a request came from, every one of
# which is written by whoever sent it. Named here because this component
# must never be taught to read one: an operator who really does front
# the install with nginx configures `advertiseUrl` and a bind host, not
# a header we believe.
FORWARDING_HEADERS = frozenset(
    {
        "forwarded",
        "x-forwarded-for",
        "x-forwarded-host",
        "x-forwarded-port",
        "x-forwarded-prefix",
        "x-forwarded-proto",
        "x-real-ip",
    }
)


def _parsed(host: str | None) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """`host` as an address, or None if it is not one.

    A value that is not an address is not evidence of anything, and it
    is a fine rate-limit bucket key for an attacker who wants an
    unbounded supply of them. `strip("[]")` because an IPv6 peer arrives
    bracketed from some transports and bare from others.
    """
    if not host:
        return None
    try:
        return ipaddress.ip_address(host.strip().strip("[]"))
    except ValueError:
        return None


def is_loopback(host: str | None) -> bool:
    """Is this address this machine?

    `::1` and IPv4-mapped loopback are, which a string comparison to
    `"127.0.0.1"` would miss. Something that is not an address at all is
    not loopback: a test transport's `"testclient"` placeholder and a
    Unix socket peer are both *unknown*, and treating unknown as local
    is how a header comes to be believed from a caller we cannot see.
    """
    parsed = _parsed(host)
    if parsed is None:
        return False
    if parsed.is_loopback:
        return True
    mapped = getattr(parsed, "ipv4_mapped", None)
    return mapped is not None and mapped.is_loopback


def peer_of(peer: str | None, supplied: str | None) -> str | None:
    """The address this request came from, as well as we can know it.

    `peer` is the TCP peer (`request.client.host`); `supplied` is
    `PEER_HEADER` if one arrived.

    The header wins **only** when the peer is loopback, which is exactly
    the case where the peer is a proxy of ours and therefore says
    nothing. Off-host the peer is the truth and the header is ignored,
    which is what makes a forged one worthless from anywhere it could be
    forged. A header that is not an address is discarded rather than
    used.
    """
    if is_loopback(peer):
        forwarded = _parsed(supplied)
        if forwarded is not None:
            return str(forwarded)
    return peer
