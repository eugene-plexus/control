"""A Job Site's own tokens, checked against the site registry (J19, J23).

A site is not a node: its key is in no trust bundle, and nothing but this
root ever checks one of its tokens. So a site token is verified here, by
looking its issuer up in the replicated registry, rather than through
`tokens.verify` and the bundle every node holds.

The shape (`siteToken` in control.yaml): EdDSA, signed by the key the site
sent at enrollment; `iss: site:<id>`, `sub: site`, `aud: control`; `iat`
and `exp` at most five minutes apart. The same clock leeway every token
here allows (`tokens.LEEWAY_SECONDS`), so a site whose clock is a little
ahead still polls.
"""

from __future__ import annotations

from typing import Any

import jwt

from . import tokens
from .applied import SiteRecord, valid_site_id

ISSUER_PREFIX = "site:"
SUBJECT = "site"
AUDIENCE = tokens.RECIPIENT_CONTROL
MAX_LIFETIME_SECONDS = 300


class SiteTokenError(Exception):
    """Why a site token was refused, in a sentence."""


def issuer(site: str) -> str:
    return ISSUER_PREFIX + site


def verify(token: str, sites: dict[str, SiteRecord]) -> SiteRecord:
    """The enrolled site that signed `token`, or `SiteTokenError`."""
    try:
        unverified: dict[str, Any] = jwt.decode(token, options={"verify_signature": False})
        header = jwt.get_unverified_header(token)
    except jwt.InvalidTokenError:
        raise SiteTokenError("This is not a site token.") from None
    if header.get("alg") != "EdDSA":
        raise SiteTokenError("A site token is signed with EdDSA.")
    claimed = unverified.get("iss")
    if not isinstance(claimed, str) or not claimed.startswith(ISSUER_PREFIX):
        raise SiteTokenError("This is not a site token.")
    site = claimed[len(ISSUER_PREFIX) :]
    record = sites.get(site) if valid_site_id(site) else None
    if record is None:
        raise SiteTokenError(
            "This site is not enrolled here: it was removed, or it left. Join it again."
        )
    try:
        key = tokens.load_public(record.tokenPublicKey)
        claims: dict[str, Any] = jwt.decode(
            token,
            key,
            algorithms=["EdDSA"],
            audience=AUDIENCE,
            issuer=claimed,
            leeway=tokens.LEEWAY_SECONDS,
            options={"require": ["iss", "sub", "aud", "iat", "exp"]},
        )
    except (jwt.InvalidTokenError, ValueError) as exc:
        raise SiteTokenError(f"The site's token did not verify ({type(exc).__name__}).") from None
    if claims.get("sub") != SUBJECT:
        raise SiteTokenError("A site token's subject is `site`.")
    iat, exp = claims.get("iat"), claims.get("exp")
    if not isinstance(iat, int | float) or not isinstance(exp, int | float):
        raise SiteTokenError("A site token carries numeric iat and exp.")
    if exp - iat > MAX_LIFETIME_SECONDS:
        raise SiteTokenError("A site token lives five minutes at most.")
    return record
