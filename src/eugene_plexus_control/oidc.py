"""Eugene as an OpenID Connect provider (C2).

Design: `specs/docs/design/sign-in-with-eugene.md`. The control root signs
people in to apps: the Authorization Code flow with PKCE (RFC 9700, OAuth
2.1), confidential clients only, RS256 tokens. What lives here:

* **The sign-in key.** A 3072-bit RSA key, made at first need on an
  unlocked active root, sealed under the master key like the root token
  key and replicated with `putOidcKey`, so a promoted standby signs people
  in with the same key. RS256 because it is the one algorithm OIDC
  requires of a provider, so every relying party supports it.
* **Three token kinds, all RS256 under that key** -- ID (`JWT`), access
  (`at+jwt`, RFC 9068) and refresh (`rt+jwt`). Hub verifiers accept EdDSA
  `ep-*` classes only, so none of them opens anything in the hub (D3,
  D12): a sign-in says who someone is and nothing more.
* **Pending sign-in requests and codes**, in this process's memory. A code
  lives 60 seconds and is good once; a pending request lives 10 minutes. A
  root restart costs only the sign-ins in flight, which is why neither is
  replicated.
* **The issuer**, which is `oidcIssuer` when the operator set one and
  otherwise the address the request arrived at through an agent (D11).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import html
import json
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlencode, urlsplit

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from . import security
from .applied import OP_PUT_OIDC_KEY, OPERATOR_NAME

ISSUER_PATH = "/oidc"
RSA_BITS = 3072

CODE_TTL_SECONDS = 60
REQUEST_TTL_SECONDS = 600
ID_TOKEN_TTL_SECONDS = 600
ACCESS_TOKEN_TTL_SECONDS = 600
REFRESH_TOKEN_TTL_SECONDS = 30 * 86400

TYP_ID = "JWT"
TYP_ACCESS = "at+jwt"
TYP_REFRESH = "rt+jwt"

#: Set by an agent's `/oidc/*` forward, after its host allowlist accepted
#: the host. Believed only beside a valid agent token (`NODE_TOKEN_HEADER`).
FORWARDED_HOST_HEADER = "x-eugene-plexus-forwarded-host"
FORWARDED_PROTO_HEADER = "x-eugene-plexus-forwarded-proto"
FORWARDED_FOR_HEADER = "x-eugene-plexus-forwarded-for"
NODE_TOKEN_HEADER = "x-eugene-plexus-node-token"

SIGN_IN_FAILURES = 5
SIGN_IN_WINDOW_SECONDS = 60

#: A person's password: the console's passphrase rule.
MIN_PASSWORD = 12


class Locked(Exception):
    """The root cannot sign: it is locked, or not the active root."""


def b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _int_b64(value: int) -> str:
    return b64url(value.to_bytes((value.bit_length() + 7) // 8, "big"))


def thumbprint(jwk: dict[str, Any]) -> str:
    """RFC 7638: SHA-256 of the required members, in order, no spaces."""
    canonical = json.dumps(
        {"e": jwk["e"], "kty": jwk["kty"], "n": jwk["n"]}, separators=(",", ":"), sort_keys=True
    )
    return b64url(hashlib.sha256(canonical.encode()).digest())


def new_signing_key() -> tuple[bytes, dict[str, Any]]:
    """(PKCS8 PEM, public JWK with `kid`) for a fresh RSA key."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=RSA_BITS)
    numbers = key.public_key().public_numbers()
    jwk: dict[str, Any] = {"kty": "RSA", "n": _int_b64(numbers.n), "e": _int_b64(numbers.e)}
    jwk.update({"kid": thumbprint(jwk), "alg": "RS256", "use": "sig"})
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    return pem, jwk


def secret_verifier(secret: str) -> str:
    """What is kept of a client secret: SHA-256, hex. The secret is 32
    random bytes, so a slow hash buys nothing a fast one does not."""
    return hashlib.sha256(secret.encode()).hexdigest()


def pkce_matches(verifier: str, challenge: str) -> bool:
    """RFC 7636 §4.6, S256."""
    if not 43 <= len(verifier) <= 128:
        return False
    computed = b64url(hashlib.sha256(verifier.encode("ascii", "ignore")).digest())
    return hmac.compare_digest(computed, challenge)


def now() -> int:
    return int(time.time())


def iso_to_epoch(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


# --------------------------------------------------------------------------- #
# who is signing in
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Subject:
    """Who a sign-in is for: the owner, or a person."""

    sub: str
    name: str
    preferred_username: str
    role: str  # "operator" | "member"

    def claims(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "preferred_username": self.preferred_username,
            "eugene_role": self.role,
        }


OWNER = Subject(sub=OPERATOR_NAME, name="Owner", preferred_username=OPERATOR_NAME, role="operator")


def subject_of(person: dict[str, Any]) -> Subject:
    return Subject(
        sub=person["id"],
        name=person.get("displayName") or person["name"],
        preferred_username=person["name"],
        role="member",
    )


def may_use(person: dict[str, Any], client_id: str) -> bool:
    apps = person.get("apps")
    return apps is None or client_id in apps


# --------------------------------------------------------------------------- #
# the provider's own state
# --------------------------------------------------------------------------- #


@dataclass
class PendingRequest:
    client_id: str
    redirect_uri: str
    state: str
    nonce: str
    challenge: str
    scope: str
    issuer: str
    expires: float


@dataclass
class IssuedCode:
    client_id: str
    redirect_uri: str
    challenge: str
    nonce: str
    scope: str
    issuer: str
    subject: Subject
    auth_at: float
    expires: float


@dataclass
class Provider:
    """One per root process: the unsealed key, pending requests, codes."""

    _keys: dict[str, Any] = field(default_factory=dict)
    _requests: dict[str, PendingRequest] = field(default_factory=dict)
    _codes: dict[str, IssuedCode] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def forget(self) -> None:
        """Drop the unsealed key: a demoted root stops being able to sign."""
        with self._lock:
            self._keys.clear()

    # --- keys ---------------------------------------------------------

    def signing_key(self, machine: Any, auth: Any) -> tuple[str, Any]:
        """The newest key, unsealed; made first if there is none."""
        if not machine.is_active:
            raise Locked("this control root is a standby; sign in through the active one")
        master = auth.master_key
        if master is None:
            raise Locked(
                "Eugene is locked. Sign in to the console once with the passphrase to unlock it."
            )
        keys = machine.state.oidc_keys
        if not keys:
            pem, jwk = new_signing_key()
            machine.append(
                OP_PUT_OIDC_KEY,
                {
                    "key": {
                        "kid": jwk["kid"],
                        "sealedKey": security.seal_b64(pem, master),
                        "publicJwk": jwk,
                        "createdAt": utcnow_iso(),
                    }
                },
            )
            keys = machine.state.oidc_keys
        newest = keys[0]
        with self._lock:
            key = self._keys.get(newest["kid"])
            if key is None:
                pem = security.open_b64(newest["sealedKey"], master)
                key = serialization.load_pem_private_key(pem, password=None)
                self._keys[newest["kid"]] = key
        return newest["kid"], key

    @staticmethod
    def jwks(state: Any) -> dict[str, Any]:
        return {"keys": [dict(k["publicJwk"]) for k in state.oidc_keys]}

    def mint(self, machine: Any, auth: Any, claims: dict[str, Any], typ: str) -> str:
        kid, key = self.signing_key(machine, auth)
        return jwt.encode(claims, key, algorithm="RS256", headers={"kid": kid, "typ": typ})

    @staticmethod
    def decode(token: str, state: Any, *, typ: str, audience: str | None) -> dict[str, Any]:
        """A token this provider issued, of kind `typ`, or `jwt.InvalidTokenError`."""
        header = jwt.get_unverified_header(token)
        if header.get("typ") != typ or header.get("alg") != "RS256":
            raise jwt.InvalidTokenError("not a token of that kind")
        public = next((k for k in state.oidc_keys if k["kid"] == header.get("kid")), None)
        if public is None:
            raise jwt.InvalidTokenError("unknown signing key")
        key = jwt.PyJWK(public["publicJwk"], algorithm="RS256").key
        required = ["iss", "sub", "aud", "exp", "iat", "jti"]
        if audience:
            return jwt.decode(
                token, key, algorithms=["RS256"], audience=audience, options={"require": required}
            )
        return jwt.decode(
            token, key, algorithms=["RS256"], options={"require": required, "verify_aud": False}
        )

    # --- pending requests and codes -----------------------------------

    def _sweep(self) -> None:
        moment = time.time()
        self._requests = {k: v for k, v in self._requests.items() if v.expires > moment}
        self._codes = {k: v for k, v in self._codes.items() if v.expires > moment}

    def open_request(self, request: PendingRequest) -> str:
        ident = secrets.token_urlsafe(24)
        with self._lock:
            self._sweep()
            self._requests[ident] = request
        return ident

    def pending(self, ident: str) -> PendingRequest | None:
        with self._lock:
            self._sweep()
            return self._requests.get(ident)

    def close_request(self, ident: str) -> None:
        with self._lock:
            self._requests.pop(ident, None)

    def issue_code(self, code: IssuedCode) -> str:
        value = secrets.token_urlsafe(32)
        with self._lock:
            self._sweep()
            self._codes[value] = code
        return value

    def take_code(self, value: str) -> IssuedCode | None:
        """Good once: taken on first use, whatever the outcome."""
        with self._lock:
            self._sweep()
            return self._codes.pop(value, None)


# --------------------------------------------------------------------------- #
# tokens for a sign-in
# --------------------------------------------------------------------------- #


def token_response(
    provider: Provider,
    machine: Any,
    auth: Any,
    *,
    issuer: str,
    client_id: str,
    subject: Subject,
    scope: str,
    auth_at: float,
    sid: str,
    nonce: str | None,
    refresh_token: str | None = None,
) -> dict[str, Any]:
    """ID and access tokens for one sign-in, and a refresh token unless one
    is being reused (refresh tokens are not rotated; D6)."""
    issued = now()
    base = {"iss": issuer, "sub": subject.sub, "aud": client_id, "iat": issued, "sid": sid}
    id_claims = {
        **base,
        "exp": issued + ID_TOKEN_TTL_SECONDS,
        "jti": secrets.token_urlsafe(12),
        "auth_time": int(auth_at),
        "azp": client_id,
        **subject.claims(),
    }
    if nonce is not None:
        id_claims["nonce"] = nonce
    access = {
        **base,
        "exp": issued + ACCESS_TOKEN_TTL_SECONDS,
        "jti": secrets.token_urlsafe(12),
        "client_id": client_id,
        "scope": scope,
    }
    answer: dict[str, Any] = {
        "access_token": provider.mint(machine, auth, access, TYP_ACCESS),
        "token_type": "Bearer",
        "expires_in": ACCESS_TOKEN_TTL_SECONDS,
        "id_token": provider.mint(machine, auth, id_claims, TYP_ID),
        "scope": scope,
    }
    if refresh_token is None:
        refresh = {
            **base,
            "exp": issued + REFRESH_TOKEN_TTL_SECONDS,
            "jti": secrets.token_urlsafe(12),
            "scope": scope,
            # Floating-point, unlike `auth_time`: a password changed in the
            # same second as a sign-in must still be told apart from it.
            "eugene_auth_at": auth_at,
        }
        refresh_token = provider.mint(machine, auth, refresh, TYP_REFRESH)
    answer["refresh_token"] = refresh_token
    return answer


def still_valid(state: Any, claims: dict[str, Any], client_id: str) -> Subject | None:
    """The subject a refresh or access token still speaks for, or None.

    None when the sign-in was revoked, the app is gone, or the person was
    deleted, disabled, taken off this app, or given a new password after
    the sign-in (D6, D9)."""
    if claims.get("sid") in state.revoked_sign_ins:
        return None
    if client_id not in state.oidc_clients:
        return None
    sub = claims.get("sub")
    if sub == OPERATOR_NAME:
        return OWNER
    person = state.people.get(sub)
    if person is None or person.get("disabled") or not may_use(person, client_id):
        return None
    auth_at = claims.get("eugene_auth_at")
    if isinstance(auth_at, (int, float)) and iso_to_epoch(person["passwordChangedAt"]) > auth_at:
        return None
    return subject_of(person)


# --------------------------------------------------------------------------- #
# the sign-in page
# --------------------------------------------------------------------------- #

_STYLE = """
:root { color-scheme: light dark; --bg: #f6f7f9; --panel: #ffffff; --text: #1d2330;
  --muted: #5b6475; --border: #d6dae2; --accent: #2f6fd6; --on-accent: #ffffff;
  --error: #b42318; }
@media (prefers-color-scheme: dark) { :root { --bg: #0f141c; --panel: #161d28;
  --text: #e7ebf2; --muted: #9aa4b5; --border: #2a3444; --accent: #8ec5ff;
  --on-accent: #0f141c; --error: #ff8a80; } }
* { box-sizing: border-box; }
body { margin: 0; min-height: 100vh; display: grid; place-items: center; padding: 16px;
  background: var(--bg); color: var(--text); font: 16px/1.5 system-ui, sans-serif; }
main { width: 100%; max-width: 380px; background: var(--panel); border: 1px solid var(--border);
  border-radius: 4px; padding: 24px; }
h1 { font-size: 1.25rem; margin: 0 0 4px; }
.sub { color: var(--muted); margin: 0 0 20px; }
label { display: block; margin: 12px 0 4px; font-weight: 600; }
input { width: 100%; padding: 8px 10px; border: 1px solid var(--border); border-radius: 4px;
  background: transparent; color: inherit; font: inherit; }
button { margin-top: 20px; width: 100%; padding: 10px; border: 0; border-radius: 4px;
  background: var(--accent); color: var(--on-accent); font: inherit; font-weight: 600;
  cursor: pointer; }
.error { color: var(--error); margin: 0 0 8px; }
.hint { color: var(--muted); font-size: 0.875rem; margin: 16px 0 0; }
details { margin-top: 16px; } summary { cursor: pointer; color: var(--accent); }
details .hint { margin-top: 8px; }
"""


def page(
    *,
    title: str,
    app_name: str | None,
    action: str | None = None,
    request_id: str | None = None,
    asks_name: bool = False,
    error: str | None = None,
    hint: str | None = None,
) -> str:
    e = html.escape
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{e(title)} · Eugene</title><style>{_STYLE}</style></head><body><main>",
        f"<h1>{e(title)}</h1>",
    ]
    if app_name:
        parts.append(f"<p class='sub'>{e(app_name)} asks who you are. Eugene checks it.</p>")
    if error:
        parts.append(f'<p class="error" role="alert">{e(error)}</p>')
    if action and request_id:
        parts.append(f"<form method='post' action='{e(action)}'>")
        parts.append(f'<input type="hidden" name="request" value="{e(request_id)}">')
        if asks_name:
            parts.append("<label for='name'>Your name</label>")
            parts.append('<input id="name" name="name" autocomplete="username" required autofocus>')
            parts.append("<label for='password'>Password</label>")
        else:
            parts.append("<label for='password'>Eugene's passphrase</label>")
        parts.append(
            '<input id="password" type="password" name="password" '
            'autocomplete="current-password" required' + ("" if asks_name else " autofocus") + ">"
        )
        if asks_name:
            # D10: a person given a password by the owner makes it their own
            # here. The owner's passphrase is not: nothing changes it yet.
            parts.append("<details><summary>Change my password</summary>")
            parts.append("<label for='new_password'>New password</label>")
            parts.append(
                f'<input id="new_password" type="password" name="new_password" '
                f'autocomplete="new-password" minlength="{MIN_PASSWORD}">'
            )
            parts.append("<label for='new_password_again'>New password again</label>")
            parts.append(
                f'<input id="new_password_again" type="password" name="new_password_again" '
                f'autocomplete="new-password" minlength="{MIN_PASSWORD}">'
            )
            parts.append(f"<p class='hint'>At least {MIN_PASSWORD} characters.</p></details>")
        parts.append("<button type='submit'>Sign in</button></form>")
    if hint:
        parts.append(f"<p class='hint'>{e(hint)}</p>")
    parts.append("</main></body></html>")
    return "".join(parts)


def page_headers(redirect_uri: str | None = None) -> dict[str, str]:
    """No framing, no scripts, no caching; the form may go to this
    provider and, because Chrome applies `form-action` to the redirect
    after a post, to the app it sends the browser back to."""
    form_action = "'self'"
    if redirect_uri:
        parts = urlsplit(redirect_uri)
        form_action += f" {parts.scheme}://{parts.netloc}"
    return {
        "Content-Security-Policy": (
            "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; "
            f"frame-ancestors 'none'; form-action {form_action}"
        ),
        "X-Frame-Options": "DENY",
        "Cache-Control": "no-store",
        "Referrer-Policy": "no-referrer",
    }


def redirect_with(uri: str, params: dict[str, str]) -> str:
    joiner = "&" if urlsplit(uri).query else "?"
    return uri + joiner + urlencode(params)
