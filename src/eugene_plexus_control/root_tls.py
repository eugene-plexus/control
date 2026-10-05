"""The TLS keys this root's nodes name presents, signed by its identity (J7a).

Troy, 2026-10-05: a Job Site pins **this root's identity key**, carried in its
join command, not a certificate. The root connects to its own nodes name,
reads the public key it is shown, and signs the list with the key the site
pinned. A site checks every connection against that list, and fetches a
fresh one, over the very connection in question, when it is shown a key it
has not seen. So a renewal needs nobody, and neither a public CA nor a DNS
name is needed.

**Where the root looks.** The entry point hands this process the nodes
origin and, when Eugene's own proxy terminates TLS, the loopback address to
dial with that name (`EUGENE_PLEXUS_CONTROL_NODES_ORIGIN`,
`EUGENE_PLEXUS_CONTROL_NODES_PROBE`). Behind an outside proxy that holds the
certificate (Nginx Proxy Manager, say), there is nothing to dial locally, so
the probe goes to the public name itself. When that cannot work, the reason
is kept and served as a 503: a site joins nothing rather than trust a key it
cannot check.

**What is kept.** Every key seen whose certificate has not expired, at most
eight, so a renewal that changes the key leaves the old one usable until it
expires and sites move over in their own time. A key is dropped only by
expiry; a key the owner believes stolen goes with its certificate.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import ipaddress
import json
import logging
import socket
import ssl
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import jwt
from cryptography import x509
from cryptography.hazmat.primitives import serialization

from . import tokens

log = logging.getLogger(__name__)

TYP_ROOT_TLS = "ep-root-tls+jwt"
PROBE_SECONDS = 300.0
PROBE_TIMEOUT_SECONDS = 5.0
STALE_SECONDS = 60.0
MAX_KEYS = 8
FILE = "root_tls.json"


def spki_pin(der: bytes) -> tuple[str, int]:
    """Base64url SHA-256 of a certificate's SubjectPublicKeyInfo, and its expiry."""
    cert = x509.load_der_x509_certificate(der)
    spki = cert.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    digest = base64.urlsafe_b64encode(hashlib.sha256(spki).digest()).rstrip(b"=").decode()
    return digest, int(cert.not_valid_after_utc.timestamp())


def _is_address(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def presented(origin: str, dial: str | None) -> tuple[str, int]:
    """Connect as a site would and read the key shown. Blocking.

    Verification is off **here only**: the point is to learn what is shown,
    and what is learned is then signed, not trusted.
    """
    parts = urlsplit(origin)
    host = parts.hostname or ""
    port = parts.port or 443
    if dial:
        target = urlsplit("//" + dial)
        address = (target.hostname or "127.0.0.1", target.port or port)
    else:
        address = (host, port)
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    with socket.create_connection(address, timeout=PROBE_TIMEOUT_SECONDS) as sock:
        server_name = None if _is_address(host) else host
        with context.wrap_socket(sock, server_hostname=server_name) as tls:
            der = tls.getpeercert(binary_form=True)
    if not der:
        raise ValueError("no certificate was presented")
    return spki_pin(der)


class RootTls:
    """What was seen, the last list signed, and why there is none when there is none."""

    def __init__(self, state_dir: Path, origin: str | None, dial: str | None) -> None:
        self.origin = origin.rstrip("/") if origin else None
        self.dial = dial or None
        self._path = Path(state_dir) / FILE
        self.keys: dict[str, int] = {}
        self.error: str | None = None
        self.checked_at = 0.0
        self.jws: str | None = None
        self._signed_keys: tuple[tuple[str, int], ...] | None = None
        self._lock = asyncio.Lock()
        self._load()

    # ----- keeping ------------------------------------------------------

    def _load(self) -> None:
        try:
            document = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            log.warning("the kept TLS key list at %s is unreadable: %s", self._path, exc)
            return
        if document.get("origin") != self.origin:
            return
        keys = document.get("keys")
        if isinstance(keys, dict):
            self.keys = {str(k): int(v) for k, v in keys.items() if isinstance(v, int)}
        jws = document.get("jws")
        self.jws = jws if isinstance(jws, str) else None
        self._signed_keys = None

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps({"origin": self.origin, "keys": self.keys, "jws": self.jws}),
                encoding="utf-8",
            )
        except OSError as exc:
            log.warning("could not keep the TLS key list at %s: %s", self._path, exc)

    def _current(self) -> tuple[tuple[str, int], ...]:
        now = int(time.time())
        live = sorted(
            ((k, v) for k, v in self.keys.items() if v > now), key=lambda kv: (-kv[1], kv[0])
        )
        return tuple(live[:MAX_KEYS])

    # ----- looking ------------------------------------------------------

    async def probe(self) -> None:
        if self.origin is None:
            return
        async with self._lock:
            try:
                pin, not_after = await asyncio.to_thread(presented, self.origin, self.dial)
            except (OSError, ValueError, ssl.SSLError) as exc:
                where = self.dial or self.origin
                self.error = (
                    f"this root could not read the certificate {self.origin} presents "
                    f"(connecting to {where}: {exc})"
                )
                if not self.keys:
                    log.warning("%s; Job Sites cannot join until it can", self.error)
            else:
                self.error = None
                if pin not in self.keys:
                    log.info("the nodes name presents a TLS key not seen before: %s", pin)
                self.keys[pin] = not_after
                self.keys = dict(self._current())
                self._save()
            self.checked_at = time.perf_counter()

    async def run(self) -> None:
        while True:
            await self.probe()
            await asyncio.sleep(PROBE_SECONDS)

    # ----- signing ------------------------------------------------------

    def signed(self, control_private_key: str | None) -> str | None:
        """The current list as a JWS, re-signed when the keys changed. None while
        nothing has been seen; the last one signed while the root is locked."""
        current = self._current()
        if not current:
            return None
        if current != self._signed_keys and control_private_key is not None:
            payload: dict[str, Any] = {
                "iat": int(time.time()),
                "origin": self.origin,
                "keys": [{"spki": k, "notAfter": v} for k, v in current],
            }
            self.jws = jwt.encode(
                payload,
                tokens.load_private(control_private_key),
                algorithm="EdDSA",
                headers={"typ": TYP_ROOT_TLS},
            )
            self._signed_keys = current
            self._save()
        return self.jws
