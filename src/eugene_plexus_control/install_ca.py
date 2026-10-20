"""The install CA: what a person's devices trust to reach Eugene over TLS (LT2).

`specs/docs/design/tls-for-clients.md` §3-§4. Clients and browsers cannot pin
a node's key as machines do (LT1), so each node's client doors present a leaf
from one CA per install, which a device trusts once.

**Made once, by the active root, the first time it is unsealed with LT2.**
ECDSA P-256, valid 20 years. Its key is sealed under the passphrase-derived
key and replicated (`setInstallCa`), so a promoted standby issues under the
same CA and no device has to trust a new one.

**Name-constrained (LT2-C2, critical).** It may sign for private addresses and
local names only, so its key, if stolen, cannot forge a certificate for a
public name: a client that honours name constraints refuses one. A dotless
name (`amish-station`) is outside every permitted subtree, which is why
Eugene's URLs use an address or `<host>.local`.

**A leaf is only ever for a node's own key and names** (`issue`). The key
must be one the node registered under its own signature (`Node.tlsPins`),
and every name the CSR asks for must be the node's own: its signed address's
host, `<name>.local`, and the loopback names. Anything else refuses the whole
request and names what was refused.
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import ipaddress
import logging
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

log = logging.getLogger(__name__)

CA_DAYS = 20 * 365 + 5
LEAF_DAYS = 90
#: Issued a little in the past, so a device whose clock runs behind takes it.
BACKDATE = datetime.timedelta(hours=1)

#: §3, LT2-C2: what the CA may sign for.
PERMITTED_NETWORKS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("::1/128"),
)
PERMITTED_SUFFIXES = ("local", "home.arpa", "internal", "lan", "localhost")
LOOPBACK_NAMES = ("localhost", "127.0.0.1", "::1")
#: A DNS label the CA signs: letters, digits and inner hyphens. No `*`, no
#: underscore, nothing a client could read as a wildcard.
_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


class Refused(Exception):
    """A request this root will not sign; `status`, `title` and the message say why."""

    def __init__(self, status: int, title: str, detail: str) -> None:
        super().__init__(detail)
        self.status, self.title, self.detail = status, title, detail


@dataclass(frozen=True)
class Issued:
    certificate: str
    names: list[str]
    not_after: datetime.datetime


def spki_pin(public_key: Any) -> str:
    """Base64url SHA-256 of a key's SubjectPublicKeyInfo: the form of `Node.tlsPins`."""
    der = public_key.public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return base64.urlsafe_b64encode(hashlib.sha256(der).digest()).rstrip(b"=").decode("ascii")


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #


def normalise(name: str) -> str:
    """One spelling per name: an address in its compressed form, a DNS name in
    lower case without a trailing dot or brackets."""
    text = name.strip().strip("[]").rstrip(".").lower()
    try:
        return str(ipaddress.ip_address(text))
    except ValueError:
        return text


def _address(name: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(name)
    except ValueError:
        return None


def permitted(name: str) -> bool:
    """Whether the CA may sign for `name` (§3)."""
    name = normalise(name)
    address = _address(name)
    if address is not None:
        return any(address in network for network in PERMITTED_NETWORKS)
    if not name or len(name) > 253 or not all(_LABEL.match(p) for p in name.split(".")):
        return False
    return any(name == suffix or name.endswith("." + suffix) for suffix in PERMITTED_SUFFIXES)


def held_by_others(node_name: str, nodes: Any) -> set[str]:
    """Every name another node holds: its address's host, its `.local` name
    and the names listed for it. Loopback names are everyone's."""
    held: set[str] = set()
    for other in nodes.values():
        if other.name == node_name:
            continue
        held.update(own_names(other.name, other.url, list(other.certificateNames)))
    return held - set(LOOPBACK_NAMES)


def own_names(node_name: str, node_url: str | None, listed: list[str] | None = None) -> list[str]:
    """Every name a node may hold (§4), whether or not the CA may sign it."""
    names: list[str] = []
    host = urlsplit(node_url or "").hostname
    if host:
        names.append(normalise(host))
    names.append(normalise(f"{node_name}.local"))
    names.extend(LOOPBACK_NAMES)
    names.extend(normalise(n) for n in listed or [])
    return list(dict.fromkeys(names))


# --------------------------------------------------------------------------- #
# The CA
# --------------------------------------------------------------------------- #


def make_ca(label: str) -> tuple[str, bytes]:
    """A new install CA: its certificate (PEM) and key (PKCS8 DER)."""
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Eugene Plexus"),
            x509.NameAttribute(NameOID.COMMON_NAME, f"Eugene Plexus {label} CA"),
        ]
    )
    now = datetime.datetime.now(datetime.UTC)
    subtrees: list[x509.GeneralName] = [x509.IPAddress(n) for n in PERMITTED_NETWORKS]
    subtrees += [x509.DNSName(s) for s in PERMITTED_SUFFIXES]
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - BACKDATE)
        .not_valid_after(now + datetime.timedelta(days=CA_DAYS))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.NameConstraints(permitted_subtrees=subtrees, excluded_subtrees=None),
            critical=True,
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .sign(key, hashes.SHA256())
    )
    pem = certificate.public_bytes(serialization.Encoding.PEM).decode("ascii")
    der = key.private_bytes(
        serialization.Encoding.DER,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    return pem, der


def label_for(control_public_key: str | None) -> str:
    """Tells two installs' CAs apart in a device's trust store."""
    if not control_public_key:
        return "install"
    digest = hashlib.sha256(control_public_key.encode("ascii")).hexdigest()
    return f"install {digest[:6]}"


# --------------------------------------------------------------------------- #
# Leaves
# --------------------------------------------------------------------------- #


def _requested(csr: x509.CertificateSigningRequest) -> list[str]:
    try:
        san = csr.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except x509.ExtensionNotFound:
        return []
    names: list[str] = []
    for item in san:
        if isinstance(item, x509.IPAddress):
            names.append(normalise(str(item.value)))
        elif isinstance(item, x509.DNSName):
            names.append(normalise(item.value))
        else:
            raise Refused(403, "Not a name this root signs", f"The CSR asks for {item}.")
    return names


def issue(
    *,
    ca_certificate: str,
    ca_key: bytes,
    csr_pem: str,
    node_name: str,
    allowed: list[str],
    registered_pins: list[str],
    now: datetime.datetime | None = None,
    others: set[str] | None = None,
) -> Issued:
    """A leaf for `node_name`'s own key and names, or `Refused`."""
    try:
        csr = x509.load_pem_x509_csr(csr_pem.encode("ascii"))
    except (ValueError, UnicodeEncodeError) as exc:
        raise Refused(400, "Not a CSR", f"The request does not hold a PEM CSR: {exc}") from exc
    if not csr.is_signature_valid:
        raise Refused(400, "Not a CSR", "The CSR's signature does not verify.")
    public_key = csr.public_key()
    if not isinstance(public_key, ec.EllipticCurvePublicKey):
        raise Refused(403, "Not this node's key", "The CSR is not for an EC key.")
    if not registered_pins:
        raise Refused(
            403,
            "Not this node's key",
            f"{node_name} has registered no machine key, so no certificate can name it. "
            "Update it to a version with a machine port.",
        )
    if spki_pin(public_key) not in registered_pins:
        raise Refused(
            403,
            "Not this node's key",
            f"The CSR is for a key {node_name} did not register; only its machine key "
            "gets a certificate.",
        )
    # A name another node holds is nobody's to sign here: not this node's,
    # even if its own record says so (adversarial review, 2026-10-20).
    taken = others or set()
    own = [n for n in allowed if permitted(n) and n not in taken]
    requested = _requested(csr) or own
    refused = [n for n in requested if n not in own]
    if refused:
        raise Refused(
            403,
            "Not this node's names",
            f"{node_name} may not hold {', '.join(refused)}: a certificate names only the "
            "node's own signed address, its .local name and the loopback names, and the "
            "install CA signs only private addresses and local names.",
        )
    ca = x509.load_pem_x509_certificate(ca_certificate.encode("ascii"))
    key = serialization.load_der_private_key(ca_key, password=None)
    if not isinstance(key, ec.EllipticCurvePrivateKey):
        raise Refused(503, "No CA", "The install CA's key is not an EC key.")
    now = now or datetime.datetime.now(datetime.UTC)
    not_after = now + datetime.timedelta(days=LEAF_DAYS)
    names = list(dict.fromkeys(requested))
    sans: list[x509.GeneralName] = []
    for name in names:
        address = _address(name)
        sans.append(x509.IPAddress(address) if address is not None else x509.DNSName(name))
    dns = [n for n in names if _address(n) is None and n != "localhost"]
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, dns[0])]) if dns else x509.Name([])
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca.subject)
        .public_key(public_key)
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - BACKDATE)
        .not_valid_after(not_after)
        # Critical when the subject is empty (RFC 5280 §4.2.1.6).
        .add_extension(x509.SubjectAlternativeName(sans), critical=not dns)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(public_key), critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(key.public_key()), critical=False
        )
        .sign(key, hashes.SHA256())
    )
    try:
        certificate.verify_directly_issued_by(ca)
    except (InvalidSignature, ValueError, TypeError) as exc:  # pragma: no cover - a bug
        raise Refused(503, "No CA", f"The leaf did not verify against the CA: {exc}") from exc
    return Issued(
        certificate=certificate.public_bytes(serialization.Encoding.PEM).decode("ascii"),
        names=names,
        not_after=not_after,
    )


# --------------------------------------------------------------------------- #
# Holding it: made once, sealed, replicated
# --------------------------------------------------------------------------- #


def ensure(auth: Any, machine: Any) -> None:
    """Make the install CA when this active, unsealed root has none, and hold
    its key in memory either way. Never raises: a root that cannot make a CA
    still serves everything else, and its doors stay plain."""
    from . import security
    from .applied import OP_SET_INSTALL_CA

    identity = machine.state.identity
    if auth.master_key is None:
        return
    if identity.sealedInstallCaKey:
        if auth.install_ca_key is None:
            try:
                auth.install_ca_key = security.open_b64(
                    identity.sealedInstallCaKey, auth.master_key
                )
            except ValueError as exc:
                log.error("the install CA's key did not open: %s", exc)
        return
    if not machine.is_active:
        return
    try:
        certificate, key = make_ca(label_for(identity.controlPublicKey))
        machine.append(
            OP_SET_INSTALL_CA,
            {"certificate": certificate, "sealedKey": security.seal_b64(key, auth.master_key)},
        )
    except Exception as exc:  # degraded mode: logged, never fatal
        log.error("could not make the install CA; client doors stay plain: %s", exc)
        return
    auth.install_ca_key = key
    log.info("made the install CA; nodes ask it for their client certificates now")
