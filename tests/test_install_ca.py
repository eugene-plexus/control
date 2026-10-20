"""The install CA and node certificates (LT2, `tls-for-clients.md` §3-§4).

* The CA is made once, at initialization, sealed and in the replicated log;
  it is a CA constrained (critical) to private addresses and local names.
* A leaf is for the asking node's registered key and own names only.
* A name constraint is what a client checks: a leaf for a public name signed
  by the CA fails path validation, and one for a node's own address passes.
"""

from __future__ import annotations

import datetime
import ipaddress

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509 import verification
from cryptography.x509.oid import NameOID
from fastapi.testclient import TestClient

from eugene_plexus_control import install_ca, security
from eugene_plexus_control.applied import (
    OP_SET_INSTALL_CA,
    AppliedState,
    ApplyError,
    apply_all,
    from_canonical,
    to_canonical,
)

from .conftest import PASSPHRASE, enroll


def _csr(key: ec.EllipticCurvePrivateKey, names: list[str]) -> str:
    sans: list[x509.GeneralName] = []
    for name in names:
        try:
            sans.append(x509.IPAddress(ipaddress.ip_address(name)))
        except ValueError:
            sans.append(x509.DNSName(name))
    builder = x509.CertificateSigningRequestBuilder().subject_name(
        x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "node")])
    )
    if sans:
        builder = builder.add_extension(x509.SubjectAlternativeName(sans), critical=False)
    return builder.sign(key, hashes.SHA256()).public_bytes(serialization.Encoding.PEM).decode()


def _verify(ca_pem: str, leaf_pem: str, subject: x509.GeneralName) -> None:
    """Path validation as a client does it, name constraints included."""
    ca = x509.load_pem_x509_certificate(ca_pem.encode())
    leaf = x509.load_pem_x509_certificate(leaf_pem.encode())
    verifier = (
        verification.PolicyBuilder().store(verification.Store([ca])).build_server_verifier(subject)  # type: ignore[arg-type]
    )
    verifier.verify(leaf, [])


@pytest.fixture(scope="module")
def ca() -> tuple[str, bytes]:
    return install_ca.make_ca("test")


def test_the_ca_is_a_ca_constrained_to_private_addresses_and_local_names(
    ca: tuple[str, bytes],
) -> None:
    cert = x509.load_pem_x509_certificate(ca[0].encode())
    assert cert.extensions.get_extension_for_class(x509.BasicConstraints).value.ca
    constraints = cert.extensions.get_extension_for_class(x509.NameConstraints)
    assert constraints.critical
    permitted = constraints.value.permitted_subtrees or []
    assert {str(n.value) for n in permitted if isinstance(n, x509.IPAddress)} == {
        str(n) for n in install_ca.PERMITTED_NETWORKS
    }
    assert {n.value for n in permitted if isinstance(n, x509.DNSName)} == set(
        install_ca.PERMITTED_SUFFIXES
    )
    assert isinstance(cert.public_key(), ec.EllipticCurvePublicKey)
    assert cert.not_valid_after_utc - cert.not_valid_before_utc > datetime.timedelta(days=7000)


@pytest.mark.parametrize(
    ("name", "ok"),
    [
        ("192.168.1.20", True),
        ("10.0.0.1", True),
        ("100.101.102.103", True),
        ("::1", True),
        ("fd00::5", True),
        ("8.8.8.8", False),
        ("2001:db8::1", False),
        ("amish-station.local", True),
        ("box.home.arpa", True),
        ("localhost", True),
        ("amish-station", False),
        ("bank.example.com", False),
        ("evil.local.example.com", False),
        ("..local", False),
    ],
)
def test_what_the_ca_may_sign_for(name: str, ok: bool) -> None:
    assert install_ca.permitted(name) is ok


def _issue(ca: tuple[str, bytes], key: ec.EllipticCurvePrivateKey, names: list[str]) -> object:
    return install_ca.issue(
        ca_certificate=ca[0],
        ca_key=ca[1],
        csr_pem=_csr(key, names),
        node_name="gpu-box",
        allowed=install_ca.own_names("gpu-box", "http://192.168.1.20:8079"),
        registered_pins=[install_ca.spki_pin(key.public_key())],
    )


def test_a_node_gets_a_leaf_for_its_own_key_and_names_that_clients_verify(
    ca: tuple[str, bytes],
) -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    issued = _issue(ca, key, ["192.168.1.20", "gpu-box.local"])
    assert issued.names == ["192.168.1.20", "gpu-box.local"]  # type: ignore[attr-defined]
    leaf = x509.load_pem_x509_certificate(issued.certificate.encode())  # type: ignore[attr-defined]
    assert install_ca.spki_pin(leaf.public_key()) == install_ca.spki_pin(key.public_key())
    days = (leaf.not_valid_after_utc - leaf.not_valid_before_utc).days
    assert days <= 398
    _verify(ca[0], issued.certificate, x509.IPAddress(ipaddress.ip_address("192.168.1.20")))  # type: ignore[attr-defined]
    _verify(ca[0], issued.certificate, x509.DNSName("gpu-box.local"))  # type: ignore[attr-defined]


def test_a_csr_without_names_gets_every_own_name_the_ca_may_sign(
    ca: tuple[str, bytes],
) -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    issued = _issue(ca, key, [])
    assert issued.names == [  # type: ignore[attr-defined]
        "192.168.1.20",
        "gpu-box.local",
        "localhost",
        "127.0.0.1",
        "::1",
    ]


def test_a_name_that_is_not_the_nodes_own_refuses_the_whole_request(
    ca: tuple[str, bytes],
) -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    with pytest.raises(install_ca.Refused) as refused:
        _issue(ca, key, ["192.168.1.20", "192.168.1.99"])
    assert refused.value.status == 403
    assert "192.168.1.99" in refused.value.detail


def test_a_public_address_the_node_holds_is_still_refused(ca: tuple[str, bytes]) -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    with pytest.raises(install_ca.Refused) as refused:
        install_ca.issue(
            ca_certificate=ca[0],
            ca_key=ca[1],
            csr_pem=_csr(key, ["203.0.113.9"]),
            node_name="vps",
            allowed=install_ca.own_names("vps", "http://203.0.113.9:8079"),
            registered_pins=[install_ca.spki_pin(key.public_key())],
        )
    assert refused.value.status == 403


def test_a_key_the_node_did_not_register_is_refused(ca: tuple[str, bytes]) -> None:
    registered = ec.generate_private_key(ec.SECP256R1())
    other = ec.generate_private_key(ec.SECP256R1())
    with pytest.raises(install_ca.Refused) as refused:
        install_ca.issue(
            ca_certificate=ca[0],
            ca_key=ca[1],
            csr_pem=_csr(other, ["192.168.1.20"]),
            node_name="gpu-box",
            allowed=install_ca.own_names("gpu-box", "http://192.168.1.20:8079"),
            registered_pins=[install_ca.spki_pin(registered.public_key())],
        )
    assert refused.value.status == 403
    assert "did not register" in refused.value.detail


def test_not_a_csr_is_a_400(ca: tuple[str, bytes]) -> None:
    with pytest.raises(install_ca.Refused) as refused:
        install_ca.issue(
            ca_certificate=ca[0],
            ca_key=ca[1],
            csr_pem="not a csr",
            node_name="gpu-box",
            allowed=[],
            registered_pins=["x"],
        )
    assert refused.value.status == 400


def test_a_leaf_for_a_public_name_signed_with_the_ca_key_fails_validation(
    ca: tuple[str, bytes],
) -> None:
    """What a thief holding the CA key gets: nothing a client accepts for a
    public name (LT2-C2). The same forgery for a private address passes,
    so it is the constraint that refuses the first."""
    ca_cert = x509.load_pem_x509_certificate(ca[0].encode())
    ca_key = serialization.load_der_private_key(ca[1], password=None)
    now = datetime.datetime.now(datetime.UTC)

    def forged(name: x509.GeneralName) -> str:
        key = ec.generate_private_key(ec.SECP256R1())
        return (
            x509.CertificateBuilder()
            .subject_name(x509.Name([]))
            .issuer_name(ca_cert.subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(now + datetime.timedelta(days=30))
            .add_extension(x509.SubjectAlternativeName([name]), critical=True)
            .add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),  # type: ignore[arg-type]
                critical=False,
            )
            .sign(ca_key, hashes.SHA256())  # type: ignore[arg-type]
            .public_bytes(serialization.Encoding.PEM)
            .decode()
        )

    public = x509.DNSName("bank.example.com")
    with pytest.raises(verification.VerificationError):
        _verify(ca[0], forged(public), public)
    private = x509.IPAddress(ipaddress.ip_address("192.168.1.50"))
    _verify(ca[0], forged(private), private)


# --------------------------------------------------------------------------- #
# Held once, sealed, replicated
# --------------------------------------------------------------------------- #


def test_set_install_ca_applies_once_and_survives_the_snapshot() -> None:
    entry = {
        "index": 1,
        "epoch": 1,
        "op": OP_SET_INSTALL_CA,
        "payload": {"certificate": "PEM", "sealedKey": "SEALED"},
    }
    state = apply_all(AppliedState(), [entry])
    assert state.identity.installCaCertificate == "PEM"
    again = from_canonical(to_canonical(state))
    assert again.identity.sealedInstallCaKey == "SEALED"
    with pytest.raises(ApplyError):
        apply_all(state, [{**entry, "index": 2}])


def test_initialization_makes_the_ca_sealed_under_the_passphrase(
    active_client: TestClient,
) -> None:
    machine = active_client.app.state.machine  # type: ignore[attr-defined]
    identity = machine.state.identity
    assert identity.installCaCertificate and identity.sealedInstallCaKey
    master = security.derive_master_key(PASSPHRASE, __import__("base64").b64decode(identity.salt))
    key = serialization.load_der_private_key(
        security.open_b64(identity.sealedInstallCaKey, master), password=None
    )
    cert = x509.load_pem_x509_certificate(identity.installCaCertificate.encode())
    assert install_ca.spki_pin(key.public_key()) == install_ca.spki_pin(cert.public_key())
    served = active_client.get("/v1/trust/ca", headers={"Authorization": ""})
    assert served.status_code == 200
    assert served.text == identity.installCaCertificate
    der = active_client.get("/v1/trust/ca?format=der", headers={"Authorization": ""})
    assert der.headers["content-type"] == "application/pkix-cert"
    assert x509.load_der_x509_certificate(der.content) == cert


def test_a_node_is_issued_a_certificate_only_for_itself(active_client: TestClient) -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    pin = install_ca.spki_pin(key.public_key())
    keys, made = enroll(
        active_client,
        "gpu-box",
        url="http://192.168.1.20:8079",
        machineUrl="https://192.168.1.20:8083",
        tlsPin=pin,
    )
    assert made.status_code == 201, made.text
    other, made = enroll(active_client, "other", url="http://192.168.1.21:8079")
    assert made.status_code == 201, made.text

    def ask(as_node: object, names: list[str], signer: ec.EllipticCurvePrivateKey = key) -> object:
        return active_client.post(
            "/v1/nodes/gpu-box/certificate",
            headers={"Authorization": "Bearer " + as_node.service_token()},  # type: ignore[attr-defined]
            json={"csr": _csr(signer, names)},
        )

    issued = ask(keys, ["192.168.1.20", "gpu-box.local"])
    assert issued.status_code == 200, issued.text  # type: ignore[attr-defined]
    body = issued.json()  # type: ignore[attr-defined]
    assert body["names"] == ["192.168.1.20", "gpu-box.local"]
    assert body["ca"] == active_client.app.state.machine.state.identity.installCaCertificate  # type: ignore[attr-defined]
    _verify(body["ca"], body["certificate"], x509.DNSName("gpu-box.local"))

    assert ask(other, ["192.168.1.20"]).status_code == 403  # type: ignore[attr-defined]
    assert ask(keys, ["192.168.1.21"]).status_code == 403  # type: ignore[attr-defined]
    stranger = ec.generate_private_key(ec.SECP256R1())
    assert ask(keys, ["192.168.1.20"], stranger).status_code == 403  # type: ignore[attr-defined]
    session = active_client.post(
        "/v1/nodes/gpu-box/certificate", json={"csr": _csr(key, ["192.168.1.20"])}
    )
    assert session.status_code == 401


def test_a_sealed_root_issues_nothing(active_client: TestClient) -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    keys, made = enroll(
        active_client,
        "gpu-box",
        url="http://192.168.1.20:8079",
        machineUrl="https://192.168.1.20:8083",
        tlsPin=install_ca.spki_pin(key.public_key()),
    )
    assert made.status_code == 201, made.text
    active_client.app.state.auth_state.install_ca_key = None  # type: ignore[attr-defined]
    response = active_client.post(
        "/v1/nodes/gpu-box/certificate",
        headers={"Authorization": "Bearer " + keys.service_token()},
        json={"csr": _csr(key, [])},
    )
    assert response.status_code == 503


def test_the_machine_list_carries_plain_off(active_client: TestClient) -> None:
    from .test_machine_tls import _machine_list

    assert _machine_list(active_client)["clientPlainHttp"] == "on"
    assert active_client.patch("/v1/config", json={"clientPlainHttp": "off"}).status_code == 200
    assert _machine_list(active_client)["clientPlainHttp"] == "off"


def test_the_owner_lists_names_a_node_certificate_may_hold(active_client: TestClient) -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    keys, made = enroll(
        active_client,
        "gpu-box",
        url="http://192.168.1.20:8079",
        machineUrl="https://192.168.1.20:8083",
        tlsPin=install_ca.spki_pin(key.public_key()),
    )
    assert made.status_code == 201, made.text
    _, made = enroll(active_client, "other", url="http://192.168.1.21:8079")
    assert made.status_code == 201, made.text

    def ask(names: list[str]) -> int:
        return active_client.post(
            "/v1/nodes/gpu-box/certificate",
            headers={"Authorization": "Bearer " + keys.service_token()},
            json={"csr": _csr(key, names)},
        ).status_code

    assert ask(["eugene.lan"]) == 403
    listed = active_client.put(
        "/v1/nodes/gpu-box/certificate-names", json={"names": ["Eugene.LAN.", "wb.lan"]}
    )
    assert listed.status_code == 200, listed.text
    assert listed.json()["certificateNames"] == ["eugene.lan", "wb.lan"]
    assert ask(["eugene.lan", "wb.lan", "192.168.1.20"]) == 200
    public = active_client.put(
        "/v1/nodes/gpu-box/certificate-names", json={"names": ["eugene.example.com"]}
    )
    assert public.status_code == 400 and "eugene.example.com" in public.text
    theirs = active_client.put(
        "/v1/nodes/gpu-box/certificate-names", json={"names": ["other.local"]}
    )
    assert theirs.status_code == 409
    again = active_client.put("/v1/nodes/other/certificate-names", json={"names": ["wb.lan"]})
    assert again.status_code == 409
    node = active_client.get("/v1/nodes/gpu-box").json()
    assert node["certificateNames"] == ["eugene.lan", "wb.lan"]
