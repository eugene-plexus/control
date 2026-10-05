"""Job Sites, slice 1 (`specs/docs/design/remote-nodes.md` §3, §5, §6).

A machine joins with a `files` invitation that names a person, who confirms
at the machine with their own password. It holds `files` and not `node`, has
no address, and is sent no inference work. Only its owner turns its helper on,
adds folders and says who may use them, from Workbench. Eugene's owner manages
membership; in production mode that is all, in dev mode they also see the
site's folders and may give themselves access (J13b).
"""

from __future__ import annotations

import base64
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jwt
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_control import node_helpers, root_tls, tokens
from eugene_plexus_control.app import create_app
from eugene_plexus_control.settings import Settings

from .conftest import PASSPHRASE, NodeKeys, login, mint_join_token, node_keys
from .test_oidc import CALLBACK, _tokens

REPORT = {"supported": True, "ready": True, "account": "isolated-file-helper"}
PUBLIC = {"X-Eugene-Plexus-Entry": "public-nodes"}
NODES_ORIGIN = "https://nodes.example.test:8443"


@pytest.fixture(autouse=True)
def quick_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(node_helpers, "POLL_SECONDS", 0.03)


@pytest.fixture
def root(tmp_path: Path) -> Iterator[TestClient]:
    """An active root whose entry point opened the public node route."""
    settings = Settings(
        config_file=tmp_path / "control.yaml",
        state_dir=tmp_path / "state",
        nodes_origin=NODES_ORIGIN,
        nodes_public=True,
    )
    with TestClient(create_app(settings)) as client:
        assert (
            client.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
        )
        client.headers["Authorization"] = f"Bearer {login(client)}"
        yield client


def agent(keys: NodeKeys) -> dict[str, str]:
    return {"Authorization": "Bearer " + keys.service_token()}


def workbench(c: TestClient) -> dict[str, Any]:
    made = c.post(
        "/v1/oidc/clients",
        json={"name": "Workbench", "owner": "app:workbench@root", "redirectUris": [CALLBACK]},
    )
    assert made.status_code == 201, made.text
    return made.json()


def auth(app: dict[str, Any]) -> tuple[str, str]:
    return app["client"]["clientId"], app["clientSecret"]


def add_person(c: TestClient, app: dict[str, Any], name: str) -> tuple[dict[str, Any], str]:
    made = c.post(
        "/v1/people",
        json={"name": name, "password": PASSPHRASE, "apps": [app["client"]["clientId"]]},
    )
    assert made.status_code == 201, made.text
    return made.json(), _tokens(c, app, name=name)["refresh_token"]


def join_site(
    c: TestClient, owner: dict[str, Any], name: str = "desk", **extra: Any
) -> tuple[NodeKeys, Any]:
    keys = node_keys(name)
    token = mint_join_token(c, grants=["files"], owner=owner["id"])
    body = keys.body(token, owner={"name": owner["name"], "password": PASSPHRASE}, **extra)
    return keys, c.post("/v1/nodes/enroll", json=body, headers=PUBLIC)


def deliver(c: TestClient, keys: NodeKeys) -> tuple[str, dict[str, Any]]:
    for _ in range(100):
        answer = c.post("/v1/node-helpers/poll", headers=agent(keys), json=REPORT)
        assert answer.status_code == 200, answer.text
        ident = answer.json()["operation"]
        if ident:
            claim = c.post(f"/v1/node-helpers/operations/{ident}/claim", headers=agent(keys))
            assert claim.status_code == 200, claim.text
            return ident, claim.json()
    pytest.fail("no file operation was delivered")


def site_with_folder(
    c: TestClient,
) -> tuple[NodeKeys, dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    """Ada's job site `desk`, its helper on, one writable folder nobody may use yet."""
    app = workbench(c)
    ada, ada_token = add_person(c, app, "ada")
    keys, joined = join_site(c, ada)
    assert joined.status_code == 201, joined.text
    enabled = c.post(
        "/oidc/job-sites/desk/enabled",
        auth=auth(app),
        json={"refreshToken": ada_token, "enabled": True},
    )
    assert enabled.status_code == 200, enabled.text
    polled = c.post("/v1/node-helpers/poll", headers=agent(keys), json=REPORT)
    assert polled.status_code == 200
    # The site is told its owner, the one person whose commands register
    # its folders; its own relay checks it again.
    assert polled.json()["siteOwner"] == ada["id"]
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(
            c.post,
            "/oidc/job-sites/desk/folders",
            auth=auth(app),
            json={
                "refreshToken": ada_token,
                "name": "Notes",
                "path": "C:\\Notes",
                "writable": True,
            },
        )
        ident, command = deliver(c, keys)
        assert command["tool"] == "inspect" and command["subject"] == ada["id"]
        done = c.post(
            f"/v1/node-helpers/operations/{ident}/result",
            headers=agent(keys),
            json={"status": "done", "result": {"path": "C:\\Notes", "identity": "vol:file"}},
        )
        assert done.status_code == 204
        folder = pending.result(timeout=5)
    assert folder.status_code == 201, folder.text
    return keys, app, ada, ada_token, folder.json()


def read(c: TestClient, app: dict[str, Any], token: str, folder: dict[str, Any]) -> Any:
    return c.post(
        "/oidc/node-helpers/execute",
        auth=auth(app),
        json={
            "refreshToken": token,
            "folderId": folder["id"],
            "tool": "read_text",
            "arguments": {"path": "a.txt"},
        },
    )


def read_through(
    c: TestClient, keys: NodeKeys, app: dict[str, Any], token: str, folder: dict[str, Any]
) -> Any:
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(read, c, app, token, folder)
        ident, _ = deliver(c, keys)
        c.post(
            f"/v1/node-helpers/operations/{ident}/result",
            headers=agent(keys),
            json={"status": "done", "result": {"text": "SITE-CONTENT"}},
        )
        return pending.result(timeout=5)


# --------------------------------------------------------------------------- joining


def test_a_job_site_invitation_names_a_person_and_holds_no_gateway(root: TestClient) -> None:
    c = root
    app = workbench(c)
    ada, _ = add_person(c, app, "ada")
    assert c.post("/v1/nodes/join-token", json={"grants": ["files"]}).status_code == 422
    both = c.post("/v1/nodes/join-token", json={"grants": ["files", "gateway"], "owner": ada["id"]})
    assert both.status_code == 422
    assert c.post("/v1/nodes/join-token", json={"owner": ada["id"]}).status_code == 422
    minted = c.post("/v1/nodes/join-token", json={"grants": ["files"], "owner": ada["id"]})
    assert minted.status_code == 201
    body = minted.json()
    assert body["owner"] == ada["id"]
    assert body["rootKey"] == c.app.state.machine.state.identity.controlPublicKey


def test_the_person_confirms_at_the_machine_and_a_wrong_try_keeps_the_token(
    root: TestClient,
) -> None:
    c = root
    app = workbench(c)
    ada, _ = add_person(c, app, "ada")
    keys = node_keys("desk")
    token = mint_join_token(c, grants=["files"], owner=ada["id"])
    unconfirmed = c.post("/v1/nodes/enroll", json=keys.body(token), headers=PUBLIC)
    assert unconfirmed.status_code == 401 and "ada" in unconfirmed.text
    wrong = keys.body(token, owner={"name": "ada", "password": "not-her-password"})
    assert c.post("/v1/nodes/enroll", json=wrong, headers=PUBLIC).status_code == 401
    someone_else = keys.body(token, owner={"name": "bo", "password": PASSPHRASE})
    assert c.post("/v1/nodes/enroll", json=someone_else, headers=PUBLIC).status_code == 401
    addressed = keys.body(
        token, owner={"name": "Ada", "password": PASSPHRASE}, url="http://203.0.113.9:8079"
    )
    assert c.post("/v1/nodes/enroll", json=addressed, headers=PUBLIC).status_code == 400
    good = keys.body(token, owner={"name": "Ada", "password": PASSPHRASE})
    joined = c.post("/v1/nodes/enroll", json=good, headers=PUBLIC)
    assert joined.status_code == 201, joined.text
    assert joined.json()["grants"] == ["files"]
    record = c.app.state.machine.state.nodes["desk"]
    assert record.grants == ("files",) and record.owner == ada["id"] and record.url is None
    listed = {
        k.issuer: k.grants
        for k in c.app.state.trust.view_for(c.app.state.machine.state).keys.values()
    }
    assert listed["node:desk"] == frozenset({"files"})


def test_through_the_public_route_only_a_job_site_may_join(root: TestClient) -> None:
    c = root
    keys = node_keys("rogue")
    token = mint_join_token(c)
    refused = c.post("/v1/nodes/enroll", json=keys.body(token), headers=PUBLIC)
    assert refused.status_code == 403
    # The token was not spent: the same machine joins from the LAN.
    assert c.post("/v1/nodes/enroll", json=keys.body(token)).status_code == 201


def test_a_job_site_token_opens_the_helper_routes_and_the_bundle_only(root: TestClient) -> None:
    c = root
    app = workbench(c)
    ada, _ = add_person(c, app, "ada")
    keys, joined = join_site(c, ada)
    assert joined.status_code == 201
    headers = agent(keys)
    assert c.post("/v1/node-helpers/poll", headers=headers, json=REPORT).status_code == 200
    for path in ("/v1/nodes", "/v1/components", "/v1/runtimes", "/v1/control/status"):
        assert c.get(path, headers=headers).status_code == 401, path
    bundle = c.get("/v1/trust/bundle", headers={**PUBLIC, **headers})
    assert bundle.status_code == 200


def test_the_bundle_needs_a_member_token_through_the_public_route(root: TestClient) -> None:
    c = root
    del c.headers["Authorization"]
    assert c.get("/v1/trust/bundle").status_code == 200
    assert c.get("/v1/trust/bundle", headers=PUBLIC).status_code == 401
    bogus = {"Authorization": "Bearer not-a-token"}
    assert c.get("/v1/trust/bundle", headers=bogus).status_code == 401


def test_a_job_site_is_shown_by_its_last_contact_not_probed(root: TestClient) -> None:
    c = root
    app = workbench(c)
    ada, _ = add_person(c, app, "ada")
    keys, _ = join_site(c, ada)
    # What the root's poller records for an address-less node: it is never
    # shown for a job site, whose status is its last contact.
    c.app.state.node_probes["desk"] = SimpleNamespace(
        reachable=False,
        error="no url recorded",
        epoch=None,
        agent_version=None,
        last_seen_at=None,
        trust_bundle_version=None,
    )
    before = c.get("/v1/nodes/desk").json()
    assert before["owner"] == ada["id"] and before["ownerName"] == "ada"
    assert before["lastContactAt"] is None and before["lastError"] is None
    c.post("/v1/node-helpers/poll", headers=agent(keys), json=REPORT)
    assert c.get("/v1/nodes/desk").json()["lastContactAt"] is not None


def test_nothing_runs_on_a_job_site(root: TestClient) -> None:
    c = root
    app = workbench(c)
    ada, _ = add_person(c, app, "ada")
    join_site(c, ada)
    placed = c.post("/v1/runtimes", json={"node": "desk", "spec": {"name": "qwen"}})
    assert placed.status_code == 409


# --------------------------------------------------------------------------- access


def test_only_the_site_owner_grants_and_the_grantee_reads(root: TestClient) -> None:
    c = root
    keys, app, _, ada_token, folder = site_with_folder(c)
    bo, bo_token = add_person(c, app, "bo")
    assert read(c, app, bo_token, folder).status_code == 403
    assert read(c, app, ada_token, folder).status_code == 403  # not even its owner, yet
    # Eugene's owner cannot grant it, by people or by folder.
    patched = c.patch(
        f"/v1/people/{bo['id']}",
        json={"helperGrants": [{"folderId": folder["id"], "writable": False}]},
    )
    assert patched.status_code == 403
    assert c.put("/v1/node-helpers/desk", json={"enabled": False}).status_code == 403
    assert c.delete(f"/v1/node-helpers/desk/folders/{folder['id']}").status_code == 403
    # Bo cannot grant himself anything on Ada's site, nor see that it exists.
    bo_grants = c.post(
        f"/oidc/job-sites/desk/folders/{folder['id']}/people",
        auth=auth(app),
        json={"refreshToken": bo_token, "people": [{"name": "bo", "writable": False}]},
    )
    assert bo_grants.status_code == 404
    granted = c.post(
        f"/oidc/job-sites/desk/folders/{folder['id']}/people",
        auth=auth(app),
        json={"refreshToken": ada_token, "people": [{"name": "BO", "writable": False}]},
    )
    assert granted.status_code == 200, granted.text
    answer = read_through(c, keys, app, bo_token, folder)
    assert answer.status_code == 200, answer.text
    assert answer.json()["jobSite"] is True and answer.json()["installMode"] == "production"
    mine = c.post("/oidc/job-sites", auth=auth(app), json={"refreshToken": ada_token}).json()
    assert mine["sites"][0]["folders"][0]["people"] == [
        {"person": bo["id"], "name": "bo", "writable": False}
    ]
    assert (
        c.post("/oidc/job-sites", auth=auth(app), json={"refreshToken": bo_token}).json()["sites"]
        == []
    )


def test_production_hides_a_site_from_eugenes_owner_and_dev_mode_shows_it(root: TestClient) -> None:
    c = root
    keys, app, _, _, folder = site_with_folder(c)
    owner_token = _tokens(c, app, name="operator")["refresh_token"]  # the passphrase, in Workbench
    listed = next(h for h in c.get("/v1/node-helpers").json()["helpers"] if h["node"] == "desk")
    assert listed["hidden"] is True and listed["folders"] == [] and "enabled" not in listed
    access = c.patch(f"/v1/node-helpers/desk/folders/{folder['id']}", json={"ownerAccess": "read"})
    assert access.status_code == 403
    assert read(c, app, owner_token, folder).status_code == 403

    switched = c.patch("/v1/config", json={"installMode": "dev"})
    assert switched.status_code == 200, switched.text
    mode = c.post("/oidc/install-mode", auth=auth(app)).json()
    assert mode["mode"] == "dev" and mode["changedAt"]
    listed = next(h for h in c.get("/v1/node-helpers").json()["helpers"] if h["node"] == "desk")
    assert listed["hidden"] is False and listed["folders"][0]["id"] == folder["id"]
    assert (
        c.patch(
            f"/v1/node-helpers/desk/folders/{folder['id']}", json={"ownerAccess": "read"}
        ).status_code
        == 200
    )
    answer = read_through(c, keys, app, owner_token, folder)
    assert answer.status_code == 200 and answer.json()["installMode"] == "dev"
    # Eugene's owner still cannot give another person the site's folder.
    assert c.put("/v1/node-helpers/desk", json={"enabled": False}).status_code == 403

    # Back to production: the owner's own grant stops at once.
    assert c.patch("/v1/config", json={"installMode": "production"}).status_code == 200
    assert read(c, app, owner_token, folder).status_code == 403


def test_the_mode_is_dated_only_when_it_changes(root: TestClient) -> None:
    c = root
    app = workbench(c)
    assert c.post("/oidc/install-mode", auth=auth(app)).json() == {
        "mode": "production",
        "changedAt": None,
    }
    assert c.post("/oidc/install-mode").status_code == 401
    c.patch("/v1/config", json={"installMode": "production"})
    assert c.post("/oidc/install-mode", auth=auth(app)).json()["changedAt"] is None
    assert c.get("/v1/config").json()["installMode"] == "production"
    assert "installModeChangedAt" not in c.get("/v1/config").json()


# --------------------------------------------------------------------------- self-service


def test_a_person_invites_their_own_machine_and_the_owner_cannot(root: TestClient) -> None:
    c = root
    app = workbench(c)
    ada, ada_token = add_person(c, app, "ada")
    owner_token = _tokens(c, app, name="operator")["refresh_token"]
    assert (
        c.post(
            "/oidc/job-sites/invite", auth=auth(app), json={"refreshToken": owner_token}
        ).status_code
        == 403
    )
    invited = c.post(
        "/oidc/job-sites/invite",
        auth=auth(app),
        json={"refreshToken": ada_token, "nodeName": "laptop"},
    )
    assert invited.status_code == 200, invited.text
    body = invited.json()
    assert body["nodesUrl"] == NODES_ORIGIN and body["owner"] == "ada"
    assert body["rootKey"] == c.app.state.machine.state.identity.controlPublicKey
    keys = node_keys("laptop")
    joined = c.post(
        "/v1/nodes/enroll",
        json=keys.body(body["token"], owner={"name": "ada", "password": PASSPHRASE}),
        headers=PUBLIC,
    )
    assert joined.status_code == 201, joined.text
    assert c.app.state.machine.state.nodes["laptop"].owner == ada["id"]
    for _ in range(3):
        c.post("/oidc/job-sites/invite", auth=auth(app), json={"refreshToken": ada_token})
    too_many = c.post("/oidc/job-sites/invite", auth=auth(app), json={"refreshToken": ada_token})
    assert too_many.status_code == 429
    left = c.post("/oidc/job-sites/laptop/leave", auth=auth(app), json={"refreshToken": ada_token})
    assert left.status_code == 204 and "laptop" not in c.app.state.machine.state.nodes


def test_no_invitation_without_the_public_route(tmp_path: Path) -> None:
    settings = Settings(config_file=tmp_path / "c.yaml", state_dir=tmp_path / "s")
    with TestClient(create_app(settings)) as c:
        c.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE})
        c.headers["Authorization"] = f"Bearer {login(c)}"
        app = workbench(c)
        _, ada_token = add_person(c, app, "ada")
        listed = c.post("/oidc/job-sites", auth=auth(app), json={"refreshToken": ada_token})
        assert listed.json()["canInvite"] is False
        invited = c.post("/oidc/job-sites/invite", auth=auth(app), json={"refreshToken": ada_token})
        assert invited.status_code == 409


# --------------------------------------------------------------------------- J7a


def test_the_tls_list_is_signed_by_the_key_a_site_pinned(
    root: TestClient, monkeypatch: Any
) -> None:
    c = root
    watcher = c.app.state.root_tls

    def shown(origin: str, dial: str | None) -> tuple[str, int]:
        return "spki-pin", 4_102_444_800

    monkeypatch.setattr(root_tls, "presented", shown)
    answer = c.get("/v1/trust/tls")
    assert answer.status_code == 200, answer.text
    pinned = c.app.state.machine.state.identity.controlPublicKey
    key = tokens.load_public(pinned)
    claims = jwt.decode(answer.json()["jws"], key, algorithms=["EdDSA"])
    # The contract's literal, not the module's constant: a site checks this one.
    assert jwt.get_unverified_header(answer.json()["jws"])["typ"] == "ep-root-tls+jwt"
    assert claims["origin"] == NODES_ORIGIN
    assert claims["keys"] == [{"spki": "spki-pin", "notAfter": 4_102_444_800}]
    assert watcher.keys == {"spki-pin": 4_102_444_800}


def test_no_tls_list_when_the_root_cannot_see_its_certificate(
    root: TestClient, monkeypatch: Any
) -> None:
    def refused(origin: str, dial: str | None) -> tuple[str, int]:
        raise OSError("connection refused")

    monkeypatch.setattr(root_tls, "presented", refused)
    answer = root.get("/v1/trust/tls")
    assert answer.status_code == 503 and "connection refused" in answer.text


def test_spki_pin_is_the_sha256_of_the_public_key_info() -> None:
    import datetime as dt
    import hashlib

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "nodes.example.test")])
    now = dt.datetime.now(dt.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(now)
        .not_valid_after(now + dt.timedelta(days=1))
        .sign(key, hashes.SHA256())
    )
    der = cert.public_bytes(serialization.Encoding.DER)
    spki = key.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    expected = base64.urlsafe_b64encode(hashlib.sha256(spki).digest()).rstrip(b"=").decode()
    assert root_tls.spki_pin(der) == (expected, int(cert.not_valid_after_utc.timestamp()))
