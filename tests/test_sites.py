"""Job Sites as their own enrollment (slice 2b.1,
`specs/docs/design/job-sites-own-enrollment.md`).

A site host joins with a site invitation that names a person, who confirms
at the machine with their own password. From then on it polls, claims and
answers with its own token, which this root checks against the site
registry and never the trust bundle. A site is not a node: it has no
address, no node key and no place in any node listing. Only its owner
decides who may use it, from Workbench. Eugene's owner manages membership;
in production that is all, in dev mode they also see what it reported and
may give themselves its folders (J13b, J33).
"""

from __future__ import annotations

import asyncio
import base64
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jwt
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from eugene_plexus_control import applied, root_tls, sites, tokens
from eugene_plexus_control.app import create_app
from eugene_plexus_control.settings import Settings

from .conftest import PASSPHRASE, login, mint_join_token, node_keys
from .test_oidc import CALLBACK, _tokens

REPORT = {"ready": True, "account": "site-host", "protocol": "mcp-2026-07-28"}
PUBLIC = {"X-Eugene-Plexus-Entry": "public-sites"}
NODES_ORIGIN = "https://nodes.example.test:8443"


@pytest.fixture(autouse=True)
def quick_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sites, "POLL_SECONDS", 0.03)


def _root(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as client:
        assert (
            client.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
        )
        client.headers["Authorization"] = f"Bearer {login(client)}"
        yield client


@pytest.fixture
def root(tmp_path: Path) -> Iterator[TestClient]:
    """An active root whose entry point has a name for machines."""
    yield from _root(
        Settings(
            config_file=tmp_path / "control.yaml",
            state_dir=tmp_path / "state",
            nodes_origin=NODES_ORIGIN,
        )
    )


@pytest.fixture
def lan_root(tmp_path: Path) -> Iterator[TestClient]:
    """An active root with no entry point at all: a LAN-only install."""
    yield from _root(Settings(config_file=tmp_path / "c.yaml", state_dir=tmp_path / "s"))


@dataclass
class SiteKeys:
    """What a site host generates before it joins, private half included."""

    key: Any

    @property
    def public(self) -> str:
        return tokens.public_b64(self.key)

    def token(self, site: str, *, ttl: int = 300, aud: str = "control", sub: str = "site") -> str:
        now = int(time.time())
        return jwt.encode(
            {"iss": f"site:{site}", "sub": sub, "aud": aud, "iat": now, "exp": now + ttl},
            self.key,
            algorithm="EdDSA",
        )

    def bearer(self, site: str) -> dict[str, str]:
        return {"Authorization": "Bearer " + self.token(site)}


def site_keys() -> SiteKeys:
    return SiteKeys(tokens.generate_private_key())


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


def invite(c: TestClient, owner: dict[str, Any], label: str = "desk") -> dict[str, Any]:
    made = c.post("/v1/sites/invitations", json={"owner": owner["id"], "label": label})
    assert made.status_code == 201, made.text
    return made.json()


def join_site(
    c: TestClient,
    owner: dict[str, Any],
    label: str = "desk",
    *,
    password: str = PASSPHRASE,
    headers: dict[str, str] | None = None,
) -> tuple[SiteKeys, Any]:
    keys = site_keys()
    token = invite(c, owner, label)["token"]
    body = {
        "token": token,
        "label": label,
        "tokenPublicKey": keys.public,
        "owner": {"name": owner["name"], "password": password},
    }
    return keys, c.post("/v1/sites/enroll", json=body, headers=headers or {})


def deliver(
    c: TestClient, keys: SiteKeys, site: str, report: dict[str, Any] | None = None
) -> tuple[str, dict[str, Any]]:
    for _ in range(100):
        polled = c.post("/v1/sites/poll", headers=keys.bearer(site), json=report or REPORT)
        assert polled.status_code == 200, polled.text
        ident = polled.json()["operation"]
        if ident:
            claim = c.post(f"/v1/sites/operations/{ident}/claim", headers=keys.bearer(site))
            assert claim.status_code == 200, claim.text
            return ident, claim.json()
    pytest.fail("no operation was delivered")


def answer(c: TestClient, keys: SiteKeys, site: str, ident: str, value: dict[str, Any]) -> None:
    done = c.post(f"/v1/sites/operations/{ident}/result", headers=keys.bearer(site), json=value)
    assert done.status_code == 204, done.text


class Report:
    """What a site holds and reports (`SiteSummary`): its own list, which
    the root keeps only as a cache (rule 2 of remote-nodes.md §3.3)."""

    def __init__(self, owner: str) -> None:
        self.summary: dict[str, Any] = {
            "owner": owner,
            "ownerInDevMode": False,
            "folders": [],
            "servers": [],
            "access": [],
        }

    def report(self) -> dict[str, Any]:
        return {**REPORT, "site": self.summary}

    def folder(self, **values: Any) -> dict[str, Any]:
        folder = {"people": [], **values}
        self.summary["folders"].append(folder)
        return folder


def manage_through(
    c: TestClient, keys: SiteKeys, site: str, held: Report, post: Any, result: Any, *, action: str
) -> tuple[dict[str, Any], Any]:
    """One relayed management action: the site claims it and answers
    `result(command)`."""
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(post)
        ident, command = deliver(c, keys, site, held.report())
        assert command["kind"] == "manage" and command["action"] == action, command
        answer(c, keys, site, ident, result(command))
        response = pending.result(timeout=5)
    c.post("/v1/sites/poll", headers=keys.bearer(site), json=held.report())
    return command, response


@dataclass
class Ada:
    keys: SiteKeys
    site: str
    app: dict[str, Any]
    person: dict[str, Any]
    token: str
    folder: dict[str, Any]
    held: Report


def ada_with_a_folder(c: TestClient) -> Ada:
    """Ada's site `desk`, with one writable folder nobody may use yet."""
    app = workbench(c)
    ada, ada_token = add_person(c, app, "ada")
    keys, joined = join_site(c, ada)
    assert joined.status_code == 201, joined.text
    site = joined.json()["id"]
    assert joined.json()["owner"] == ada["id"]
    held = Report(ada["id"])
    assert c.post("/v1/sites/poll", headers=keys.bearer(site), json=held.report()).json() == {
        "operation": None
    }

    def added(command: dict[str, Any]) -> dict[str, Any]:
        assert command["subject"] == ada["id"]
        assert command["site"] == site
        folder = held.folder(
            id="a" * 32, name="Notes", path="C:/Notes", identity="vol:file", writable=True
        )
        return {"status": "done", "result": folder}

    _, folder = manage_through(
        c,
        keys,
        site,
        held,
        lambda: c.post(
            f"/oidc/job-sites/{site}/folders",
            auth=auth(app),
            json={"refreshToken": ada_token, "name": "Notes", "path": "C:/Notes", "writable": True},
        ),
        added,
        action="folder.add",
    )
    assert folder.status_code == 201, folder.text
    return Ada(keys, site, app, ada, ada_token, folder.json(), held)


def rpc(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": 3,
        "method": method,
        "params": {
            **(params or {}),
            "_meta": {"io.modelcontextprotocol/protocolVersion": "2026-07-28"},
        },
    }


def read(c: TestClient, a: Ada, token: str) -> Any:
    return c.post(
        "/oidc/sites/mcp",
        auth=auth(a.app),
        json={
            "refreshToken": token,
            "site": a.site,
            "server": "files",
            "request": rpc(
                "tools/call",
                {"name": "read_text", "arguments": {"folder": "Notes", "path": "a.txt"}},
            ),
        },
    )


def read_through(c: TestClient, a: Ada, token: str) -> tuple[dict[str, Any], Any]:
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(read, c, a, token)
        ident, command = deliver(c, a.keys, a.site, a.held.report())
        answer(
            c,
            a.keys,
            a.site,
            ident,
            {"status": "done", "response": {"jsonrpc": "2.0", "id": 3, "result": {"t": "SITE"}}},
        )
        return command, pending.result(timeout=5)


# --------------------------------------------------------------------------- joining


def test_a_site_invitation_names_a_person_and_is_listed_with_the_join_tokens(
    root: TestClient,
) -> None:
    app = workbench(root)
    ada, _ = add_person(root, app, "ada")
    made = invite(root, ada)
    assert made["ownerName"] == "ada" and made["joinUrl"] == NODES_ORIGIN
    assert made["rootKey"] == root.app.state.machine.state.identity.controlPublicKey
    listed = root.get("/v1/nodes/join-tokens").json()["tokens"]
    assert [(t["kind"], t["owner"], t["nodeName"]) for t in listed] == [("site", ada["id"], "desk")]
    nobody = root.post("/v1/sites/invitations", json={"owner": "nobody"})
    assert nobody.status_code == 422


def test_a_site_invitation_does_not_join_a_node_nor_a_node_token_a_site(root: TestClient) -> None:
    app = workbench(root)
    ada, _ = add_person(root, app, "ada")
    token = invite(root, ada)["token"]
    keys = node_keys("desk")
    as_node = root.post("/v1/nodes/enroll", json=keys.body(token))
    assert as_node.status_code == 401 and "job site" in as_node.text
    node_token = mint_join_token(root)
    as_site = root.post(
        "/v1/sites/enroll",
        json={
            "token": node_token,
            "label": "desk",
            "tokenPublicKey": site_keys().public,
            "owner": {"name": "ada", "password": PASSPHRASE},
        },
    )
    assert as_site.status_code == 401 and "node" in as_site.text
    # Neither refusal spent its token.
    assert root.post("/v1/nodes/enroll", json=keys.body(node_token)).status_code == 201
    joined = root.post(
        "/v1/sites/enroll",
        json={
            "token": token,
            "label": "desk",
            "tokenPublicKey": site_keys().public,
            "owner": {"name": "ada", "password": PASSPHRASE},
        },
    )
    assert joined.status_code == 201, joined.text


def test_the_person_confirms_at_the_machine_and_a_wrong_try_keeps_the_invitation(
    root: TestClient,
) -> None:
    app = workbench(root)
    ada, _ = add_person(root, app, "ada")
    keys = site_keys()
    token = invite(root, ada)["token"]
    body = {
        "token": token,
        "label": "desk",
        "tokenPublicKey": keys.public,
        "owner": {"name": "ada", "password": "not the password at all"},
    }
    assert root.post("/v1/sites/enroll", json=body).status_code == 401
    body["owner"] = {"name": "ada", "password": PASSPHRASE}
    joined = root.post("/v1/sites/enroll", json=body)
    assert joined.status_code == 201, joined.text
    value = joined.json()
    assert value["id"].startswith("s-") and len(value["id"]) == 28
    assert value["owner"] == ada["id"] and value["ownerName"] == "ada"
    replay = root.post("/v1/sites/enroll", json={**body, "tokenPublicKey": site_keys().public})
    assert replay.status_code == 409
    # The same key twice is refused before the token is looked at.
    assert root.post("/v1/sites/enroll", json=body).status_code == 400


def test_from_outside_only_a_site_joins(root: TestClient) -> None:
    keys = node_keys("far")
    node = root.post("/v1/nodes/enroll", json=keys.body(mint_join_token(root)), headers=PUBLIC)
    assert node.status_code == 403
    old = root.post(
        "/v1/nodes/enroll",
        json=keys.body(mint_join_token(root)),
        headers={"X-Eugene-Plexus-Entry": "public-nodes"},
    )
    assert old.status_code == 403
    app = workbench(root)
    ada, _ = add_person(root, app, "ada")
    _, joined = join_site(root, ada, headers=PUBLIC)
    assert joined.status_code == 201, joined.text


def test_a_site_is_not_a_node_and_its_key_is_in_no_bundle(root: TestClient) -> None:
    app = workbench(root)
    ada, _ = add_person(root, app, "ada")
    keys, joined = join_site(root, ada)
    site = joined.json()["id"]
    assert all(n["name"] != "desk" for n in root.get("/v1/nodes").json()["nodes"])
    listed = root.app.state.trust.view_for(root.app.state.machine.state).keys.values()
    assert all(tokens.public_b64(entry.public) != keys.public for entry in listed)
    # Its token opens its own four routes and nothing else here.
    assert root.get("/v1/nodes", headers=keys.bearer(site)).status_code == 401
    assert root.get("/v1/trust/bundle", headers=keys.bearer(site)).status_code == 401
    assert root.get("/v1/sites", headers=keys.bearer(site)).status_code == 401
    assert root.post("/v1/sites/poll", headers=keys.bearer(site), json=REPORT).status_code == 200


def test_a_site_token_is_short_lived_addressed_here_and_signed_by_its_own_key(
    root: TestClient,
) -> None:
    app = workbench(root)
    ada, _ = add_person(root, app, "ada")
    keys, joined = join_site(root, ada)
    site = joined.json()["id"]

    def poll(token: str) -> int:
        return root.post(
            "/v1/sites/poll", headers={"Authorization": f"Bearer {token}"}, json=REPORT
        ).status_code

    assert poll(keys.token(site)) == 200
    assert poll(keys.token(site, ttl=3600)) == 401
    assert poll(keys.token(site, aud="node:desk")) == 401
    assert poll(keys.token(site, sub="agent")) == 401
    assert poll(site_keys().token(site)) == 401
    assert poll(keys.token("s-" + "a" * 26)) == 401


def test_removing_a_site_or_its_leaving_ends_its_token_at_once(root: TestClient) -> None:
    app = workbench(root)
    ada, ada_token = add_person(root, app, "ada")
    keys, joined = join_site(root, ada)
    site = joined.json()["id"]
    assert root.delete(f"/v1/sites/{site}").status_code == 204
    assert root.post("/v1/sites/poll", headers=keys.bearer(site), json=REPORT).status_code == 401
    assert root.delete(f"/v1/sites/{site}").status_code == 204  # already gone
    keys, joined = join_site(root, ada, "desk2")
    second = joined.json()["id"]
    assert second != site
    assert root.post("/v1/sites/leave", headers=keys.bearer(second)).status_code == 204
    assert root.get("/v1/sites").json()["sites"] == []
    # And from Workbench, by its owner.
    keys, joined = join_site(root, ada, "desk3")
    third = joined.json()["id"]
    left = root.post(
        f"/oidc/job-sites/{third}/leave", auth=auth(app), json={"refreshToken": ada_token}
    )
    assert left.status_code == 204
    assert root.post("/v1/sites/poll", headers=keys.bearer(third), json=REPORT).status_code == 401


def test_deleting_the_owner_removes_their_sites(root: TestClient) -> None:
    app = workbench(root)
    ada, _ = add_person(root, app, "ada")
    keys, joined = join_site(root, ada)
    site = joined.json()["id"]
    assert root.delete(f"/v1/people/{ada['id']}").status_code == 204
    assert root.get("/v1/sites").json()["sites"] == []
    assert root.post("/v1/sites/poll", headers=keys.bearer(site), json=REPORT).status_code == 401


# --------------------------------------------------------------------------- using a site


def test_only_the_site_owner_grants_and_the_grantee_reads(root: TestClient) -> None:
    c = root
    a = ada_with_a_folder(c)
    bo, bo_token = add_person(c, a.app, "bo")
    assert read(c, a, bo_token).status_code == 403
    assert read(c, a, a.token).status_code == 403  # not even its owner, yet
    # Bo cannot grant himself anything on Ada's site, nor learn that it exists.
    bo_grants = c.post(
        f"/oidc/job-sites/{a.site}/folders/{a.folder['id']}/people",
        auth=auth(a.app),
        json={"refreshToken": bo_token, "people": [{"name": "bo", "writable": False}]},
    )
    assert bo_grants.status_code == 404
    assert not c.app.state.site_broker.jobs

    def granted(command: dict[str, Any]) -> dict[str, Any]:
        assert command["subject"] == a.person["id"]
        assert command["arguments"] == {
            "id": a.folder["id"],
            "people": [{"subject": bo["id"], "writable": False}],
        }
        a.held.summary["folders"][0]["people"] = command["arguments"]["people"]
        return {"status": "done", "result": a.held.summary["folders"][0]}

    _, given = manage_through(
        c,
        a.keys,
        a.site,
        a.held,
        lambda: c.post(
            f"/oidc/job-sites/{a.site}/folders/{a.folder['id']}/people",
            auth=auth(a.app),
            json={"refreshToken": a.token, "people": [{"name": "BO", "writable": False}]},
        ),
        granted,
        action="folder.people",
    )
    assert given.status_code == 200, given.text
    assert given.json()["people"] == [{"person": bo["id"], "name": "bo", "writable": False}]
    listed = c.post("/oidc/sites/servers", auth=auth(a.app), json={"refreshToken": bo_token})
    assert listed.json()["servers"] == [
        {
            "site": a.site,
            "label": "desk",
            "server": "files",
            "name": "Files",
            "kind": "files",
            "available": True,
            "reason": None,
            "folders": [{"id": a.folder["id"], "name": "Notes", "writable": False}],
        }
    ]
    command, answered = read_through(c, a, bo_token)
    assert command["kind"] == "mcp" and command["grants"] == []
    assert command["subject"] == bo["id"]
    assert command["site"] == a.site and command["enrolledAt"]
    assert answered.status_code == 200, answered.text
    assert answered.json()["installMode"] == "production"
    mine = c.post("/oidc/job-sites", auth=auth(a.app), json={"refreshToken": a.token}).json()
    assert mine["sites"][0]["folders"][0]["people"] == [
        {"person": bo["id"], "name": "bo", "writable": False}
    ]
    assert (
        c.post("/oidc/job-sites", auth=auth(a.app), json={"refreshToken": bo_token}).json()["sites"]
        == []
    )


def test_the_sites_refusal_comes_back_in_its_own_words(root: TestClient) -> None:
    c = root
    a = ada_with_a_folder(c)
    _, refused = manage_through(
        c,
        a.keys,
        a.site,
        a.held,
        lambda: c.post(
            f"/oidc/job-sites/{a.site}/folders/{a.folder['id']}/people",
            auth=auth(a.app),
            json={"refreshToken": a.token, "people": [{"name": "ada", "writable": True}]},
        ),
        lambda _: {"status": "failed", "message": "Only this machine's owner changes it."},
        action="folder.people",
    )
    assert refused.status_code == 422 and "Only this machine's owner" in refused.text


def test_nothing_queued_for_one_enrollment_reaches_another(root: TestClient) -> None:
    """A site removed while a call waits: the call ends, and the site that
    joins again (a new id) cannot claim it."""
    c = root
    a = ada_with_a_folder(c)
    a.held.summary["folders"][0]["people"] = [{"subject": a.person["id"], "writable": True}]
    c.post("/v1/sites/poll", headers=a.keys.bearer(a.site), json=a.held.report())
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(read, c, a, a.token)
        for _ in range(100):
            if c.app.state.site_broker.jobs:
                break
            time.sleep(0.01)
        (ident,) = list(c.app.state.site_broker.jobs)
        assert c.delete(f"/v1/sites/{a.site}").status_code == 204
        ended = pending.result(timeout=5)
    assert ended.status_code == 503
    keys, joined = join_site(c, a.person, "desk")
    again = joined.json()["id"]
    assert again != a.site
    claim = c.post(f"/v1/sites/operations/{ident}/claim", headers=keys.bearer(again))
    assert claim.status_code == 404


def test_a_result_larger_than_the_limit_is_refused(root: TestClient) -> None:
    c = root
    a = ada_with_a_folder(c)
    a.held.summary["folders"][0]["people"] = [{"subject": a.person["id"], "writable": True}]
    c.post("/v1/sites/poll", headers=a.keys.bearer(a.site), json=a.held.report())
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(read, c, a, a.token)
        ident, _ = deliver(c, a.keys, a.site, a.held.report())
        big = c.post(
            f"/v1/sites/operations/{ident}/result",
            headers=a.keys.bearer(a.site),
            json={"status": "done", "message": None, "result": {"x": "y" * 80_000}},
        )
        assert big.status_code == 413
        answer(c, a.keys, a.site, ident, {"status": "failed", "message": "too big"})
        assert pending.result(timeout=5).json()["status"] == "failed"


# --------------------------------------------------------------------------- the console


def test_production_shows_membership_only_and_dev_mode_shows_the_rest(root: TestClient) -> None:
    c = root
    a = ada_with_a_folder(c)
    (listed,) = c.get("/v1/sites").json()["sites"]
    assert listed["id"] == a.site and listed["ownerName"] == "ada" and listed["online"] is True
    assert listed["dev"] is None
    production = c.patch(
        f"/v1/sites/{a.site}/folders/{a.folder['id']}", json={"ownerAccess": "read"}
    )
    assert production.status_code == 409
    assert c.patch("/v1/config", json={"installMode": "dev"}).status_code == 200
    (listed,) = c.get("/v1/sites").json()["sites"]
    assert listed["dev"]["ownerInDevMode"] is False
    assert listed["dev"]["folders"] == [
        {
            "id": "a" * 32,
            "name": "Notes",
            "path": "C:/Notes",
            "writable": True,
            "ownerAccess": "none",
        }
    ]
    given = c.patch(f"/v1/sites/{a.site}/folders/{a.folder['id']}", json={"ownerAccess": "read"})
    assert given.status_code == 200, given.text
    assert given.json()["dev"]["folders"][0]["ownerAccess"] == "read"
    # The owner's grant reaches Workbench only once the site's owner lets them in (J6e).
    operator = _tokens(c, a.app, name="operator")["refresh_token"]
    servers = c.post("/oidc/sites/servers", auth=auth(a.app), json={"refreshToken": operator})
    assert servers.json()["servers"] == []
    a.held.summary["ownerInDevMode"] = True
    c.post("/v1/sites/poll", headers=a.keys.bearer(a.site), json=a.held.report())
    servers = c.post("/oidc/sites/servers", auth=auth(a.app), json={"refreshToken": operator})
    assert servers.json()["servers"][0]["folders"] == [
        {"id": "a" * 32, "name": "Notes", "writable": False}
    ]
    # Back to production: the grant stops at once, and the view goes.
    assert c.patch("/v1/config", json={"installMode": "production"}).status_code == 200
    servers = c.post("/oidc/sites/servers", auth=auth(a.app), json={"refreshToken": operator})
    assert servers.json()["servers"] == []
    assert c.get("/v1/sites").json()["sites"][0]["dev"] is None


def test_a_node_says_which_sites_it_hosts_for_display(root: TestClient) -> None:
    from .conftest import enroll

    app = workbench(root)
    ada, _ = add_person(root, app, "ada")
    _, joined = join_site(root, ada)
    site = joined.json()["id"]
    amish, made = enroll(root, "amish")
    assert made.status_code == 201
    other, made = enroll(root, "other")
    assert made.status_code == 201

    def say(keys: Any, name: str, ids: list[str]) -> int:
        return root.put(
            f"/v1/nodes/{name}/hosted-sites",
            headers={"Authorization": "Bearer " + keys.service_token()},
            json={"sites": ids},
        ).status_code

    assert say(other, "amish", [site]) == 403
    assert say(amish, "amish", [site, "s-" + "b" * 26]) == 204
    assert root.get("/v1/sites").json()["sites"][0]["hostNode"] == "amish"
    assert say(amish, "amish", []) == 204
    assert root.get("/v1/sites").json()["sites"][0]["hostNode"] is None


# --------------------------------------------------------------------------- Workbench invites


def test_a_person_invites_their_own_machine_and_the_owner_cannot(root: TestClient) -> None:
    app = workbench(root)
    _, ada_token = add_person(root, app, "ada")
    made = root.post("/oidc/job-sites/invite", auth=auth(app), json={"refreshToken": ada_token})
    assert made.status_code == 200, made.text
    assert made.json()["joinUrl"] == NODES_ORIGIN and made.json()["owner"] == "ada"
    operator = _tokens(root, app, name="operator")["refresh_token"]
    refused = root.post("/oidc/job-sites/invite", auth=auth(app), json={"refreshToken": operator})
    assert refused.status_code == 403


def test_a_lan_only_install_invites_through_the_address_its_owner_set(
    lan_root: TestClient,
) -> None:
    c = lan_root
    app = workbench(c)
    _, ada_token = add_person(c, app, "ada")
    listed = c.post("/oidc/job-sites", auth=auth(app), json={"refreshToken": ada_token})
    assert listed.json()["canInvite"] is False
    assert (
        c.post("/oidc/job-sites/invite", auth=auth(app), json={"refreshToken": ada_token})
    ).status_code == 409
    assert c.patch("/v1/config", json={"siteJoinUrl": "http://192.168.1.5:8083"}).status_code == 200
    listed = c.post("/oidc/job-sites", auth=auth(app), json={"refreshToken": ada_token})
    assert listed.json()["canInvite"] is True
    made = c.post("/oidc/job-sites/invite", auth=auth(app), json={"refreshToken": ada_token})
    assert made.status_code == 200 and made.json()["joinUrl"] == "http://192.168.1.5:8083"


# --------------------------------------------------------------------------- old logs


def test_a_slice_2_job_site_and_its_node_folders_still_replay_and_grant_nothing(
    root: TestClient,
) -> None:
    """J20, J34: old entries apply, and what they describe is retired."""
    machine = root.app.state.machine
    keys = node_keys("oldsite")
    machine.append(
        applied.OP_ENROLL_NODE,
        {
            "name": "oldsite",
            "role": "agent",
            "publicKey": keys.public,
            "signingPublicKey": keys.signing_public,
            "tokenPublicKey": tokens.public_b64(keys.token.key),
            "grants": ["files"],
            "owner": None,
            "url": None,
            "enrolledAt": "2026-10-05T12:00:00+00:00",
        },
    )
    machine.append(
        applied.OP_PUT_NODE_HELPER,
        {"helper": {"node": "oldsite", "enabled": True, "folders": []}},
    )
    assert all(n["name"] != "oldsite" for n in root.get("/v1/nodes").json()["nodes"])
    assert root.get("/v1/nodes/oldsite").status_code == 404
    listed = root.app.state.trust.view_for(machine.state).keys.values()
    assert all(entry.issuer != "node:oldsite" for entry in listed)
    # Its own token opens nothing, and it may not give itself an address (control#6).
    assert (
        root.get(
            "/v1/nodes", headers={"Authorization": "Bearer " + keys.service_token()}
        ).status_code
        == 401
    )
    announced = root.patch(
        "/v1/nodes/oldsite",
        json={"url": "http://203.0.113.9:8079", "sequence": 1, "signature": "x"},
    )
    assert announced.status_code == 403
    placed = root.post(
        "/v1/runtimes", json={"node": "oldsite", "spec": {"name": "x", "modelPath": "/m.gguf"}}
    )
    assert placed.status_code == 409 and "retired" in placed.text
    state = machine.state
    assert applied.from_canonical(applied.to_canonical(state)) == state
    assert applied.to_canonical(state)["nodeHelpers"] == []
    machine.append(applied.OP_REVOKE_NODE, {"name": "oldsite"})
    assert "oldsite" not in machine.state.nodes


def test_a_site_survives_the_snapshot(root: TestClient) -> None:
    app = workbench(root)
    ada, _ = add_person(root, app, "ada")
    join_site(root, ada)
    state = root.app.state.machine.state
    assert len(state.sites) == 1
    assert applied.from_canonical(applied.to_canonical(state)) == state


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


def test_an_offer_nobody_claimed_is_offered_again(
    root: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A site host restarted mid-poll: the old poll was offered the call and
    its answer went nowhere. The restarted host must still get it."""
    monkeypatch.setattr(sites, "OFFER_SECONDS", 0.05)
    c = root
    a = ada_with_a_folder(c)
    a.held.summary["folders"][0]["people"] = [{"subject": a.person["id"], "writable": True}]
    c.post("/v1/sites/poll", headers=a.keys.bearer(a.site), json=a.held.report())
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(read, c, a, a.token)
        lost = None
        for _ in range(100):
            polled = c.post("/v1/sites/poll", headers=a.keys.bearer(a.site), json=a.held.report())
            lost = polled.json()["operation"]
            if lost:
                break
        assert lost, "the call was never offered"
        time.sleep(0.1)  # the poll that took it is gone; nobody claims it
        ident, _ = deliver(c, a.keys, a.site, a.held.report())
        assert ident == lost
        answer(
            c,
            a.keys,
            a.site,
            ident,
            {"status": "done", "response": {"jsonrpc": "2.0", "id": 3, "result": {}}},
        )
        assert pending.result(timeout=5).json()["status"] == "done"


# --------------------------------------------------------------------------- the broker, directly


def test_the_broker_keeps_reports_to_the_enrollment_that_made_them() -> None:
    broker = sites.Broker()
    old = {"site": "s-a", "enrolledAt": "2026-10-06T10:00:00+00:00"}
    new = {"site": "s-a", "enrolledAt": "2026-10-06T11:00:00+00:00"}
    asyncio.run(broker.poll(old, {"ready": True, "protocol": sites.PROTOCOL}))
    assert broker.report(old) is not None and broker.last_contact(old) is not None
    assert broker.report(new) is None and broker.last_contact(new) is None
    assert broker.status(new)["online"] is False
    assert broker.status(old)["online"] is True


def test_a_site_claims_and_answers_only_its_own_operations() -> None:
    async def go() -> None:
        broker = sites.Broker()
        job = sites.Job(
            "op-1",
            "s-a",
            lambda: {"kind": "mcp"},
            asyncio.get_running_loop().create_future(),
            time.perf_counter() + 20,
            state="offered",
        )
        broker.jobs[job.id] = job
        with pytest.raises(HTTPException) as other:
            broker.claim("s-b", "op-1")
        assert other.value.status_code == 404 and job.state == "offered"
        assert broker.claim("s-a", "op-1")["id"] == "op-1"
        with pytest.raises(HTTPException):
            broker.finish("s-b", "op-1", {"status": "done"})
        assert not job.result.done()
        broker.finish("s-a", "op-1", {"status": "done"})
        assert job.result.done()

    asyncio.run(go())


def test_a_call_is_checked_against_the_enrollment_it_was_queued_for(root: TestClient) -> None:
    from eugene_plexus_control.routes import sites as site_routes

    a = ada_with_a_folder(root)
    machine = root.app.state.machine
    bound = sites.binding(machine.state.sites[a.site])
    assert site_routes._same_site(machine, bound).id == a.site
    with pytest.raises(HTTPException) as changed:
        site_routes._same_site(machine, {**bound, "enrolledAt": "2000-01-01T00:00:00+00:00"})
    assert changed.value.status_code == 409


def test_a_site_needs_an_owner_who_is_on_the_install(root: TestClient) -> None:
    with pytest.raises(applied.ApplyError, match="owner"):
        root.app.state.machine.append(
            applied.OP_ENROLL_SITE,
            {
                "id": sites.new_site_id(),
                "label": "desk",
                "owner": "nobody",
                "tokenPublicKey": site_keys().public,
                "enrolledAt": "2026-10-06T10:00:00+00:00",
            },
        )


# --------------------------------------------------------------------------- joining, more


def test_another_persons_name_does_not_confirm_the_join(root: TestClient) -> None:
    app = workbench(root)
    ada, _ = add_person(root, app, "ada")
    add_person(root, app, "grace")
    keys, token = site_keys(), invite(root, ada)["token"]
    body = {
        "token": token,
        "label": "desk",
        "tokenPublicKey": keys.public,
        "owner": {"name": "grace", "password": PASSPHRASE},
    }
    assert root.post("/v1/sites/enroll", json=body).status_code == 401
    body["owner"] = {"name": "ada", "password": PASSPHRASE}
    assert root.post("/v1/sites/enroll", json=body).status_code == 201, "the invitation was kept"


def test_a_locked_root_does_not_spend_the_invitation(root: TestClient) -> None:
    app = workbench(root)
    ada, _ = add_person(root, app, "ada")
    keys, token = site_keys(), invite(root, ada)["token"]
    body = {
        "token": token,
        "label": "desk",
        "tokenPublicKey": keys.public,
        "owner": {"name": "ada", "password": PASSPHRASE},
    }
    held = root.app.state.auth_state.signing_key
    root.app.state.auth_state.signing_key = None
    try:
        assert root.post("/v1/sites/enroll", json=body).status_code == 503
    finally:
        root.app.state.auth_state.signing_key = held
    assert root.post("/v1/sites/enroll", json=body).status_code == 201


def test_a_person_may_hold_three_open_invitations_and_no_more(root: TestClient) -> None:
    app = workbench(root)
    _, ada_token = add_person(root, app, "ada")
    codes = [
        root.post("/oidc/job-sites/invite", auth=auth(app), json={"refreshToken": ada_token})
        for _ in range(4)
    ]
    assert [c.status_code for c in codes] == [200, 200, 200, 429]


def test_eugenes_owner_gets_production_mode_not_a_silent_nothing(root: TestClient) -> None:
    """In production the refusal says so; it is not the same answer as having
    no grant, because the owner can do something about it (switch to dev)."""
    a = ada_with_a_folder(root)
    operator = _tokens(root, a.app, name="operator")["refresh_token"]
    refused = read(root, a, operator)
    assert refused.status_code == 403
    assert "production mode" in refused.text.lower()


def test_a_change_of_mode_is_dated(root: TestClient) -> None:
    app = workbench(root)
    assert root.post("/oidc/install-mode", auth=auth(app)).json()["changedAt"] is None
    assert root.patch("/v1/config", json={"installMode": "dev"}).status_code == 200
    assert root.post("/oidc/install-mode", auth=auth(app)).json()["changedAt"]


def test_deleting_the_owner_ends_a_call_queued_for_their_site(root: TestClient) -> None:
    """Their site goes with them, and so does what was waiting for it: the
    caller hears at once, rather than after the call's 20 s."""
    c = root
    a = ada_with_a_folder(c)
    a.held.summary["folders"][0]["people"] = [{"subject": a.person["id"], "writable": True}]
    bo, bo_token = add_person(c, a.app, "bo")
    a.held.summary["folders"][0]["people"].append({"subject": bo["id"], "writable": False})
    c.post("/v1/sites/poll", headers=a.keys.bearer(a.site), json=a.held.report())
    with ThreadPoolExecutor() as pool:
        started = time.perf_counter()
        pending = pool.submit(read, c, a, bo_token)
        for _ in range(100):
            if c.app.state.site_broker.jobs:
                break
            time.sleep(0.01)
        assert c.delete(f"/v1/people/{a.person['id']}").status_code == 204
        ended = pending.result(timeout=10)
    assert ended.status_code == 503 and time.perf_counter() - started < 10


def test_a_change_the_site_holds_is_202_not_a_refusal_and_names_its_people(
    root: TestClient,
) -> None:
    """J14a: a site whose owner has a key holds a change that gives access
    until they approve it at the machine (J50). Workbench is told so, with
    the site's words, and the site is sent how this root names each person,
    for display only (J54)."""
    c = root
    a = ada_with_a_folder(c)
    bo, _ = add_person(c, a.app, "bo")
    words = "Waiting for your approval on desk, at http://127.0.0.1:8079/link/approve."
    command, held = manage_through(
        c,
        a.keys,
        a.site,
        a.held,
        lambda: c.post(
            f"/oidc/job-sites/{a.site}/folders/{a.folder['id']}/people",
            auth=auth(a.app),
            json={"refreshToken": a.token, "people": [{"name": "bo", "writable": False}]},
        ),
        lambda _: {"status": "held", "message": words},
        action="folder.people",
    )
    assert command["names"] == {bo["id"]: "bo"}
    assert command["arguments"]["people"] == [{"subject": bo["id"], "writable": False}]
    assert held.status_code == 202, held.text
    assert held.json() == {"held": True, "message": words}
    assert held.headers["cache-control"] == "no-store"
    # The site's own signing state is shown as it reported it.
    page = "http://127.0.0.1:8079/link/approve"
    a.held.summary["signing"] = {"state": "signed", "held": 1, "approvePage": page}
    c.post("/v1/sites/poll", headers=a.keys.bearer(a.site), json=a.held.report())
    listed = c.post("/oidc/job-sites", auth=auth(a.app), json={"refreshToken": a.token})
    (site,) = listed.json()["sites"]
    assert site["signing"] == {"state": "signed", "held": 1, "approvePage": page}


# --------------------------------------------------------------------------- passkeys (J14a.3)


def _passkey_site(c: TestClient) -> Ada:
    a = ada_with_a_folder(c)
    a.held.summary["signing"] = {"state": "unconfirmed", "held": 0, "passkeys": True}
    c.post("/v1/sites/poll", headers=a.keys.bearer(a.site), json=a.held.report())
    return a


PAIRING = {
    "credentialId": "Y3JlZA",
    "publicKey": "cHVibGlj",
    "alg": -7,
    "rpId": "workbench.example",
    "label": "Chrome on laptop",
    "mac": "m" * 43,
}


def test_a_passkey_is_carried_to_the_site_and_the_code_never_is(root: TestClient) -> None:
    """The root carries the public half and the MAC, exactly as sent; it
    has no field for the code and refuses a body that carries one."""
    c = root
    a = _passkey_site(c)
    pinned = {"id": "a" * 32, **{k: PAIRING[k] for k in ("credentialId", "alg", "rpId", "label")}}
    command, paired = manage_through(
        c,
        a.keys,
        a.site,
        a.held,
        lambda: c.post(
            f"/oidc/job-sites/{a.site}/passkeys",
            auth=auth(a.app),
            json={"refreshToken": a.token, **PAIRING},
        ),
        lambda _: {"status": "done", "result": {**pinned, "addedAt": "2026-10-07T12:00:00Z"}},
        action="passkey.pair",
    )
    assert command["subject"] == a.person["id"]
    assert command["arguments"] == PAIRING
    assert paired.status_code == 201, paired.text
    assert paired.json()["id"] == "a" * 32
    smuggled = c.post(
        f"/oidc/job-sites/{a.site}/passkeys",
        auth=auth(a.app),
        json={"refreshToken": a.token, **PAIRING, "code": "ABCDE-FGHJK"},
    )
    assert smuggled.status_code == 422


def test_held_changes_are_listed_and_approved_through_the_root(root: TestClient) -> None:
    c = root
    a = _passkey_site(c)
    listing = {
        "subject": a.person["id"],
        "keys": ["b" * 32],
        "passkeys": [],
        "items": [{"id": "rules", "action": "rules.confirm", "words": ["x"], "heldAt": None}],
    }
    command, listed = manage_through(
        c,
        a.keys,
        a.site,
        a.held,
        lambda: c.post(
            f"/oidc/job-sites/{a.site}/held",
            auth=auth(a.app),
            json={"refreshToken": a.token, "key": "b" * 32},
        ),
        lambda _: {"status": "done", "result": listing},
        action="held.list",
    )
    assert command["arguments"] == {"key": "b" * 32}
    assert listed.status_code == 200 and listed.json() == listing
    approval = {
        "envelope": "{}x",
        "key": "b" * 32,
        "credentialId": "Y3JlZA",
        "authenticatorData": "YXV0aA",
        "clientDataJSON": "Y2xpZW50",
        "signature": "c2ln",
    }
    command, approved = manage_through(
        c,
        a.keys,
        a.site,
        a.held,
        lambda: c.post(
            f"/oidc/job-sites/{a.site}/held/abc123/approve",
            auth=auth(a.app),
            json={"refreshToken": a.token, **approval},
        ),
        lambda _: {"status": "done", "result": {}},
        action="held.approve",
    )
    # The change named in the address, not `rules`, which a test approving
    # only the rules could not tell apart.
    assert command["arguments"] == {"id": "abc123", **approval}
    assert approved.status_code == 200 and approved.json()["id"] == a.site
    # The site's refusal comes back in its words.
    _, refused = manage_through(
        c,
        a.keys,
        a.site,
        a.held,
        lambda: c.post(
            f"/oidc/job-sites/{a.site}/held/rules/approve",
            auth=auth(a.app),
            json={"refreshToken": a.token, **approval},
        ),
        lambda _: {"status": "failed", "message": "The passkey signed something else."},
        action="held.approve",
    )
    assert refused.status_code == 422 and "signed something else" in refused.text
    command, rejected = manage_through(
        c,
        a.keys,
        a.site,
        a.held,
        lambda: c.post(
            f"/oidc/job-sites/{a.site}/held/abc123/reject",
            auth=auth(a.app),
            json={"refreshToken": a.token},
        ),
        lambda _: {"status": "done", "result": {}},
        action="held.reject",
    )
    assert command["arguments"] == {"id": "abc123"} and rejected.status_code == 204
    command, removed = manage_through(
        c,
        a.keys,
        a.site,
        a.held,
        lambda: c.post(
            f"/oidc/job-sites/{a.site}/passkeys/{'b' * 32}/remove",
            auth=auth(a.app),
            json={"refreshToken": a.token},
        ),
        lambda _: {"status": "done", "result": {}},
        action="passkey.remove",
    )
    assert command["arguments"] == {"id": "b" * 32} and removed.status_code == 204
    _, gone = manage_through(
        c,
        a.keys,
        a.site,
        a.held,
        lambda: c.post(
            f"/oidc/job-sites/{a.site}/passkeys/{'b' * 32}/remove",
            auth=auth(a.app),
            json={"refreshToken": a.token},
        ),
        lambda _: {"status": "failed", "message": "That passkey is not paired with this machine."},
        action="passkey.remove",
    )
    assert gone.status_code == 422 and "not paired" in gone.text
    not_a_key = c.post(
        f"/oidc/job-sites/{a.site}/passkeys/abc123/remove",
        auth=auth(a.app),
        json={"refreshToken": a.token},
    )
    assert not_a_key.status_code == 422
    bad = c.post(
        f"/oidc/job-sites/{a.site}/held/NOT_AN_ID/reject",
        auth=auth(a.app),
        json={"refreshToken": a.token},
    )
    assert bad.status_code == 422


def test_a_site_that_does_not_take_passkeys_is_told_to_update(root: TestClient) -> None:
    """A site older than J14a.3 would refuse the action as unreadable;
    the root says what to do instead, and queues nothing."""
    c = root
    a = ada_with_a_folder(c)
    for path, body in (
        ("passkeys", PAIRING),
        ("held", {}),
        ("held/abc/reject", {}),
        (f"passkeys/{'b' * 32}/remove", {}),
    ):
        answer_ = c.post(
            f"/oidc/job-sites/{a.site}/{path}",
            auth=auth(a.app),
            json={"refreshToken": a.token, **body},
        )
        assert answer_.status_code == 503, (path, answer_.text)
        assert "does not take passkeys yet" in answer_.text
    polled = c.post("/v1/sites/poll", headers=a.keys.bearer(a.site), json=a.held.report())
    assert polled.json() == {"operation": None}


def test_only_the_sites_owner_reaches_its_passkeys(root: TestClient) -> None:
    c = root
    a = _passkey_site(c)
    _, bo_token = add_person(c, a.app, "bo")
    for path in ("passkeys", "held", f"passkeys/{'b' * 32}/remove"):
        body = PAIRING if path == "passkeys" else {}
        refused = c.post(
            f"/oidc/job-sites/{a.site}/{path}",
            auth=auth(a.app),
            json={"refreshToken": bo_token, **body},
        )
        assert refused.status_code == 404, refused.text
