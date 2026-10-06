"""Linking a person to an OS account on a site's machine (slice 2b.2,
`specs/docs/design/job-sites-own-enrollment.md` §3.2, J36, J37).

The agent's link page signs the person in to Eugene through the built-in
public client `eugene-site-link`; a Linux system install checks a person's
sign-in through the root with the site's own token; and a person (or the
site's owner) removes a link through the agent that hosts the site.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_control import oidc, security
from eugene_plexus_control.settings import Settings

from .conftest import PASSPHRASE, enroll
from .test_oidc import _challenge, _request_id
from .test_sites import (
    NODES_ORIGIN,
    Ada,
    Report,
    _root,
    ada_with_a_folder,
    add_person,
    auth,
    join_site,
    quick_poll,  # noqa: F401  (autouse fixture)
    site_keys,
    workbench,
)

LINK = oidc.SITE_LINK_CLIENT
CALLBACK = "http://127.0.0.1:51234/link/callback"


@pytest.fixture
def root(tmp_path: Path) -> Iterator[TestClient]:
    yield from _root(
        Settings(
            config_file=tmp_path / "control.yaml",
            state_dir=tmp_path / "state",
            nodes_origin=NODES_ORIGIN,
        )
    )


def _link_authorize(c: TestClient, **extra: str) -> Any:
    params = {
        "response_type": "code",
        "client_id": LINK,
        "redirect_uri": CALLBACK,
        "scope": "openid profile",
        "state": "the-state",
        "nonce": "the-nonce",
        "code_challenge": _challenge("v" * 50),
        "code_challenge_method": "S256",
        **extra,
    }
    return c.get("/oidc/authorize", params=params, follow_redirects=False)


def _link_code(c: TestClient, *, name: str, password: str = PASSPHRASE) -> str:
    page = _link_authorize(c)
    assert page.status_code == 200, page.text
    done = c.post(
        "/oidc/authorize",
        data={"request": _request_id(page.text), "name": name, "password": password},
        follow_redirects=False,
    )
    assert done.status_code == 302, done.text
    assert done.headers["location"].startswith(CALLBACK + "?")
    return parse_qs(urlsplit(done.headers["location"]).query)["code"][0]


def _trade(c: TestClient, code: str, **change: Any) -> Any:
    data = {
        "grant_type": "authorization_code",
        "client_id": LINK,
        "code": code,
        "code_verifier": "v" * 50,
        "redirect_uri": CALLBACK,
        **change,
    }
    return c.post("/oidc/token", data=data)


# ------------------------------------------------------------------ the built-in client


@pytest.mark.parametrize(
    "uri",
    [
        "http://127.0.0.1:1/link/callback",
        "http://127.0.0.1:65535/link/callback",
        "http://127.0.0.1:51234/link/callback",
    ],
)
def test_the_link_page_may_use_any_loopback_port(root: TestClient, uri: str) -> None:
    assert _link_authorize(root, redirect_uri=uri).status_code == 200


@pytest.mark.parametrize(
    "uri",
    [
        "http://localhost:51234/link/callback",
        "http://[::1]:51234/link/callback",
        "http://127.0.0.2:51234/link/callback",
        "https://127.0.0.1:51234/link/callback",
        "http://127.0.0.1/link/callback",
        "http://127.0.0.1:0/link/callback",
        "http://127.0.0.1:65536/link/callback",
        "http://127.0.0.1:012/link/callback",
        "http://127.0.0.1:51234/other",
        "http://127.0.0.1:51234/link/callback/",
        "http://127.0.0.1:51234/link/callback?x=1",
        "http://127.0.0.1:51234/link/callback#frag",
        "http://user@127.0.0.1:51234/link/callback",
        "http://127.0.0.1:51234@evil.example/link/callback",
        "http://127.0.0.1:51234/link/callback\n",
        "",
    ],
)
def test_any_other_return_address_is_a_page_and_never_a_redirect(
    root: TestClient, uri: str
) -> None:
    answer = _link_authorize(root, redirect_uri=uri)
    assert answer.status_code == 400
    assert "location" not in answer.headers


def test_state_nonce_and_pkce_are_required_of_the_link_page_too(root: TestClient) -> None:
    for missing in ("state", "nonce"):
        answer = _link_authorize(root, **{missing: ""})
        assert answer.status_code == 302
        assert "error=invalid_request" in answer.headers["location"]
    plain = _link_authorize(root, code_challenge_method="plain")
    assert plain.status_code == 302 and "error=invalid_request" in plain.headers["location"]


def test_it_is_in_no_list_and_cannot_be_edited_or_removed(root: TestClient) -> None:
    workbench(root)
    listed = root.get("/v1/oidc/clients").json()["clients"]
    assert LINK not in {c["clientId"] for c in listed}
    assert root.delete(f"/v1/oidc/clients/{LINK}").status_code == 404
    edited = root.put(
        f"/v1/oidc/clients/{LINK}/redirect-uris", json={"redirectUris": ["http://127.0.0.1:9/x"]}
    )
    assert edited.status_code in (404, 405)
    # And nothing about it is in the replicated state.
    assert LINK not in root.app.state.machine.state.oidc_clients  # type: ignore[attr-defined]


def test_every_person_may_use_it_whatever_apps_they_were_given(root: TestClient) -> None:
    app = workbench(root)
    add_person(root, app, "ada")  # allowed only Workbench
    got = _trade(root, _link_code(root, name="ada"))
    assert got.status_code == 200, got.text
    body = got.json()
    assert "refresh_token" not in body and body["id_token"] and body["access_token"]
    claims = oidc.Provider.decode(
        body["id_token"],
        root.app.state.machine.state,  # type: ignore[attr-defined]
        typ=oidc.TYP_ID,
        audience=LINK,
    )
    assert claims["preferred_username"] == "ada" and claims["nonce"] == "the-nonce"


def test_a_person_with_no_apps_at_all_may_still_link(root: TestClient) -> None:
    made = root.post("/v1/people", json={"name": "bo", "password": PASSPHRASE, "apps": []})
    assert made.status_code == 201, made.text
    assert _trade(root, _link_code(root, name="bo")).status_code == 200


def test_eugenes_owner_is_refused_with_the_may_not_use_page(root: TestClient) -> None:
    app = workbench(root)
    add_person(root, app, "ada")
    page = _link_authorize(root)
    refused = root.post(
        "/oidc/authorize",
        data={"request": _request_id(page.text), "name": "operator", "password": PASSPHRASE},
        follow_redirects=False,
    )
    assert refused.status_code == 403 and "You may not use" in refused.text
    assert "location" not in refused.headers


def test_a_wrong_password_on_the_link_page_is_the_ordinary_answer(root: TestClient) -> None:
    app = workbench(root)
    add_person(root, app, "ada")
    page = _link_authorize(root)
    answer = root.post(
        "/oidc/authorize",
        data={"request": _request_id(page.text), "name": "ada", "password": "nope-nope-nope-no"},
        follow_redirects=False,
    )
    assert answer.status_code == 200 and "do not match" in answer.text


def test_the_link_page_trades_a_code_and_nothing_else(root: TestClient) -> None:
    app = workbench(root)
    add_person(root, app, "ada")
    # A secret in the form is refused, and the refusal does not spend the code.
    code = _link_code(root, name="ada")
    assert _trade(root, code, client_secret="anything").status_code == 401
    # So is a Basic header, even one naming this client.
    refused = root.post(
        "/oidc/token",
        auth=(LINK, "x"),
        data={
            "grant_type": "authorization_code",
            "client_id": LINK,
            "code": code,
            "code_verifier": "v" * 50,
            "redirect_uri": CALLBACK,
        },
    )
    assert refused.status_code == 401 and refused.json()["error"] == "invalid_client"
    assert _trade(root, code).status_code == 200
    # No refresh grant.
    grant = root.post(
        "/oidc/token",
        data={"grant_type": "refresh_token", "client_id": LINK, "refresh_token": "x"},
    )
    assert grant.status_code == 400 and grant.json()["error"] == "unsupported_grant_type"
    # A wrong redirect or verifier, and a reused code, are refused.
    wrong = "http://127.0.0.1:1/link/callback"
    assert _trade(root, _link_code(root, name="ada"), redirect_uri=wrong).status_code == 400
    assert _trade(root, _link_code(root, name="ada"), code_verifier="w" * 50).status_code == 400
    used = _link_code(root, name="ada")
    assert _trade(root, used).status_code == 200
    assert _trade(root, used).status_code == 400


def test_a_code_is_for_the_client_it_was_issued_to(root: TestClient) -> None:
    app = workbench(root)
    add_person(root, app, "ada")
    link_code = _link_code(root, name="ada")
    stolen = root.post(
        "/oidc/token",
        auth=auth(app),
        data={
            "grant_type": "authorization_code",
            "code": link_code,
            "code_verifier": "v" * 50,
            "redirect_uri": CALLBACK,
        },
    )
    assert stolen.status_code == 400 and stolen.json()["error"] == "invalid_grant"
    # And the link page cannot redeem another app's code.
    redirect = "http://127.0.0.1:9/callback"
    page = root.get(
        "/oidc/authorize",
        params={
            "response_type": "code",
            "client_id": app["client"]["clientId"],
            "redirect_uri": redirect,
            "scope": "openid",
            "state": "s",
            "nonce": "n",
            "code_challenge": _challenge("v" * 50),
            "code_challenge_method": "S256",
        },
    )
    done = root.post(
        "/oidc/authorize",
        data={"request": _request_id(page.text), "name": "ada", "password": PASSPHRASE},
        follow_redirects=False,
    )
    code = parse_qs(urlsplit(done.headers["location"]).query)["code"][0]
    refused = _trade(root, code, redirect_uri=redirect)
    assert refused.status_code == 400 and refused.json()["error"] == "invalid_grant"


def test_the_link_pages_access_token_speaks_for_nobody_at_userinfo(root: TestClient) -> None:
    app = workbench(root)
    add_person(root, app, "ada")
    body = _trade(root, _link_code(root, name="ada")).json()
    userinfo = root.get(
        "/oidc/userinfo", headers={"Authorization": "Bearer " + body["access_token"]}
    )
    assert userinfo.status_code == 401


# ------------------------------------------------------------------ the site's own check


def _joined(root: TestClient) -> tuple[Ada, dict[str, str]]:
    a = ada_with_a_folder(root)
    return a, a.keys.bearer(a.site)


def _check(c: TestClient, headers: dict[str, str], name: str, password: str) -> Any:
    return c.post(
        "/v1/sites/links/check", headers=headers, json={"name": name, "password": password}
    )


def test_a_site_checks_a_persons_sign_in_and_nothing_is_recorded(root: TestClient) -> None:
    a, headers = _joined(root)
    bo, _ = add_person(root, a.app, "bo")
    state = root.app.state.machine.state  # type: ignore[attr-defined]
    ok = _check(root, headers, "Bo", PASSPHRASE)
    assert ok.status_code == 200, ok.text
    assert ok.json() == {"subject": bo["id"], "name": "bo"}
    assert root.app.state.machine.state is state  # type: ignore[attr-defined]


def test_a_wrong_name_and_a_wrong_password_are_the_same_answer(root: TestClient) -> None:
    a, headers = _joined(root)
    add_person(root, a.app, "bo")
    wrong_password = _check(root, headers, "bo", "not-the-password")
    wrong_name = _check(root, headers, "nobody", PASSPHRASE)
    assert wrong_password.status_code == wrong_name.status_code == 401
    assert wrong_password.json() == wrong_name.json()


def test_a_disabled_person_and_the_owner_are_refused(root: TestClient) -> None:
    a, headers = _joined(root)
    bo, _ = add_person(root, a.app, "bo")
    assert root.patch(f"/v1/people/{bo['id']}", json={"disabled": True}).status_code == 200
    assert _check(root, headers, "bo", PASSPHRASE).status_code == 403
    assert _check(root, headers, "operator", PASSPHRASE).status_code == 403


def test_a_site_cannot_use_the_check_to_guess_faster_than_a_sign_in(root: TestClient) -> None:
    a, headers = _joined(root)
    add_person(root, a.app, "bo")
    add_person(root, a.app, "cy")
    for _ in range(oidc.SIGN_IN_FAILURES):
        assert _check(root, headers, "bo", "wrong-wrong-wrong").status_code == 401
    # The name is limited now, whatever the password; so is the site, for any name.
    assert _check(root, headers, "bo", PASSPHRASE).status_code == 429
    assert _check(root, headers, "cy", PASSPHRASE).status_code == 429
    # The site's limit is its own: a second site is not held by it.
    other, joined = join_site(root, a.person, "second")
    assert joined.status_code == 201, joined.text
    second = other.bearer(joined.json()["id"])
    assert _check(root, second, "cy", PASSPHRASE).status_code == 200
    # The name's limit is the sign-in page's own, and no site's: a second site
    # cannot try the name the first has just been held on.
    assert _check(root, second, "bo", PASSPHRASE).status_code == 429


def test_only_an_enrolled_sites_token_may_check(root: TestClient) -> None:
    a, headers = _joined(root)
    add_person(root, a.app, "bo")
    body = {"name": "bo", "password": PASSPHRASE}
    operator = {"Authorization": root.headers.pop("Authorization")}
    try:
        assert root.post("/v1/sites/links/check", json=body).status_code == 401
        assert root.post("/v1/sites/links/check", headers=operator, json=body).status_code == 401
        stranger = site_keys().bearer(a.site)
        assert root.post("/v1/sites/links/check", headers=stranger, json=body).status_code == 401
        assert root.post("/v1/sites/links/check", headers=headers, json=body).status_code == 200
    finally:
        root.headers.update(operator)


def test_an_unknown_name_costs_a_password_verification(
    root: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, headers = _joined(root)
    calls: list[str] = []
    real = security.verify_passphrase

    def counting(password: str, verifier: str) -> bool:
        calls.append(verifier)
        return real(password, verifier)

    monkeypatch.setattr(security, "verify_passphrase", counting)
    _check(root, headers, "nobody", "whatever-whatever")
    assert len(calls) == 1


# ------------------------------------------------------------------ removing a link


class FakeNodes:
    """The root's node client, answering as an agent would."""

    def __init__(self) -> None:
        self.code: int = 204
        self.body: Any = None
        self.raises: Exception | None = None
        self.calls: list[tuple[str, str, str | None]] = []

    async def delete(self, url: str, path: str, token: str | None) -> tuple[int, Any]:
        self.calls.append((url, path, token))
        if self.raises:
            raise self.raises
        return self.code, self.body

    async def aclose(self) -> None:
        return None


def _hosted(root: TestClient, a: Ada) -> FakeNodes:
    """Node `amish`, with an address, hosting Ada's site."""
    keys, made = enroll(root, "amish", url="http://192.0.2.10:8079")
    assert made.status_code == 201, made.text
    said = root.put(
        "/v1/nodes/amish/hosted-sites",
        headers={"Authorization": "Bearer " + keys.service_token()},
        json={"sites": [a.site]},
    )
    assert said.status_code == 204, said.text
    fake = FakeNodes()
    root.app.state.nodes_client = fake  # type: ignore[attr-defined]
    return fake


def _remove(c: TestClient, a: Ada, token: str, **fields: Any) -> Any:
    return c.post(
        "/oidc/sites/link/remove",
        auth=auth(a.app),
        json={"refreshToken": token, "site": a.site, **fields},
    )


def _share_with(root: TestClient, a: Ada, person: dict[str, Any]) -> None:
    a.held.summary["folders"][0]["people"] = [{"subject": person["id"], "writable": False}]
    root.post("/v1/sites/poll", headers=a.keys.bearer(a.site), json=a.held.report())


def test_a_person_removes_their_own_link_through_the_hosting_agent(root: TestClient) -> None:
    a = ada_with_a_folder(root)
    fake = _hosted(root, a)
    bo, bo_token = add_person(root, a.app, "bo")
    _share_with(root, a, bo)
    done = _remove(root, a, bo_token)
    assert done.status_code == 204, done.text
    ((url, path, token),) = fake.calls
    assert url.rstrip("/") == "http://192.0.2.10:8079" and path == f"/v1/site/links/{bo['id']}"
    assert token  # the root's own, addressed to that node
    # Naming themselves is the same as naming nobody.
    assert _remove(root, a, bo_token, person=bo["id"]).status_code == 204
    assert fake.calls[1][1] == f"/v1/site/links/{bo['id']}"


def test_a_person_cannot_remove_anothers_link_but_the_owner_can(root: TestClient) -> None:
    a = ada_with_a_folder(root)
    fake = _hosted(root, a)
    bo, bo_token = add_person(root, a.app, "bo")
    _share_with(root, a, bo)
    assert _remove(root, a, bo_token, person=a.person["id"]).status_code == 403
    assert fake.calls == []
    owner = _remove(root, a, a.token, person=bo["id"])
    assert owner.status_code == 204, owner.text
    assert fake.calls[0][1] == f"/v1/site/links/{bo['id']}"
    assert _remove(root, a, a.token).status_code == 204
    assert fake.calls[1][1] == f"/v1/site/links/{a.person['id']}"


def test_a_site_one_may_not_use_is_not_found(root: TestClient) -> None:
    a = ada_with_a_folder(root)
    fake = _hosted(root, a)
    _, cy_token = add_person(root, a.app, "cy")  # no grant, no link
    assert _remove(root, a, cy_token).status_code == 404
    unknown = root.post(
        "/oidc/sites/link/remove",
        auth=auth(a.app),
        json={"refreshToken": cy_token, "site": "s-" + "a" * 26},
    )
    assert unknown.status_code == 404
    assert fake.calls == []


def test_a_server_grant_or_a_link_is_enough_to_know_the_site(root: TestClient) -> None:
    a = ada_with_a_folder(root)
    fake = _hosted(root, a)
    bo, bo_token = add_person(root, a.app, "bo")
    cy, cy_token = add_person(root, a.app, "cy")
    a.held.summary["access"] = [{"subject": bo["id"], "server": "calc", "tools": [{"name": "add"}]}]
    a.held.summary["links"] = [{"subject": cy["id"], "accountName": "HOST\\cy", "available": True}]
    root.post("/v1/sites/poll", headers=a.keys.bearer(a.site), json=a.held.report())
    assert _remove(root, a, bo_token).status_code == 204
    assert _remove(root, a, cy_token).status_code == 204
    assert len(fake.calls) == 2


def test_no_hosting_node_is_a_conflict_and_an_agents_409_is_relayed(root: TestClient) -> None:
    a = ada_with_a_folder(root)
    fake = FakeNodes()
    root.app.state.nodes_client = fake  # type: ignore[attr-defined]
    none = _remove(root, a, a.token)
    assert none.status_code == 409 and "no node hosts this site" in none.text.lower()
    assert fake.calls == []
    fake = _hosted(root, a)
    fake.code = 409
    fake.body = {
        "detail": {"title": "Elevated only", "detail": "Links here go with the one-liner."}
    }
    refused = _remove(root, a, a.token)
    assert refused.status_code == 409
    assert "one-liner" in refused.text


def test_an_agent_that_does_not_answer_is_a_503(root: TestClient) -> None:
    a = ada_with_a_folder(root)
    fake = _hosted(root, a)
    fake.raises = httpx.ConnectError("refused")
    assert _remove(root, a, a.token).status_code == 503
    fake.raises = None
    fake.code, fake.body = 500, {"detail": "boom"}
    assert _remove(root, a, a.token).status_code == 503


def test_removal_needs_the_workbench_client_and_a_live_sign_in(root: TestClient) -> None:
    a = ada_with_a_folder(root)
    _hosted(root, a)
    no_client = root.post("/oidc/sites/link/remove", json={"refreshToken": a.token, "site": a.site})
    assert no_client.status_code == 401
    assert _remove(root, a, "not-a-token").status_code == 401


# ------------------------------------------------------------------ what Workbench is told


def _with_links(root: TestClient, a: Ada, bo: dict[str, Any]) -> None:
    a.held.summary["folders"][0]["people"] = [
        {"subject": a.person["id"], "writable": True},
        {"subject": bo["id"], "writable": False},
    ]
    a.held.summary["links"] = [
        {"subject": a.person["id"], "accountName": "HOST\\ada", "available": True}
    ]
    a.held.summary["linkPage"] = "http://127.0.0.1:8079/link"
    a.held.summary["sharing"] = True
    root.post("/v1/sites/poll", headers=a.keys.bearer(a.site), json=a.held.report())


def test_the_owners_page_carries_the_links_the_link_page_and_sharing(root: TestClient) -> None:
    a = ada_with_a_folder(root)
    bo, _ = add_person(root, a.app, "bo")
    _with_links(root, a, bo)
    listed = root.post("/oidc/job-sites", auth=auth(a.app), json={"refreshToken": a.token})
    (site,) = listed.json()["sites"]
    assert site["links"] == [
        {"subject": a.person["id"], "accountName": "HOST\\ada", "available": True}
    ]
    assert site["linkPage"] == "http://127.0.0.1:8079/link" and site["sharing"] is True


def test_each_grant_says_whether_this_person_linked_and_as_whom_they_run(
    root: TestClient,
) -> None:
    a = ada_with_a_folder(root)
    bo, bo_token = add_person(root, a.app, "bo")
    _with_links(root, a, bo)
    unlinked = root.post("/oidc/sites/servers", auth=auth(a.app), json={"refreshToken": bo_token})
    (grant,) = unlinked.json()["servers"]
    # Bo has not linked: they run as the owner, and are told where to link.
    assert grant["linked"] is False
    assert grant["account"] == "HOST\\ada"
    assert grant["linkPage"] == "http://127.0.0.1:8079/link"
    mine = root.post("/oidc/sites/servers", auth=auth(a.app), json={"refreshToken": a.token})
    (own,) = mine.json()["servers"]
    assert own["linked"] is True and own["account"] == "HOST\\ada" and own["linkPage"] is None
    # Bo links: their own account, and no link page.
    a.held.summary["links"].append(
        {"subject": bo["id"], "accountName": "HOST\\bo", "available": False, "reason": "x"}
    )
    root.post("/v1/sites/poll", headers=a.keys.bearer(a.site), json=a.held.report())
    (linked,) = root.post(
        "/oidc/sites/servers", auth=auth(a.app), json={"refreshToken": bo_token}
    ).json()["servers"]
    assert linked["linked"] is True and linked["account"] == "HOST\\bo"
    assert linked["linkPage"] is None


def test_an_owner_who_has_not_linked_leaves_the_account_empty(root: TestClient) -> None:
    a = ada_with_a_folder(root)
    bo, bo_token = add_person(root, a.app, "bo")
    _with_links(root, a, bo)
    a.held.summary["links"] = []
    root.post("/v1/sites/poll", headers=a.keys.bearer(a.site), json=a.held.report())
    (grant,) = root.post(
        "/oidc/sites/servers", auth=auth(a.app), json={"refreshToken": bo_token}
    ).json()["servers"]
    assert grant["linked"] is False and grant["account"] is None


def test_an_old_report_with_no_links_still_lists(root: TestClient) -> None:
    a = ada_with_a_folder(root)
    assert isinstance(a.held, Report) and "links" not in a.held.summary
    _share_with(root, a, a.person)
    (grant,) = root.post(
        "/oidc/sites/servers", auth=auth(a.app), json={"refreshToken": a.token}
    ).json()["servers"]
    assert "linked" not in grant and "account" not in grant and "linkPage" not in grant
    (site,) = root.post("/oidc/job-sites", auth=auth(a.app), json={"refreshToken": a.token}).json()[
        "sites"
    ]
    assert "links" not in site
