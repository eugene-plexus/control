"""Row 38: alert rules, SLOs and the webhook, on the active root.

`specs/docs/design/rows/row-38.md`. A rule fires after its duration and
resolves after the same quiet time; a failed webhook is tried three times
and then shown; the evaluator runs only on the active root; the webhook
address is sealed and never shown back; a person with an operations power
reads alerts but never an SLO set on a client key.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_control import alerts
from eugene_plexus_control.alerts import (
    AlertBook,
    AlertService,
    MachineReading,
    Readings,
    WebhookSender,
    complete,
    conditions,
    slo_statuses,
)
from eugene_plexus_control.app import create_app, start_root_tasks
from eugene_plexus_control.applied import canonical_bytes

from .conftest import PASSPHRASE, login, settings_for
from .test_administration import delegate

T0 = datetime(2026, 11, 16, 12, 0, tzinfo=UTC)
HOT = {"id": "gpu-hot", "kind": "gpu-hot", "enabled": True, "threshold": 85.0, "forMinutes": 5,
       "machine": None, "model": None, "sloId": None}  # fmt: skip


def _gpu(celsius: float | None, *, reachable: bool = True) -> Readings:
    gpus = None if celsius is None else (("gpu:cuda:0", "RTX 5090", celsius),)
    return Readings(
        machines=(MachineReading(name="nas", label="nas", reachable=reachable, gpus=gpus),)
    )


def _step(book: AlertBook, minute: int, readings: Readings) -> list[dict[str, Any]]:
    rules = [HOT]
    return book.step(
        T0 + timedelta(minutes=minute),
        conditions(rules, readings),
        rules=rules,
        machines={"nas"},
        whole=complete(rules, readings),
    )


def test_a_rule_fires_after_its_duration_and_resolves() -> None:
    book = AlertBook(None)
    for minute in range(5):
        assert _step(book, minute, _gpu(88)) == [], f"fired after {minute} minutes"
    fired = _step(book, 5, _gpu(88))
    assert [(a["state"], a["value"], a["machine"]) for a in fired] == [("firing", 88.0, "nas")]
    assert fired[0]["message"] == "RTX 5090 on nas is at 88 °C"
    assert len(book.firing()) == 1
    # A minute's dip does not resolve it, and a minute back over does not fire it again.
    assert _step(book, 6, _gpu(80)) == []
    assert _step(book, 7, _gpu(90)) == []
    for minute in range(8, 13):
        assert _step(book, minute, _gpu(70)) == [], f"resolved after {minute - 8} quiet minutes"
    resolved = _step(book, 13, _gpu(70))
    assert [a["state"] for a in resolved] == ["resolved"]
    assert book.firing() == [] and book.recent[0]["resolvedAt"]


def test_an_unread_machine_leaves_its_alerts_as_they_were() -> None:
    book = AlertBook(None)
    for minute in range(6):
        _step(book, minute, _gpu(88))
    assert len(book.firing()) == 1
    # The host history did not answer for half an hour: unknown is not clear.
    for minute in range(6, 36):
        assert _step(book, minute, _gpu(None)) == []
    assert len(book.firing()) == 1


def test_a_failure_rate_fires_at_once_and_needs_ten_requests() -> None:
    rule = {"id": "failure-rate", "kind": "failure-rate", "enabled": True, "threshold": 5.0,
            "forMinutes": 10, "machine": None, "model": None, "sloId": None}  # fmt: skip
    book = AlertBook(None)

    def step(minute: int, requests: int, failed: int) -> list[dict[str, Any]]:
        readings = Readings(failures={10: {None: (requests, failed)}})
        return book.step(
            T0 + timedelta(minutes=minute),
            conditions([rule], readings),
            rules=[rule],
            machines=set(),
        )

    assert step(0, 4, 4) == [], "four requests are not a failure rate"
    fired = step(1, 20, 10)
    assert fired[0]["state"] == "firing" and fired[0]["value"] == 50.0
    assert "50% of requests to all models failed in the last 10 minutes" in fired[0]["message"]


def test_the_alert_state_is_kept_across_a_restart(tmp_path: Path) -> None:
    book = AlertBook(tmp_path / "alerts.json")
    for minute in range(6):
        _step(book, minute, _gpu(88))
    again = AlertBook(tmp_path / "alerts.json")
    assert [a["id"] for a in again.firing()] == ["gpu-hot:nas:gpu:cuda:0"]


def _failing_then(answers: list[int]) -> tuple[WebhookSender, list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(answers.pop(0) if answers else 200)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def no_wait(_: float) -> None:
        return None

    return WebhookSender(internal=client, egress=client, sleep=no_wait), seen


def test_a_failed_webhook_is_retried_then_shown() -> None:
    sender, seen = _failing_then([500, 500, 500])
    alert = {"ruleId": "gpu-hot", "message": "RTX 5090 on nas is at 88 °C", "value": 88.0,
             "threshold": 85.0, "machine": "nas", "model": None, "link": "/hosts?node=nas"}  # fmt: skip
    body = alerts.webhook_post(alert, state="firing", console="https://eugene.lan")
    assert asyncio.run(sender.post("http://hooks.lan/x", body)) is False
    assert len(seen) == 3, "three tries"
    status = sender.status(True)
    assert status["state"] == "failing" and status["lastError"] == "HTTP 500"
    assert seen[0]["text"] == "Firing: RTX 5090 on nas is at 88 °C"
    assert seen[0]["url"] == "https://eugene.lan/hosts?node=nas"
    assert {"rule", "state", "value", "threshold", "machine", "model", "url", "at"} <= set(seen[0])
    # It answers again: the failure clears.
    assert asyncio.run(sender.post("http://hooks.lan/x", body)) is True
    assert sender.status(True)["state"] == "ok" and sender.status(True)["lastError"] is None
    assert sender.status(False)["state"] == "off"


def test_slo_budget_and_burn_rate() -> None:
    now = T0
    hourly = [
        {"bucketStart": (now - timedelta(days=d)).isoformat(), "model": "qwen", "requests": 1000,
         "errors": 2, "source": "rollup" if d > 5 else "raw"}
        for d in range(1, 21)
    ]  # fmt: skip
    statuses = slo_statuses(
        [
            {"id": "qwen-99", "name": None, "model": "qwen", "clientKeyId": None, "windowDays": 30,
             "successPercent": 99.0, "firstTokenSeconds": 2.0, "firstTokenPercentile": 90},
            {"id": "strict", "name": "Strict", "model": None, "clientKeyId": None, "windowDays": 7,
             "successPercent": 99.95, "firstTokenSeconds": None, "firstTokenPercentile": None},
        ],
        now=now,
        hourly=hourly,
        last_hour=[{"model": "qwen", "requests": 100, "errors": 5}],
        first_token={"qwen": {"p50": 900, "p90": 2500, "p99": 4000, "max": 5000}},
        kept_since=now - timedelta(days=5),
    )  # fmt: skip
    qwen, strict = statuses
    assert (qwen["requests"], qwen["failures"]) == (20_000, 40)
    assert qwen["budgetLeftPercent"] == 80.0  # 40 of 200 allowed failures spent
    assert qwen["burnRate"] == 5.0  # 5% failing against 1% allowed
    assert qwen["firstToken"]["measuredSeconds"] == 2.5 and qwen["firstToken"]["met"] is False
    assert qwen["firstToken"]["keptDays"] == 5.0
    assert qwen["state"] == "at-risk", "the first-token target is not met"
    assert qwen["coveredDays"] == 20.0
    # The bucket that starts exactly seven days back is inside the window.
    assert (strict["requests"], strict["failures"]) == (7000, 14)
    assert strict["state"] == "missed"  # 14 failures against 3.5 allowed


class _NotPolling:
    def cancel(self) -> None:
        return None


def test_the_evaluator_moves_with_the_root(tmp_path: Path) -> None:
    app = create_app(settings_for(tmp_path / "root"))
    with TestClient(app) as client:
        assert (
            client.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE}).status_code == 204
        )
        service: AlertService = app.state.alerts
        assert service.running(), "the active, unlocked root evaluates"
        assert app.state.alerts_task is not None and not app.state.alerts_task.done()
    standby = create_app(settings_for(tmp_path / "standby", role="standby"))
    with TestClient(standby) as client:
        assert standby.state.alerts_task is None, "a standby does not evaluate"
        assert not standby.state.alerts.running()
        # A promotion starts the root's tasks, the evaluator among them.
        standby.state.node_poller = _NotPolling()  # this test is about the evaluator only
        assert client.portal is not None
        client.portal.call(start_root_tasks, standby)
        assert standby.state.alerts_task is not None


def test_the_owner_sets_rules_and_the_webhook_is_never_shown_back(
    active_client: TestClient,
) -> None:
    c = active_client
    shown = c.get("/v1/alerting")
    assert shown.status_code == 200, shown.text
    assert shown.json()["defaults"] is True
    assert {r["id"] for r in shown.json()["rules"]} == {
        "machine-down", "disk-full", "gpu-hot", "failure-rate", "slo-burn"
    }  # fmt: skip
    secret = "https://hooks.slack.com/services/T000/B000/s3cr3t-path"
    body = {
        "rules": [{"id": "hot", "kind": "gpu-hot", "threshold": 80, "forMinutes": 2}],
        "slos": [{"id": "all", "successPercent": 99.5, "windowDays": 30}],
        "webhookUrl": secret,
    }
    saved = c.put("/v1/alerting", json=body)
    assert saved.status_code == 200, saved.text
    assert saved.json()["webhook"] == {"configured": True, "host": "hooks.slack.com"}
    assert saved.json()["defaults"] is False
    assert secret not in saved.text and secret not in c.get("/v1/alerting").text
    machine = c.app.state.machine  # type: ignore[attr-defined]
    assert secret.encode() not in canonical_bytes(machine.state), "sealed in the replicated state"
    assert c.app.state.alerts._webhook() == secret  # type: ignore[attr-defined]
    # Absent keeps it; null removes it.
    kept = c.put("/v1/alerting", json={"rules": body["rules"], "slos": []})
    assert kept.json()["webhook"]["configured"] is True
    gone = c.put("/v1/alerting", json={"rules": body["rules"], "slos": [], "webhookUrl": None})
    assert gone.json()["webhook"] == {"configured": False, "host": None}


@pytest.mark.parametrize(
    ("body", "words"),
    [
        ({"rules": [], "slos": [], "webhookUrl": "file:///etc/passwd"}, "http:// or https://"),
        (
            {"rules": [{"id": "a", "kind": "gpu-hot", "threshold": 80}] * 2, "slos": []},
            "Two rules",
        ),
        (
            {"rules": [{"id": "b", "kind": "slo-burn", "threshold": 2, "sloId": "x"}], "slos": []},
            "not set",
        ),
        ({"rules": [{"id": "c", "kind": "nonsense"}], "slos": []}, "rules"),
    ],
)
def test_bad_alert_settings_are_refused_with_the_cause(
    active_client: TestClient, body: dict[str, Any], words: str
) -> None:
    refused = active_client.put("/v1/alerting", json=body)
    assert refused.status_code == 400, refused.text
    assert words in refused.json()["detail"]["detail"]


def test_a_person_with_operations_power_reads_alerts_but_not_key_slos(
    active_client: TestClient,
) -> None:
    c = active_client
    service: AlertService = c.app.state.alerts  # type: ignore[attr-defined]
    service._cache.slos = (
        {"id": "all", "clientKeyId": None, "state": "met", "windowDays": 30,
         "successPercent": 99.0, "requests": 1, "failures": 0},
        {"id": "acme", "clientKeyId": "ck-acme", "state": "met", "windowDays": 30,
         "successPercent": 99.0, "requests": 1, "failures": 0},
    )  # fmt: skip
    service.book.active = {
        "slo-burn:acme": {"id": "slo-burn:acme", "ruleId": "slo-burn", "kind": "slo-burn",
                          "message": "SLO acme burns", "sloId": "acme", "firing": True,
                          "startedAt": "2026-11-16T12:00:00Z"},
        "machine-down:gpu-b": {"id": "machine-down:gpu-b", "ruleId": "machine-down",
                               "kind": "machine-down", "message": "gpu-b does not answer",
                               "machine": "gpu-b", "firing": True,
                               "startedAt": "2026-11-16T12:00:00Z"},
        "machine-down:gpu-a": {"id": "machine-down:gpu-a", "ruleId": "machine-down",
                               "kind": "machine-down", "message": "gpu-a does not answer",
                               "machine": "gpu-a", "firing": True,
                               "startedAt": "2026-11-16T12:00:00Z"},
    }  # fmt: skip
    owner = c.get("/v1/alerts")
    assert owner.status_code == 200, owner.text
    assert len(owner.json()["firing"]) == 3 and len(owner.json()["slos"]) == 2
    _, dana = delegate(c, operateModels=True, operationsMachines=["gpu-a"])
    seen = c.get("/v1/alerts", headers=dana)
    assert seen.status_code == 200, seen.text
    assert [a["id"] for a in seen.json()["firing"]] == ["machine-down:gpu-a"]
    assert [s["id"] for s in seen.json()["slos"]] == ["all"]
    # The rules and the webhook stay the owner's.
    assert c.get("/v1/alerting", headers=dana).status_code in (401, 403)
    assert c.put("/v1/alerting", headers=dana, json={"rules": [], "slos": []}).status_code in (
        401,
        403,
    )
    assert c.post("/v1/alerting/test", headers=dana).status_code in (401, 403)


def test_a_person_without_operations_power_reads_no_alerts(active_client: TestClient) -> None:
    _, manager = delegate(active_client, managePeople=True)
    assert active_client.get("/v1/alerts", headers=manager).status_code == 403


def test_a_passphrase_change_seals_the_webhook_again(active_client: TestClient) -> None:
    c = active_client
    secret = "http://ha.lan:8123/api/webhook/eugene"
    assert (
        c.put("/v1/alerting", json={"rules": [], "slos": [], "webhookUrl": secret}).status_code
        == 200
    )
    changed = c.post(
        "/v1/auth/passphrase",
        json={"current": PASSPHRASE, "new": "another-horse-battery-staple"},
    )
    assert changed.status_code == 200, changed.text
    c.headers["Authorization"] = (
        f"Bearer {login(c, 'another-horse-battery-staple')}"  # the old session ended
    )
    assert c.app.state.alerts._webhook() == secret  # type: ignore[attr-defined]


def test_the_root_reads_the_gateway_through_its_agent_with_its_own_token(
    active_client: TestClient,
) -> None:
    """The failure rate comes from the gateway's metrics summary, read through
    this machine's agent's proxy with a token addressed to the gateway's
    machine; a gateway that refuses is named, and the alerts it feeds stay."""
    import time

    import jwt

    app = active_client.app
    service: AlertService = app.state.alerts  # type: ignore[attr-defined]
    app.state.settings.agent_url = "http://agent.test:9079"  # type: ignore[attr-defined]
    service._cache.gateway_node = "gpu-a"
    service._cache.gateway_node_at = time.perf_counter()
    seen: list[httpx.Request] = []
    answer = {"status": 200}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if answer["status"] != 200:
            return httpx.Response(answer["status"], json={"detail": "off"})
        return httpx.Response(
            200,
            json={"groups": [{"model": "qwen", "requests": 30, "errors": 12},
                             {"model": "gemma", "requests": 10, "errors": 0}]},
        )  # fmt: skip

    service._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    rules = [alerts._rule(r) for r in alerts.DEFAULT_RULES if r["kind"] == "failure-rate"]
    readings = asyncio.run(service.read(rules, []))
    assert readings.failures == {10: {"qwen": (30, 12), "gemma": (10, 0), None: (40, 12)}}
    request = seen[0]
    assert str(request.url).startswith("http://agent.test:9079/api/proxy/gateway/v1/metrics?")
    claims = jwt.decode(
        request.headers["authorization"].removeprefix("Bearer "),
        options={"verify_signature": False},
    )
    assert (claims["iss"], claims["sub"], claims["aud"]) == ("control", "control", ["node:gpu-a"])
    answer["status"] = 503
    refused = asyncio.run(service.read(rules, []))
    assert refused.failures is None
    assert refused.problems == ("Gateway metrics: request metrics are off on the gateway",)
