"""Alerts and SLOs, evaluated by the active root (row 38).

`specs/docs/design/rows/row-38.md`. Once a minute the active root reads
what it needs, decides which rules hold, and moves each alert between
pending, firing and resolved. An alert shows in the console (`GET
/v1/alerts`, Home) and, when the owner set a webhook, is posted to it.

**Where it runs.** The design put the evaluator in the root machine's agent.
It runs here instead, in the active control root, for one reason: the reads
it needs already take this root's own service token. Every agent's
`GET /v1/node/history` accepts it (the root polls `GET /v1/node` the same
way), and the gateway's metrics summary takes it on two reads (row 38). An
agent has no credential the gateway or another machine's agent takes for
these, and giving it one would be three new paths into those machines. The
root runs on exactly one machine, moves on a takeover (`start_root_tasks`
runs on a promotion), and does not evaluate while it is a standby or locked.

**What it reads.** The machines' reachability from the node poller
(`node_probes`); each machine's newest disk and GPU temperature readings
from its host history; the gateway's metrics summary through this machine's
agent (`/api/proxy/gateway/...`, a token addressed to the gateway's machine,
which the proxy passes unchanged). A read that fails is named in `problems`
and leaves the alerts it feeds as they were: an unknown is never "clear".

**What it keeps.** The alert state lives in this root's state directory,
not in the replicated log: what a root observed is not replicated
(`nodes_client.NodeProbe`). A root that takes over starts empty and fires
what still holds after its `forMinutes`.

**The webhook** is egress: off until the owner sets one. Each post is tried
three times, in a task of its own, so a slow or failing destination never
holds up an evaluation. Its address is a secret, sealed in the replicated
state, opened here only to post.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import httpx

from . import _private_files, security
from ._http import egress_client, internal_client, is_internal, machine_named

log = logging.getLogger(__name__)

EVALUATE_EVERY_SECONDS = 60.0
"""How often the root evaluates the rules."""
SLO_EVERY_SECONDS = 300.0
"""How often the SLOs are read again: a 30-day window moves slowly, and the
read is the gateway's heaviest."""
COMPONENTS_EVERY_SECONDS = 600.0
"""How often the root asks which machine runs the gateway."""
RECENT_KEPT = 100
WEBHOOK_TRIES = 3
WEBHOOK_TIMEOUT_SECONDS = 10.0
MIN_REQUESTS = 10
"""A failure rate over fewer requests than this is not evaluated as one."""
BURN_LOOKBACK_MINUTES = 60
STATE_FILE = "alerts.json"

#: The rules in effect until the owner saves their own (`AlertingSettings.defaults`).
DEFAULT_RULES: tuple[dict[str, Any], ...] = (
    {"id": "machine-down", "kind": "machine-down", "enabled": True, "threshold": 0,
     "forMinutes": 5, "machine": None},
    {"id": "disk-full", "kind": "disk-full", "enabled": True, "threshold": 90,
     "forMinutes": 10, "machine": None},
    {"id": "gpu-hot", "kind": "gpu-hot", "enabled": True, "threshold": 85,
     "forMinutes": 5, "machine": None},
    {"id": "failure-rate", "kind": "failure-rate", "enabled": True, "threshold": 5,
     "forMinutes": 10, "model": None},
    {"id": "slo-burn", "kind": "slo-burn", "enabled": True, "threshold": 14.4,
     "forMinutes": 5, "sloId": None},
)  # fmt: skip

MACHINE_KINDS = frozenset({"machine-down", "disk-full", "gpu-hot"})


# --------------------------------------------------------------------------- #
# settings
# --------------------------------------------------------------------------- #


def _rule(raw: dict[str, Any]) -> dict[str, Any]:
    """A rule with every field present, the contract's defaults filled in."""
    return {
        "id": str(raw["id"]),
        "kind": str(raw["kind"]),
        "enabled": raw.get("enabled", True) is not False,
        "threshold": float(raw.get("threshold") or 0),
        "forMinutes": int(raw.get("forMinutes") or 5),
        "machine": raw.get("machine") or None,
        "model": raw.get("model") or None,
        "sloId": raw.get("sloId") or None,
    }


def _slo(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(raw["id"]),
        "name": raw.get("name") or None,
        "model": raw.get("model") or None,
        "clientKeyId": raw.get("clientKeyId") or None,
        "windowDays": int(raw.get("windowDays") or 30),
        "successPercent": float(raw["successPercent"]),
        "firstTokenSeconds": raw.get("firstTokenSeconds"),
        "firstTokenPercentile": raw.get("firstTokenPercentile"),
    }


def effective(alerting: dict[str, Any] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The rules and SLO targets in effect: the owner's, or the built-in rules
    and no SLOs until the owner saves any."""
    if alerting is None:
        return [_rule(r) for r in DEFAULT_RULES], []
    return (
        [_rule(r) for r in alerting.get("rules") or []],
        [_slo(s) for s in alerting.get("slos") or []],
    )


def settings_view(alerting: dict[str, Any] | None) -> dict[str, Any]:
    """`AlertingSettings`: what is in effect, with the webhook's host only."""
    rules, slos = effective(alerting)
    configured = bool((alerting or {}).get("sealedWebhookUrl"))
    return {
        "rules": rules,
        "slos": slos,
        "webhook": {
            "configured": configured,
            "host": (alerting or {}).get("webhookHost") if configured else None,
        },
        "defaults": alerting is None,
    }


def webhook_host(url: str) -> str:
    parts = urlsplit(url)
    return parts.hostname or ""


def check_update(rules: list[dict[str, Any]], slos: list[dict[str, Any]]) -> str | None:
    """Why the owner's rules and targets cannot be saved, or None. The contract
    holds each field's range; this holds what spans fields."""
    seen: set[str] = set()
    for rule in rules:
        if rule["id"] in seen:
            return f"Two rules are called {rule['id']!r}. Give each rule its own id."
        seen.add(rule["id"])
        kind = rule["kind"]
        if rule.get("machine") and kind not in MACHINE_KINDS:
            return f"Rule {rule['id']!r}: only a machine rule names a machine."
        if rule.get("model") and kind != "failure-rate":
            return f"Rule {rule['id']!r}: only a failure-rate rule names a model."
        if rule.get("sloId") and kind != "slo-burn":
            return f"Rule {rule['id']!r}: only an slo-burn rule names an SLO."
        if (
            kind in ("disk-full", "failure-rate")
            and not 0 < float(rule.get("threshold") or 0) <= 100
        ):
            return f"Rule {rule['id']!r}: the threshold is a percent, above 0 and at most 100."
        if kind in ("gpu-hot", "slo-burn") and not float(rule.get("threshold") or 0) > 0:
            return f"Rule {rule['id']!r}: set a threshold above 0."
    ids: set[str] = set()
    for slo in slos:
        if slo["id"] in ids:
            return f"Two SLOs are called {slo['id']!r}. Give each SLO its own id."
        ids.add(slo["id"])
        if slo.get("clientKeyId") and slo.get("model"):
            return f"SLO {slo['id']!r}: set a model or a client key, not both."
        if slo.get("clientKeyId") and slo.get("firstTokenSeconds") is not None:
            return f"SLO {slo['id']!r}: a client key's SLO has a success target only."
        if (slo.get("firstTokenSeconds") is None) != (slo.get("firstTokenPercentile") is None):
            return f"SLO {slo['id']!r}: set both the first-token time and its share, or neither."
    for rule in rules:
        if rule.get("sloId") and rule["sloId"] not in ids:
            return f"Rule {rule['id']!r} names SLO {rule['sloId']!r}, which is not set."
    return None


def check_webhook_url(url: str) -> str | None:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return "The webhook must be an http:// or https:// address."
    if any(ord(c) < 0x21 for c in url):
        return "The webhook address has a space or a control character in it."
    return None


# --------------------------------------------------------------------------- #
# readings and conditions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MachineReading:
    name: str
    label: str
    reachable: bool | None
    """None: not polled yet."""
    error: str | None = None
    disks: tuple[tuple[str, float], ...] | None = None
    """(volume, percent used); None when the host history could not be read."""
    gpus: tuple[tuple[str, str, float], ...] | None = None
    """(series key, card name, °C) for each card that reports a temperature."""


@dataclass(frozen=True)
class Readings:
    machines: tuple[MachineReading, ...] = ()
    failures: dict[int, dict[str | None, tuple[int, int]]] | None = None
    """Look-back minutes -> model (None: every model) -> (requests, failed).
    None when the gateway could not be read."""
    slos: tuple[dict[str, Any], ...] | None = None
    """`SloStatus` documents; None when they could not be read."""
    problems: tuple[str, ...] = ()


@dataclass(frozen=True)
class Condition:
    key: str
    rule: dict[str, Any]
    holds: bool | None
    """None: not known this minute. The alert stays as it was."""
    message: str
    value: float | None = None
    machine: str | None = None
    model: str | None = None
    slo_id: str | None = None
    link: str | None = None


def _num(value: float) -> str:
    return f"{value:.0f}" if abs(value - round(value)) < 0.05 else f"{value:.1f}"


def complete(rules: list[dict[str, Any]], readings: Readings) -> set[tuple[str, str | None]]:
    """The (rule, machine) pairs read whole this minute: an alert of one of
    them that `conditions` did not name again (a volume gone, an SLO removed)
    is clear, not unknown."""
    out: set[tuple[str, str | None]] = set()
    for rule in rules:
        kind = rule["kind"]
        if kind == "slo-burn" and readings.slos is not None:
            out.add((rule["id"], None))
        for m in readings.machines:
            known = {
                "machine-down": m.reachable is not None,
                "disk-full": m.disks is not None,
                "gpu-hot": m.gpus is not None,
            }.get(kind, False)
            if known:
                out.add((rule["id"], m.name))
    return out


def conditions(rules: list[dict[str, Any]], readings: Readings) -> list[Condition]:
    """What each enabled rule says about each thing it watches, this minute."""
    out: list[Condition] = []
    for rule in rules:
        if not rule["enabled"]:
            continue
        kind, rid, threshold = rule["kind"], rule["id"], rule["threshold"]
        if kind in MACHINE_KINDS:
            for m in readings.machines:
                if rule["machine"] and m.name != rule["machine"]:
                    continue
                link = f"/hosts?node={quote(m.name)}"
                if kind == "machine-down":
                    out.append(
                        Condition(
                            key=f"{rid}:{m.name}",
                            rule=rule,
                            holds=None if m.reachable is None else not m.reachable,
                            message=f"{m.label} does not answer"
                            + (f": {m.error}" if m.error else ""),
                            machine=m.name,
                            link=link,
                        )
                    )
                elif kind == "disk-full" and m.disks is not None:
                    for volume, used in m.disks:
                        out.append(
                            Condition(
                                key=f"{rid}:{m.name}:{volume}",
                                rule=rule,
                                holds=used >= threshold,
                                message=f"Disk {volume} on {m.label} is {_num(used)}% full",
                                value=round(used, 1),
                                machine=m.name,
                                link=link,
                            )
                        )
                elif kind == "gpu-hot" and m.gpus is not None:
                    for key, name, celsius in m.gpus:
                        out.append(
                            Condition(
                                key=f"{rid}:{m.name}:{key}",
                                rule=rule,
                                holds=celsius >= threshold,
                                message=f"{name} on {m.label} is at {_num(celsius)} °C",
                                value=round(celsius, 1),
                                machine=m.name,
                                link=link,
                            )
                        )
        elif kind == "failure-rate":
            model = rule["model"]
            minutes = rule["forMinutes"]
            what = model or "all models"
            window = None if readings.failures is None else readings.failures.get(minutes)
            requests, failed = (0, 0) if window is None else window.get(model, (0, 0))
            share = 100.0 * failed / requests if requests else 0.0
            out.append(
                Condition(
                    key=f"{rid}:{model or '*'}",
                    rule=rule,
                    holds=None
                    if window is None
                    else requests >= MIN_REQUESTS and share >= threshold,
                    message=f"{_num(share)}% of requests to {what} failed in the last "
                    f"{minutes} minutes ({failed} of {requests})",
                    value=round(share, 2),
                    model=model,
                    link="/metrics",
                )
            )
        elif kind == "slo-burn":
            for slo in readings.slos or ():
                if rule["sloId"] and slo["id"] != rule["sloId"]:
                    continue
                burn = slo.get("burnRate")
                name = slo.get("name") or slo["id"]
                out.append(
                    Condition(
                        key=f"{rid}:{slo['id']}",
                        rule=rule,
                        holds=burn is not None and burn >= threshold,
                        message=f"SLO {name} is spending its error budget "
                        f"{_num(burn or 0)} times faster than its window allows",
                        value=None if burn is None else round(burn, 2),
                        model=slo.get("model"),
                        slo_id=slo["id"],
                        link="/dashboards#slos",
                    )
                )
    return out


# --------------------------------------------------------------------------- #
# the alert book
# --------------------------------------------------------------------------- #


def _iso(at: datetime) -> str:
    return at.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class AlertBook:
    """Each alert's state, and the resolved ones, kept in one file.

    A rule fires after its condition held for `forMinutes` (a failure rate is
    already a rate over that many minutes, so it fires at once) and resolves
    after the condition was clear for `forMinutes`, so a value that flaps
    neither fires nor resolves every minute."""

    def __init__(self, path: Path | None) -> None:
        self._path = path
        self.active: dict[str, dict[str, Any]] = {}
        self.recent: list[dict[str, Any]] = []
        if path is not None and path.is_file():
            try:
                kept = json.loads(path.read_text(encoding="utf-8"))
                self.active = dict(kept.get("active") or {})
                self.recent = list(kept.get("recent") or [])[:RECENT_KEPT]
            except (OSError, ValueError, AttributeError) as exc:
                log.warning(
                    "the alert state in %s could not be read (%s); starting empty", path, exc
                )

    def _save(self) -> None:
        if self._path is None:
            return
        try:
            _private_files.write_private_text(
                self._path, json.dumps({"active": self.active, "recent": self.recent})
            )
        except OSError as exc:
            log.warning("could not keep the alert state in %s: %s", self._path, exc)

    def firing(self) -> list[dict[str, Any]]:
        return [_public(a) for a in self.active.values() if a.get("firing")]

    def step(
        self,
        now: datetime,
        found: list[Condition],
        *,
        rules: list[dict[str, Any]],
        machines: set[str],
        whole: set[tuple[str, str | None]] = frozenset(),  # type: ignore[assignment]
    ) -> list[dict[str, Any]]:
        """Move every alert one minute on; return the alerts that fired or
        resolved, as `Alert` documents, for the webhook. `whole` is what
        `complete` says was read whole."""
        changed: list[dict[str, Any]] = []
        live = {r["id"]: r for r in rules if r["enabled"]}
        named = {c.key for c in found}
        for key, entry in self.active.items():
            rule = live.get(str(entry.get("ruleId")))
            if (
                key not in named
                and rule is not None
                and (rule["id"], entry.get("machine")) in whole
            ):
                found = [
                    *found,
                    Condition(
                        key=key,
                        rule=rule,
                        holds=False,
                        message=str(entry.get("message") or ""),
                        value=entry.get("value"),
                        machine=entry.get("machine"),
                        model=entry.get("model"),
                        slo_id=entry.get("sloId"),
                        link=entry.get("link"),
                    ),
                ]
        for c in found:
            item: dict[str, Any] | None = self.active.get(c.key)
            if c.holds is None:
                continue
            hold = timedelta(
                minutes=0 if c.rule["kind"] == "failure-rate" else c.rule["forMinutes"]
            )
            quiet = timedelta(minutes=c.rule["forMinutes"])
            if c.holds:
                if item is None:
                    item = self.active[c.key] = {"pendingSince": _iso(now), "firing": False}
                item.update(_details(c))
                item["clearSince"] = None
                if not item["firing"] and now - _parse(item["pendingSince"]) >= hold:
                    item["firing"] = True
                    item["startedAt"] = _iso(now)
                    changed.append(_public(item))
            elif item is not None:
                if not item["firing"]:
                    del self.active[c.key]
                    continue
                item["value"] = c.value
                item["clearSince"] = item.get("clearSince") or _iso(now)
                if now - _parse(item["clearSince"]) >= quiet:
                    del self.active[c.key]
                    done = {**_public(item), "state": "resolved", "resolvedAt": _iso(now)}
                    self.recent.insert(0, done)
                    changed.append(done)
        # A rule removed or turned off, or a machine removed: its alerts end
        # quietly. Nothing else that went unread this minute ends.
        for key in list(self.active):
            entry = self.active[key]
            gone = entry.get("ruleId") not in live or (
                entry.get("machine") is not None and entry["machine"] not in machines
            )
            if gone:
                del self.active[key]
                if entry.get("firing"):
                    self.recent.insert(
                        0, {**_public(entry), "state": "resolved", "resolvedAt": _iso(now)}
                    )
        del self.recent[RECENT_KEPT:]
        self._save()
        return changed


def _details(c: Condition) -> dict[str, Any]:
    return {
        "id": c.key,
        "ruleId": c.rule["id"],
        "kind": c.rule["kind"],
        "message": c.message,
        "value": c.value,
        "threshold": None if c.rule["kind"] == "machine-down" else c.rule["threshold"],
        "machine": c.machine,
        "model": c.model,
        "sloId": c.slo_id,
        "link": c.link,
    }


_PUBLIC = ("id", "ruleId", "kind", "message", "value", "threshold", "machine", "model", "sloId",
           "link", "startedAt", "resolvedAt")  # fmt: skip


def _public(entry: dict[str, Any]) -> dict[str, Any]:
    out = {k: entry.get(k) for k in _PUBLIC}
    out["state"] = entry.get("state") or ("firing" if entry.get("firing") else "resolved")
    out["startedAt"] = out["startedAt"] or entry.get("pendingSince")
    return out


# --------------------------------------------------------------------------- #
# SLOs
# --------------------------------------------------------------------------- #


def slo_statuses(
    slos: list[dict[str, Any]],
    *,
    now: datetime,
    hourly: list[dict[str, Any]],
    last_hour: list[dict[str, Any]],
    first_token: dict[str | None, dict[str, Any] | None],
    kept_since: datetime | None,
    clients: list[dict[str, Any]] | None = None,
    clients_last_hour: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """`SloStatus` for each target.

    `hourly` is `GET /v1/metrics?groupBy=model&bucket=hour` over the longest
    window; `last_hour` the same, unbucketed, over the last hour (the burn
    rate); `first_token` each model's (None: all models) `ttftMs` over the
    requests the gateway keeps whole, which begin at `kept_since`; `clients`
    `GET /v1/metrics/clients` over the same kept requests."""
    earliest = min(
        (_parse(g["bucketStart"]) for g in hourly if g.get("bucketStart") and g.get("requests")),
        default=None,
    )
    out: list[dict[str, Any]] = []
    for slo in slos:
        window = timedelta(days=slo["windowDays"])
        start = now - window
        model, key = slo["model"], slo["clientKeyId"]
        allowed_share = 1 - slo["successPercent"] / 100
        key_name = None
        if key:
            row = next((c for c in clients or [] if c.get("clientKeyId") == key), None)
            requests, failed = (int(row["requests"]), int(row["failed"])) if row else (0, 0)
            key_name = row.get("clientKeyName") if row else None
            hour = next((c for c in clients_last_hour or [] if c.get("clientKeyId") == key), None)
            h_requests, h_failed = (int(hour["requests"]), int(hour["failed"])) if hour else (0, 0)
            reach = kept_since or start
        else:
            requests = failed = 0
            for g in hourly:
                if model and g.get("model") != model:
                    continue
                if g.get("bucketStart") and _parse(g["bucketStart"]) < start:
                    continue
                requests += int(g.get("requests") or 0)
                failed += int(g.get("errors") or 0)
            h_requests = sum(
                int(g.get("requests") or 0)
                for g in last_hour
                if not model or g.get("model") == model
            )
            h_failed = sum(
                int(g.get("errors") or 0) for g in last_hour if not model or g.get("model") == model
            )
            reach = earliest or now
        covered = max(0.0, min(window.total_seconds(), (now - max(reach, start)).total_seconds()))
        measured = 100.0 * (requests - failed) / requests if requests else None
        allowed = requests * allowed_share
        if not requests:
            budget = None
        elif allowed > 0:
            budget = 100.0 * (1 - failed / allowed)
        else:
            budget = 100.0 if failed == 0 else -100.0
        burn = (h_failed / h_requests) / allowed_share if h_requests and allowed_share > 0 else None
        first = None
        if slo.get("firstTokenSeconds") is not None and slo.get("firstTokenPercentile"):
            pct = int(slo["firstTokenPercentile"])
            spread = first_token.get(model)
            measured_ms = (spread or {}).get(f"p{pct}")
            kept_from = max(kept_since or start, start)
            first = {
                "percentile": pct,
                "targetSeconds": float(slo["firstTokenSeconds"]),
                "measuredSeconds": None if measured_ms is None else round(measured_ms / 1000, 3),
                "met": None
                if measured_ms is None
                else measured_ms / 1000 <= float(slo["firstTokenSeconds"]),
                "keptDays": round((now - kept_from).total_seconds() / 86400, 2),
            }
        if not requests:
            state = "no-data"
        elif budget is not None and budget <= 0:
            state = "missed"
        elif (budget is not None and budget < 25) or (first is not None and first["met"] is False):
            state = "at-risk"
        else:
            state = "met"
        out.append(
            {
                "id": slo["id"],
                "name": slo["name"],
                "model": model,
                "clientKeyId": key,
                "clientKeyName": key_name,
                "windowDays": slo["windowDays"],
                "successPercent": slo["successPercent"],
                "requests": requests,
                "failures": failed,
                "measuredPercent": None if measured is None else round(measured, 3),
                "budgetLeftPercent": None if budget is None else round(budget, 1),
                "burnRate": None if burn is None else round(burn, 2),
                "coveredDays": round(covered / 86400, 2),
                "firstToken": first,
                "state": state,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# the webhook
# --------------------------------------------------------------------------- #


def webhook_post(alert: dict[str, Any], *, state: str, console: str | None) -> dict[str, Any]:
    """`AlertWebhookPost`. `text` and `content` carry the one line for Slack,
    Discord and ntfy; the rest is for anything that reads fields."""
    words = {"firing": "Firing", "resolved": "Resolved", "test": "Test"}[state]
    line = f"{words}: {alert['message']}"
    link = alert.get("link")
    return {
        "rule": alert.get("ruleId") or "test",
        "state": state,
        "value": alert.get("value"),
        "threshold": alert.get("threshold"),
        "machine": alert.get("machine"),
        "model": alert.get("model"),
        "url": f"{console.rstrip('/')}{link}" if console and link else None,
        "at": _iso(datetime.now(UTC)),
        "text": line,
        "content": line,
    }


def _cause(exc: BaseException) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    if isinstance(exc, httpx.TimeoutException):
        return "no answer in time"
    if isinstance(exc, httpx.ConnectError):
        text = str(exc).strip()
        return "connection refused" if "refused" in text.lower() else (text or "could not connect")
    if isinstance(exc, httpx.HTTPError):
        return type(exc).__name__
    return type(exc).__name__


class WebhookSender:
    """Posts to the webhook, three tries each, and remembers how it went.

    Two clients for the life of the root, chosen per address: a webhook on
    this LAN (Home Assistant) goes direct, one on the internet (Slack) goes
    through the person's proxy, as every other egress does."""

    def __init__(
        self,
        *,
        internal: httpx.AsyncClient | None = None,
        egress: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._internal = internal
        self._egress = egress
        self._sleep = sleep
        self.last_attempt: str | None = None
        self.last_success: str | None = None
        self.last_error: str | None = None

    def _client(self, url: str) -> httpx.AsyncClient:
        if is_internal(url) or machine_named(url):
            if self._internal is None:
                self._internal = internal_client()
            return self._internal
        if self._egress is None:
            self._egress = egress_client()
        return self._egress

    async def aclose(self) -> None:
        for client in (self._internal, self._egress):
            if client is not None:
                await client.aclose()

    def status(self, configured: bool) -> dict[str, Any]:
        if not configured:
            return {"state": "off", "lastAttemptAt": None, "lastSuccessAt": None, "lastError": None}
        return {
            "state": "failing" if self.last_error else "ok",
            "lastAttemptAt": self.last_attempt,
            "lastSuccessAt": self.last_success,
            "lastError": self.last_error,
        }

    async def post(self, url: str, body: dict[str, Any], *, tries: int = WEBHOOK_TRIES) -> bool:
        cause = "not sent"
        for attempt in range(tries):
            self.last_attempt = _iso(datetime.now(UTC))
            try:
                response = await self._client(url).post(
                    url, json=body, timeout=WEBHOOK_TIMEOUT_SECONDS, follow_redirects=False
                )
                response.raise_for_status()
            except (httpx.HTTPError, ValueError) as exc:
                cause = _cause(exc)
                if attempt + 1 < tries:
                    await self._sleep(2.0**attempt)
                continue
            if self.last_error:
                log.info("the alert webhook answers again")
            self.last_success = _iso(datetime.now(UTC))
            self.last_error = None
            return True
        if self.last_error != cause:
            log.warning("the alert webhook did not answer after %d tries: %s", tries, cause)
        self.last_error = cause
        return False


# --------------------------------------------------------------------------- #
# the service
# --------------------------------------------------------------------------- #


@dataclass
class _Cache:
    gateway_node: str | None = None
    gateway_node_at: float = 0.0
    slos: tuple[dict[str, Any], ...] | None = None
    slos_at: float = 0.0
    slo_problem: str | None = None


@dataclass
class AlertService:
    """The evaluator: one per control process, running only while it is the
    active root and unlocked."""

    app: Any
    state_dir: Path | None
    sender: WebhookSender = field(default_factory=WebhookSender)
    book: AlertBook = field(init=False)
    evaluated_at: str | None = None
    problems: list[str] = field(default_factory=list)
    _cache: _Cache = field(default_factory=_Cache)
    _client: httpx.AsyncClient | None = None
    _deliveries: set[asyncio.Task[None]] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.book = AlertBook(None if self.state_dir is None else self.state_dir / STATE_FILE)

    # ----- what the routes read -----------------------------------------

    def running(self) -> bool:
        machine = self.app.state.machine
        return bool(machine.is_active and self.app.state.auth_state.signing_key is not None)

    def view(self) -> dict[str, Any]:
        configured = bool((self.app.state.machine.state.alerting or {}).get("sealedWebhookUrl"))
        return {
            "evaluating": self.running(),
            "evaluatedAt": self.evaluated_at,
            "problems": list(self.problems),
            "firing": self.book.firing(),
            "recent": list(self.book.recent),
            "slos": list(self._cache.slos or ()),
            "webhook": self.sender.status(configured),
        }

    # ----- reading ----------------------------------------------------

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = internal_client()
        return self._client

    async def aclose(self) -> None:
        for task in list(self._deliveries):
            task.cancel()
        if self._client is not None:
            await self._client.aclose()
        await self.sender.aclose()

    def _token(self, node: str) -> str | None:
        from .routes.nodes import service_token_for

        return service_token_for(self.app.state.auth_state, node)

    async def _gateway_node(self) -> str | None:
        """The machine that runs the gateway, asked every ten minutes."""
        if time.perf_counter() - self._cache.gateway_node_at < COMPONENTS_EVERY_SECONDS and (
            self._cache.gateway_node
        ):
            return self._cache.gateway_node
        from .routes.topology import _targets

        collected = await self.app.state.nodes_client.collect(
            _targets(self.app.state.machine), self._token, path="/v1/components", key="components"
        )
        found = next((i["node"] for i in collected.items if i.get("kind") == "gateway"), None)
        self._cache.gateway_node = found
        self._cache.gateway_node_at = time.perf_counter()
        return found

    async def _gateway(self, path: str) -> Any:
        """One read of the gateway, through this machine's agent's proxy, with a
        token addressed to the gateway's machine (passed through unchanged)."""
        agent = getattr(self.app.state.settings, "agent_url", None)
        if not agent:
            raise LookupError(
                "this root was not started by an agent, so it cannot reach the gateway"
            )
        node = await self._gateway_node()
        if node is None:
            raise LookupError("no machine runs a gateway")
        token = self._token(node)
        if token is None:
            raise LookupError("this root is locked")
        response = await self._http().get(
            f"{agent.rstrip('/')}/api/proxy/gateway{path}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15.0,
        )
        if response.status_code == 503:
            raise LookupError("request metrics are off on the gateway")
        response.raise_for_status()
        return response.json()

    async def _host(self, name: str, url: str | None) -> tuple[tuple, tuple] | None:
        if not url:
            return None
        since = _iso(datetime.now(UTC) - timedelta(minutes=5))
        path = (
            f"/v1/node/history?from={quote(since)}&metric=diskFree&metric=diskTotal"
            "&metric=gpuTemperature"
        )
        try:
            body = await self.app.state.nodes_client.get_json(url, path, self._token(name))
        except (httpx.HTTPError, ValueError):
            return None
        free: dict[str, float] = {}
        total: dict[str, float] = {}
        gpus: list[tuple[str, str, float]] = []
        for series in body.get("series") or []:
            latest = series.get("latest")
            if not isinstance(latest, (int, float)) or series.get("unavailable"):
                continue
            if series.get("metric") == "diskFree" and series.get("volume"):
                free[series["volume"]] = float(latest)
            elif series.get("metric") == "diskTotal" and series.get("volume"):
                total[series["volume"]] = float(latest)
            elif series.get("metric") == "gpuTemperature":
                index = series.get("deviceIndex")
                name_ = series.get("deviceName") or f"GPU {index if index is not None else ''}"
                gpus.append((str(series.get("key")), str(name_).strip(), float(latest)))
        disks = tuple(
            (volume, 100.0 * (1 - free[volume] / size))
            for volume, size in sorted(total.items())
            if size > 0 and volume in free
        )
        return disks, tuple(gpus)

    async def read(self, rules: list[dict[str, Any]], slos: list[dict[str, Any]]) -> Readings:
        from .routes.topology import _targets

        machine = self.app.state.machine
        probes = getattr(self.app.state, "node_probes", {}) or {}
        problems: list[str] = []
        targets = _targets(machine)
        wants_host = any(r["enabled"] and r["kind"] in ("disk-full", "gpu-hot") for r in rules)
        hosts = await asyncio.gather(
            *(
                self._host(n, u)
                if wants_host and (probes.get(n) is None or probes[n].reachable)
                else _none()
                for n, u in targets
            )
        )
        machines: list[MachineReading] = []
        for (name, _), host in zip(targets, hosts, strict=True):
            record = machine.state.nodes[name]
            probe = probes.get(name)
            if wants_host and host is None and (probe is None or probe.reachable):
                problems.append(f"Host history of {record.label or name}: no answer")
            machines.append(
                MachineReading(
                    name=name,
                    label=record.label or name,
                    reachable=None if probe is None else bool(probe.reachable),
                    error=None if probe is None else probe.error,
                    disks=None if host is None else host[0],
                    gpus=None if host is None else host[1],
                )
            )

        read: dict[int, dict[str | None, tuple[int, int]]] = {}
        failures: dict[int, dict[str | None, tuple[int, int]]] | None = read
        lookbacks = {r["forMinutes"] for r in rules if r["enabled"] and r["kind"] == "failure-rate"}
        try:
            for minutes in sorted(lookbacks):
                read[minutes] = await self._failures(minutes)
        except (httpx.HTTPError, ValueError, LookupError) as exc:
            problems.append(f"Gateway metrics: {_reason(exc)}")
            failures = None

        wants_slos = bool(slos)
        if wants_slos and (
            failures is None
            or self._cache.slos is None
            or time.perf_counter() - self._cache.slos_at >= SLO_EVERY_SECONDS
            or {s["id"] for s in self._cache.slos} != {s["id"] for s in slos}
        ):
            try:
                self._cache.slos = tuple(await self._slos(slos))
                self._cache.slo_problem = None
            except (httpx.HTTPError, ValueError, LookupError) as exc:
                self._cache.slos = None
                self._cache.slo_problem = f"SLOs: {_reason(exc)}"
            self._cache.slos_at = time.perf_counter()
        if not wants_slos:
            self._cache.slos = ()
        if self._cache.slo_problem and wants_slos:
            problems.append(self._cache.slo_problem)
        return Readings(
            machines=tuple(machines),
            failures=failures,
            slos=self._cache.slos,
            problems=tuple(problems),
        )

    async def _failures(self, minutes: int) -> dict[str | None, tuple[int, int]]:
        since = _iso(datetime.now(UTC) - timedelta(minutes=minutes))
        body = await self._gateway(f"/v1/metrics?groupBy=model&since={quote(since)}")
        out: dict[str | None, tuple[int, int]] = {}
        total = [0, 0]
        for g in body.get("groups") or []:
            requests, errors = int(g.get("requests") or 0), int(g.get("errors") or 0)
            model = g.get("model")
            before = out.get(model, (0, 0))
            out[model] = (before[0] + requests, before[1] + errors)
            total[0] += requests
            total[1] += errors
        out[None] = (total[0], total[1])
        return out

    async def _slos(self, slos: list[dict[str, Any]]) -> list[dict[str, Any]]:
        now = datetime.now(UTC)
        longest = max(s["windowDays"] for s in slos)
        since = _iso(now - timedelta(days=longest))
        hourly = await self._gateway(f"/v1/metrics?groupBy=model&bucket=hour&since={quote(since)}")
        groups = hourly.get("groups") or []
        raw = [g for g in groups if g.get("source", "raw") == "raw" and g.get("bucketStart")]
        kept_since = min((_parse(g["bucketStart"]) for g in raw), default=None)
        if kept_since is None or not any(g.get("source") == "rollup" for g in groups):
            kept_since = now - timedelta(days=longest)
        hour_ago = quote(_iso(now - timedelta(minutes=BURN_LOOKBACK_MINUTES)))
        last_hour = (await self._gateway(f"/v1/metrics?groupBy=model&since={hour_ago}")).get(
            "groups"
        ) or []
        first_token: dict[str | None, dict[str, Any] | None] = {}
        if any(s.get("firstTokenSeconds") is not None for s in slos):
            kept = quote(_iso(kept_since))
            by_model = await self._gateway(f"/v1/metrics?groupBy=model&since={kept}")
            total = await self._gateway(f"/v1/metrics?groupBy=total&since={kept}")
            for g in by_model.get("groups") or []:
                first_token[g.get("model")] = g.get("ttftMs")
            whole: dict[str, Any] = next(iter(total.get("groups") or []), {})
            first_token[None] = whole.get("ttftMs")
        clients = clients_hour = None
        if any(s.get("clientKeyId") for s in slos):
            kept = quote(_iso(max(kept_since, now - timedelta(days=longest))))
            clients = (await self._gateway(f"/v1/metrics/clients?since={kept}")).get("clients")
            clients_hour = (await self._gateway(f"/v1/metrics/clients?since={hour_ago}")).get(
                "clients"
            )
        return slo_statuses(
            slos,
            now=now,
            hourly=groups,
            last_hour=last_hour,
            first_token=first_token,
            kept_since=kept_since,
            clients=clients,
            clients_last_hour=clients_hour,
        )

    # ----- evaluating -------------------------------------------------

    def _webhook(self) -> str | None:
        alerting = self.app.state.machine.state.alerting or {}
        sealed = alerting.get("sealedWebhookUrl")
        if not sealed:
            return None
        master, _ = self.app.state.auth_state.sealing_key()
        if master is None:
            return None
        try:
            return security.open_b64(sealed, master).decode("utf-8")
        except ValueError:
            log.warning("the alert webhook address does not open with this root's key")
            return None

    def _console(self) -> str | None:
        settings = self.app.state.settings
        origin = getattr(settings, "agent_public_origin", None)
        if origin:
            return str(origin)
        record = self.app.state.machine.state.nodes.get(self.app.state.node_name or "")
        return str(record.url) if record is not None and record.url else None

    async def tick(self, now: datetime | None = None) -> list[dict[str, Any]]:
        """One evaluation. Returns what fired or resolved."""
        now = now or datetime.now(UTC)
        state = self.app.state.machine.state
        rules, slos = effective(state.alerting)
        readings = await self.read(rules, slos)
        found = conditions(rules, readings)
        changed = self.book.step(
            now, found, rules=rules, machines=set(state.nodes), whole=complete(rules, readings)
        )
        self.problems = list(readings.problems)
        self.evaluated_at = _iso(now)
        url = self._webhook()
        if url and changed:
            self._deliver(url, changed)
        return changed

    def _deliver(self, url: str, changed: list[dict[str, Any]]) -> None:
        console = self._console()

        async def send() -> None:
            for alert in changed:
                await self.sender.post(
                    url, webhook_post(alert, state=alert["state"], console=console)
                )

        task = asyncio.create_task(send(), name="control-alert-webhook")
        self._deliveries.add(task)
        task.add_done_callback(self._deliveries.discard)

    async def test(self) -> dict[str, Any]:
        url = self._webhook()
        if url is None:
            raise LookupError("no webhook is set")
        alert = {"message": "Eugene can post alerts here.", "link": "/alerts", "ruleId": "test"}
        await self.sender.post(
            url, webhook_post(alert, state="test", console=self._console()), tries=1
        )
        return self.sender.status(True)

    async def run(self) -> None:
        """Once a minute while this is the active, unlocked root. Never dies."""
        while True:
            try:
                if self.running():
                    await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("alert evaluation failed: %s", exc)
            await asyncio.sleep(EVALUATE_EVERY_SECONDS)


async def _none() -> None:
    return None


def _reason(exc: BaseException) -> str:
    if isinstance(exc, LookupError):
        return str(exc.args[0]) if exc.args else "not available"
    return _cause(exc)


__all__ = [
    "DEFAULT_RULES",
    "AlertBook",
    "AlertService",
    "Condition",
    "MachineReading",
    "Readings",
    "WebhookSender",
    "check_update",
    "check_webhook_url",
    "conditions",
    "effective",
    "settings_view",
    "slo_statuses",
    "webhook_host",
    "webhook_post",
]
