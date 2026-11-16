"""Row 40: what each key, person and group used, their allowances, and the prices.

The root counts every served request of the install, whichever gateway served
it. A gateway says what a request used when it releases the request's lease
(`AdmissionUsage`); the root prices it, and the release's own log entry
carries the count (`putClientAdmission`'s `usage`), so a standby counts the
same and a promotion keeps the counters.

**Money is exact.** A price is a whole number of millionths of a dollar for a
million tokens, which is one pico-dollar a token, so a request's cost is the
integer `prompt * input + completion * output` pico-dollars. Counters keep
pico-dollars; the API says dollars as a decimal string with twelve places.
Nothing here is a float.

**Whose price list.** The owner's rate for a model, kept here (`setModelPrice`),
wins. Without one, the provider's published price as the serving gateway's
driver reported it (`ModelPriceHint.providerPrice`: OpenRouter's listing, or
Eugene's table for OpenAI, Anthropic and Gemini). Without either, a model every
backend of which is local costs nothing, and any other model has an unknown
cost: it is counted as unpriced, never as free, and a money allowance refuses
it.

**Who is counted.** The key's lineage always (a rotated key shares its first
key's counters), the person the root verified at `acquire`, and each of that
person's groups then. An allowance is checked before a request runs; a request
that crosses the line while it runs finishes, and the next is refused.

Counters are kept per UTC day for `DAYS_KEPT` days and per calendar month for
`MONTHS_KEPT` months, and each holds at most `MODELS_KEPT` models (the rest
under `*`), so the replicated state stays bounded.
"""

from __future__ import annotations

import calendar
import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

from .client_admission import USAGE_LIMITS, AdmissionRefusal, lineage_of, validate_allowances

DAYS_KEPT = 62
MONTHS_KEPT = 13
MODELS_KEPT = 32
OTHER_MODELS = "*"
#: Running requests whose verified person is remembered until their release.
MAX_OPEN = 50_000
#: A request counts for its key, its person and at most this many groups.
MAX_GROUPS = 64
PICOS_PER_MICRO = 10**6
PICOS_PER_DOLLAR = 10**12
#: Row 40 raises the log version: `setModelPrice`, and allowances in a key's
#: limits or a group's policies, stop an older standby's copy.
LOG_VERSION_NEEDED = 2
#: The largest rate, in micro-dollars a million tokens ($1,000,000).
MAX_RATE = 10**12
TOTALS = ("requests", "promptTokens", "completionTokens", "costPicos", "unpriced")
#: The group policies this module owns (`GroupPolicies`), the same names as a
#: key's allowances.
GROUP_ALLOWANCES = USAGE_LIMITS


# --------------------------------------------------------------------------- #
# periods
# --------------------------------------------------------------------------- #


def day_of(when: datetime) -> str:
    return when.astimezone(UTC).date().isoformat()


def month_of(when: datetime) -> str:
    moment = when.astimezone(UTC)
    return f"{moment.year:04d}-{moment.month:02d}"


def _month_index(month: str) -> int:
    year, number = month.split("-")
    return int(year) * 12 + int(number) - 1


def _next_month(month: str) -> date:
    index = _month_index(month) + 1
    return date(index // 12, index % 12 + 1, 1)


def resets_at(kind: str, day: str, month: str) -> datetime:
    """When an allowance of `kind` counted in `day` or `month` starts again."""
    if kind == "tokensPerDay":
        start = date.fromisoformat(day) + timedelta(days=1)
    else:
        start = _next_month(month)
    return datetime(start.year, start.month, start.day, tzinfo=UTC)


def _valid_day(text: Any) -> bool:
    try:
        return isinstance(text, str) and date.fromisoformat(text).isoformat() == text
    except ValueError:
        return False


def _valid_month(text: Any) -> bool:
    if not isinstance(text, str) or len(text) != 7 or text[4] != "-":
        return False
    try:
        year, number = int(text[:4]), int(text[5:])
    except ValueError:
        return False
    return year >= 1 and 1 <= number <= 12 and text == f"{year:04d}-{number:02d}"


# --------------------------------------------------------------------------- #
# prices and cost
# --------------------------------------------------------------------------- #


def valid_price(price: Any) -> dict[str, int]:
    if (
        not isinstance(price, dict)
        or set(price) != {"inputPerMillionMicros", "outputPerMillionMicros"}
        or any(
            type(price[k]) is not int or not 0 <= price[k] <= MAX_RATE
            for k in ("inputPerMillionMicros", "outputPerMillionMicros")
        )
    ):
        raise ValueError("invalid model price")
    return {k: int(price[k]) for k in ("inputPerMillionMicros", "outputPerMillionMicros")}


def validate_prices(raw: Any) -> dict[str, dict[str, int]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict) or len(raw) > 10_000:
        raise ValueError("invalid model prices")
    out: dict[str, dict[str, int]] = {}
    for model, price in raw.items():
        if not isinstance(model, str) or not 1 <= len(model) <= 256:
            raise ValueError("invalid priced model")
        out[model] = valid_price(price)
    return out


def _hint_price(hint: Any) -> dict[str, int] | None:
    found = hint.get("providerPrice") if isinstance(hint, dict) else None
    try:
        return valid_price(found) if found is not None else None
    except ValueError:
        return None


def price_of(
    prices: dict[str, dict[str, int]], model: str, hint: Any
) -> tuple[dict[str, int] | None, str]:
    """The rate `model` is counted at, and whose it is: `owner`, `provider`,
    `local` (no rate, every backend local: free) or `unknown` (no rate)."""
    owner = prices.get(model)
    if owner is not None:
        return owner, "owner"
    provider = _hint_price(hint)
    if provider is not None:
        return provider, "provider"
    if isinstance(hint, dict) and hint.get("locality") == "local":
        return {"inputPerMillionMicros": 0, "outputPerMillionMicros": 0}, "local"
    return None, "unknown"


def cost_picos(price: dict[str, int] | None, prompt: int, completion: int) -> int | None:
    """Exact cost in pico-dollars, or None when the price is unknown."""
    if price is None:
        return None
    return prompt * price["inputPerMillionMicros"] + completion * price["outputPerMillionMicros"]


def dollars(picos: int) -> str:
    """Exact US dollars with twelve places, as the API says them."""
    return f"{Decimal(picos).scaleb(-12):.12f}"


def money_words(picos: int) -> str:
    """$4.00, or as many places as it takes: $0.010512."""
    amount = Decimal(picos).scaleb(-12).normalize()
    exponent = amount.as_tuple().exponent
    places = max(2, -exponent) if isinstance(exponent, int) and amount else 2
    return f"${amount:,.{places}f}"


# --------------------------------------------------------------------------- #
# the replicated counters
# --------------------------------------------------------------------------- #


def empty() -> dict[str, Any]:
    return {"counters": {}, "open": {}, "pruned": ""}


def _nonnegative(value: Any) -> bool:
    return type(value) is int and value >= 0


def _check_totals(totals: Any, *, with_models: bool) -> None:
    keys = set(TOTALS) | ({"models"} if with_models else set())
    if not isinstance(totals, dict) or set(totals) != keys:
        raise ValueError("invalid usage totals")
    if not all(_nonnegative(totals[k]) for k in TOTALS):
        raise ValueError("invalid usage count")
    if with_models:
        models = totals["models"]
        if not isinstance(models, dict) or len(models) > MODELS_KEPT + 1:
            raise ValueError("invalid usage models")
        for model, inner in models.items():
            if not isinstance(model, str) or not model:
                raise ValueError("invalid usage model")
            _check_totals(inner, with_models=False)


def validate(raw: Any) -> dict[str, Any]:
    """The counters as a snapshot holds them (`clientUsage`), checked."""
    if raw is None:
        return empty()
    if not isinstance(raw, dict) or set(raw) != {"counters", "open", "pruned"}:
        raise ValueError("invalid usage state")
    counters, opened, pruned = raw["counters"], raw["open"], raw["pruned"]
    if not isinstance(counters, dict) or not isinstance(opened, dict):
        raise ValueError("invalid usage state")
    if pruned != "" and not _valid_day(pruned):
        raise ValueError("invalid usage pruning day")
    for subject, periods in counters.items():
        if not isinstance(subject, str) or subject.split(":", 1)[0] not in (
            "key",
            "person",
            "group",
        ):
            raise ValueError("invalid usage subject")
        if not isinstance(periods, dict) or len(periods) > DAYS_KEPT + MONTHS_KEPT + 2:
            raise ValueError("invalid usage periods")
        for period, totals in periods.items():
            if not isinstance(period, str) or not (
                (period.startswith("d:") and _valid_day(period[2:]))
                or (period.startswith("m:") and _valid_month(period[2:]))
            ):
                raise ValueError("invalid usage period")
            _check_totals(totals, with_models=True)
    if len(opened) > MAX_OPEN:
        raise ValueError("usage state exceeds its bound")
    for request_id, held in opened.items():
        _check_open(request_id, held)
    return raw


def _check_open(request_id: Any, held: Any) -> None:
    if (
        not isinstance(request_id, str)
        or not isinstance(held, dict)
        or set(held) != {"person", "groups"}
        or not isinstance(held["person"], str)
        or not held["person"]
        or not isinstance(held["groups"], list)
        or len(held["groups"]) > MAX_GROUPS
        or any(not isinstance(g, str) or not g for g in held["groups"])
    ):
        raise ValueError("invalid open usage")


def _zero(*, with_models: bool) -> dict[str, Any]:
    out: dict[str, Any] = dict.fromkeys(TOTALS, 0)
    if with_models:
        out["models"] = {}
    return out


def _added(totals: dict[str, Any] | None, delta: dict[str, int], model: str) -> dict[str, Any]:
    new = dict(totals) if totals is not None else _zero(with_models=True)
    for k in TOTALS:
        new[k] = new[k] + delta[k]
    models = dict(new["models"])
    name = model if model in models or len(models) < MODELS_KEPT else OTHER_MODELS
    inner = dict(models.get(name) or _zero(with_models=False))
    for k in TOTALS:
        inner[k] = inner[k] + delta[k]
    models[name] = inner
    new["models"] = models
    return new


def _pruned(periods: dict[str, Any], day: str, month: str) -> dict[str, Any]:
    oldest_day = (date.fromisoformat(day) - timedelta(days=DAYS_KEPT - 1)).isoformat()
    oldest_month = _month_index(month) - (MONTHS_KEPT - 1)
    return {
        period: totals
        for period, totals in periods.items()
        if (period.startswith("d:") and period[2:] >= oldest_day)
        or (period.startswith("m:") and _month_index(period[2:]) >= oldest_month)
    }


def opened(usage: dict[str, Any], record: Any) -> dict[str, Any]:
    """Remember the person (and their groups) an acquire verified, by request."""
    request_id, held = (
        record.get("requestId"),
        {
            "person": record.get("person"),
            "groups": record.get("groups"),
        },
    )
    _check_open(request_id, held)
    if len(usage["open"]) >= MAX_OPEN and request_id not in usage["open"]:
        raise ValueError("usage state exceeds its bound")
    return {**usage, "open": {**usage["open"], request_id: held}}


def counted(usage: dict[str, Any], record: Any) -> dict[str, Any]:
    """Add one released request's use to each of its subjects.

    `record` is what the writer put in the log: the subjects, the UTC day and
    month it was counted in, the model that served it, its tokens and its cost
    in pico-dollars (None: unknown). Pure, so every replica adds the same."""
    if not isinstance(record, dict):
        raise ValueError("invalid usage record")
    day, month, model = record.get("day"), record.get("month"), record.get("model")
    subjects = record.get("subjects")
    prompt, completion = record.get("promptTokens"), record.get("completionTokens")
    cost = record.get("costPicos")
    if (
        not _valid_day(day)
        or not _valid_month(month)
        or not isinstance(model, str)
        or not model
        or not isinstance(subjects, list)
        or not 1 <= len(subjects) <= MAX_GROUPS + 2
        or len(set(subjects)) != len(subjects)
        or not _nonnegative(prompt)
        or not _nonnegative(completion)
        or not (cost is None or _nonnegative(cost))
    ):
        raise ValueError("invalid usage record")
    day, month = str(day), str(month)
    delta: dict[str, int] = {
        "requests": 1,
        "promptTokens": int(prompt or 0),
        "completionTokens": int(completion or 0),
        "costPicos": int(cost or 0),
        "unpriced": 1 if cost is None else 0,
    }
    counters = dict(usage["counters"])
    pruned = str(usage.get("pruned") or "")
    if day > pruned:
        # Once a day, every subject loses what is past keeping.
        counters = {s: kept for s, p in counters.items() if (kept := _pruned(p, day, month))}
        pruned = day
    for subject in subjects:
        if not isinstance(subject, str) or subject.split(":", 1)[0] not in (
            "key",
            "person",
            "group",
        ):
            raise ValueError("invalid usage subject")
        periods = dict(counters.get(subject) or {})
        periods[f"d:{day}"] = _added(periods.get(f"d:{day}"), delta, model)
        periods[f"m:{month}"] = _added(periods.get(f"m:{month}"), delta, model)
        counters[subject] = periods
    request_id = record.get("requestId")
    remaining = {k: v for k, v in usage["open"].items() if k != request_id}
    return {"counters": counters, "open": remaining, "pruned": pruned}


def settled(usage: dict[str, Any], ledger: dict[str, Any]) -> dict[str, Any]:
    """Forget the person of a request whose lease is gone from the ledger
    (it expired, or a release that carried no usage ended it)."""
    if not usage["open"]:
        return usage
    live = {
        rid for bucket in ledger["buckets"].values() for rid, e in bucket.items() if e["active"]
    }
    kept = {rid: held for rid, held in usage["open"].items() if rid in live}
    return usage if len(kept) == len(usage["open"]) else {**usage, "open": kept}


def totals_of(usage: dict[str, Any], subject: str, period: str) -> dict[str, Any]:
    found = usage["counters"].get(subject, {}).get(period)
    return found if found is not None else _zero(with_models=True)


# --------------------------------------------------------------------------- #
# allowances
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Allowance:
    subject: str
    kind: str
    limit: int
    #: The words for whose it is: "This key", "Your group Sales".
    whose: str
    #: For a person's allowance, the group that sets it.
    set_by: str | None = None

    def used(self, usage: dict[str, Any], day: str, month: str) -> int:
        period = f"d:{day}" if self.kind == "tokensPerDay" else f"m:{month}"
        totals = totals_of(usage, self.subject, period)
        if self.kind == "costPerMonthMicros":
            return int(totals["costPicos"])
        return int(totals["promptTokens"]) + int(totals["completionTokens"])

    def spent(self, usage: dict[str, Any], day: str, month: str, *, costs: bool = True) -> bool:
        """Whether the next request is refused for it. A money allowance never
        refuses a request that costs nothing (`costs` False)."""
        used = self.used(usage, day, month)
        if self.kind == "costPerMonthMicros":
            return costs and used >= self.limit * PICOS_PER_MICRO
        return used >= self.limit

    def words(self, usage: dict[str, Any], day: str, month: str) -> str:
        used = self.used(usage, day, month)
        when = resets_at(self.kind, day, month)
        if self.kind == "costPerMonthMicros":
            amount = f"{money_words(used)} of its {money_words(self.limit * PICOS_PER_MICRO)}"
        else:
            amount = f"{used:,} of its {self.limit:,} tokens"
        if self.kind == "tokensPerDay":
            span = f"{date.fromisoformat(day).day} {calendar.month_name[int(day[5:7])]}"
            resets = f"at 00:00 UTC on {when.day} {calendar.month_name[when.month]}"
        else:
            span = calendar.month_name[int(month[5:])]
            resets = f"on {when.day} {calendar.month_name[when.month]}"
        return f"{self.whose} used {amount} for {span}. It resets {resets}."


def groups_of(state: Any, person_id: str | None) -> list[dict[str, Any]]:
    """The person's groups now, by name, at most `MAX_GROUPS`."""
    person = state.people.get(person_id or "")
    ids = (person or {}).get("groups") or []
    found = [state.permission_groups[g] for g in ids if g in state.permission_groups]
    return sorted(found, key=lambda g: (g["name"].casefold(), g["id"]))[:MAX_GROUPS]


def key_allowances(key: dict[str, Any] | None, lineage: str) -> list[Allowance]:
    limits = validate_allowances((key or {}).get("limits") or {})
    return [
        Allowance(f"key:{lineage}", kind, limit, "This key")
        for kind, limit in sorted(limits.items(), key=lambda kv: USAGE_LIMITS.index(kv[0]))
    ]


def group_allowances(groups: Iterable[dict[str, Any]]) -> dict[str, list[Allowance]]:
    """For each allowance kind, every group of the person that sets it."""
    out: dict[str, list[Allowance]] = {}
    for g in groups:
        policies = g.get("policies") or {}
        for kind in GROUP_ALLOWANCES:
            if (limit := policies.get(kind)) is not None:
                out.setdefault(kind, []).append(
                    Allowance(
                        f"group:{g['id']}", kind, int(limit), f"Your group {g['name']}", g["id"]
                    )
                )
    return out


def _seconds_until(when: datetime, now: datetime) -> int:
    return max(1, math.ceil((when - now).total_seconds()))


def refuse_if_spent(
    state: Any,
    *,
    key: dict[str, Any] | None,
    key_id: str,
    person_id: str | None,
    model: str,
    hint: Any,
    now: datetime,
    money: bool = True,
) -> None:
    """Raise the `allowance` refusal when an allowance that applies is spent.

    `money` False checks the token allowances only (a `check` on arrival,
    which has no price for the model yet).

    The key's own allowances each apply. For a person, each kind of allowance
    their groups set applies as the most generous of those groups: refused
    only when every group that sets it has spent it. While any money allowance
    applies, a model with no price is refused, because its cost cannot be
    counted; a model that costs nothing never meets a money allowance."""
    usage = state.client_usage
    day, month = day_of(now), month_of(now)
    price, _ = price_of(state.model_prices, model, hint)
    costs = (
        price is None or price["inputPerMillionMicros"] > 0 or price["outputPerMillionMicros"] > 0
    )
    lineage = lineage_of(key, key_id)
    own = key_allowances(key, lineage)
    by_kind = group_allowances(groups_of(state, person_id)) if person_id else {}
    if not money:
        own = [a for a in own if a.kind != "costPerMonthMicros"]
        by_kind.pop("costPerMonthMicros", None)
    for allowance in own:
        if allowance.spent(usage, day, month, costs=costs):
            _refuse(allowance, usage, day, month, now)
    for kind in GROUP_ALLOWANCES:
        setting = by_kind.get(kind) or []
        if setting and all(a.spent(usage, day, month, costs=costs) for a in setting):
            _refuse(setting[0], usage, day, month, now)
    money = any(a.kind == "costPerMonthMicros" for a in own) or bool(
        by_kind.get("costPerMonthMicros")
    )
    if money and price is None:
        raise AdmissionRefusal(
            429,
            f"No price is set for {model}, so a money limit cannot allow it. "
            "The owner sets one in Models.",
            None,
            reason="allowance",
        )


def _refuse(
    allowance: Allowance, usage: dict[str, Any], day: str, month: str, now: datetime
) -> None:
    when = resets_at(allowance.kind, day, month)
    raise AdmissionRefusal(
        429,
        allowance.words(usage, day, month),
        _seconds_until(when, now),
        reason="allowance",
    )


def subjects_for(
    state: Any, key: dict[str, Any] | None, key_id: str, held: dict[str, Any] | None
) -> list[str]:
    """Whom a released request counts for: its key's lineage, and the person
    (and their groups) the acquire verified."""
    out = [f"key:{lineage_of(key, key_id)}"]
    if held:
        out.append(f"person:{held['person']}")
        out.extend(f"group:{g}" for g in held["groups"])
    return out


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #


def _totals_view(totals: dict[str, Any]) -> dict[str, Any]:
    models = sorted(totals["models"].items(), key=lambda kv: (-kv[1]["requests"], kv[0]))
    return {
        "requests": totals["requests"],
        "promptTokens": totals["promptTokens"],
        "completionTokens": totals["completionTokens"],
        "tokens": totals["promptTokens"] + totals["completionTokens"],
        "costUsd": dollars(totals["costPicos"]),
        "unpricedRequests": totals["unpriced"],
        "models": [
            {
                "model": model,
                "requests": inner["requests"],
                "promptTokens": inner["promptTokens"],
                "completionTokens": inner["completionTokens"],
                "costUsd": dollars(inner["costPicos"]),
                "unpricedRequests": inner["unpriced"],
            }
            for model, inner in models
        ],
    }


def _sum(usage: dict[str, Any], subject: str, days: list[str]) -> dict[str, Any]:
    total = _zero(with_models=True)
    for day in days:
        found = usage["counters"].get(subject, {}).get(f"d:{day}")
        if found is None:
            continue
        for model, inner in found["models"].items():
            total = _added(total, {k: inner[k] for k in TOTALS}, model)
    return total


def _allowance_view(a: Allowance, usage: dict[str, Any], day: str, month: str) -> dict[str, Any]:
    used = a.used(usage, day, month)
    if a.kind == "costPerMonthMicros":
        limit = a.limit * PICOS_PER_MICRO
        view: dict[str, Any] = {
            "kind": "costPerMonth",
            "limit": dollars(limit),
            "used": dollars(used),
            "left": dollars(max(0, limit - used)),
        }
    else:
        view = {
            "kind": a.kind,
            "limit": str(a.limit),
            "used": str(used),
            "left": str(max(0, a.limit - used)),
        }
    view["spent"] = a.spent(usage, day, month)
    view["resetsAt"] = resets_at(a.kind, day, month).isoformat().replace("+00:00", "Z")
    if a.set_by is not None:
        view["setBy"] = a.set_by
    return view


def key_name(state: Any, lineage: str) -> str:
    """The name of the newest key of a lineage, or the lineage's id."""
    found = [
        k
        for kid, k in state.client_keys.items()
        if lineage_of(k, kid) == lineage and not k.get("revokedAt")
    ] or [k for kid, k in state.client_keys.items() if lineage_of(k, kid) == lineage]
    if not found:
        return lineage
    return str(max(found, key=lambda k: str(k.get("createdAt") or ""))["name"])


def subject_view(
    state: Any,
    subject: str,
    *,
    day: str,
    month: str,
    days: list[str] | None,
    allowances: list[Allowance],
) -> dict[str, Any]:
    usage = state.client_usage
    kind, _, ident = subject.partition(":")
    if kind == "key":
        name = key_name(state, ident)
    elif kind == "person":
        name = str((state.people.get(ident) or {}).get("name") or ident)
    else:
        name = str((state.permission_groups.get(ident) or {}).get("name") or ident)
    view: dict[str, Any] = {
        "kind": kind,
        "id": ident,
        "name": name,
        "day": _totals_view(totals_of(usage, subject, f"d:{day}")),
        "month": _totals_view(totals_of(usage, subject, f"m:{month}")),
        "allowances": [_allowance_view(a, usage, day, month) for a in allowances],
    }
    if days is not None:
        view["range"] = _totals_view(_sum(usage, subject, days))
    return view


def allowances_of(state: Any, subject: str) -> list[Allowance]:
    """The allowances shown beside a subject: a key's own, a group's own, and
    for a person every allowance their groups set."""
    kind, _, ident = subject.partition(":")
    if kind == "key":
        keys = [
            (kid, k)
            for kid, k in state.client_keys.items()
            if lineage_of(k, kid) == ident and not k.get("revokedAt")
        ]
        newest = max(keys, key=lambda kv: str(kv[1].get("createdAt") or ""), default=None)
        return key_allowances(newest[1], ident) if newest else []
    if kind == "group":
        group = state.permission_groups.get(ident)
        if group is None:
            return []
        return [
            Allowance(subject, a.kind, a.limit, a.whose)
            for allowances in group_allowances([group]).values()
            for a in allowances
        ]
    return [a for kinds in group_allowances(groups_of(state, ident)).values() for a in kinds]


def report(
    state: Any,
    subjects: Iterable[str],
    *,
    day: str,
    month: str,
    first: str | None = None,
    last: str | None = None,
) -> dict[str, Any]:
    days: list[str] | None = None
    if first is not None and last is not None:
        start, end = date.fromisoformat(first), date.fromisoformat(last)
        days = [(start + timedelta(days=n)).isoformat() for n in range((end - start).days + 1)]
    out: dict[str, Any] = {
        "day": day,
        "month": month,
        "kept": {"days": DAYS_KEPT, "months": MONTHS_KEPT},
        "subjects": [
            subject_view(
                state,
                subject,
                day=day,
                month=month,
                days=days,
                allowances=allowances_of(state, subject),
            )
            for subject in subjects
        ],
    }
    if days is not None:
        out["from"], out["to"] = first, last
    return out


def known_subjects(state: Any) -> list[str]:
    """Every key lineage, person and group that has counters or an allowance."""
    found = set(state.client_usage["counters"])
    for kid, k in state.client_keys.items():
        if not k.get("revokedAt"):
            found.add(f"key:{lineage_of(k, kid)}")
    found.update(f"person:{p}" for p in state.people)
    found.update(f"group:{g}" for g in state.permission_groups)
    order = {"key": 0, "group": 1, "person": 2}
    return sorted(found, key=lambda s: (order[s.split(":", 1)[0]], s))
