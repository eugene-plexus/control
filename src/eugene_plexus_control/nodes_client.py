"""Asking the nodes. The control root's only outbound direction.

This component spawns nothing, so everything it knows about what is
*running* comes from asking each enrolled host's agent. Three calls:

  * `GET /v1/runtimes` and `GET /v1/components` — assembled into the
    union views.
  * `GET /v1/node` — the node's own identity and, crucially, the epoch
    it has recorded.
  * `POST /v1/runtimes` — a declaration forwarded to the host that will
    actually run it, because the operator should not have to know which
    host owns what.

**A node that does not answer is `down`, not `out`.** Nothing is
reassigned on the strength of a failed request; the node is named in
`unreachableNodes` and the view is honestly partial. A dashboard showing
fewer runtimes than exist, with no indication why, is worse than one
that says which host it could not reach — and a control plane that
un-declared a runtime because a poll timed out would be a control plane
that reacts to a network blip by deleting configuration.

Every node is asked **concurrently**, with a per-request timeout from
config. Sequentially, one unreachable host in another building would
make the whole dashboard wait for it in turn.

`GET /v1/node` may 404 on an agent that predates M5. That is treated as
"this agent has no node identity yet" rather than as an error, because
during a rolling upgrade it is the true answer.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class NodeProbe:
    """What one node said about itself, or why it did not.

    Everything here is an **observation** and is deliberately kept out
    of applied state — a standby cannot have seen what this root saw,
    so replicating any of it would manufacture drift out of two hosts
    honestly reporting different things."""

    name: str
    reachable: bool
    epoch: int | None = None
    agent_version: str | None = None
    os: str | None = None
    arch: str | None = None
    devices: list[dict[str, Any]] | None = None
    last_seen_at: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class NodeCollection:
    """A union view plus the hosts that could not be asked."""

    items: list[dict[str, Any]]
    unreachable: list[str]


class NodesClient:
    """One httpx client for the process, so connections are reused.

    The timeout is re-read from config on every call rather than baked
    into the client, so a `PATCH /v1/config` takes effect without a
    restart — which is the standing rule for anything an operator can
    tune.
    """

    def __init__(self, *, timeout_provider: Any) -> None:
        self._client = httpx.AsyncClient()
        self._timeout_provider = timeout_provider

    async def aclose(self) -> None:
        await self._client.aclose()

    @property
    def _timeout(self) -> float:
        try:
            return float(self._timeout_provider())
        except Exception:
            return 5.0

    async def _get(self, url: str, path: str, token: str | None) -> Any:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        response = await self._client.get(
            f"{url.rstrip('/')}{path}", headers=headers, timeout=self._timeout
        )
        response.raise_for_status()
        return response.json()

    async def probe(self, name: str, url: str | None, token: str | None) -> NodeProbe:
        """Ask one node who it is and which epoch it has recorded.

        The epoch is the interesting field. A node whose `lastSeenEpoch`
        trails this root's is the visible symptom of a host that was
        partitioned during a promotion — bounded, self-healing, and
        surfaced on the dashboard rather than hidden.
        """
        if not url:
            return NodeProbe(name=name, reachable=False, error="no url recorded for this node")
        try:
            body = await self._get(url, "/v1/node", token)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                # An agent that predates M5 has no node surface. It is
                # reachable and supervising; it simply has nothing to
                # say about identity yet.
                return NodeProbe(
                    name=name,
                    reachable=True,
                    last_seen_at=datetime.now(UTC).isoformat(),
                    error="agent has no /v1/node yet",
                )
            return NodeProbe(name=name, reachable=False, error=_describe(exc))
        except (httpx.HTTPError, ValueError) as exc:
            return NodeProbe(name=name, reachable=False, error=_describe(exc))

        if not isinstance(body, dict):
            return NodeProbe(name=name, reachable=False, error="malformed /v1/node response")
        epoch = body.get("epoch")
        devices = body.get("devices")
        return NodeProbe(
            name=name,
            reachable=True,
            epoch=int(epoch) if isinstance(epoch, int) else None,
            agent_version=_str_or_none(body.get("agentVersion")),
            os=_str_or_none(body.get("os")),
            arch=_str_or_none(body.get("arch")),
            devices=devices if isinstance(devices, list) else None,
            last_seen_at=datetime.now(UTC).isoformat(),
        )

    async def probe_all(
        self, targets: list[tuple[str, str | None]], token: str | None
    ) -> dict[str, NodeProbe]:
        results = await asyncio.gather(
            *(self.probe(name, url, token) for name, url in targets), return_exceptions=False
        )
        return {probe.name: probe for probe in results}

    async def collect(
        self,
        targets: list[tuple[str, str | None]],
        token: str | None,
        *,
        path: str,
        key: str,
    ) -> NodeCollection:
        """Ask every node for a list and tag each item with its node.

        `return_exceptions=True` on the gather, because one host
        refusing must not take the view down — the whole reason the
        response has an `unreachableNodes` field.
        """

        async def one(name: str, url: str | None) -> tuple[str, list[dict[str, Any]] | None]:
            if not url:
                return name, None
            try:
                body = await self._get(url, path, token)
            except (httpx.HTTPError, ValueError) as exc:
                log.info("node %s did not answer %s: %s", name, path, _describe(exc))
                return name, None
            if not isinstance(body, dict) or not isinstance(body.get(key), list):
                log.info("node %s returned a malformed %s response", name, path)
                return name, None
            tagged: list[dict[str, Any]] = []
            for item in body[key]:
                if isinstance(item, dict):
                    # `node` is filled from the node we asked, never
                    # from anything the agent declared. An agent
                    # supervises only its own host, so a `node` it
                    # reported could contradict reality; this one
                    # cannot.
                    tagged.append({**item, "node": name})
            return name, tagged

        gathered = await asyncio.gather(*(one(n, u) for n, u in targets), return_exceptions=True)

        items: list[dict[str, Any]] = []
        unreachable: list[str] = []
        for index, result in enumerate(gathered):
            name = targets[index][0]
            if isinstance(result, BaseException):
                log.warning("collecting %s from node %s raised: %s", path, name, result)
                unreachable.append(name)
                continue
            _, payload = result
            if payload is None:
                unreachable.append(name)
            else:
                items.extend(payload)
        return NodeCollection(items=items, unreachable=sorted(unreachable))

    async def post_json(self, url: str, path: str, token: str | None, body: dict[str, Any]) -> Any:
        """POST to one node and raise on anything but success.

        Used by key rotation, where "did this host take the new key" has
        to be a yes or a no — a host that answered 500 is stale in
        exactly the way a host that timed out is stale, and both belong
        in `nodesPending`.
        """
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        response = await self._client.post(
            f"{url.rstrip('/')}{path}", headers=headers, json=body, timeout=self._timeout
        )
        response.raise_for_status()
        try:
            return response.json()
        except ValueError:
            return None

    async def create_runtime(
        self, url: str, token: str | None, spec: dict[str, Any]
    ) -> tuple[int, Any]:
        """Forward a runtime declaration to the node that will run it.

        Returns the agent's status and body rather than raising, because
        the useful failures here are the agent's own: does this model
        path exist, is this port free, does the engine adapter know
        these flags. Only the host can answer those, so this endpoint's
        errors are largely relayed.
        """
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        response = await self._client.post(
            f"{url.rstrip('/')}/v1/runtimes",
            headers=headers,
            json=spec,
            timeout=self._timeout,
        )
        try:
            return response.status_code, response.json()
        except ValueError:
            return response.status_code, {"detail": response.text}


def _describe(exc: BaseException) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}{_problem_words(exc.response)}"
    if isinstance(exc, httpx.TimeoutException):
        return "timed out"
    return f"{type(exc).__name__}: {exc}"


def _problem_words(response: httpx.Response) -> str:
    """`: <title>: <detail>` from a Problem body, or nothing.

    "HTTP 401" alone was the whole record of a node refusing this root's
    tokens as "not yet valid (iat)"; the refusing agent had written the
    reason into the body every time. Bounded, because a body is what the
    far side chose to send.
    """
    try:
        body = response.json()
    except ValueError:
        return ""
    problem = body.get("detail") if isinstance(body, dict) else None
    if not isinstance(problem, dict):
        return ""
    words = ": ".join(
        str(problem[key]) for key in ("title", "detail") if isinstance(problem.get(key), str)
    )
    return f": {words[:240]}" if words else ""


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
