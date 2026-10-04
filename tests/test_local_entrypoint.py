"""The root reaches its own supervising agent without public DNS or hairpin NAT."""

from __future__ import annotations

import httpx

from eugene_plexus_control.nodes_client import NodesClient


async def test_only_the_exact_configured_origin_uses_loopback_with_auth_unchanged():
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(200, json={"epoch": 1, "devices": [], "ok": True})

    client = NodesClient(
        timeout_provider=lambda: 5,
        local_agent_url="http://127.0.0.1:8079",
        local_agent_public_origin="https://eugene.example:8443",
    )
    await client._client.aclose()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(respond), trust_env=False)
    try:
        own = "https://eugene.example:8443/"
        assert (await client.probe("root", own, "for-root")).reachable
        await client.post_json(own, "/v1/trust/bundle", "for-root", {"bundle": "signed"})
        await client.create_runtime(own, "for-root", {"name": "model"})
        await client.probe("worker", "https://worker.example:8443", "for-worker")
        await client.probe("another", "https://eugene.example:9443", "for-another")
        assert [str(call.url) for call in calls] == [
            "http://127.0.0.1:8079/v1/node",
            "http://127.0.0.1:8079/v1/trust/bundle",
            "http://127.0.0.1:8079/v1/runtimes",
            "https://worker.example:8443/v1/node",
            "https://eugene.example:9443/v1/node",
        ]
        assert [call.headers["authorization"] for call in calls] == [
            "Bearer for-root",
            "Bearer for-root",
            "Bearer for-root",
            "Bearer for-worker",
            "Bearer for-another",
        ]
    finally:
        await client.aclose()
