"""Install-wide topology: the union views, and forwarding a declaration.

Two shapes of truth live here and they are not the same thing.

**Declarations** are applied state — what *should* be running where.
They come from the log and survive a host being unreachable.

**The union views** are assembled by asking every enrolled node's agent
what is actually running. They cannot come from the log, because a
process's status is not a fact the control root can know without asking.

A node that does not answer contributes to `unreachableNodes` rather
than silently contributing nothing: a dashboard showing fewer runtimes
than exist, with no indication why, is worse than one that names the
host it could not reach. And nothing is un-declared because a poll
failed — that is `down`, not `out`.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Request, status

from .._generated.models import (
    ComponentPlacement,
    ComponentPlacementList,
    EngineKind,
    RuntimePlacement,
    RuntimePlacementList,
    RuntimePlacementSpec,
)
from ..applied import OP_PUT_RUNTIME, normalize_url
from ..auth_state import AuthState
from ..dependencies import problem, require_authorized, require_operator
from ..state_machine import NotActive, StateMachine

log = logging.getLogger(__name__)

router = APIRouter(tags=["topology"])


@router.get(
    "/v1/components",
    response_model=ComponentPlacementList,
    dependencies=[Depends(require_authorized)],
)
async def list_components(request: Request) -> ComponentPlacementList:
    """Every component in the install, across every node.

    Each entry carries the `node` it runs on, which an agent's own
    `/v1/components` cannot report because it only knows its own host.

    Returns **503** when no node is reachable, which is distinct from an
    empty list: empty means nothing is declared, 503 means we cannot
    tell. Conflating them would have a total network outage look like a
    freshly wiped install.
    """
    machine: StateMachine = request.app.state.machine
    auth: AuthState = request.app.state.auth_state
    client = request.app.state.nodes_client

    from .nodes import service_token_for

    targets = _targets(machine)
    if not targets:
        return ComponentPlacementList(components=[], unreachableNodes=[])

    collected = await client.collect(
        targets,
        lambda node: service_token_for(auth, node),
        path="/v1/components",
        key="components",
    )

    if collected.unreachable and len(collected.unreachable) == len(targets):
        raise problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "No node reachable",
            "None of the "
            f"{len(targets)} enrolled node(s) answered. This is not an empty install — "
            "it is an install we cannot currently see.",
        )

    declared = {(c.node, c.name): c for c in machine.state.components.values()}
    components: list[ComponentPlacement] = []
    seen: set[tuple[str, str]] = set()

    for item in collected.items:
        key = (str(item.get("node")), str(item.get("name")))
        seen.add(key)
        components.append(
            ComponentPlacement.model_validate(
                {
                    "node": key[0],
                    "name": key[1],
                    "kind": item.get("kind"),
                    "url": item.get("url"),
                    # A string here rather than a copy of the agent's
                    # enum: the agent owns component status, and
                    # duplicating its enum in this document would create
                    # two definitions of one fact.
                    "status": str(item.get("status")) if item.get("status") else None,
                }
            )
        )

    # Declared-but-not-reported components are still part of the
    # install. A component on a node that answered but did not list it
    # is a real discrepancy an operator should see, not a row to hide.
    for key, record in declared.items():
        if key in seen or record.node in collected.unreachable:
            continue
        components.append(
            ComponentPlacement.model_validate(
                {
                    "node": record.node,
                    "name": record.name,
                    "kind": record.kind,
                    "url": record.url,
                    "status": "undeclared_on_node",
                }
            )
        )

    return ComponentPlacementList(
        components=sorted(components, key=lambda c: (c.node, c.name)),
        unreachableNodes=collected.unreachable,
    )


@router.get(
    "/v1/runtimes",
    response_model=RuntimePlacementList,
    dependencies=[Depends(require_authorized)],
)
async def list_runtimes(request: Request) -> RuntimePlacementList:
    """The union of every node's runtimes, each tagged with its node.

    `modelAlias` is what a client asks the gateway for, and it is unique
    across the **install** rather than the node: two nodes serving the
    same alias is how replicas are expressed, and the gateway load
    balances across them. So duplicates here are not a bug to
    de-duplicate.
    """
    machine: StateMachine = request.app.state.machine
    auth: AuthState = request.app.state.auth_state
    client = request.app.state.nodes_client

    from .nodes import service_token_for

    targets = _targets(machine)
    if not targets:
        return RuntimePlacementList(runtimes=[], unreachableNodes=[])

    collected = await client.collect(
        targets,
        lambda node: service_token_for(auth, node),
        path="/v1/runtimes",
        key="runtimes",
    )

    runtimes = [
        RuntimePlacement(
            node=str(item.get("node")),
            name=str(item.get("name")),
            modelAlias=item.get("modelAlias"),
            status=str(item.get("status")) if item.get("status") else None,
            url=item.get("url"),
            engine=_engine(item.get("engine")),
        )
        for item in collected.items
        if item.get("name")
    ]
    return RuntimePlacementList(
        runtimes=sorted(runtimes, key=lambda r: (r.node, r.name)),
        unreachableNodes=collected.unreachable,
    )


@router.post(
    "/v1/runtimes",
    response_model=RuntimePlacement,
    status_code=201,
    dependencies=[Depends(require_operator)],
)
async def create_runtime(request: Request, body: RuntimePlacementSpec) -> RuntimePlacement:
    """Declare a runtime on a node, forwarded to that node's agent.

    Accepted here and forwarded, which is what makes "the operator talks
    to the control root" true rather than aspirational. `node` omitted
    means the node this control root runs on, so a single-host install
    never has to name anything.

    Order: **forward first, record second.** Validation that needs the
    host — does this model path exist, is this port free, does the
    engine adapter know these flags — can only happen on the agent, so
    recording a declaration the agent then refused would put a runtime
    in the install's topology that never existed. A 502 means the agent
    refused or was unreachable, and `detail` carries what it said.
    """
    machine: StateMachine = request.app.state.machine
    auth: AuthState = request.app.state.auth_state
    client = request.app.state.nodes_client

    from .nodes import _not_active, service_token_for

    if not machine.is_active:
        raise _not_active(machine)

    node_name = body.node or getattr(request.app.state, "node_name", None)
    if node_name is None:
        raise problem(
            status.HTTP_400_BAD_REQUEST,
            "No target node",
            "This control root does not know which node it runs on, so `node` cannot be "
            "defaulted. Name it explicitly.",
        )
    record = machine.state.nodes.get(node_name)
    if record is None:
        raise problem(
            status.HTTP_404_NOT_FOUND,
            "No such node",
            f"No node named {node_name!r} is enrolled. GET /v1/nodes lists them.",
        )
    if not record.url:
        raise problem(
            status.HTTP_502_BAD_GATEWAY,
            "Node has no address",
            f"Node {node_name!r} has no agent URL recorded, so nothing can be forwarded to it.",
        )

    spec = dict(body.spec)
    name = spec.get("name")
    if not isinstance(name, str) or not name:
        raise problem(
            status.HTTP_400_BAD_REQUEST,
            "Runtime needs a name",
            "The agent's RuntimeSpec requires `name`, and it is what the declaration is "
            "keyed by here.",
        )

    status_code, payload = await client.create_runtime(
        str(record.url), service_token_for(auth, node_name), spec
    )
    if status_code >= 400:
        raise problem(
            status.HTTP_502_BAD_GATEWAY,
            "Node refused the runtime",
            f"The agent on {node_name!r} answered {status_code}: {_detail(payload)}",
        )

    try:
        machine.append(OP_PUT_RUNTIME, {"node": node_name, "spec": spec})
    except NotActive as exc:
        raise _not_active(machine) from exc

    reported = payload if isinstance(payload, dict) else {}
    return RuntimePlacement(
        node=node_name,
        name=name,
        modelAlias=reported.get("modelAlias") or spec.get("modelAlias"),
        status=str(reported.get("status")) if reported.get("status") else None,
        url=reported.get("url"),
        engine=_engine(reported.get("engine") or spec.get("engine")),
    )


def _targets(machine: StateMachine) -> list[tuple[str, str | None]]:
    return [
        (record.name, normalize_url(record.url))
        for record in sorted(machine.state.nodes.values(), key=lambda r: r.name)
    ]


def _engine(value: Any) -> EngineKind | None:
    if not isinstance(value, str):
        return None
    try:
        return EngineKind(value)
    except ValueError:
        # An engine kind this build does not know is reported as absent
        # rather than as an error. During a rolling upgrade a newer agent
        # legitimately runs an engine this control root has never heard
        # of, and refusing the whole union view over it would be worse
        # than one blank field.
        log.info("node reported unknown engine kind %r", value)
        return None


def _detail(payload: Any) -> str:
    if isinstance(payload, dict):
        return str(payload.get("detail") or payload.get("title") or payload)
    return str(payload)
