"""POST /v1/people/known: which people this root still has, for a member node (row 53).

The install's store keeps files for people the root verified, and deletes
what a removed person kept. It learns who was removed by asking its own
machine's agent, which asks here with its node token (`sub: agent`): the
store's own token cannot leave its machine. The answer is the one
`/oidc/subjects` gives an app, and it is withheld the same way while this
root cannot vouch for its list: locked, or a standby that may be behind.
Only the ids sent are answered, so nobody can list people with it.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, status

from .._generated.models import PeopleKnownRequest, PeopleKnownResult
from ..applied import OPERATOR_NAME
from ..dependencies import problem, require_node_actor

router = APIRouter(tags=["auth"])


@router.post(
    "/v1/people/known",
    response_model=PeopleKnownResult,
    dependencies=[Depends(require_node_actor)],
)
async def people_known(request: Request, body: PeopleKnownRequest) -> PeopleKnownResult:
    machine = request.app.state.machine
    if not machine.is_active:
        raise problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Standby",
            "This control root is a standby and may be behind; ask the active one.",
        )
    if request.app.state.auth_state.master_key is None:
        raise problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Locked",
            "Eugene is locked. Sign in to the console once with the passphrase to unlock it.",
        )
    people = machine.state.people
    known: list[str] = []
    unknown: list[str] = []
    for subject in dict.fromkeys(str(s.root) for s in body.subjects):
        (known if subject == OPERATOR_NAME or subject in people else unknown).append(subject)
    return PeopleKnownResult(known=known, unknown=unknown)
