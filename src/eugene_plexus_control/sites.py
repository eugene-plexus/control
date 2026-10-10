"""Job Sites: the registry's reading side, and the broker that carries each
site's operations (`docs/design/job-sites-own-enrollment.md`).

A Job Site is its own enrollment, held by its site host (J19, J23). It
polls this root with its own token, claims what is queued for it, and
answers. Each operation is an envelope (`SiteOperation`): one MCP request
of the 2026-07-28 revision to one of the site's servers, or one of its
management actions. The site's policy is final (J8).

The registry is replicated; operations, their contents and a site's own
report never are. Delivery is at most once: an offered operation must be
claimed, permission is checked again, and a claimed operation is never put
back in the queue. Losing a tool call's answer is uncertain, not permission
to run it again.

An operation is bound to *(site id, `enrolledAt`)*: a site removed and
joined again is a new site, with a new id, and nothing queued for the old
one reaches it.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException

from .applied import SiteRecord
from .dependencies import problem

log = logging.getLogger(__name__)
JOB_SECONDS = 20.0
POLL_SECONDS = 8.0
ONLINE_SECONDS = 25.0
MAX_JOBS = 128
MAX_JOBS_PER_SITE = 8
#: How long an offered operation waits for its claim before it is offered
#: again. A site claims at once; an offer it never claimed went to a poll
#: whose site host was gone (restarted mid-poll), and must not strand the call.
OFFER_SECONDS = 5.0

OPERATOR = "operator"
FILES = "files"
PRODUCTION = "production"
DEV = "dev"


def new_site_id() -> str:
    """`s-` and 26 lowercase base32 characters: 130 random bits."""
    return "s-" + base64.b32encode(secrets.token_bytes(17)).decode("ascii")[:26].lower()


def install_mode(state: Any) -> str:
    """`production` unless the owner chose dev (J13, J18). An install that
    never recorded one is in production."""
    return DEV if state.config.get("installMode") == DEV else PRODUCTION


def mode_view(state: Any) -> dict[str, Any]:
    return {"mode": install_mode(state), "changedAt": state.config.get("installModeChangedAt")}


def binding(record: SiteRecord) -> dict[str, str]:
    """What an operation is bound to: this enrollment of this site."""
    return {"site": record.id, "enrolledAt": record.enrolledAt}


def person_name(state: Any, person: str) -> str:
    return str(state.people.get(person, {}).get("name") or "someone who has left")


def unique_names(names: list[str]) -> list[str]:
    """Each name as a site's `folder` argument takes it: a name already
    taken, ignoring case, by one before it reads `Name (2)`, `Name (3)`...
    The site names a person's view the same way (`SiteWorkspace.name`)."""
    taken: set[str] = set()
    out: list[str] = []
    for name in names:
        candidate, n = name, 1
        while candidate.casefold() in taken:
            n += 1
            candidate = f"{name} ({n})"
        taken.add(candidate.casefold())
        out.append(candidate)
    return out


def workspace_view(
    summary: dict[str, Any] | None, subject: str
) -> list[tuple[str, dict[str, Any], dict[str, str], bool]]:
    """Every workspace in `subject`'s view of a site, in its order: their own,
    then those the site's owner shared with them. Each with its name there,
    their rules in it, and whether it is their own."""
    workspaces = (summary or {}).get("workspaces") or []
    owner = (summary or {}).get("owner")
    rows: list[tuple[dict[str, Any], dict[str, str], bool]] = [
        (w, w["rules"], True) for w in workspaces if w["holder"] == subject
    ]
    for workspace in workspaces:
        if workspace["holder"] != owner or workspace["holder"] == subject:
            continue
        for person in workspace.get("people") or []:
            if person["subject"] == subject:
                rows.append(
                    (workspace, {"read": person["read"], "change": person["change"]}, False)
                )
    names = unique_names([w["name"] for w, _, _ in rows])
    return [(name, w, rules, mine) for name, (w, rules, mine) in zip(names, rows, strict=True)]


def linked(summary: dict[str, Any] | None, subject: str) -> dict[str, Any] | None:
    """`subject`'s link on a site's machine, as it last reported."""
    return next(
        (e for e in (summary or {}).get("links") or [] if e.get("subject") == subject), None
    )


def dev_grants(
    state: Any, record: SiteRecord, summary: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Eugene's owner's own grants on a site, named as the site last
    reported its workspaces, and **only while the install is in dev mode**
    (J13b): switching to production ends them at once, because this is
    checked at every use. The site honours them only if its owner let
    Eugene's owner in there (J6e), which it checks itself. A grant names one
    of the site owner's workspaces by id alone (J76)."""
    if install_mode(state) != DEV or summary is None:
        return []
    out: list[dict[str, Any]] = []
    owners = {w["id"]: (name, w) for name, w, _, _ in workspace_view(summary, record.owner)}
    for grant in record.devGrants:
        found = owners.get(grant["folderId"])
        if found is not None:
            name, workspace = found
            out.append(
                {
                    "folderId": workspace["id"],
                    "name": name,
                    "writable": bool(grant["writable"] and workspace["writable"]),
                }
            )
    return out


def site_folders_for(
    summary: dict[str, Any] | None, subject: str, *, own: bool = True
) -> list[dict[str, Any]]:
    """The workspaces on a site `subject` may use, as it last reported
    them, named as the site names their view. A cache: the
    site checks every call against its own copy again (rule 2 of
    remote-nodes.md §3.3). `own` False leaves out their own workspaces, for a
    person without `use-job-sites` (J77); so does no link, since only their
    own worker opens them."""
    mine_ok = own and linked(summary, subject) is not None
    return [
        {
            "id": w["id"],
            "name": name,
            "writable": rules["change"] != "deny",
            "mine": mine,
        }
        for name, w, rules, mine in workspace_view(summary, subject)
        if (mine_ok or not mine) and not (rules["read"] == rules["change"] == "deny")
    ]


def site_local_servers_for(summary: dict[str, Any] | None, subject: str) -> list[dict[str, Any]]:
    """The local servers on a site that `subject` has tools on, as it last
    reported them."""
    granted = {
        entry["server"]
        for entry in (summary or {}).get("access") or []
        if entry["subject"] == subject and entry.get("tools")
    }
    return [
        server
        for server in (summary or {}).get("servers") or []
        if server["kind"] == "local" and server["id"] in granted and server.get("enabled")
    ]


def link_fields(summary: dict[str, Any] | None, subject: str, owner: str) -> dict[str, Any]:
    """What one person is told about their own OS account on a site's machine
    (J27, §3.2): whether they linked it, which account their calls run as
    (their own once linked; until then the owner's, if the owner linked one),
    and where to link when they have not. From the site's last report, so a
    cache like the rest of the summary."""
    links = (summary or {}).get("links") or []
    mine = next((entry for entry in links if entry.get("subject") == subject), None)
    if mine is not None:
        return {"linked": True, "account": mine.get("accountName"), "linkPage": None}
    theirs = next((entry for entry in links if entry.get("subject") == owner), None)
    return {
        "linked": False,
        "account": theirs.get("accountName") if theirs else None,
        "linkPage": (summary or {}).get("linkPage"),
    }


def may_use_site(summary: dict[str, Any] | None, subject: str) -> bool:
    """Whether `subject` holds anything on a site, or is linked there, as it
    last reported: the test for being told a site exists at all."""
    held = (
        bool(site_folders_for(summary, subject))
        or bool(workspace_view(summary, subject))
        or any(entry.get("subject") == subject for entry in (summary or {}).get("access") or [])
        or any(entry.get("subject") == subject for entry in (summary or {}).get("links") or [])
    )
    return held


def envelope(state: Any, bound: dict[str, str], subject: str, **fields: Any) -> dict[str, Any]:
    """What a site claims (`SiteOperation`), less the id and deadline the
    broker adds at the claim."""
    return {
        **bound,
        "subject": subject,
        "installMode": install_mode(state),
        "kind": "mcp",
        "server": None,
        "request": None,
        "grants": [],
        "action": None,
        "arguments": None,
        **fields,
    }


@dataclass
class Job:
    id: str
    site: str
    check: Callable[[], dict[str, Any]]
    result: asyncio.Future[dict[str, Any]]
    deadline: float
    state: str = "queued"
    write: bool = False
    offered_at: float = 0.0


class Broker:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.reports: dict[str, tuple[float, str, dict[str, Any]]] = {}
        self.changed = asyncio.Event()
        self.used: dict[str, float] = {}

    def report(self, bound: dict[str, str]) -> dict[str, Any] | None:
        """What this enrollment of the site last reported, however long ago;
        None if it never has. A report from an earlier enrollment is never
        answered for a later one."""
        seen = self.reports.get(bound["site"])
        if seen is None or seen[1] != bound["enrolledAt"]:
            return None
        return seen[2]

    def summary(self, bound: dict[str, str]) -> dict[str, Any] | None:
        """The site's own folders, servers and list, as it last reported them."""
        report = self.report(bound)
        site = report.get("site") if report else None
        return site if isinstance(site, dict) else None

    def last_contact(self, bound: dict[str, str]) -> float | None:
        """Seconds since the site last polled, or None if it never has."""
        seen = self.reports.get(bound["site"])
        if seen is None or seen[1] != bound["enrolledAt"]:
            return None
        return time.perf_counter() - seen[0]

    def status(self, bound: dict[str, str]) -> dict[str, Any]:
        ago = self.last_contact(bound)
        report = self.report(bound)
        if ago is None or ago > ONLINE_SECONDS or report is None:
            return {
                "online": False,
                "ready": False,
                "reason": "This site is offline or has not connected yet.",
            }
        return {**report, "online": True}

    def forget(self, site: str) -> None:
        """A removed site: its report goes, and so does anything queued."""
        self.reports.pop(site, None)
        for job in [j for j in self.jobs.values() if j.site == site]:
            if not job.result.done():
                job.result.set_exception(
                    problem(503, "Site removed", "This site left the install.")
                )
        self.changed.set()

    def availability(self, bound: dict[str, str]) -> tuple[bool, str | None]:
        status = self.status(bound)
        if not status.get("ready"):
            return False, status.get("reason") or "This site is not ready."
        return True, None

    async def submit(
        self,
        bound: dict[str, str],
        check: Callable[[], dict[str, Any]],
        *,
        write: bool = False,
        operation_id: str | None = None,
    ) -> dict[str, Any]:
        check()
        available, reason = self.availability(bound)
        if not available:
            raise problem(503, "Site unavailable", reason or "This site is not ready.")
        if (
            len(self.jobs) >= MAX_JOBS
            or sum(j.site == bound["site"] for j in self.jobs.values()) >= MAX_JOBS_PER_SITE
        ):
            raise problem(429, "Site busy", "Wait for this site's operations to finish.")
        now = time.perf_counter()
        self.used = {key: until for key, until in self.used.items() if until > now}
        ident = operation_id or secrets.token_urlsafe(24)
        if ident in self.used or ident in self.jobs:
            raise problem(409, "Operation already used", "This operation cannot be replayed.")
        if len(self.used) >= 4096:
            raise problem(429, "Too many operations", "Wait before starting more operations.")
        self.used[ident] = now + 300
        job = Job(
            ident,
            bound["site"],
            check,
            asyncio.get_running_loop().create_future(),
            time.perf_counter() + JOB_SECONDS,
            write=write,
        )
        self.jobs[job.id] = job
        self.changed.set()
        try:
            result = await asyncio.wait_for(asyncio.shield(job.result), JOB_SECONDS)
            # Do not disclose a result after access was withdrawn during execution.
            try:
                check()
            except HTTPException:
                if write and job.state == "running":
                    return {
                        "status": "uncertain",
                        "message": "Access changed after this call started. "
                        "Check before trying again.",
                    }
                raise
            return result
        except TimeoutError:
            if job.state == "running" and write:
                return {
                    "status": "uncertain",
                    "message": "The site did not confirm this call. Check before trying again.",
                }
            raise problem(
                504,
                "Site timed out",
                "The site did not finish this operation. Check its connection.",
            ) from None
        finally:
            self.jobs.pop(job.id, None)
            job.result.cancel()

    async def poll(self, bound: dict[str, str], report: dict[str, Any]) -> str | None:
        self.reports[bound["site"]] = (time.perf_counter(), bound["enrolledAt"], report)
        deadline = time.perf_counter() + POLL_SECONDS
        while True:
            self.changed.clear()
            if report.get("ready"):
                now = time.perf_counter()
                for job in self.jobs.values():
                    stale = job.state == "offered" and now - job.offered_at > OFFER_SECONDS
                    if job.site == bound["site"] and (job.state == "queued" or stale):
                        job.state, job.offered_at = "offered", now
                        return job.id
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return None
            try:
                await asyncio.wait_for(self.changed.wait(), remaining)
            except TimeoutError:
                return None

    def claim(self, site: str, ident: str) -> dict[str, Any]:
        job = self.jobs.get(ident)
        if (
            job is None
            or job.site != site
            or job.state != "offered"
            or time.perf_counter() >= job.deadline
        ):
            raise problem(
                404, "Operation unavailable", "This operation expired or was already claimed."
            )
        try:
            command = job.check()
        except HTTPException as exc:
            if not job.result.done():
                job.result.set_exception(exc)
            raise
        job.state = "running"
        return {
            **command,
            "id": job.id,
            "expiresAt": time.time() + max(0, job.deadline - time.perf_counter()),
        }

    def finish(self, site: str, ident: str, result: dict[str, Any]) -> None:
        job = self.jobs.get(ident)
        if job is None or job.site != site or job.state != "running" or job.result.done():
            raise problem(
                404, "Operation unavailable", "This operation ended or belongs to another site."
            )
        job.result.set_result(result)
        log.info(
            "site operation completed: site=%s operation=%s status=%s",
            site,
            ident,
            result["status"],
        )

    def cancel(self, ident: str) -> None:
        now = time.perf_counter()
        self.used = {key: until for key, until in self.used.items() if until > now}
        if ident not in self.used and len(self.used) >= 4096:
            raise problem(429, "Too many operations", "Wait before cancelling more operations.")
        self.used[ident] = now + 300
        job = self.jobs.pop(ident, None)
        if job is not None and not job.result.done():
            job.result.set_result(
                {
                    "status": "uncertain" if job.write and job.state == "running" else "failed",
                    "message": (
                        "The operation was stopped. A call already started may finish; "
                        "check before trying again."
                        if job.write and job.state == "running"
                        else "The operation was stopped."
                    ),
                }
            )


def broker(request: Any) -> Broker:
    found = getattr(request.app.state, "site_broker", None)
    if found is None:
        found = Broker()
        request.app.state.site_broker = found
    return found
