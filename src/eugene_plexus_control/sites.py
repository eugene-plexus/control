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

OPERATOR = "operator"
FILES = "files"
PROTOCOL = "mcp-2026-07-28"
PRODUCTION = "production"
DEV = "dev"
NEEDS_UPDATE = (
    "This site's host is older than this root and speaks no MCP. Update Eugene on that "
    "machine, then try again."
)


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


def dev_grants(
    state: Any, record: SiteRecord, summary: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Eugene's owner's own grants on a site, named as the site last
    reported its folders, and **only while the install is in dev mode**
    (J13b): switching to production ends them at once, because this is
    checked at every use. The site honours them only if its owner let
    Eugene's owner in there (J6e), which it checks itself."""
    if install_mode(state) != DEV or summary is None:
        return []
    folders = {f["id"]: f for f in summary.get("folders") or []}
    out: list[dict[str, Any]] = []
    for grant in record.devGrants:
        folder = folders.get(grant["folderId"])
        if folder is not None:
            out.append(
                {
                    "folderId": folder["id"],
                    "name": folder["name"],
                    "path": folder["path"],
                    "identity": folder["identity"],
                    "writable": bool(grant["writable"] and folder["writable"]),
                }
            )
    return out


def site_folders_for(summary: dict[str, Any] | None, subject: str) -> list[dict[str, Any]]:
    """The folders on a site's own list for `subject`, as it last reported
    them. A cache: the site checks every call against its own copy again
    (rule 2 of remote-nodes.md §3.3)."""
    out = []
    for folder in (summary or {}).get("folders") or []:
        person = next((p for p in folder.get("people") or [] if p["subject"] == subject), None)
        if person is not None:
            out.append(
                {
                    "id": folder["id"],
                    "name": folder["name"],
                    "writable": bool(person["writable"] and folder["writable"]),
                }
            )
    return out


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
        if status.get("protocol") != PROTOCOL:
            return False, NEEDS_UPDATE
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
                for job in self.jobs.values():
                    if job.site == bound["site"] and job.state == "queued":
                        job.state = "offered"
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
