"""Bounded, ephemeral delivery of operations to machines' site hosts.

Each operation is an envelope (`HelperOperation`): one MCP request of the
2026-07-28 revision to one of the machine's servers, or one of its host's
management actions (Job Sites J6, `remote-nodes.md` §3.4). The machine's
host decides whether it runs, and on a job site its policy is final (J8).

Configuration is replicated; jobs, their contents and a site's own report
never are. Delivery is at most once: an offered job must be claimed,
authorization is checked again, and a claimed job is never put back in the
queue. Losing a tool call's answer is uncertain, not permission to run it
again.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException

from .dependencies import problem

log = logging.getLogger(__name__)
JOB_SECONDS = 20.0
POLL_SECONDS = 8.0
ONLINE_SECONDS = 25.0
MAX_JOBS = 128


def configuration(state: Any, node: str) -> dict[str, Any]:
    member = state.nodes.get(node)
    if member is None or not member.signingPublicKey:
        raise problem(404, "Node unavailable", "This enrolled machine is no longer available.")
    current = state.node_helpers.get(node)
    if (
        current is not None
        and current["nodeKey"] == member.signingPublicKey
        and current["enrolledAt"] == member.enrolledAt
    ):
        return dict(current)
    return {
        "node": node,
        "nodeKey": member.signingPublicKey,
        "enrolledAt": member.enrolledAt,
        "enabled": False,
        "folders": [],
    }


def find_folder(state: Any, folder_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    for node in state.nodes:
        if not state.nodes[node].signingPublicKey:
            continue
        config = configuration(state, node)
        for folder in config["folders"]:
            if folder["id"] == folder_id:
                return config, folder
    raise problem(403, "Folder unavailable", "This folder is no longer granted to you.")


PRODUCTION = "production"
DEV = "dev"


def install_mode(state: Any) -> str:
    """`production` unless the owner chose dev (J13, J18). An install that
    never recorded one is in production: a new install starts there, and an
    old one had no job sites, so it has nothing to hide or show."""
    return DEV if state.config.get("installMode") == DEV else PRODUCTION


def mode_view(state: Any) -> dict[str, Any]:
    return {"mode": install_mode(state), "changedAt": state.config.get("installModeChangedAt")}


def is_site(state: Any, node: str) -> bool:
    """A Job Site: its folders and their grants are its owner's (J11)."""
    record = state.nodes.get(node)
    return record is not None and "files" in record.grants


OPERATOR = "operator"
FILES = "files"
PROTOCOL = "mcp-2026-07-28"


def unique_names(names: list[str]) -> list[str]:
    """Each folder's name as the file server's `folder` argument takes it: a
    name already taken on the machine, ignoring case, by a folder registered
    before it reads `Name (2)`, `Name (3)`... (J6g). The site host applies the
    same rule to what it is given."""
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


def folder_names(config: dict[str, Any]) -> dict[str, str]:
    folders = config["folders"]
    names = unique_names([f["name"] for f in folders])
    return {f["id"]: name for f, name in zip(folders, names, strict=True)}


def node_grants(state: Any, subject: str, node: str) -> list[dict[str, Any]]:
    """`subject`'s grants on ordinary node `node`, as the host takes them
    (`SiteGrantHint`): the root's grant is final there (J6d).

    Eugene's owner has the folders they gave themselves (`ownerAccess`); a
    person, their `helperGrants`, which Eugene's owner writes. A job site's
    folders are its own and are never here."""
    if is_site(state, node):
        return []
    config = configuration(state, node)
    names = folder_names(config)
    if subject == OPERATOR:
        chosen = [
            (f, f["ownerAccess"] == "write")
            for f in config["folders"]
            if f["ownerAccess"] != "none"
        ]
    else:
        held = {
            g["folderId"]: bool(g["writable"])
            for g in state.people.get(subject, {}).get("helperGrants") or []
        }
        chosen = [(f, held[f["id"]]) for f in config["folders"] if f["id"] in held]
    return [
        {
            "folderId": f["id"],
            "name": names[f["id"]],
            "path": f["path"],
            "identity": f["identity"],
            "writable": bool(writable and f["writable"]),
        }
        for f, writable in chosen
    ]


def dev_grants(state: Any, node: str, summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Eugene's owner's own grants on job site `node`, named as the site last
    reported its folders, and **only while the install is in dev mode**
    (J13b): switching to production ends them at once, because this is
    checked at every use. The site honours them only if its owner let
    Eugene's owner in there (J6e), which it checks itself."""
    if install_mode(state) != DEV or summary is None or not is_site(state, node):
        return []
    folders = {f["id"]: f for f in summary.get("folders") or []}
    out: list[dict[str, Any]] = []
    for grant in configuration(state, node).get("devGrants") or []:
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
    """The folders on a job site's own list for `subject`, as it last
    reported them. A cache: the site checks every call against its own copy
    again (rule 2 of §3.3)."""
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
    """The local servers on a job site that `subject` has tools on, as it
    last reported them."""
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


def envelope(state: Any, config: dict[str, Any], subject: str, **fields: Any) -> dict[str, Any]:
    """What a machine claims (`HelperOperation`), less the id and deadline
    the broker adds at the claim."""
    return {
        "node": config["node"],
        "nodeKey": config["nodeKey"],
        "enrolledAt": config["enrolledAt"],
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


def check_grants(state: Any, values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    for value in values:
        config, folder = find_folder(state, value["folderId"])
        if is_site(state, config["node"]):
            raise problem(
                403,
                "A job site's folder",
                f"{config['node']} is a job site: only its owner gives people its folders, "
                "from Workbench.",
            )
        if value["folderId"] in seen:
            raise problem(422, "Duplicate folder", "Assign each folder only once.")
        if value["writable"] and not folder["writable"]:
            raise problem(422, "Read-only folder", "This machine's folder permits reads only.")
        seen.add(value["folderId"])
    return values


@dataclass
class Job:
    id: str
    node: str
    check: Callable[[], dict[str, Any]]
    result: asyncio.Future[dict[str, Any]]
    deadline: float
    state: str = "queued"
    write: bool = False


class Broker:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.reports: dict[str, tuple[float, str, str, dict[str, Any]]] = {}
        self.changed = asyncio.Event()
        self.used: dict[str, float] = {}

    def report(self, node: str) -> dict[str, Any] | None:
        """What the machine last reported, however long ago; None if never."""
        seen = self.reports.get(node)
        return seen[3] if seen is not None else None

    def summary(self, node: str) -> dict[str, Any] | None:
        """A job site's own folders, servers and list, as it last reported them."""
        report = self.report(node)
        site = report.get("site") if report else None
        return site if isinstance(site, dict) else None

    def status(self, config: dict[str, Any]) -> dict[str, Any]:
        seen = self.reports.get(config["node"])
        if (
            seen is None
            or time.perf_counter() - seen[0] > ONLINE_SECONDS
            or seen[1:3] != (config["nodeKey"], config["enrolledAt"])
        ):
            return {
                "online": False,
                "ready": False,
                "supported": None,
                "reason": "This machine's file helper is offline or has not connected yet.",
            }
        return {**seen[3], "online": True}

    async def submit(
        self,
        config: dict[str, Any],
        check: Callable[[], dict[str, Any]],
        *,
        write: bool = False,
        operation_id: str | None = None,
    ) -> dict[str, Any]:
        check()
        status = self.status(config)
        if not config["enabled"] or not status.get("ready"):
            raise problem(
                503,
                "File helper unavailable",
                status.get("reason")
                or "Enable this machine's file helper and wait for it to be ready.",
            )
        if (
            len(self.jobs) >= MAX_JOBS
            or sum(j.node == config["node"] for j in self.jobs.values()) >= 8
        ):
            raise problem(429, "File helper busy", "Wait for existing file operations to finish.")
        now = time.perf_counter()
        self.used = {key: until for key, until in self.used.items() if until > now}
        ident = operation_id or secrets.token_urlsafe(24)
        if ident in self.used or ident in self.jobs:
            raise problem(409, "Operation already used", "This file operation cannot be replayed.")
        if len(self.used) >= 4096:
            raise problem(429, "Too many operations", "Wait before starting more file operations.")
        self.used[ident] = now + 300
        job = Job(
            ident,
            config["node"],
            check,
            asyncio.get_running_loop().create_future(),
            time.perf_counter() + JOB_SECONDS,
            write=write,
        )
        self.jobs[job.id] = job
        self.changed.set()
        try:
            result = await asyncio.wait_for(asyncio.shield(job.result), JOB_SECONDS)
            # Do not disclose a read result after access was withdrawn during execution.
            try:
                check()
            except HTTPException:
                if write and job.state == "running":
                    return {
                        "status": "uncertain",
                        "message": "Access changed after this write started. "
                        "Check the file before retrying.",
                    }
                raise
            return result
        except TimeoutError:
            if job.state == "running" and write:
                return {
                    "status": "uncertain",
                    "message": "The machine did not confirm this write. "
                    "Check the file before retrying.",
                }
            raise problem(
                504,
                "File helper timed out",
                "The machine did not finish this operation. Check its connection.",
            ) from None
        finally:
            self.jobs.pop(job.id, None)
            job.result.cancel()

    async def poll(self, config: dict[str, Any], report: dict[str, Any]) -> str | None:
        self.reports[config["node"]] = (
            time.perf_counter(),
            config["nodeKey"],
            config["enrolledAt"],
            report,
        )
        deadline = time.perf_counter() + POLL_SECONDS
        while True:
            self.changed.clear()
            if config["enabled"] and report["ready"]:
                for job in self.jobs.values():
                    if job.node == config["node"] and job.state == "queued":
                        job.state = "offered"
                        return job.id
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return None
            try:
                await asyncio.wait_for(self.changed.wait(), remaining)
            except TimeoutError:
                return None

    def claim(self, node: str, ident: str) -> dict[str, Any]:
        job = self.jobs.get(ident)
        if (
            job is None
            or job.node != node
            or job.state != "offered"
            or time.perf_counter() >= job.deadline
        ):
            raise problem(
                409, "Operation unavailable", "This operation expired or was already claimed."
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

    def finish(self, node: str, ident: str, result: dict[str, Any]) -> None:
        job = self.jobs.get(ident)
        if job is None or job.node != node or job.state != "running" or job.result.done():
            raise problem(
                409, "Operation unavailable", "This operation ended or belongs to another node."
            )
        job.result.set_result(result)
        log.info(
            "node file operation completed: node=%s operation=%s status=%s",
            node,
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
                        "The operation was stopped. A write already started may finish; "
                        "check the file before retrying."
                        if job.write and job.state == "running"
                        else "The file operation was stopped."
                    ),
                }
            )


def broker(request: Any) -> Broker:
    found = getattr(request.app.state, "node_helper_broker", None)
    if found is None:
        found = Broker()
        request.app.state.node_helper_broker = found
    return found
