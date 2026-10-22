"""Every machine's TLS door and key, signed by this root's identity (LT1).

`docs/design/tls-between-machines.md` §3.2. Each node registers its machine
port and the pin of its machine key under its own signature (its signed
address announcement) or its join token (enrolment), and this root records
both in the node's record. What it publishes here is that record for every
node, plus the keys of the machine port in front of this root itself, which
its own machine's agent hands it (`Settings.machine_pins`), signed with the
identity key every node pinned at enrolment. So a node trusts a pin because
the root it pinned says that node registered it, and nothing on the network
can add one.

**A separate list, not a field on the trust bundle or on the Job Sites'
list.** The bundle is parsed by every node of every version, and an older
one might refuse a field it does not know. The Job Sites' list
(`root_tls`) answers 503 whenever this root cannot read its nodes name,
which is a Job Site's business and must not stop machine traffic.

**Kept on disk**, so a sealed root (no identity key in memory) still serves
the last list it signed, exactly as it serves the last bundle. `version`
only ever rises, so a node can refuse a list older than the one it holds.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import jwt

from . import tokens

log = logging.getLogger(__name__)

TYP_MACHINES = "ep-machines+jwt"
FILE = "machine_list.json"

#: `machineEncryption`, the control config field.
ENCRYPTION_KEY = "machineEncryption"
AUTOMATIC, REQUIRED, OFF = "automatic", "required", "off"


def required(config: dict[str, Any]) -> bool:
    return config.get(ENCRYPTION_KEY) == REQUIRED


#: `clientPlainHttp`, the control config field (LT2-C4): `on` until the owner
#: turns it off.
CLIENT_PLAIN_KEY = "clientPlainHttp"


def client_plain(config: dict[str, Any]) -> str:
    return "off" if config.get(CLIENT_PLAIN_KEY) == "off" else "on"


def content(
    state: Any, control_pins: list[str], *, control: dict[str, Any] | None = None
) -> dict[str, Any]:
    """What the list says, before it is dated and signed. `control` is
    `control_block`'s answer (row 4); without it the list names no root."""
    signed: dict[str, Any] = {
        "required": required(state.config),
        "clientPlainHttp": client_plain(state.config),
        # Row 3: this root records a next key (`tlsNextPin`), so a node may
        # change its machine key. A node offers no change without it.
        "keyChange": True,
        "controlPins": sorted(set(control_pins)),
        "nodes": [
            {
                "name": record.name,
                "url": record.url,
                "machineUrl": record.machineUrl,
                "pins": list(record.tlsPins),
            }
            for record in sorted(state.nodes.values(), key=lambda r: r.name)
        ],
    }
    if control is not None:
        signed["control"] = control
    return signed


# --------------------------------------------------------------------------- #
# Row 4: where this root is, for the nodes that must follow a promotion
# --------------------------------------------------------------------------- #

#: The door file is read again at most this often (it is a few hundred bytes).
_DOOR_RECHECK_SECONDS = 1.0
_DOOR_URLS: dict[str, tuple[float, str | None]] = {}


def door_url(path: str | None) -> str | None:
    """The machine port's URL in the door file this root's agent writes
    (`{"pins": [...], "url": ...}`, row 3), or None when it names none or
    cannot be read."""
    if not path:
        return None
    now = time.perf_counter()
    cached = _DOOR_URLS.get(path)
    if cached is not None and now - cached[0] < _DOOR_RECHECK_SECONDS:
        return cached[1]
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        url = document.get("url") if isinstance(document, dict) else None
        found = url if isinstance(url, str) and url.strip() else None
    except (OSError, ValueError):
        found = None
    _DOOR_URLS[path] = (now, found)
    return found


def _plain(url: str | None) -> str | None:
    """`http://host:port` of a machine port's `https://host:port`: the form
    a node's machine table carries to TLS under `/control`."""
    origin = _origin(url)
    if origin is None:
        return None
    return "http://" + origin.split("://", 1)[1]


def control_block(
    state: Any, *, node: str | None, door_file: str | None, nodes_origin: str | None
) -> dict[str, Any]:
    """`MachineList.control`: this root's epoch, its node and where nodes
    reach it (row 4).

    The address is this machine's port in its plain form, from the door
    file its agent rewrites while it runs, else from the machine port this
    root's own node announced, else this root's nodes name. A promoted
    standby signs its first list before its agent has written a door file
    for it, so its own node record is what names it then."""
    url = _plain(door_url(door_file))
    if url is None and node is not None:
        record = state.nodes.get(node)
        url = _plain(getattr(record, "machineUrl", None))
    if url is None and nodes_origin:
        url = nodes_origin.strip().rstrip("/") or None
    return {"epoch": int(state.epoch), "node": node, "url": url}


def signed(app: Any) -> str | None:
    """This root's current list, signed if it changed: what
    `GET /v1/trust/machines` serves and what a promotion pushes."""
    from .routes.nodes import machine_pins

    settings = app.state.settings
    machine = app.state.machine
    publisher: MachineListPublisher = app.state.machine_list
    return publisher.current(
        machine.state,
        machine_pins(settings),
        app.state.auth_state.control_private_key,
        control=control_block(
            machine.state,
            node=getattr(app.state, "node_name", None),
            door_file=getattr(settings, "machine_door_file", None),
            nodes_origin=getattr(settings, "nodes_origin", None),
        ),
    )


async def push(app: Any, jws: str, *, names: list[str] | None = None) -> dict[str, bool]:
    """POST the list to every node with an address (`names` limits it).

    **No bearer**: the list is signed with the identity every node pinned,
    the mirror of the trust bundle's push (`trust.push`). A node that is
    down takes it from the next root it hears, or by its own pull."""
    import asyncio

    import httpx

    machine = app.state.machine
    client = app.state.nodes_client
    results: dict[str, bool] = {}

    async def one(name: str, url: str) -> None:
        try:
            await client.post_json(url, "/v1/node/machine-list", None, {"jws": jws})
            results[name] = True
        except (httpx.HTTPError, ValueError) as exc:
            log.info("node %s did not take the machine list yet: %s", name, exc)
            results[name] = False

    targets = [
        (record.name, str(record.url))
        for record in machine.state.nodes.values()
        if record.url and (names is None or record.name in names)
    ]
    await asyncio.gather(*(one(name, url) for name, url in targets))
    return results


def _origin(url: str | None) -> str | None:
    from urllib.parse import urlsplit

    parts = urlsplit(url or "")
    if not parts.hostname or parts.scheme not in ("http", "https"):
        return None
    host = parts.hostname.lower()
    host = f"[{host}]" if ":" in host else host
    return f"{parts.scheme}://{host}:{parts.port or (443 if parts.scheme == 'https' else 80)}"


def _loopback(origin: str) -> bool:
    """An address on this machine: nothing to encrypt, nothing to refuse."""
    import ipaddress
    from urllib.parse import urlsplit

    host = (urlsplit(origin).hostname or "").strip("[]")
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def door_of(url: str | None) -> str | None:
    """`host:port` of an `https` machine URL, normalised as tables key it."""
    origin = _origin(url)
    return origin.removeprefix("https://") if origin and origin.startswith("https://") else None


def install_table(state: Any, own_node: str | None) -> None:
    """The active root's own machine table, from its own records (LT1).

    The root is where every pin is recorded, so it never waits for its
    agent's copy of the list: a node it has just enrolled is reached over
    TLS by the very next call, the trust bundle pushed to it included.
    """
    from ._http import parse_machine_table, set_machine_table

    machines: dict[str, list[str]] = {}
    routes: dict[str, str | None] = {}
    names: dict[str, str] = {}
    local: list[str] = []
    for record in sorted(state.nodes.values(), key=lambda r: r.name):
        if record.name == own_node:
            # This root's own machine, under its LAN address: not a crossing.
            if record.url:
                local.append(str(record.url))
            continue
        agent = _origin(record.url)
        door = _origin(record.machineUrl)
        carried = None
        if door is not None and door.startswith("https://") and record.tlsPins:
            hostport = door.removeprefix("https://")
            if hostport in machines:
                # Two records claim one door (written before the root checked
                # for that): neither is trusted there.
                log.warning(
                    "%s and %s both registered %s; neither is trusted there",
                    names.get(hostport),
                    record.name,
                    hostport,
                )
                machines[hostport] = []
                continue
            machines[hostport] = list(record.tlsPins)
            names[hostport] = record.name
            carried = str(record.machineUrl).rstrip("/") + "/agent"
        if agent is not None and agent.startswith("http://") and not _loopback(agent):
            if agent in routes and names.get(agent) != record.name:
                # Two records claim one address (written before the root
                # refused that): neither is carried there.
                log.warning("%s and %s both claim %s; neither is trusted there",
                            names.get(agent), record.name, agent)  # fmt: skip
                routes[agent] = None
                continue
            routes[agent] = carried
            names[agent] = record.name
    set_machine_table(
        parse_machine_table(
            {
                "machines": machines,
                "routes": routes,
                "names": names,
                "required": required(state.config),
                "local": local,
            }
        )
    )


def link(record: Any, probe: Any | None) -> dict[str, Any]:
    """`Node.machineTls`: how this root last reached the node (an observation)."""
    from ._http import machine_route

    if not record.machineUrl or not record.tlsPins:
        return {
            "state": "no_key",
            "detail": "It runs a version without a machine port; update it.",
        }
    if probe is None:
        return {"state": "unknown", "detail": None}
    if probe.hosts_control:
        # This root reaches its own machine on loopback, so it has observed
        # nothing about the machine port; other machines' views say that.
        return {"state": "unknown", "detail": "This root's own machine."}
    if probe.pin_refused:
        return {
            "state": "refused",
            "detail": "Its machine port presented a key it did not register "
            f"({probe.pin_refused}). Nothing was sent to it.",
        }
    if not probe.reachable:
        return {"state": "unreachable", "detail": probe.error}
    if record.url and machine_route(record.url):
        return {"state": "encrypted", "detail": None}
    return {
        "state": "unknown",
        "detail": "This root has not yet taken its key from the machine list.",
    }


#: `first_signed`'s key for the machine port in front of this root.
CONTROL_KEY = "@control"


def _pin_sets(listed: dict[str, Any] | None) -> dict[str, tuple[str, ...]]:
    """Each node's pins in a list's content, and this root's own."""
    if not listed:
        return {}
    sets = {
        str(n.get("name")): tuple(sorted(n.get("pins") or ()))
        for n in listed.get("nodes") or ()
        if isinstance(n, dict)
    }
    sets[CONTROL_KEY] = tuple(sorted(listed.get("controlPins") or ()))
    return sets


def key_change(
    record: Any,
    *,
    nodes: dict[str, Any],
    probes: dict[str, Any],
    publisher: MachineListPublisher,
    control_pins: list[str],
    hosts_control: bool,
) -> dict[str, Any] | None:
    """`Node.machineKeyChange` (row 3): who this root sees holding a list
    that names both of a node's keys. An observation, never replicated.

    `listVersion` is the first list that named both keys (and, for the
    machine in front of this root, both in `controlPins` too); a machine
    holding that list or a later one has them. The node switches once
    `waitingOn` is empty. Unreachable machines are not waited for: each
    takes the list when it is back, the root's own key included
    (`machine_trust.pull` fetches it in plain HTTP when the key it holds is
    refused, and takes it only if it verifies). A machine without a machine
    key talks plain HTTP and is not concerned.
    """
    if not record.tlsNextPin:
        return None
    pins = tuple(sorted(record.tlsPins))
    version = publisher.first_signed(record.name, pins)
    if hosts_control and version is not None:
        wanted = tuple(sorted(set(control_pins)))
        if not set(pins) <= set(wanted):
            version = None
        else:
            door = publisher.first_signed(CONTROL_KEY, wanted)
            version = None if door is None else max(version, door)
    waiting: list[str] = []
    unreachable: list[str] = []
    not_ready: list[str] = []
    capable = publisher.list_capable
    for other in sorted(nodes.values(), key=lambda r: r.name):
        if other.name == record.name or not other.tlsPins:
            continue
        probe = probes.get(other.name)
        held = getattr(probe, "list_version", None)
        if held is None and other.name not in capable:
            # Review blocker 2: an agent older than row 3 cannot take the
            # root's new key by itself if it misses the change. Named even
            # while it is off; the root's own change waits for it.
            not_ready.append(other.name)
        if probe is not None and not probe.reachable:
            unreachable.append(other.name)
            continue
        if version is None or held is None or held < version:
            waiting.append(other.name)
    return {
        "nextPin": record.tlsNextPin,
        "listVersion": version,
        "waitingOn": waiting,
        "unreachable": unreachable,
        "notReady": not_ready,
    }


class MachineListPublisher:
    """The last list signed, re-signed when what it says changes.

    It also keeps, for each node and for this root's own machine port, the
    version that first named its pins as they are now (row 3), so a machine
    key change can tell which machines hold a list that names both keys."""

    def __init__(self, state_dir: Path) -> None:
        self._path = Path(state_dir) / FILE
        self._lock = threading.Lock()
        self.jws: str | None = None
        self._content: dict[str, Any] | None = None
        self._version = 0
        self._first_signed: dict[str, int] = {}
        #: Nodes this root has seen report a machine list version (row 3):
        #: their agents take the root's new key by themselves after missing
        #: a change. Kept in this root's file, so a node that is off when
        #: this root restarts still counts.
        self.list_capable: set[str] = set()
        try:
            kept = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            log.warning("the kept machine list at %s is unreadable: %s", self._path, exc)
            return
        capable = kept.get("listCapable") if isinstance(kept, dict) else None
        if isinstance(capable, list):
            self.list_capable = {str(n) for n in capable if isinstance(n, str)}
        if isinstance(kept, dict) and isinstance(kept.get("jws"), str):
            self.jws = kept["jws"]
            self._content = kept.get("content") if isinstance(kept.get("content"), dict) else None
            self._version = int(kept.get("version") or 0)
            first = kept.get("firstSigned")
            if isinstance(first, dict):
                self._first_signed = {
                    str(k): int(v) for k, v in first.items() if isinstance(v, int)
                }

    def note_list_capable(self, names: set[str]) -> None:
        """Remember nodes seen reporting a list version; written when it grows."""
        with self._lock:
            if names <= self.list_capable:
                return
            self.list_capable |= names
            self._save()

    def first_signed(self, name: str, pins: tuple[str, ...]) -> int | None:
        """The version that first named `pins` for `name` (a node, or
        `CONTROL_KEY`), when the last list signed names exactly those; None
        when no list signed yet does."""
        with self._lock:
            if _pin_sets(self._content).get(name) != tuple(sorted(pins)):
                return None
            return self._first_signed.get(name)

    def current(
        self,
        state: Any,
        control_pins: list[str],
        control_private_key: str | None,
        *,
        control: dict[str, Any] | None = None,
    ) -> str | None:
        """The list as a JWS: re-signed when its content changed and this root
        holds its identity key; the last one signed otherwise; None when it
        has never signed one."""
        wanted = content(state, control_pins, control=control)
        with self._lock:
            if wanted == self._content and self.jws is not None:
                return self.jws
            if control_private_key is None:
                return self.jws
            # `version` strictly rises, so a node that holds one never takes
            # an older one for the same; `iat` is only when it was signed.
            # It follows the replicated log index (a thousand versions to an
            # entry), so a promoted standby, whose own file starts empty,
            # signs above everything the root it replaced signed.
            self._version = max(self._version + 1, int(state.index) * 1000)
            self.jws = jwt.encode(
                {"iat": int(time.time()), "version": self._version, **wanted},
                tokens.load_private(control_private_key),
                algorithm="EdDSA",
                headers={"typ": TYP_MACHINES},
            )
            # Row 3: a pin set unchanged since the last list keeps the
            # version that first named it; a new or changed one starts here.
            # With no last list (a promoted standby, a new file), every set
            # starts here, so a key change waits for every machine again.
            before, after = _pin_sets(self._content), _pin_sets(wanted)
            self._first_signed = {
                key: (
                    self._first_signed[key]
                    if before.get(key) == pins and key in self._first_signed
                    else self._version
                )
                for key, pins in after.items()
            }
            self._content = wanted
            self._save()
            return self.jws

    def _save(self) -> None:
        """The list and what is known about it, kept beside the log. Called
        with the lock held."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(
                    {
                        "jws": self.jws,
                        "content": self._content,
                        "version": self._version,
                        "firstSigned": self._first_signed,
                        "listCapable": sorted(self.list_capable),
                    }
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            log.warning("could not keep the machine list at %s: %s", self._path, exc)
