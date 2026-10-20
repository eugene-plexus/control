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


def content(state: Any, control_pins: list[str]) -> dict[str, Any]:
    """What the list says, before it is dated and signed."""
    return {
        "required": required(state.config),
        "clientPlainHttp": client_plain(state.config),
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


class MachineListPublisher:
    """The last list signed, re-signed when what it says changes."""

    def __init__(self, state_dir: Path) -> None:
        self._path = Path(state_dir) / FILE
        self._lock = threading.Lock()
        self.jws: str | None = None
        self._content: dict[str, Any] | None = None
        self._version = 0
        try:
            kept = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            log.warning("the kept machine list at %s is unreadable: %s", self._path, exc)
            return
        if isinstance(kept, dict) and isinstance(kept.get("jws"), str):
            self.jws = kept["jws"]
            self._content = kept.get("content") if isinstance(kept.get("content"), dict) else None
            self._version = int(kept.get("version") or 0)

    def current(
        self, state: Any, control_pins: list[str], control_private_key: str | None
    ) -> str | None:
        """The list as a JWS: re-signed when its content changed and this root
        holds its identity key; the last one signed otherwise; None when it
        has never signed one."""
        wanted = content(state, control_pins)
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
            self._content = wanted
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._path.write_text(
                    json.dumps({"jws": self.jws, "content": wanted, "version": self._version}),
                    encoding="utf-8",
                )
            except OSError as exc:
                log.warning("could not keep the machine list at %s: %s", self._path, exc)
            return self.jws
