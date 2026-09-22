"""Runtime configuration — which, here, is replicated state.

Implements the shared Eugene Plexus config protocol (`GET
/v1/config/schema`, `GET /v1/config`, `PATCH /v1/config`) with one
structural difference from every other component: **a PATCH is a log
entry.**

That follows from what a standby is for. A promoted standby has to come
up configured the way it would have been had it been active all along —
same standby endpoints to replicate to, same security mode, same poll
intervals. Config held in a local file on the active root only would be
config the standby silently lacks, and the first anyone would learn of
it is during a failover. So `patchConfig` is one of the nine ops, values
are validated *before* the entry is created, and the on-disk YAML is a
write-through cache the process reads at boot before it has replayed
anything.

The fields themselves are deliberately few. Ports are the agent's
(topology owns them, and two sources of truth on a port is the trap
where the agent spawns at one and the component's own config claims
another), and there are no `sensitive` fields — this component holds
keys, but it holds them in applied state under the master key, not as
config values an operator types.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Any

import yaml

from . import _private_files
from ._generated.models import (
    ConfigDocument,
    ConfigField,
    ConfigFieldError,
    ConfigSchema,
    ConfigUpdateRequest,
    ConfigValueType,
)

log = logging.getLogger(__name__)

CATEGORY_LABELS: dict[str, str] = {
    "setup": "Setup",
    "security": "Security",
    "replication": "Replication",
    "topology": "Topology",
    "ui": "Appearance",
    "logging": "Logging",
}

FIELDS: list[ConfigField] = [
    ConfigField(
        key="firstRunComplete",
        label="First-run setup complete",
        description=(
            "Set by the first-run wizard's final step. While false the "
            "UI routes operators to /setup; flipping it back to false "
            "re-enters the wizard on the next reload. Safe to leave "
            "alone unless you want to redo setup."
        ),
        category="setup",
        valueType=ConfigValueType.boolean,
        default=False,
    ),
    ConfigField(
        key="securityMode",
        label="Security mode",
        description=(
            "How this host keeps its master encryption key between "
            "restarts. 'Prompt on startup' keeps it in process memory "
            "only, so a restart needs the passphrase again. 'OS keyring' "
            "stores it in this machine's Credential Manager, Keychain or "
            "Secret Service for auto-unlock.\n\n"
            "'Passphrase file' reads it from a file you mount, for hosts "
            "that have no keyring at all — which is every container.\n\n"
            "Worth knowing before you pick: **OS keyring is host-bound.** "
            "The key lives in *this* machine's store, so a standby "
            "control root on another host cannot inherit auto-unlock and "
            "will ask for the passphrase when you promote it. That is "
            "consistent with promotion being something a person does "
            "anyway — but it is better read here than discovered during "
            "a failover. **A passphrase file is not host-bound**: mount "
            "the same secret on the standby and a failover needs nobody "
            "either.\n\n"
            "'Passphrase file' also needs "
            "`EUGENE_PLEXUS_CONTROL_PASSPHRASE_FILE` set to the path, "
            "because the root has to find it before it is unlocked. "
            "Selecting this mode without that variable leaves the root "
            "locked and says so in the log.\n\n"
            "Both auto-unlock modes trade the same thing away: whoever "
            "can read the store — this machine's keyring, or that file — "
            "can unlock the install without knowing the passphrase. "
            "They buy an install that comes back from a power cut with "
            "nobody present, which is why they exist and why neither is "
            "the default."
        ),
        category="security",
        valueType=ConfigValueType.enum,
        default="prompt_on_startup",
        enumValues=["prompt_on_startup", "os_keyring", "passphrase_file"],
        enumLabels=[
            "Prompt on startup",
            "OS keyring auto-unlock",
            "Passphrase file auto-unlock",
        ],
        requiresRestart=True,
    ),
    ConfigField(
        key="standbyUrls",
        label="Standby control roots",
        description=(
            "Where this install's warm standbys are. Each one pulls the "
            "log from here and can be promoted if this host is not "
            "coming back.\n\n"
            "More than one standby is a configuration, not a mechanism — "
            "list as many as you want, or none. What matters is that "
            "`GET /v1/control/status` reports how far behind each one "
            "is, so you find out whether failover is safe *before* you "
            "need it rather than during."
        ),
        category="replication",
        valueType=ConfigValueType.url_list,
        default=[],
    ),
    ConfigField(
        key="nodePollIntervalSeconds",
        label="Node poll interval",
        description=(
            "How often this root checks whether each enrolled host is "
            "reachable and which epoch it has seen. Reachability is only "
            "ever reported, never acted on: a node that stops answering "
            "is `down`, not `out`, and nothing is reassigned on the "
            "strength of a failed poll."
        ),
        category="topology",
        valueType=ConfigValueType.integer,
        default=15,
        minimum=2,
        maximum=600,
    ),
    ConfigField(
        key="nodeRequestTimeoutSeconds",
        label="Node request timeout",
        description=(
            "How long to wait for one host's agent before treating it as "
            "unreachable for this request. Keep it short: the union "
            "runtime view asks every node, and a dashboard that hangs on "
            "the slowest host is worse than one that names it as "
            "unreachable."
        ),
        category="topology",
        valueType=ConfigValueType.number,
        default=5.0,
        minimum=0.5,
        maximum=60.0,
    ),
    ConfigField(
        key="joinTokenTtlSeconds",
        label="Default join-token lifetime",
        description=(
            "How long a freshly minted enrollment token stays usable. "
            "Short by default: a join token is the one credential that "
            "grants membership, so the window in which it is useful is "
            "the window in which it is dangerous. A request can override "
            "this per token."
        ),
        category="security",
        valueType=ConfigValueType.integer,
        default=900,
        minimum=30,
        maximum=86400,
    ),
    ConfigField(
        key="uiTheme",
        label="Theme",
        description="Color theme for the Eugene Plexus UI.",
        category="ui",
        valueType=ConfigValueType.enum,
        default="auto",
        enumValues=["light", "dark", "auto"],
        enumLabels=["Light", "Dark", "Auto (follow system)"],
    ),
    ConfigField(
        key="uiFontSize",
        label="Font size",
        description="Base font size for the Eugene Plexus UI.",
        category="ui",
        valueType=ConfigValueType.enum,
        default="medium",
        enumValues=["small", "medium", "large"],
        enumLabels=["Small", "Medium", "Large"],
    ),
    ConfigField(
        key="logLevel",
        label="Log level",
        description=(
            "How chatty this component's terminal output is. `DEBUG` "
            "names every log entry as it is applied and every node poll, "
            "which is the fastest way to see why a standby is behind."
        ),
        category="logging",
        valueType=ConfigValueType.enum,
        default="INFO",
        enumValues=["DEBUG", "INFO", "WARNING", "ERROR"],
        requiresRestart=True,
    ),
]

# NOTE on what is NOT in this schema:
#   * The bind port — the agent owns it via EUGENE_PLEXUS_CONTROL_BIND_PORT.
#   * The role (control vs standby) — an env var, because a role a config
#     PATCH could change would be a second writer reachable over HTTP.
#   * Where the active root is, for a standby — same reason.

_FIELDS_BY_KEY: dict[str, ConfigField] = {f.key: f for f in FIELDS}


def as_schema() -> ConfigSchema:
    return ConfigSchema(component="control", fields=list(FIELDS), categories=CATEGORY_LABELS)


def defaults() -> dict[str, Any]:
    return {f.key: f.default for f in FIELDS if f.default is not None}


def validate_patch(request: ConfigUpdateRequest) -> tuple[dict[str, Any], list[ConfigFieldError]]:
    """Split a PATCH into accepted values and per-key rejections.

    Validation happens **here**, before an entry exists, and never at
    apply time. An entry already in the log has to apply the same way on
    every replica, including one running a build whose validation rules
    have since changed — so apply takes what the writer accepted.
    """
    accepted: dict[str, Any] = {}
    rejected: list[ConfigFieldError] = []

    for key, value in request.model_dump().items():
        field = _FIELDS_BY_KEY.get(key)
        if field is None:
            rejected.append(ConfigFieldError(key=key, message="unknown field"))
            continue
        error = _validate_value(field, value)
        if error is not None:
            rejected.append(ConfigFieldError(key=key, message=error))
            continue
        if value is None and field.default is not None:
            accepted[key] = field.default
        else:
            accepted[key] = value

    return accepted, rejected


def requires_restart(keys: list[str]) -> list[str]:
    return sorted(k for k in keys if (_FIELDS_BY_KEY.get(k) and _FIELDS_BY_KEY[k].requiresRestart))


def as_document(values: dict[str, Any]) -> ConfigDocument:
    merged = {**defaults(), **values}
    return ConfigDocument.model_validate(merged)


def effective(values: dict[str, Any]) -> dict[str, Any]:
    """Applied config over the built-in defaults.

    A field absent from applied state is one nobody has ever set, which
    is not the same as one set to its default — but for reading purposes
    they are the same value, and the config document is a reading
    surface."""
    return {**defaults(), **values}


def _validate_value(field: ConfigField, value: Any) -> str | None:
    """None if valid, otherwise an operator-readable error message."""
    if value is None:
        return None  # null clears to default

    vt = field.valueType

    if vt in (ConfigValueType.string, ConfigValueType.url, ConfigValueType.file_path):
        if not isinstance(value, str):
            return f"expected string, got {type(value).__name__}"
        if field.pattern is not None and re.search(field.pattern, value) is None:
            return f"value does not match pattern {field.pattern!r}"
        return None

    if vt == ConfigValueType.url_list:
        if not isinstance(value, list):
            return f"expected a list of URLs, got {type(value).__name__}"
        seen: dict[str, int] = {}
        for position, item in enumerate(value):
            if not isinstance(item, str):
                return f"entry {position} is {type(item).__name__}, expected a string"
            if not item.strip():
                return f"entry {position} is empty"
            if item in seen:
                return f"entry {position} duplicates entry {seen[item]} ({item!r})"
            seen[item] = position
        return None

    if vt == ConfigValueType.integer:
        if isinstance(value, bool) or not isinstance(value, int):
            return f"expected integer, got {type(value).__name__}"
        if field.minimum is not None and value < field.minimum:
            return f"must be >= {field.minimum}"
        if field.maximum is not None and value > field.maximum:
            return f"must be <= {field.maximum}"
        return None

    if vt in (ConfigValueType.number, ConfigValueType.duration):
        if isinstance(value, bool) or not isinstance(value, int | float):
            return f"expected number, got {type(value).__name__}"
        if field.minimum is not None and value < field.minimum:
            return f"must be >= {field.minimum}"
        if field.maximum is not None and value > field.maximum:
            return f"must be <= {field.maximum}"
        return None

    if vt == ConfigValueType.boolean:
        if not isinstance(value, bool):
            return f"expected boolean, got {type(value).__name__}"
        return None

    if vt == ConfigValueType.enum:
        if not isinstance(value, str):
            return f"expected string, got {type(value).__name__}"
        allowed = field.enumValues or []
        if value not in allowed:
            return f"must be one of {allowed}"
        return None

    return f"unsupported valueType: {vt}"


class BootstrapConfig:
    """The write-through YAML copy of applied config.

    Exists for one narrow reason: the process has to know its log level
    and open its state directory before it has replayed anything, and
    the replayed answer is the authority the moment it is available.
    Every applied `patchConfig` is written here so the next boot starts
    from the same place.

    Never read after startup. If this file and applied state disagree,
    applied state wins — the log is the install, this is a cache.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._values: dict[str, Any] = defaults()

    @property
    def values(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._values)

    def get(self, key: str) -> Any:
        with self._lock:
            return self._values.get(key)

    def load(self) -> None:
        with self._lock:
            if not self._path.exists():
                self._values = defaults()
                return
            raw = yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
            if not isinstance(raw, dict):
                raise ValueError(f"config file {self._path} must be a YAML mapping at the root")
            merged = defaults()
            for key, value in raw.items():
                if key in _FIELDS_BY_KEY:
                    merged[key] = value
            self._values = merged

    def write_through(self, values: dict[str, Any]) -> None:
        """Persist applied config. Best-effort by design.

        A failure here must not fail the PATCH: the entry is already in
        the log, which is the authority, and refusing the operator's
        change because a cache could not be updated would be the wrong
        way round."""
        with self._lock:
            self._values = {**defaults(), **values}
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                # Private and replaced rather than rewritten: see
                # `_private_files`. Nothing here is sealed material, but
                # it sits beside what is, and a truncate-then-write that
                # is killed halfway leaves a boot reading an empty cache.
                _private_files.write_private_text(
                    self._path,
                    yaml.safe_dump(self._values, sort_keys=True, default_flow_style=False),
                )
            except OSError as exc:
                log.warning(
                    "could not write the bootstrap config cache %s (%s); applied state is "
                    "unaffected, but the next boot will start from stale values",
                    self._path,
                    exc,
                )
