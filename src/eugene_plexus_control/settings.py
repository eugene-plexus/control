"""Startup-time settings, sourced from environment variables.

Distinct from the runtime *config* (see `config.py`), which is editable
via `PATCH /v1/config` at runtime and is replicated through the log.
These only control bootstrap: where state lives on this host, which
interface to bind, whether to skip persisted config. If it is tunable by
an operator it is a config field and the UI exposes it; env vars are for
the things that have to be known before there is a config to read.

The control root is the trust root, so — unlike every other component —
it receives no signing key, no service token and no master key from the
agent that spawns it. It *mints* the signing key and derives the master
key from the operator's passphrase. What the agent threads in is the
same three bootstrap values it gives any child: config path, bind port,
safe-mode flag.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EUGENE_PLEXUS_CONTROL_",
        env_file=None,
        case_sensitive=False,
    )

    config_file: Path = Path("control.yaml")
    """Where the *bootstrap* copy of runtime config is persisted.

    Control config is replicated state — it lives in the log and the
    snapshot like everything else. This file exists so the process can
    find its log level and its state directory before it has replayed
    anything, and is written through on every applied `patchConfig` so
    the next boot starts from the same place."""

    state_dir: Path = Path("control-state")
    """Directory holding `log.jsonl`, `snapshot.json` and the install
    identity. One directory rather than one file because the log is
    append-only and the snapshot is replaced atomically, and those want
    different write patterns."""

    bind_host: str = "127.0.0.1"
    """Network interface to bind. An install that spans hosts has to set
    this — a standby cannot pull a log it cannot reach, and an agent in
    another building cannot enroll against loopback. It stays on
    loopback by default because a single-host install is the common case
    and a trust root should not open itself to the LAN without being
    asked."""

    safe_mode: bool = False
    """Skip the persisted config at startup and run on built-in
    defaults. Set by the agent as EUGENE_PLEXUS_CONTROL_SAFE_MODE=1
    after a boot failed on bad config, so the operator can reach
    /v1/config to fix it.

    Safe mode does **not** skip the log. Applied state is the install,
    not a preference: a control root that came up ignoring its own node
    registry would hand out wrong answers rather than degraded ones."""

    role: str = "control"
    """`control` or `standby`. Which one this process is, is
    configuration and not an election outcome — nothing here votes.

    An env var rather than a config field because it has to be known
    before the log is opened, and because a standby that could be
    switched to active by a config PATCH would be a second writer
    reachable over HTTP. Promotion is `POST /v1/control/promote` with
    the passphrase, and nothing else."""

    active_url: str | None = None
    """For a standby: where the active root is, so the follower knows
    what to pull from. Ignored when `role` is `control`."""

    follow_interval_seconds: float = 2.0
    """How often a standby pulls `GET /v1/control/log`. Pulling rather
    than being pushed to is what keeps one writer and lets the standby
    measure its own lag honestly."""

    passphrase_file: Path | None = None
    """Where to read the operator passphrase for unattended unlock.

    Read only when `securityMode` is `passphrase_file`; see that module
    for why this is a path and not the passphrase itself. Bootstrap
    rather than config because it has to be known before the root is
    unlocked, and because in a container it is fixed by the image and
    the secret mount rather than chosen at runtime — the same reason
    `state_dir` and `bind_host` live here.

    The config field's own description names this variable, so an
    operator who flips the mode in the UI is told what else to set."""

    allowed_hosts: str | None = None
    """Extra names first-run setup and sign-in answer to, comma-separated;
    `*` for any. See `trusted_host` for the rule and why it exists.

    Bootstrap rather than config for two reasons: the routes it guards
    are the ones an install that has no config yet must answer, and in
    normal use nobody needs it -- the browser reaches this root through
    an agent's proxy, which dials it by address. It is the override for
    an operator who points something straight at this port by a DNS
    name of their own."""


def load_settings() -> Settings:
    return Settings()
