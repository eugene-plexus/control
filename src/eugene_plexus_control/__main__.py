"""Entrypoint: `python -m eugene_plexus_control`."""

from __future__ import annotations

import contextlib
import logging
import os

import uvicorn

from .app import create_app
from .config import BootstrapConfig
from .settings import load_settings

# Bind port when the agent doesn't supply one — i.e. standalone runs.
# Under supervision the topology owns it, which is the only source of
# truth: a port in this component's own config would be a second one,
# and the agent spawning at 8083 while the config claims 9000 is a
# failure nobody can see from either side.
#
# 8083 rather than the agent's 8079: the agent keeps that port because
# it appears in every existing config and acceptance script and there
# are now N agents, while this component is new and can take a free
# number.
_DEFAULT_PORT = 8083


def _resolve_port() -> int:
    env_port = os.environ.get("EUGENE_PLEXUS_CONTROL_BIND_PORT")
    return int(env_port) if env_port else _DEFAULT_PORT


def main() -> None:
    settings = load_settings()

    # Bootstrap the config cache only to discover the log level. The
    # authoritative values arrive when the log is replayed a moment
    # later, in the lifespan.
    bootstrap = BootstrapConfig(settings.config_file)
    if not settings.safe_mode:
        # A broken config must not stop logging from being configured;
        # the lifespan reports it properly.
        with contextlib.suppress(Exception):
            bootstrap.load()

    log_level = str(bootstrap.get("logLevel") or "INFO").upper()
    # uvicorn configures only its own loggers, so without this our
    # warnings arrive bare and untimestamped. force=True overrides any
    # basicConfig uvicorn already applied.
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    uvicorn.run(
        create_app(settings),
        host=settings.bind_host,
        port=_resolve_port(),
        log_level=log_level.lower(),
        # **Trust no forwarding header from anyone.** uvicorn's default
        # is `"127.0.0.1"`, and every browser reaches this root through
        # a node agent's loopback proxy -- so the default trusted
        # `X-Forwarded-For` from exactly the peer that is always ours,
        # and any caller could pick the login limiter's bucket. What the
        # proxy saw arrives as `peer.PEER_HEADER` instead. See `peer.py`;
        # review §6.1 #1, roadmap R1.2.
        forwarded_allow_ips=[],
    )


if __name__ == "__main__":
    main()
