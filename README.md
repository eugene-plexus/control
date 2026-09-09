# Eugene Plexus — `control`

[![CI](https://github.com/eugene-plexus/control/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/eugene-plexus/control/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg)](https://www.python.org)

The **control root** of an [Eugene Plexus](https://github.com/eugene-plexus) install. Exactly one is active, plus any number of warm standbys. It holds what is inherently singular — the trust root, the node registry, install-wide topology, and the replicated control-state log — and it **spawns nothing**.

Supervision is the [`agent`](https://github.com/eugene-plexus/agent)'s job, and there is one agent per host. On whichever host holds the control root, `control` is itself just another local component in that agent's topology. That is deliberate: if the control root supervised processes it would need a second copy of the supervision machinery, and *"components share schemas, not code"* means a real second copy rather than an import. One supervisor implementation, running everywhere; one control root, supervised like anything else.

## The property this component exists to protect

**A control root that dies stops management, not inference.**

Once things are running, the data path needs no control root: the gateway routes from topology it already holds plus driver health, each driver holds its own config on disk, and engines are supervised by their local agent. So `control` going down pauses config edits, new runtimes, enrollment and UI login — and a chat completion still succeeds.

That was an accident of the architecture before this repo existed. [`tests/test_survives_control_root_loss.py`](tests/test_survives_control_root_loss.py) and [`specs/scripts/m5-acceptance.sh`](https://github.com/eugene-plexus/specs/blob/main/scripts/m5-acceptance.sh) are what make it a guarantee: the second of those kills the control root mid-run and asserts a real completion still comes back through a real engine.

## One writer, one ordered log

Every control-state change — enroll a node, revoke one, declare or delete a runtime, patch a component's config, rotate the signing key, promote — goes through **one** choke point that stamps a gapless monotonic `index`. Applied state is a deterministic function of the entries applied so far. Standbys **pull** from `GET /v1/control/log`; the active root never pushes.

Three rules hold the whole thing up, and each is asserted by a test rather than trusted:

| Rule | Why | Where |
|---|---|---|
| `apply()` never reads the clock, the filesystem, or the network | otherwise replay diverges from the original and a standby's state is a guess | [`applied.py`](src/eugene_plexus_control/applied.py) |
| Index is the only ordering; timestamps are informational | keeps clock skew between buildings out of the correctness argument | [`log_store.py`](src/eugene_plexus_control/log_store.py) |
| Liveness is **not** replicated state | `reachable`, `lastSeenAt`, `lastSeenEpoch` are one root's observations, not facts about the install | [`applied.py`](src/eugene_plexus_control/applied.py) |

**Replay equivalence** — a standby's applied state must be byte-identical to the active root's after applying the same log — is a required deliverable of this milestone, not a nice-to-have. It is what makes a promotion safe rather than hopeful, and it is what would later let Raft be dropped in underneath: the log, the apply function, the snapshot and the epoch already exist, so consensus would be a swap of the transport and the election, not a redesign.

This is **not** Raft and does not pretend to be. There is no quorum, no voting, and no automatic promotion. See §5 and §11 of the design doc for why that is a decision rather than an omission.

## Promotion is an operator act, always

There is no `mon_osd_down_out_interval` here. A control root is never marked `out` on a timer, because automatic promotion without quorum is the definition of split-brain.

An operator calls `POST /v1/control/promote` **on the standby**, with the passphrase. The standby verifies it against the replicated verifier, derives the master key from the replicated salt, confirms how far it has applied, increments the epoch, appends a `promote` entry, and begins accepting writes. **Refusing to promote is a valid outcome** — a standby that is behind reports the gap and returns 409, and `force` exists for the operator who has decided that losing the tail beats staying down.

Every control root carries a monotonic `epoch`, and agents record the highest they have seen and refuse to go backwards. A returning old root at epoch 5, against agents at 6, is fenced permanently — no election, no quorum, and no agent needing to agree with any other agent.

The known and accepted cost: an agent partitioned *during* a promotion still trusts the old root until it reconnects. Bounded to the partitioned nodes, visible as an epoch disagreement on `GET /v1/nodes`, and self-healing. It must not be "fixed" with automatic promotion.

## Per-node sealing

Each node has an identity keypair generated at enrollment whose private half never leaves it. A secret is sealed to the node whose component will read it — topology already says which node that is — plus a **recovery recipient** held here and itself sealed under the passphrase-derived key.

| Compromise | Yields |
|---|---|
| One node | that node's secrets only |
| This process | nothing directly — the recovery key is sealed under the passphrase |
| This process **and** the passphrase | everything. Unavoidable, and what the passphrase *is* |

`securityMode: os_keyring` is **host-bound**: the master key sits in *this* machine's Credential Manager or Keychain, so a standby cannot inherit auto-unlock and will require the passphrase at promotion. That is consistent with promotion being a human act anyway, but the wizard has to say so in words rather than let it be discovered during a failover.

## Status

**M5, partial.** Landed and tested: the single-writer ordered log, the deterministic apply function, snapshots and compaction, the standby follower, epoch fencing, promotion, node enrollment and revocation-as-rotation, two-recipient sealing, the trust root's auth surface, the config trio, and the install-wide topology views.

Not here yet: serving the UI (still the agent's job until the assets move), the log-shaped side of `securityMode: os_keyring` auto-unlock on a standby, recovery-from-a-dead-node as an operator flow, and the wizard copy that has to explain the keyring/HA tension. `POST /v1/runtimes` forwards to a node's agent; a node that is `down` during a key rotation is re-keyed on reconnect.

**Explicitly out of scope, by decision and not by neglect:** Raft, quorum, automatic promotion, multi-writer control state, and migrating an existing single-host install.

## Wire contract

Defined in [`specs/openapi/control.yaml`](https://github.com/eugene-plexus/specs/blob/main/openapi/control.yaml). Port **8083**.

| | |
|---|---|
| `GET /v1/nodes` | Every host, with its identity, role and last-seen epoch |
| `GET`/`DELETE /v1/nodes/{name}` | Read one / revoke it, **which rotates the signing key** |
| `POST /v1/nodes/join-token` | Mint a single-use, short-lived, node-scoped token |
| `POST /v1/nodes/enroll` | Called by an agent; join-token authenticated, no session |
| `GET /v1/control/status` | Role, epoch, applied index, and every standby's lag |
| `GET /v1/control/log` | Standbys pull entries after an index |
| `GET /v1/control/snapshot` | Bootstrap a standby; recover past compaction |
| `POST /v1/control/promote` | Operator action on the standby, passphrase required |
| `POST`/`GET /v1/control/rotate-key` | Explicit rotation / its progress |
| `GET /v1/components` | Install-wide, each entry tagged with its node |
| `GET`/`POST /v1/runtimes` | The union across nodes / declare one, forwarded to an agent |
| `GET /v1/auth/status`, `POST /v1/auth/initialize`, `POST /v1/auth/login` | The trust root |
| `GET`/`PATCH /v1/config`, `GET /v1/config/schema` | The standard config trio |
| `GET /healthz` | Liveness. Healthy on a standby too — replicating is doing its job |

Generated Pydantic models live in `src/eugene_plexus_control/_generated/` and are committed. They are regenerated from the specs commit pinned in [`SPECS_REF`](SPECS_REF):

```bash
python scripts/codegen.py
```

CI re-runs that and fails if the result differs from what is committed.

## Running it

Under supervision, the local agent spawns this like any other component and threads its config path, bind port and safe-mode flag in by env var. Standalone:

```bash
pip install -e ".[dev]"
python -m eugene_plexus_control          # binds 127.0.0.1:8083
pytest
```

Startup-only settings come from `EUGENE_PLEXUS_CONTROL_*` env vars — where the state directory lives, which interface to bind, safe mode. Everything an operator would want to tune is a config field reachable from the UI, because if it is tunable the UI exposes it. Bad config never stops startup: the config endpoints are how a broken config gets repaired, so refusing to boot over one would lock the operator out of the fix.

## License

Apache 2.0. Contributions under the [DCO](CONTRIBUTING.md) — `git commit -s`, no CLA.
