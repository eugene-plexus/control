# Contributing to Eugene Plexus `control`

Thanks for your interest. This service implements the `control` OpenAPI contract from [`eugene-plexus/specs`](https://github.com/eugene-plexus/specs) — please read this before opening a PR.

## Developer Certificate of Origin (DCO)

We use the [Developer Certificate of Origin](https://developercertificate.org/) instead of a CLA. **Every commit must be signed off** with `git commit -s`:

```
Signed-off-by: Your Name <your.email@example.com>
```

The name and email must match your `git config user.name` and `git config user.email`. CI blocks PRs whose commits are missing matching sign-offs.

If you forgot to sign off, fix the most recent commit:

```bash
git commit --amend -s --no-edit
```

…or for a whole branch:

```bash
git rebase --signoff main
```

The full DCO text is in [the specs CONTRIBUTING.md](https://github.com/eugene-plexus/specs/blob/main/CONTRIBUTING.md).

## Wire contract changes go in `specs`, not here

If your change touches the HTTP API — endpoints, request/response shapes, schemas — it belongs in [`eugene-plexus/specs`](https://github.com/eugene-plexus/specs), not here. Land that PR first; bump `SPECS_REF` and re-run codegen here in a follow-up.

This consumer generates from **two** documents, `control.yaml` and `agent.yaml`.
Audit both inputs when updating the pin, including shared-schema output. See the
[consumer update workflow](https://github.com/eugene-plexus/specs/blob/main/CONTRIBUTING.md#consumer-updates).

## Read this before touching the log

Four rules hold this component up. Each has a test, and a change that breaks one is a change that makes promotion unsafe rather than a change that fails a test.

**1. Every control-state mutation goes through `Writer.append()`.** One writer, one ordered path, one place that stamps an index. If mutations start writing files from ten places, replication is over. `LogOp` is a closed enum in the contract precisely so that adding a mutation is a visible act: *"an operation that is not in this list is an operation that would not replicate."*

**2. `apply()` is pure.** No clock, no filesystem, no network, no randomness. Every timestamp in applied state comes from the entry's payload, never from `now()`. This is what makes a standby's state comparable to the active root's, and it is the single easiest rule to break by accident — one `datetime.now()` in an apply path and replay equivalence is silently false.

**3. Liveness is not replicated state.** `reachable`, `lastSeenAt`, `lastSeenEpoch` and rotation progress are one root's *observations*. They are layered on when rendering a response and are deliberately absent from `AppliedState` and from the canonical bytes. Putting an observation in the log would make two roots disagree about the same install and call it drift.

**4. Index is the only ordering.** `LogEntry.at` exists so an operator can read the log. It is never compared, sorted on, or used to resolve anything.

## Local setup

```bash
git clone https://github.com/eugene-plexus/control
cd control
python -m venv .venv
. .venv/bin/activate           # or: .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

Note that the local [`agent`](https://github.com/eugene-plexus/agent) spawns every component with its own `sys.executable`, so for a supervised run this package has to be installed into the agent's venv as well — a component that is missing from there fails at spawn rather than at import.

## Git hooks

We use [pre-commit](https://pre-commit.com/) to auto-format staged files with Ruff before they reach CI. Enable it once per clone:

```bash
pip install pre-commit
pre-commit install
```

After that, `git commit` runs `ruff check --fix` and `ruff format` on staged Python files; if a hook reformats anything, re-stage and commit again.

## Style

- **Python 3.12+** features are fine. We use `from __future__ import annotations` only where it materially helps; otherwise just use modern syntax.
- **Ruff** for lint and format. `ruff check .` and `ruff format .` should both be clean before you push. CI enforces.
- **Mypy strict** for type-checking. New code must type-check; the `_generated/` directory is excluded.
- **No comments explaining what code does** — let names do the work. Reserve comments for *why* a non-obvious choice was made.
- **Async-first** on the request path. The one deliberate exception is the log append, which is synchronous and fsync'd inside the writer's lock: it is the serialization point of the whole component, so moving it off the event loop would buy concurrency we do not want.

## Running checks

```bash
ruff check .                       # lint
ruff format --check .              # formatting
mypy src/                          # type-check
pytest                             # tests
python scripts/codegen.py          # regenerate models from pinned specs
git diff --exit-code src/eugene_plexus_control/_generated/   # codegen freshness
```

On Windows, never write `SPECS_REF` with PowerShell's `Set-Content -Encoding utf8` — the BOM becomes part of the ref and the archive URL 404s. Use `[IO.File]::WriteAllText`.

## Reporting issues

File issues at <https://github.com/eugene-plexus/control/issues>. Useful issues include:

- A sequence of mutations whose replay is **not** byte-identical on a standby. That is the most valuable bug report this repo can receive.
- Spec-vs-impl divergence (the impl drifted from `control.yaml`).
- Anything that makes a control-root outage stop inference rather than only stopping management.

**Security.** This is a security design that has had no adversarial review, and that is stated plainly as the largest risk in the M5 design doc. Enrollment, revocation-as-rotation and the sealing recipients each have failure modes that are invisible until someone attacks them. If you have found one, please report it privately rather than in a public issue.

Cross-component architecture questions belong in
[specs issues](https://github.com/eugene-plexus/specs/issues).
