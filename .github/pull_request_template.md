<!-- Thanks for contributing to Eugene Plexus / control! -->

## Summary

<!-- One or two sentences describing what changes and why. -->

## Type of change

- [ ] Wire up an existing spec endpoint
- [ ] Log / apply / snapshot core
- [ ] Trust root: enrollment, rotation, sealing
- [ ] Refactor / cleanup
- [ ] Tooling / CI / docs
- [ ] Bump SPECS_REF (regenerated models included)

## Checklist

- [ ] Every commit is signed off (`git commit -s`, or `git rebase --signoff main` for an existing branch). CI will block PRs without DCO sign-offs — see [CONTRIBUTING.md](../CONTRIBUTING.md).
- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] `mypy src/` passes
- [ ] `pytest` passes
- [ ] If `SPECS_REF` changed, `python scripts/codegen.py` was run and the regenerated `_generated/` is included in this PR.
- [ ] If wire contract changed, the matching PR landed in `eugene-plexus/specs` first and `SPECS_REF` is bumped here.

## If this PR touches the log, apply, or snapshot

- [ ] Every new mutation goes through `Writer.append()` and has a `LogOp` in the contract.
- [ ] `apply()` still reads no clock, no filesystem, no network — every timestamp comes from the payload.
- [ ] No liveness observation (`reachable`, `lastSeenAt`, `lastSeenEpoch`, rotation progress) entered `AppliedState`.
- [ ] `tests/test_replay_equivalence.py` passes, and if the change adds a mutation, it is exercised there.
