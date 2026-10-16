"""Prove U1 checks reject policy regressions; mutate disposable copies only."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MUTATIONS = [
    (
        "empty-groups-inherit-defaults",
        "permissions.py",
        'if person.get("groups") is None:',
        'if not person.get("groups"):',
        "test_explicit_group_mode_never_inherits_individual_defaults",
    ),
    (
        "refresh-ignores-group-revocation",
        "oidc.py",
        "not may_use(state, person, client_id)",
        "False",
        "test_group_edit_revokes_refresh_and_new_sign_in",
    ),
    (
        "snapshot-loses-groups",
        "applied.py",
        "state.permission_groups[key] for key in sorted(state.permission_groups)",
        "state.permission_groups[key] for key in []",
        "test_group_state_survives_snapshot_and_rejects_dangling_members",
    ),
    (
        "member-can-manage-groups",
        "routes/people.py",
        'tags=["people"], dependencies=[Depends(require_operator)]',
        'tags=["people"]',
        "test_group_edit_revokes_refresh_and_new_sign_in",
    ),
    (
        "code-exchange-ignores-revocation",
        "routes/oidc.py",
        "current = oidc.still_valid(",
        "current = code_subject or oidc.still_valid(",
        "test_group_revocation_between_approval_and_code_exchange",
    ),
]


def main() -> None:
    base = Path(tempfile.mkdtemp(prefix="ep-u1-sabotage-"))
    failures = []
    for label, filename, before, after, test in MUTATIONS:
        copy = base / label
        shutil.copytree(ROOT / "src", copy / "src", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(
            ROOT / "tests", copy / "tests", ignore=shutil.ignore_patterns("__pycache__")
        )
        shutil.copy2(ROOT / "pyproject.toml", copy / "pyproject.toml")
        target = copy / "src/eugene_plexus_control" / filename
        source = target.read_text(encoding="utf-8")
        assert source.count(before) == 1, (label, "mutation anchor changed")
        target.write_text(source.replace(before, after), encoding="utf-8", newline="\n")
        env = {k: v for k, v in os.environ.items() if not k.startswith("EUGENE_PLEXUS_")}
        env["PYTHONPATH"] = str(copy / "src")
        run = subprocess.run(
            [sys.executable, "-m", "pytest", f"tests/test_permission_groups.py::{test}", "-q"],
            cwd=copy,
            env=env,
            text=True,
            capture_output=True,
        )
        (copy / "result.log").write_text(run.stdout + run.stderr, encoding="utf-8")
        caught = run.returncode == 1 and "1 failed" in run.stdout
        print(f"{'CAUGHT' if caught else 'MISSED'} {label}", flush=True)
        if not caught:
            failures.append(label)
    print(f"Artifacts: {base}")
    if failures:
        raise SystemExit(f"Sabotage escaped: {failures}")


if __name__ == "__main__":
    main()
