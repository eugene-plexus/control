"""The conversion of what an older build retired (`retired_conversion`).

Between v0.1.0 and v0.2 the operator-managed node folders and a Job Site
joining as a node were retired. A root that ran the builds in between still
holds them: a retired Job Site in its registry, and the old fields in its
snapshot and its log. At start it is converted once (the files are kept as
`*.v0.1.bak`, each retired node is revoked, a snapshot writes the current
form, a record is left), and after that only the current form is read.

**The fixtures are the old format as that build wrote it, not as today's code
thinks it looked.** The nine entries and the snapshot below are what the
build at control `d4a7dda` (2026-10-05, the last before slice 2b.1 retired
them) wrote when its own `StateMachine` was driven through these
enrollments: the old `NodeHelper` validation refused a first guess at them,
which is how the folder's `person` (not `subject`) was found. They are kept
as literals so that a later change to `to_canonical` cannot quietly change
what "old" means. The first build that had node folders (`6f3686f`) wrote a
smaller form of them, and is a fixture of its own. To check or regenerate:
`git archive <commit> src`, put it first on `sys.path`, and drive its
`StateMachine` through the same entries. Keys are placeholders built from readable text at run
time, never literals: a scanner cannot tell a literal fake key from a real
one.

The cases that matter, in order: it converts what it found and nothing else;
it is a no-op the second time and on a root with nothing retired; a kill at
any step is finished by the next start; a failure degrades, naming the step
and what the filesystem said; and a standby that follows the root converges,
whether it reads the new entries or has to take the snapshot.
"""

from __future__ import annotations

import base64
import functools
import json
import logging
import os
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_control import retired_conversion, security
from eugene_plexus_control.app import create_app
from eugene_plexus_control.applied import is_retired_site
from eugene_plexus_control.log_store import LogStore
from eugene_plexus_control.replication import Follower
from eugene_plexus_control.retired_conversion import BACKUP_SUFFIX, MARKER_FILENAME
from eugene_plexus_control.state_machine import ROLE_STANDBY, StateMachine

from .conftest import PASSPHRASE, login, machine_at, settings_for, standby_at
from .test_replication_follower import _LiveServer

WHEN = "2026-10-05T12:00:00+00:00"
IDENTITY_FIELDS = (
    "salt",
    "passphraseVerifier",
    "sealedSigningKey",
    "rootTokenPublicKey",
    "sealedControlKey",
    "controlPublicKey",
    "sealedRecoveryKey",
    "signingKeyId",
)


def ph(label: str) -> str:
    """A stand-in for a key, computed from readable text (see the docstring)."""
    return base64.b64encode(f"THIS-IS-NOT-A-REAL-KEY::{label}".encode()).decode("ascii")


@functools.cache
def a_verifier() -> str:
    return security.hash_passphrase("a person's password")


def placeholder_identity() -> dict[str, Any]:
    return {
        "salt": base64.b64encode(b"0123456789abcdef").decode("ascii"),
        "passphraseVerifier": a_verifier(),
        "sealedSigningKey": ph("signing"),
        "rootTokenPublicKey": ph("root-token-public"),
        "sealedControlKey": ph("control-identity"),
        "controlPublicKey": ph("control-public"),
        "sealedRecoveryKey": ph("recovery"),
        "signingKeyId": "1",
    }


_REAL_IDENTITY: dict[str, Any] = {}


@pytest.fixture
def real_identity(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The identity fields of a root that really was initialized, so a root
    started on these files can be signed in to with PASSPHRASE."""
    if not _REAL_IDENTITY:
        directory = tmp_path_factory.mktemp("identity")
        with TestClient(create_app(settings_for(directory))) as client:
            made = client.post("/v1/auth/initialize", json={"passphrase": PASSPHRASE})
            assert made.status_code == 204, made.text
        snapshot = json.loads((directory / "state" / "snapshot.json").read_text(encoding="utf-8"))
        _REAL_IDENTITY.update({k: snapshot[k] for k in IDENTITY_FIELDS})
    return dict(_REAL_IDENTITY)


# --------------------------------------------------------------------------- #
# The old format, as control@d4a7dda wrote it
# --------------------------------------------------------------------------- #


def dump(document: Any) -> str:
    """The serializer the log and the snapshot have always used."""
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def gpu_box() -> dict[str, Any]:
    return {
        "advertiseSequence": 0,
        "agentVersion": "0.1.0",
        "arch": "x64",
        "devices": [
            {"index": 0, "kind": "cuda", "memoryTotalBytes": 25769803776, "name": "RTX 3090"}
        ],
        "enrolledAt": WHEN,
        "grants": ["node"],
        "name": "gpu-box",
        "os": "linux",
        "owner": None,
        "publicKey": ph("gpu-box-sealing"),
        "reachable": False,
        "role": "agent",
        "signingPublicKey": ph("gpu-box-signing"),
        "tokenPublicKey": ph("gpu-box-token"),
        "url": "http://192.0.2.7:59079/",
    }


def job_site(name: str = "oldsite", owner: str = "p-ada") -> dict[str, Any]:
    """A Job Site that joined as a node: files and nothing else, no address."""
    return {
        "advertiseSequence": 0,
        "agentVersion": "0.1.0",
        "arch": "x64",
        "devices": [],
        "enrolledAt": WHEN,
        "grants": ["files"],
        "name": name,
        "os": "windows",
        "owner": owner,
        "publicKey": ph(f"{name}-sealing"),
        "reachable": False,
        "role": "agent",
        "signingPublicKey": ph(f"{name}-signing"),
        "tokenPublicKey": ph(f"{name}-token"),
        "url": None,
    }


def enrollment(node: dict[str, Any]) -> dict[str, Any]:
    """The payload the enroll route appended for `node`."""
    keep = (
        "agentVersion",
        "arch",
        "devices",
        "enrolledAt",
        "grants",
        "name",
        "os",
        "owner",
        "publicKey",
        "role",
        "signingPublicKey",
        "tokenPublicKey",
        "url",
    )
    return {k: node[k] for k in keep}


def person(person_id: str, name: str, grants: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "apps": [],
        "createdAt": WHEN,
        "disabled": False,
        "helperGrants": grants,
        "id": person_id,
        "name": name,
        "passwordChangedAt": WHEN,
        "passwordVerifier": a_verifier(),
    }


def helper_of_gpu_box(*, enabled: bool) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "enrolledAt": WHEN,
        "folders": (
            [
                {
                    "id": "f-notes",
                    "identity": "device:inode:birth",
                    "name": "Notes",
                    "ownerAccess": "none",
                    "path": "/srv/notes",
                    "writable": False,
                }
            ]
            if enabled
            else []
        ),
        "node": "gpu-box",
        "nodeKey": ph("gpu-box-signing"),
    }


def helper_of_job_site() -> dict[str, Any]:
    return {
        "devGrants": [{"folderId": "f-desk", "writable": False}],
        "enabled": True,
        "enrolledAt": WHEN,
        "folders": [
            {
                "id": "f-desk",
                "identity": "volume:file-id",
                "name": "Desk",
                "ownerAccess": "read",
                "path": "C:/Users/ada/Desktop",
                "people": [{"person": "p-ada", "writable": True}],
                "writable": True,
            }
        ],
        "node": "oldsite",
        "nodeKey": ph("oldsite-signing"),
    }


COMPONENT = {
    "kind": "inference-driver",
    "name": "left",
    "node": "gpu-box",
    "url": "http://192.0.2.7:59081/",
}


def old_entries(extra_sites: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    """The entries the dev build wrote, indexes 1 to 9 (and then one more
    Job Site for each name in `extra_sites`)."""
    ada = person("p-ada", "ada", [{"folderId": "f-notes", "writable": False}])
    steps: list[tuple[str, dict[str, Any]]] = [
        ("enrollNode", enrollment(gpu_box())),
        ("putPerson", {"person": ada}),
        ("putPerson", {"person": person("p-bob", "bob", [])}),
        ("enrollNode", enrollment(job_site())),
        ("putNodeHelper", {"helper": helper_of_gpu_box(enabled=True)}),
        ("putNodeHelper", {"helper": helper_of_job_site()}),
        ("putComponent", dict(COMPONENT)),
        (
            "putPerson",
            {
                "person": {
                    **ada,
                    "helperGrants": [
                        {"folderId": "f-notes", "writable": False},
                        {"folderId": "f-desk", "writable": True},
                    ],
                }
            },
        ),
        ("putNodeHelper", {"helper": helper_of_gpu_box(enabled=False)}),
    ]
    steps += [("enrollNode", enrollment(job_site(name))) for name in extra_sites]
    return [
        {
            "at": f"2026-10-05T12:00:{number:02d}+00:00",
            "epoch": 1,
            "index": number,
            "op": op,
            "payload": payload,
        }
        for number, (op, payload) in enumerate(steps, start=1)
    ]


def identity_snapshot(identity: dict[str, Any]) -> dict[str, Any]:
    """The snapshot `seed_identity` wrote: index 0, nothing yet."""
    return {
        "clientAdmission": {"buckets": {}, "clock": 0.0},
        "clientKeys": [],
        "components": [],
        "config": {},
        "epoch": 1,
        "index": 0,
        "nodeHelpers": [],
        "nodes": [],
        "oidcClients": [],
        "oidcKeys": [],
        "people": [],
        "revokedSessions": [],
        "revokedSignIns": [],
        "runtimes": [],
        **identity,
    }


def compacted_snapshot(identity: dict[str, Any]) -> dict[str, Any]:
    """The snapshot after the first seven entries, as a dev build that had
    compacted wrote it: the folders in `nodeHelpers`, the grants on the
    people, the Job Site in `nodes`."""
    return {
        **identity_snapshot(identity),
        "components": [dict(COMPONENT)],
        "index": 7,
        "nodeHelpers": [helper_of_gpu_box(enabled=True), helper_of_job_site()],
        "nodes": [gpu_box(), job_site()],
        "people": [
            person("p-ada", "ada", [{"folderId": "f-notes", "writable": False}]),
            person("p-bob", "bob", []),
        ],
    }


def write_root(directory: Path, snapshot: dict[str, Any], entries: list[dict[str, Any]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "snapshot.json").write_bytes(dump(snapshot).encode("utf-8"))
    (directory / "log.jsonl").write_bytes("".join(dump(e) + "\n" for e in entries).encode("utf-8"))


def write_old_root(
    directory: Path,
    *,
    era: str = "compacted",
    identity: dict[str, Any] | None = None,
    extra_sites: tuple[str, ...] = (),
) -> None:
    """A dev build's state directory. `log`: nothing compacted yet, the whole
    history in the log. `compacted`: the snapshot holds the first seven
    entries and the log the rest."""
    identity = identity or placeholder_identity()
    entries = old_entries(extra_sites)
    if era == "log":
        write_root(directory, identity_snapshot(identity), entries)
    else:
        write_root(directory, compacted_snapshot(identity), entries[7:])


FIRST = "2026-10-04T12:00:00+00:00"


def first_build_node() -> dict[str, Any]:
    """An ordinary node as the first build that had node folders (control
    `6f3686f`, 2026-10-04) wrote it: no `owner` yet."""
    return {
        "advertiseSequence": 0,
        "agentVersion": "0.1.0",
        "arch": "x64",
        "devices": [],
        "enrolledAt": FIRST,
        "grants": ["node"],
        "name": "gpu-box",
        "os": "linux",
        "publicKey": ph("gpu-box-sealing"),
        "reachable": False,
        "role": "agent",
        "signingPublicKey": ph("gpu-box-signing"),
        "tokenPublicKey": ph("gpu-box-token"),
        "url": "http://192.0.2.7:59079/",
    }


def first_build_person(grants: list[dict[str, Any]]) -> dict[str, Any]:
    return {**person("p-ada", "ada", grants), "createdAt": FIRST, "passwordChangedAt": FIRST}


def first_build_helper(folders: list[dict[str, Any]]) -> dict[str, Any]:
    """A folder had no `people` and the helper no `devGrants` yet."""
    return {
        "enabled": True,
        "enrolledAt": FIRST,
        "folders": folders,
        "node": "gpu-box",
        "nodeKey": ph("gpu-box-signing"),
    }


def first_build_folder(
    folder_id: str, name: str, *, writable: bool, owner: str, identity: str
) -> dict[str, Any]:
    return {
        "id": folder_id,
        "identity": identity,
        "name": name,
        "ownerAccess": owner,
        "path": f"/srv/{name.lower()}",
        "writable": writable,
    }


def write_first_build_root(directory: Path, *, identity: dict[str, Any] | None = None) -> None:
    """A root that ran the first build and nothing after it: an ordinary node
    with folders and a person with grants, and no Job Site anywhere. Three
    entries are in the snapshot (index 3) and two are in the log."""
    notes_read_only = first_build_folder(
        "f-notes", "Notes", writable=False, owner="none", identity="device:inode:birth"
    )
    snapshot = {
        **identity_snapshot(identity or placeholder_identity()),
        "index": 3,
        "nodeHelpers": [first_build_helper([notes_read_only])],
        "nodes": [first_build_node()],
        "people": [first_build_person([{"folderId": "f-notes", "writable": False}])],
    }
    tail = [
        (
            "putNodeHelper",
            {
                "helper": first_build_helper(
                    [
                        first_build_folder(
                            "f-notes",
                            "Notes",
                            writable=True,
                            owner="read",
                            identity="device:inode:birth",
                        ),
                        first_build_folder(
                            "f-scans",
                            "Scans",
                            writable=False,
                            owner="none",
                            identity="device:inode:birth2",
                        ),
                    ]
                )
            },
        ),
        (
            "putPerson",
            {
                "person": first_build_person(
                    [
                        {"folderId": "f-notes", "writable": True},
                        {"folderId": "f-scans", "writable": False},
                    ]
                )
            },
        ),
    ]
    entries = [
        {
            "at": f"2026-10-04T12:00:{number:02d}+00:00",
            "epoch": 1,
            "index": number,
            "op": op,
            "payload": payload,
        }
        for number, (op, payload) in enumerate(tail, start=4)
    ]
    write_root(directory, snapshot, entries)


def retired_on_disk(directory: Path) -> list[str]:
    """Everything retired that either file still carries, read as plain JSON:
    independent of the code under test, so a bug in its finder cannot hide."""
    found: list[str] = []
    snapshot_path = directory / "snapshot.json"
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if snapshot.get("nodeHelpers"):
            found.append("snapshot: nodeHelpers")
        found += [
            f"snapshot: person {p['id']} has helperGrants"
            for p in snapshot.get("people") or []
            if "helperGrants" in p
        ]
        found += [
            f"snapshot: node {n['name']} holds files"
            for n in snapshot.get("nodes") or []
            if "files" in (n.get("grants") or [])
        ]
    log_path = directory / "log.jsonl"
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            payload = entry.get("payload") or {}
            if entry["op"] == "putNodeHelper":
                found.append(f"log {entry['index']}: putNodeHelper")
            if entry["op"] == "enrollNode" and "files" in (payload.get("grants") or []):
                found.append(f"log {entry['index']}: enrollNode holds files")
            if entry["op"] == "putPerson" and "helperGrants" in (payload.get("person") or {}):
                found.append(f"log {entry['index']}: putPerson has helperGrants")
    return found


def tree(directory: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir()) if p.is_file()}


def start(directory: Path, *, role: str = "control") -> tuple[LogStore, StateMachine]:
    """What the lifespan does before it converts: open and replay."""
    store = LogStore(directory)
    machine = StateMachine(store, role=role)
    machine.load()
    return store, machine


def retired_nodes(machine: StateMachine) -> list[str]:
    return sorted(n for n, r in machine.state.nodes.items() if is_retired_site(r))


def copy_name(name: str) -> str:
    return name + BACKUP_SUFFIX


# --------------------------------------------------------------------------- #
# The fixture is the old format
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("era", ["log", "compacted"])
def test_the_fixture_is_a_root_that_holds_what_was_retired(tmp_path: Path, era: str) -> None:
    """Guards the other tests: a fixture that held nothing retired would let
    every one of them pass without converting anything. Today's code still
    loads it, as it must for the root to be converted at all."""
    write_old_root(tmp_path / "state", era=era)

    assert retired_on_disk(tmp_path / "state")
    _, machine = start(tmp_path / "state")
    assert machine.state.index == 9
    assert retired_nodes(machine) == ["oldsite"]


# --------------------------------------------------------------------------- #
# It converts what it found, once
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("era", ["log", "compacted"])
def test_a_dev_build_root_is_converted(tmp_path: Path, era: str) -> None:
    """Both shapes a dev build's files can be in: everything still in the log,
    and everything folded into a snapshot that carries the old fields."""
    directory = tmp_path / "state"
    write_old_root(directory, era=era)
    before = tree(directory)
    store, machine = start(directory)

    done = retired_conversion.convert(store, machine)

    assert done is not None
    assert done.revoked == ("oldsite",)
    assert done.resumed is False
    # The registry: the retired Job Site is gone, and only it.
    assert sorted(machine.state.nodes) == ["gpu-box"]
    assert list(machine.state.components) == ["gpu-box/left"]
    assert sorted(p["name"] for p in machine.state.people.values()) == ["ada", "bob"]
    assert all("helperGrants" not in p for p in machine.state.people.values())
    # One entry was appended, like any other, and the log was folded.
    assert machine.state.index == 10
    assert store.load().entries == []
    assert json.loads((directory / "snapshot.json").read_text(encoding="utf-8"))["index"] == 10
    # Nothing retired is left in either file, and a fresh start reads the same.
    assert retired_on_disk(directory) == []
    assert machine_at(directory).canonical() == machine.canonical()
    # The files as they were, byte for byte.
    for name in ("log.jsonl", "snapshot.json"):
        assert (directory / copy_name(name)).read_bytes() == before[name]
    # The record.
    record = json.loads((directory / MARKER_FILENAME).read_text(encoding="utf-8"))
    assert record["convertedFrom"] == "v0.1"
    assert record["revokedNodes"] == ["oldsite"]
    assert record["copies"] == ["log.jsonl.v0.1.bak", "snapshot.json.v0.1.bak"]
    assert record["appliedIndex"] == 10
    assert record["resumedAfterAnInterruptedRun"] is False
    assert sorted(p.name for p in directory.iterdir()) == sorted(
        [
            "log.jsonl",
            "snapshot.json",
            copy_name("log.jsonl"),
            copy_name("snapshot.json"),
            MARKER_FILENAME,
        ]
    )


def test_the_first_build_that_had_node_folders_is_converted_with_nothing_to_revoke(
    tmp_path: Path,
) -> None:
    """A root that ran only the earliest of those builds holds folders and
    grants and no Job Site. It is the likelier real case, and it is converted
    all the same: the snapshot is the thing that has to change."""
    directory = tmp_path / "state"
    write_first_build_root(directory)
    before = tree(directory)
    assert retired_on_disk(directory)
    store, machine = start(directory)
    assert machine.state.index == 5 and retired_nodes(machine) == []

    done = retired_conversion.convert(store, machine)

    assert done is not None and done.revoked == ()
    assert machine.state.index == 5, "nothing was revoked, so nothing was appended"
    assert sorted(machine.state.nodes) == ["gpu-box"]
    assert retired_on_disk(directory) == []
    assert store.load().entries == []
    assert machine_at(directory).canonical() == machine.canonical()
    for name in ("log.jsonl", "snapshot.json"):
        assert (directory / copy_name(name)).read_bytes() == before[name]
    record = json.loads((directory / MARKER_FILENAME).read_text(encoding="utf-8"))
    assert record["revokedNodes"] == [] and record["appliedIndex"] == 5
    store, machine = start(directory)
    assert retired_conversion.convert(store, machine) is None


def test_every_retired_node_is_revoked_and_no_other_node_is(tmp_path: Path) -> None:
    """A node that holds `gateway` as well as `node` is not a Job Site."""
    directory = tmp_path / "state"
    write_old_root(directory, extra_sites=("attic-site", "desk-site"))
    store, machine = start(directory)
    machine.append(
        "enrollNode",
        {
            "name": "attic",
            "role": "agent",
            "grants": ["gateway", "node"],
            "url": "http://192.0.2.2:59079/",
            "enrolledAt": WHEN,
        },
    )
    assert retired_nodes(machine) == ["attic-site", "desk-site", "oldsite"]

    done = retired_conversion.convert(store, machine)

    assert done is not None and done.revoked == ("attic-site", "desk-site", "oldsite")
    assert sorted(machine.state.nodes) == ["attic", "gpu-box"]
    assert machine.state.nodes["attic"].grants == ("gateway", "node")
    assert retired_on_disk(directory) == []


def test_the_converted_files_are_private(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A copy of the log and the snapshot holds the sealed keys, the salt and
    the verifier, as they do: created asking for 0600, and through a
    temporary name so a kill leaves a whole copy or none."""
    directory = tmp_path / "state"
    write_old_root(directory)
    store, machine = start(directory)
    calls: list[tuple[str, int, int]] = []
    real_open = os.open

    def spy(path: Any, flags: int, mode: int = 0o777, *args: Any, **kw: Any) -> int:
        calls.append((os.fspath(path), flags, mode))
        return real_open(path, flags, mode, *args, **kw)

    monkeypatch.setattr(os, "open", spy)

    retired_conversion.convert(store, machine)

    for name in (copy_name("log.jsonl"), copy_name("snapshot.json"), MARKER_FILENAME):
        made = [(f, m) for p, f, m in calls if Path(p).name.startswith(f".{name}.")]
        assert made, f"{name} was not written through a private temporary file"
        assert all(m == 0o600 and f & os.O_EXCL for f, m in made)
    assert sorted(p.name for p in directory.iterdir() if p.name.endswith(".tmp")) == []


# --------------------------------------------------------------------------- #
# A second start does nothing; so does a root with nothing retired
# --------------------------------------------------------------------------- #


def test_a_second_start_changes_nothing(tmp_path: Path) -> None:
    directory = tmp_path / "state"
    write_old_root(directory)
    store, machine = start(directory)
    assert retired_conversion.run(store, machine) is None
    converted = tree(directory)

    store, machine = start(directory)
    assert retired_conversion.convert(store, machine) is None
    assert retired_conversion.run(store, machine) is None

    assert tree(directory) == converted, "a file was written, or rewritten, by the second start"
    assert machine.state.index == 10


def test_a_root_with_nothing_retired_is_left_alone(tmp_path: Path) -> None:
    """Every root that never ran those builds, which is nearly all of them."""
    directory = tmp_path / "state"
    machine = machine_at(directory)
    machine.seed_identity(placeholder_identity())
    machine.append(
        "enrollNode",
        {"name": "gpu-box", "role": "agent", "url": "http://192.0.2.7:59079/", "enrolledAt": WHEN},
    )
    machine.append("putComponent", {**COMPONENT})
    machine.snapshot_now()
    machine.append("patchConfig", {"values": {"uiTheme": "dark"}})
    before = tree(directory)

    store, again = start(directory)
    assert retired_conversion.convert(store, again) is None
    assert retired_conversion.run(store, again) is None

    assert tree(directory) == before
    assert again.state.index == machine.state.index


def test_a_directory_that_does_not_exist_yet_is_not_created(tmp_path: Path) -> None:
    """A first start: there is nothing to convert and nothing to write."""
    directory = tmp_path / "state"
    store, machine = start(directory)

    assert retired_conversion.run(store, machine) is None
    assert not directory.exists()


def test_a_standby_is_not_converted(tmp_path: Path) -> None:
    """It cannot append. It converges by following the root, below."""
    directory = tmp_path / "state"
    write_old_root(directory)
    before = tree(directory)
    store, machine = start(directory, role=ROLE_STANDBY)

    assert retired_conversion.convert(store, machine) is None
    assert retired_conversion.run(store, machine) is None

    assert tree(directory) == before
    assert retired_nodes(machine) == ["oldsite"]


# --------------------------------------------------------------------------- #
# Each kind of retired data is found, alone
# --------------------------------------------------------------------------- #


def _clean_root(directory: Path, identity: dict[str, Any]) -> None:
    """A root in today's form with a node, a person and a component, snapshot
    at index 3."""
    snapshot = {
        **identity_snapshot(identity),
        "components": [dict(COMPONENT)],
        "index": 3,
        "nodes": [gpu_box()],
        "people": [
            {k: v for k, v in person("p-ada", "ada", []).items() if k != "helperGrants"},
        ],
        "sites": [],
    }
    write_root(directory, snapshot, [])


def _with_snapshot(directory: Path, change: Callable[[dict[str, Any]], None]) -> None:
    path = directory / "snapshot.json"
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    change(snapshot)
    path.write_bytes(dump(snapshot).encode("utf-8"))


def _with_tail(directory: Path, *entries: tuple[str, dict[str, Any]]) -> None:
    snapshot = json.loads((directory / "snapshot.json").read_text(encoding="utf-8"))
    first = snapshot["index"] + 1
    lines = [
        dump(
            {
                "at": WHEN,
                "epoch": snapshot["epoch"],
                "index": first + offset,
                "op": op,
                "payload": payload,
            }
        )
        + "\n"
        for offset, (op, payload) in enumerate(entries)
    ]
    (directory / "log.jsonl").write_bytes("".join(lines).encode("utf-8"))


def _a_node_folder_entry(directory: Path) -> None:
    _with_tail(directory, ("putNodeHelper", {"helper": helper_of_gpu_box(enabled=True)}))


def _a_person_entry_with_grants(directory: Path) -> None:
    clean = {k: v for k, v in person("p-ada", "ada", []).items() if k != "helperGrants"}
    _with_tail(directory, ("putPerson", {"person": {**clean, "helperGrants": []}}))


def _a_job_site_enrolled_and_revoked_in_the_log(directory: Path) -> None:
    _with_tail(
        directory,
        ("enrollNode", enrollment(job_site())),
        ("revokeNode", {"name": "oldsite"}),
    )


def _folders_in_the_snapshot(directory: Path) -> None:
    _with_snapshot(directory, lambda s: s.update(nodeHelpers=[helper_of_gpu_box(enabled=True)]))


def _grants_on_a_person_in_the_snapshot(directory: Path) -> None:
    def change(snapshot: dict[str, Any]) -> None:
        snapshot["people"] = [person("p-ada", "ada", [])]

    _with_snapshot(directory, change)


def _a_revoked_job_site_the_snapshot_still_names(directory: Path) -> None:
    """The snapshot carries the Job Site, and the tail that follows revokes it:
    the registry is clean and the file is not."""
    _with_snapshot(directory, lambda s: s["nodes"].append(job_site()))
    _with_tail(directory, ("revokeNode", {"name": "oldsite"}))


def _only_a_job_site_in_the_registry(directory: Path) -> None:
    """What a root that compacted long ago holds: nothing but the node."""
    _with_snapshot(directory, lambda s: s["nodes"].append(job_site()))


KINDS: dict[str, tuple[Callable[[Path], None], tuple[str, ...]]] = {
    "a node-folder entry in the log": (_a_node_folder_entry, ()),
    "a person's folder grants in the log": (_a_person_entry_with_grants, ()),
    "a Job Site enrolled and revoked in the log": (_a_job_site_enrolled_and_revoked_in_the_log, ()),
    "node folders in the snapshot": (_folders_in_the_snapshot, ()),
    "a person's folder grants in the snapshot": (_grants_on_a_person_in_the_snapshot, ()),
    "a revoked Job Site the snapshot still names": (
        _a_revoked_job_site_the_snapshot_still_names,
        (),
    ),
    "a Job Site in the registry and nothing else": (_only_a_job_site_in_the_registry, ("oldsite",)),
}


@pytest.mark.parametrize("kind", list(KINDS))
def test_each_kind_of_retired_data_is_found_on_its_own(tmp_path: Path, kind: str) -> None:
    """A root can hold any one of them without the others, and each is a
    reason to convert. Only a Job Site in the registry is something to
    revoke; the rest are folded away by the snapshot."""
    directory = tmp_path / "state"
    _clean_root(directory, placeholder_identity())
    plant, revoke = KINDS[kind]
    plant(directory)
    assert retired_on_disk(directory), "the case planted nothing"
    before = tree(directory)
    store, machine = start(directory)

    done = retired_conversion.convert(store, machine)

    assert done is not None, f"{kind}: not found"
    assert done.revoked == revoke
    assert retired_on_disk(directory) == []
    assert (directory / MARKER_FILENAME).is_file()
    for name in ("log.jsonl", "snapshot.json"):
        assert (directory / copy_name(name)).read_bytes() == before[name]
    assert retired_nodes(machine) == []
    assert machine_at(directory).canonical() == machine.canonical()
    # And once is enough.
    store, machine = start(directory)
    assert retired_conversion.convert(store, machine) is None


# --------------------------------------------------------------------------- #
# A kill at any step is finished by the next start
# --------------------------------------------------------------------------- #


class Killed(BaseException):
    """The process went away. A `BaseException`, so nothing in the code under
    test that catches `Exception` can swallow it and carry on."""


def _kill_after_first_copy(m: pytest.MonkeyPatch) -> None:
    real = retired_conversion._private_files.write_private_bytes
    count = 0

    def write(path: Path, data: bytes) -> None:
        nonlocal count
        real(path, data)
        count += 1
        if count == 1:
            raise Killed

    m.setattr(retired_conversion._private_files, "write_private_bytes", write)


def _kill_after_the_copies(m: pytest.MonkeyPatch) -> None:
    def kill(directory: Path) -> None:
        raise Killed

    m.setattr(retired_conversion, "_fsync_directory", kill)


def _kill_after_first_revoke(m: pytest.MonkeyPatch) -> None:
    real = StateMachine.append
    count = 0

    def append(self: StateMachine, op: str, payload: dict[str, Any], **kw: Any) -> dict[str, Any]:
        nonlocal count
        count += 1
        if count == 2:
            raise Killed
        return real(self, op, payload, **kw)

    m.setattr(StateMachine, "append", append)


def _kill_before_the_snapshot(m: pytest.MonkeyPatch) -> None:
    def kill(self: StateMachine) -> None:
        raise Killed

    m.setattr(StateMachine, "snapshot_now", kill)


def _kill_between_the_snapshot_and_the_compaction(m: pytest.MonkeyPatch) -> None:
    def kill(self: LogStore, through_index: int) -> None:
        raise Killed

    m.setattr(LogStore, "_compact", kill)


def _kill_before_the_record(m: pytest.MonkeyPatch) -> None:
    def kill(path: Path, text: str, **kw: Any) -> None:
        raise Killed

    m.setattr(retired_conversion._private_files, "write_private_text", kill)


def _snapshot_index(directory: Path) -> int:
    return int(json.loads((directory / "snapshot.json").read_text(encoding="utf-8"))["index"])


def _log_indexes(directory: Path) -> list[int]:
    return [int(e["index"]) for e in LogStore(directory).load().entries]


def _disk_after_the_first_copy(directory: Path, original: dict[str, bytes]) -> None:
    assert sorted(n for n in tree(directory) if n.endswith(BACKUP_SUFFIX)) == [
        copy_name("log.jsonl")
    ]
    assert tree(directory)["log.jsonl"] == original["log.jsonl"]


def _disk_after_both_copies(directory: Path, original: dict[str, bytes]) -> None:
    killed = tree(directory)
    assert len([n for n in killed if n.endswith(BACKUP_SUFFIX)]) == 2
    assert killed["log.jsonl"] == original["log.jsonl"]
    assert killed["snapshot.json"] == original["snapshot.json"]


def _disk_after_the_first_revoke(directory: Path, original: dict[str, bytes]) -> None:
    assert len(tree(directory)["log.jsonl"]) > len(original["log.jsonl"])
    assert len(retired_nodes(machine_at(directory))) == 1, "one of the two was revoked, and durably"


def _disk_after_the_revokes(directory: Path, original: dict[str, bytes]) -> None:
    assert retired_nodes(machine_at(directory)) == []
    assert retired_on_disk(directory), "the files still carry what was retired"
    assert tree(directory)["snapshot.json"] == original["snapshot.json"]


def _disk_between_the_snapshot_and_the_compaction(
    directory: Path, original: dict[str, bytes]
) -> None:
    """Both files are good and the log still holds entries the snapshot covers:
    the state a root that cannot start would be left in without
    `StateMachine.load` skipping them."""
    indexes = _log_indexes(directory)
    assert indexes and indexes[0] <= _snapshot_index(directory) == indexes[-1]
    assert retired_on_disk(directory), "the log was not compacted yet"
    assert machine_at(directory).state.index == _snapshot_index(directory)


def _disk_after_the_compaction(directory: Path, original: dict[str, bytes]) -> None:
    assert _log_indexes(directory) == []
    assert retired_on_disk(directory) == []


CRASHES: dict[
    str, tuple[Callable[[pytest.MonkeyPatch], None], Callable[[Path, dict[str, bytes]], None]]
] = {
    "after the first copy": (_kill_after_first_copy, _disk_after_the_first_copy),
    "after both copies": (_kill_after_the_copies, _disk_after_both_copies),
    "after the first of two revokes": (_kill_after_first_revoke, _disk_after_the_first_revoke),
    "after the revokes, before the snapshot": (_kill_before_the_snapshot, _disk_after_the_revokes),
    "between the snapshot and the compaction": (
        _kill_between_the_snapshot_and_the_compaction,
        _disk_between_the_snapshot_and_the_compaction,
    ),
    "after the compaction, before the record": (
        _kill_before_the_record,
        _disk_after_the_compaction,
    ),
}


@pytest.mark.parametrize("era", ["log", "compacted"])
@pytest.mark.parametrize("point", list(CRASHES))
def test_a_kill_midway_is_finished_by_the_next_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, point: str, era: str
) -> None:
    directory = tmp_path / "state"
    write_old_root(directory, era=era, extra_sites=("attic-site",))
    original = tree(directory)
    store, machine = start(directory)
    assert retired_nodes(machine) == ["attic-site", "oldsite"]
    kill, disk = CRASHES[point]

    with monkeypatch.context() as m:
        kill(m)
        with pytest.raises(Killed):
            retired_conversion.convert(store, machine)

    # The kill really landed where it was meant to, and nothing says it is done.
    disk(directory, original)
    assert not (directory / MARKER_FILENAME).exists()

    # The next start can start: whatever the kill left loads.
    store, machine = start(directory)
    assert retired_conversion.run(store, machine) is None

    # Finished: only the current form, the files as they were kept, a record.
    assert retired_nodes(machine) == []
    assert retired_on_disk(directory) == []
    assert machine_at(directory).canonical() == machine.canonical()
    for name in ("log.jsonl", "snapshot.json"):
        assert (directory / copy_name(name)).read_bytes() == original[name], (
            "the copy is of a file that was already partly converted, not of the original"
        )
    record = json.loads((directory / MARKER_FILENAME).read_text(encoding="utf-8"))
    assert record["resumedAfterAnInterruptedRun"] is True
    assert sorted(machine.state.nodes) == ["gpu-box"]
    # And a third start has nothing left to do.
    again = tree(directory)
    store, machine = start(directory)
    assert retired_conversion.convert(store, machine) is None
    assert tree(directory) == again


def test_a_copy_that_is_already_there_is_not_replaced(tmp_path: Path) -> None:
    """It is the files as they were. This run's files may already be partly
    converted, and a copy made now would be a copy of that."""
    directory = tmp_path / "state"
    write_old_root(directory)
    original = tree(directory)
    (directory / copy_name("log.jsonl")).write_bytes(original["log.jsonl"])
    (directory / copy_name("snapshot.json")).write_bytes(original["snapshot.json"])
    store, machine = start(directory)
    machine.append("patchConfig", {"values": {"uiTheme": "dark"}})

    done = retired_conversion.convert(store, machine)

    assert done is not None and done.resumed is True
    for name in ("log.jsonl", "snapshot.json"):
        assert (directory / copy_name(name)).read_bytes() == original[name]


# --------------------------------------------------------------------------- #
# A failure degrades, and says why
# --------------------------------------------------------------------------- #


def tree_with_dirs(directory: Path) -> dict[str, bytes | None]:
    return {p.name: (p.read_bytes() if p.is_file() else None) for p in sorted(directory.iterdir())}


def test_a_failure_to_keep_a_copy_degrades_and_changes_nothing(tmp_path: Path) -> None:
    """A directory has the copy's name: a real failure of the filesystem's
    kind, rather than a mock. Nothing is touched when the copies cannot be
    kept, because the copies are what make every later step safe."""
    directory = tmp_path / "state"
    write_old_root(directory)
    (directory / copy_name("log.jsonl")).mkdir()
    before = tree_with_dirs(directory)
    store, machine = start(directory)

    cause = retired_conversion.run(store, machine)

    assert cause is not None
    assert cause.startswith("keeping a copy of the log and the snapshot: OSError:")
    assert copy_name("log.jsonl") in cause and "in the way" in cause
    assert tree_with_dirs(directory) == before
    assert not (directory / MARKER_FILENAME).exists()
    assert retired_nodes(machine) == ["oldsite"] and machine.state.index == 9

    # Fixed, the next start converts.
    (directory / copy_name("log.jsonl")).rmdir()
    store, machine = start(directory)
    assert retired_conversion.run(store, machine) is None
    assert retired_on_disk(directory) == []
    assert (directory / MARKER_FILENAME).is_file()


def test_a_failure_to_write_the_snapshot_degrades_and_the_next_start_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    directory = tmp_path / "state"
    write_old_root(directory)
    original = tree(directory)
    store, machine = start(directory)

    def full(self: StateMachine) -> None:
        raise OSError(28, "No space left on device")

    with monkeypatch.context() as m, caplog.at_level(logging.ERROR):
        m.setattr(StateMachine, "snapshot_now", full)
        cause = retired_conversion.run(store, machine)

    assert cause is not None
    assert cause.startswith("writing the snapshot that folds the log: OSError:")
    assert "No space left on device" in cause
    # The root is still the install it was: the revoke is in its log and its
    # registry, the copies are kept, and nothing says it was converted.
    assert retired_nodes(machine) == []
    assert machine.state.index == 10
    assert not (directory / MARKER_FILENAME).exists()
    for name in ("log.jsonl", "snapshot.json"):
        assert (directory / copy_name(name)).read_bytes() == original[name]
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR] == [
        "could not convert this root's retired data (" + cause + "). It keeps serving what it has "
        "and tries again at the next start. The files as they were are kept as "
        "log.jsonl.v0.1.bak, snapshot.json.v0.1.bak."
    ]

    store, machine = start(directory)
    assert retired_conversion.run(store, machine) is None
    assert retired_on_disk(directory) == []
    record = json.loads((directory / MARKER_FILENAME).read_text(encoding="utf-8"))
    assert record["resumedAfterAnInterruptedRun"] is True


def test_a_failure_to_append_names_the_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "state"
    write_old_root(directory)
    store, machine = start(directory)

    def refuse(self: StateMachine, op: str, payload: dict[str, Any], **kw: Any) -> dict[str, Any]:
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(StateMachine, "append", refuse)

    cause = retired_conversion.run(store, machine)

    assert (
        cause
        == "revoking the retired Job Site node 'oldsite': PermissionError: [Errno 13] Permission denied"
    )
    assert not (directory / MARKER_FILENAME).exists()
    assert retired_nodes(machine) == ["oldsite"]


def test_a_failure_to_record_it_leaves_the_conversion_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the record is missing. The next start sees copies with nothing
    recording them and writes it."""
    directory = tmp_path / "state"
    write_old_root(directory)
    store, machine = start(directory)

    def refuse(path: Path, text: str, **kw: Any) -> None:
        raise PermissionError(13, "Permission denied", str(path))

    with monkeypatch.context() as m:
        m.setattr(retired_conversion._private_files, "write_private_text", refuse)
        cause = retired_conversion.run(store, machine)
    assert cause is not None and cause.startswith(f"recording the conversion in {MARKER_FILENAME}")
    assert retired_on_disk(directory) == []

    store, machine = start(directory)
    assert retired_conversion.run(store, machine) is None
    assert (directory / MARKER_FILENAME).is_file()


def test_a_bug_in_the_finder_degrades_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`run` is the one thing startup calls, and nothing may escape it."""
    directory = tmp_path / "state"
    write_old_root(directory)
    store, machine = start(directory)

    def broken(store: LogStore, machine: StateMachine) -> Any:
        raise ValueError("unexpected shape")

    monkeypatch.setattr(retired_conversion, "find", broken)

    assert (
        retired_conversion.run(store, machine)
        == "looking for retired data: ValueError: unexpected shape"
    )


# --------------------------------------------------------------------------- #
# The root that starts: it serves either way
# --------------------------------------------------------------------------- #


def _signed_in(client: TestClient) -> TestClient:
    client.headers["Authorization"] = f"Bearer {login(client)}"
    return client


def test_a_root_started_on_old_files_converts_and_serves(
    tmp_path: Path, real_identity: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    settings = settings_for(tmp_path / "root")
    write_old_root(Path(settings.state_dir), identity=real_identity)

    with caplog.at_level(logging.INFO), TestClient(create_app(settings)) as client:
        health = client.get("/healthz").json()
        assert health["status"] == "ok"
        assert health["details"]["appliedIndex"] == 10
        assert "conversionFailed" not in health["details"]
        listed = _signed_in(client).get("/v1/nodes")
        assert [n["name"] for n in listed.json()["nodes"]] == ["gpu-box"]

    assert retired_on_disk(Path(settings.state_dir)) == []
    messages = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("converted this root's retired data: revoked 1") for m in messages)


def test_a_root_that_cannot_convert_still_starts_and_says_so(
    tmp_path: Path, real_identity: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    settings = settings_for(tmp_path / "root")
    directory = Path(settings.state_dir)
    write_old_root(directory, identity=real_identity)
    (directory / copy_name("snapshot.json")).mkdir()

    with caplog.at_level(logging.ERROR), TestClient(create_app(settings)) as client:
        health = client.get("/healthz").json()
        assert health["status"] == "degraded"
        cause = health["details"]["conversionFailed"]
        assert cause.startswith("keeping a copy of the log and the snapshot: OSError:")
        assert copy_name("snapshot.json") in cause
        # It is the install it was: it serves what it has.
        listed = _signed_in(client).get("/v1/nodes")
        assert listed.status_code == 200
        assert [n["name"] for n in listed.json()["nodes"]] == ["gpu-box"]

    assert any(
        "could not convert this root's retired data" in r.getMessage() for r in caplog.records
    )
    assert not (directory / MARKER_FILENAME).exists()

    # Fixed, the next start converts and the report clears.
    (directory / copy_name("snapshot.json")).rmdir()
    with TestClient(create_app(settings)) as client:
        health = client.get("/healthz").json()
        assert health["status"] == "ok" and "conversionFailed" not in health["details"]
    assert (directory / MARKER_FILENAME).is_file()


def test_a_standby_process_started_on_old_files_does_not_convert(
    tmp_path: Path, real_identity: dict[str, Any]
) -> None:
    settings = settings_for(tmp_path / "standby", role="standby")
    directory = Path(settings.state_dir)
    write_old_root(directory, identity=real_identity)
    before = tree(directory)

    with TestClient(create_app(settings)) as client:
        assert client.get("/healthz").json()["details"]["role"] == "standby"

    assert tree(directory) == before


# --------------------------------------------------------------------------- #
# A standby that follows the root converges
# --------------------------------------------------------------------------- #


def _standby_copy(tmp_path: Path, name: str, *, entries: int | None) -> StateMachine:
    """A standby holding a copy of the root's files from before the conversion,
    cut to its first `entries` entries when it is behind (era `log`)."""
    directory = tmp_path / name
    shutil.copytree(tmp_path / "pristine", directory)
    if entries is not None:
        lines = (directory / "log.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
        (directory / "log.jsonl").write_text("".join(lines[:entries]), encoding="utf-8", newline="")
    return standby_at(directory)


@pytest.fixture
def old_root(tmp_path: Path, real_identity: dict[str, Any]) -> Iterator[Path]:
    """The root's directory, in a dev build's form, and a pristine copy."""
    settings = settings_for(tmp_path / "active")
    write_old_root(Path(settings.state_dir), era="log", identity=real_identity)
    shutil.copytree(Path(settings.state_dir), tmp_path / "pristine")
    yield Path(settings.state_dir)


def _token(server: _LiveServer) -> str:
    with httpx.Client(base_url=server.url) as client:
        return str(
            client.post("/v1/auth/login", json={"passphrase": PASSPHRASE}).json()["sessionToken"]
        )


@pytest.mark.parametrize("entries", [None, 3], ids=["caught up", "far behind"])
async def test_a_standby_behind_the_compaction_takes_the_snapshot(
    tmp_path: Path, old_root: Path, caplog: pytest.LogCaptureFixture, entries: int | None
) -> None:
    """After the conversion the root's log starts after the revoke it wrote,
    so even a standby that had everything before asks for entries that are no
    longer there: a 409, and the snapshot instead. Its own files come out in
    the current form, because installing a snapshot rewrites them."""
    standby = _standby_copy(tmp_path, "standby", entries=entries)
    assert retired_on_disk(tmp_path / "standby")
    app = create_app(settings_for(tmp_path / "active"))
    with _LiveServer(app) as server:
        root: StateMachine = app.state.machine
        assert retired_nodes(root) == []
        token = _token(server)
        follower = Follower(
            standby,
            active_url=server.url,
            interval_seconds=0.01,
            token_provider=lambda: token,
        )
        try:
            with caplog.at_level(logging.INFO):
                await follower.tick()
            assert any(
                "fell behind the active root's compaction" in r.getMessage() for r in caplog.records
            )
            assert standby.state.index == root.state.index == 10
            assert standby.canonical() == root.canonical()
            assert retired_nodes(standby) == []
            assert retired_on_disk(tmp_path / "standby") == []
            assert machine_at(tmp_path / "standby").canonical() == root.canonical()
        finally:
            await follower.stop()


async def test_a_standby_follows_the_revoke_when_the_conversion_could_not_compact(
    tmp_path: Path,
    old_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The conversion's appends are ordinary ones. When the snapshot step
    fails, the revoke is in the log like any other entry and a caught-up
    standby reads it, with no snapshot: that is why a conversion appends
    rather than rewrites. The next start of the root finishes, and the
    standby is where the root is."""
    standby = _standby_copy(tmp_path, "standby", entries=None)
    assert retired_nodes(standby) == ["oldsite"]

    def full(self: StateMachine) -> None:
        raise OSError(28, "No space left on device")

    app = create_app(settings_for(tmp_path / "active"))
    with monkeypatch.context() as m:
        m.setattr(StateMachine, "snapshot_now", full)
        with _LiveServer(app) as server:
            assert app.state.conversion_failure is not None
            root: StateMachine = app.state.machine
            token = _token(server)
            follower = Follower(
                standby,
                active_url=server.url,
                interval_seconds=0.01,
                token_provider=lambda: token,
            )
            try:
                with caplog.at_level(logging.INFO):
                    await follower.tick()
                assert not any("re-snapshotting" in r.getMessage() for r in caplog.records)
                assert standby.state.index == root.state.index == 10
                assert standby.canonical() == root.canonical()
                assert retired_nodes(standby) == []
            finally:
                await follower.stop()

    # The root's next start finishes the job; the standby, already there,
    # needs nothing from it.
    again = create_app(settings_for(tmp_path / "active"))
    with _LiveServer(again) as server:
        assert again.state.conversion_failure is None
        token = _token(server)
        follower = Follower(
            standby,
            active_url=server.url,
            interval_seconds=0.01,
            token_provider=lambda: token,
        )
        try:
            await follower.tick()
            assert standby.canonical() == again.state.machine.canonical()
        finally:
            await follower.stop()
    assert retired_on_disk(old_root) == []
    assert (old_root / MARKER_FILENAME).is_file()
