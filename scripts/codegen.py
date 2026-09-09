"""Regenerate Pydantic models from the pinned specs commit.

Reads `SPECS_REF` (a single line: the git SHA of `eugene-plexus/specs`),
downloads the OpenAPI tree at that SHA, and runs `datamodel-code-generator`
to produce Pydantic v2 models under
`src/eugene_plexus_control/_generated/`.

The generated files are committed to the repo so builds are reproducible
without network access. CI re-runs this script and fails the build if the
working tree differs.

Usage:
    python scripts/codegen.py

To bump to a newer specs commit, overwrite `SPECS_REF` and re-run. Write
it with `[IO.File]::WriteAllText` on Windows, never PowerShell's
`Set-Content -Encoding utf8` — the BOM it prepends becomes part of the
ref and the archive URL 404s.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPECS_REF_FILE = REPO_ROOT / "SPECS_REF"
GENERATED_DIR = REPO_ROOT / "src" / "eugene_plexus_control" / "_generated"
WORKING_DIR = REPO_ROOT / ".codegen-cache"

SPECS_TARBALL_URL_TEMPLATE = "https://github.com/eugene-plexus/specs/archive/{ref}.tar.gz"

# Two inputs, two modules. control.yaml pulls what it needs from
# components/common.yaml through local $refs, so `models.py` alone would
# be enough for everything this component *serves* — but the control
# root also *reads* the agent's surface (the union runtime view is
# assembled by calling each node's `/v1/runtimes`), and those responses
# are the agent's schemas.
#
# openapi/control.yaml first appears in specs 811112b (M5), so these
# paths are only valid at a pin at or after that commit. The path is a
# coordinate inside the pinned snapshot rather than a name: it moves
# with SPECS_REF and never on its own.
SPECS_TO_GENERATE = [
    ("openapi/control.yaml", "models.py"),
    ("openapi/agent.yaml", "agent_models.py"),
]


def read_specs_ref() -> str:
    ref = SPECS_REF_FILE.read_text().strip()
    if not ref:
        sys.exit(f"error: {SPECS_REF_FILE} is empty")
    return ref


def download_specs(ref: str) -> Path:
    """Download specs tarball and extract to WORKING_DIR/specs."""
    url = SPECS_TARBALL_URL_TEMPLATE.format(ref=ref)
    print(f"fetching {url}")

    if WORKING_DIR.exists():
        shutil.rmtree(WORKING_DIR)
    WORKING_DIR.mkdir(parents=True)

    with urllib.request.urlopen(url) as response:
        data = response.read()

    extracted_root = None
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(WORKING_DIR)
        for member in tar.getnames():
            top_level = member.split("/", 1)[0]
            if top_level.startswith("specs-"):
                extracted_root = WORKING_DIR / top_level
                break

    if extracted_root is None or not extracted_root.is_dir():
        sys.exit(f"error: could not locate specs-* directory in tarball from {url}")

    return extracted_root


def write_init(generated_dir: Path, ref: str) -> None:
    """Write the package __init__.py with the pinned ref recorded."""
    content = f'''"""Generated Pydantic v2 models — DO NOT EDIT BY HAND.

Regenerate with:

    python scripts/codegen.py

Source: https://github.com/eugene-plexus/specs at commit {ref}
"""

SPECS_REF = "{ref}"
'''
    (generated_dir / "__init__.py").write_text(content, encoding="utf-8")


def run_codegen(specs_root: Path, ref: str) -> None:
    """Run datamodel-code-generator for each configured spec."""
    if GENERATED_DIR.exists():
        shutil.rmtree(GENERATED_DIR)
    GENERATED_DIR.mkdir(parents=True)

    write_init(GENERATED_DIR, ref)

    for input_rel, output_name in SPECS_TO_GENERATE:
        input_path = specs_root / input_rel
        if not input_path.is_file():
            sys.exit(f"error: expected {input_path} in extracted specs tarball")

        output_path = GENERATED_DIR / output_name
        print(f"generating {output_path.relative_to(REPO_ROOT)} from {input_rel}")

        cmd = [
            sys.executable,
            "-m",
            "datamodel_code_generator",
            "--input",
            str(input_path),
            "--input-file-type",
            "openapi",
            "--output",
            str(output_path),
            "--output-model-type",
            "pydantic_v2.BaseModel",
            "--target-python-version",
            "3.12",
            "--use-standard-collections",
            "--use-union-operator",
            "--use-schema-description",
            "--field-constraints",
            "--collapse-root-models",
            "--disable-timestamp",
        ]
        subprocess.run(cmd, check=True)


def main() -> None:
    ref = read_specs_ref()
    print(f"specs ref: {ref}")

    specs_root = download_specs(ref)

    try:
        run_codegen(specs_root, ref)
    finally:
        if WORKING_DIR.exists():
            shutil.rmtree(WORKING_DIR)

    print(f"wrote {GENERATED_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
