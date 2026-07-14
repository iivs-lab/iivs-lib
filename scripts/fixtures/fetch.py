# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Download and extract the Koala test fixtures into `tests/fixtures/`.

Reads `scripts/fixtures/lock.json` (the pinned repo, release, and per-sample
SHA256), downloads each sample's `.tar.xz` from the private data repo's release
with the GitHub CLI (`gh`, which must be authenticated), verifies its checksum,
and extracts it. A sample already present with a matching checksum is left alone.
`pack.py` is the inverse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
LOCK = HERE / "lock.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="re-download present samples"
    )
    args = parser.parse_args()

    lock = json.loads(LOCK.read_text())
    repo, tag = lock["repository"], lock["release"]
    FIXTURES.mkdir(parents=True, exist_ok=True)

    for name, meta in lock["samples"].items():
        target = FIXTURES / name
        if target.is_dir() and any(target.iterdir()) and not args.force:
            print(f"{name}: present, skipping")
            continue

        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / meta["archive"]
            print(f"{name}: downloading {meta['archive']} from {repo}@{tag} ...")
            subprocess.run(
                [
                    "gh",
                    "release",
                    "download",
                    tag,
                    "-R",
                    repo,
                    "-p",
                    meta["archive"],
                    "-D",
                    tmp,
                ],
                check=True,
            )
            digest = _sha256(archive)
            if digest != meta["sha256"]:
                msg = f"{name}: checksum mismatch\n  want {meta['sha256']}\n  got  {digest}"
                raise SystemExit(msg)

            print(f"{name}: checksum OK, extracting ...")
            shutil.rmtree(target, ignore_errors=True)
            with tarfile.open(archive, "r:xz") as tar:
                tar.extractall(FIXTURES, filter="data")

    print("done")


if __name__ == "__main__":
    main()
