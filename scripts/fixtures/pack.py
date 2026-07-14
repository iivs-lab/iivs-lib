# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Pack each Koala sample under `tests/fixtures/` into a `.tar.xz` for release.

Writes `<sample>.tar.xz` to the output dir (default the git-ignored
`.cache/fixtures-dist/`), records each archive's SHA256 in
`scripts/fixtures/lock.json`, and prints the `gh` commands to publish them to the
private data repo. Run this after regenerating fixtures; `fetch.py` is the inverse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
LOCK = HERE / "lock.json"
DEFAULT_OUT = ROOT / ".cache" / "fixtures-dist"
_DEFAULT_LOCK: dict[str, object] = {
    "repository": "iivs-lab/iivs-lib-fixtures",
    "release": "v1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--preset", type=int, default=9, help="xz preset 0-9")
    parser.add_argument(
        "--extreme",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="add the xz EXTREME flag (a little smaller, much slower)",
    )
    args = parser.parse_args()
    preset = args.preset | (lzma.PRESET_EXTREME if args.extreme else 0)

    args.out.mkdir(parents=True, exist_ok=True)
    lock: dict[str, object] = dict(_DEFAULT_LOCK)
    if LOCK.exists():
        lock.update(json.loads(LOCK.read_text()))

    samples: dict[str, dict[str, str]] = {}
    for sample in sorted(p for p in FIXTURES.iterdir() if p.is_dir()):
        archive = args.out / f"{sample.name}.tar.xz"
        print(f"packing {sample.name} -> {archive.name} ...")
        with tarfile.open(archive, "w:xz", preset=preset) as tar:
            tar.add(sample, arcname=sample.name)
        digest = _sha256(archive)
        print(f"  {archive.stat().st_size / 1e6:.1f} MB  sha256={digest[:16]}...")
        samples[sample.name] = {"archive": archive.name, "sha256": digest}

    lock["samples"] = samples
    LOCK.write_text(json.dumps(lock, indent=2) + "\n")
    print(f"\nwrote {LOCK.relative_to(ROOT)}")

    repo, tag = lock["repository"], lock["release"]
    assets = " ".join(f'"{args.out / s["archive"]}"' for s in samples.values())
    # Immutable releases lock assets at publish, so create a draft, attach the
    # assets, then publish.
    print("\n# publish (draft -> attach assets -> publish):")
    print(
        f"gh release create {tag} -R {repo} -t {tag} -n 'Koala test fixtures' --draft"
    )
    print(f"gh release upload {tag} -R {repo} {assets}")
    print(f"gh release edit {tag} -R {repo} --draft=false")


if __name__ == "__main__":
    main()
