from __future__ import annotations

import os
from pathlib import Path

import pytest


def _acquisition_dirs() -> list[Path]:
    """Real Koala acquisition folders under `$IIVS_KOALA_DATA`.

    Each acquisition is an immediate subdirectory holding a `Phase/` or `Holograms/`
    tree. Returns `[]` when the variable is unset or points nowhere usable, which
    turns the parametrized `koala_acq` fixture into a skip.
    """
    root = os.environ.get("IIVS_KOALA_DATA")
    if not root:
        return []
    base = Path(root)
    if not base.is_dir():
        return []
    return sorted(
        d
        for d in base.iterdir()
        if d.is_dir() and ((d / "Phase").is_dir() or (d / "Holograms").is_dir())
    )


@pytest.fixture(params=_acquisition_dirs(), ids=lambda p: p.name)
def koala_acq(request: pytest.FixtureRequest) -> Path:
    """A real Lyncée Tec Koala acquisition root, one per fixture parameter.

    Opt-in: set `IIVS_KOALA_DATA` to a directory of acquisition folders. Unset (CI,
    or a contributor without the proprietary data) leaves the parameter set empty,
    so every test requesting this fixture is skipped rather than failing.
    """
    return request.param
