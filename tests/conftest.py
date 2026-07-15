from __future__ import annotations

from pathlib import Path

import pytest

# Real Koala time-lapses live in the git-ignored `tests/fixtures/` (one is ~GB, so it
# never lives in the repo). When the folder is absent or holds none, the real-data suite
# skips instead of failing.
_FIXTURES_ROOT = Path(__file__).parent / "fixtures"


def _is_timelapse(path: Path) -> bool:
    """Whether `path` is a Koala time-lapse folder (holds `Phase/` or `Holograms/`)."""
    if not path.is_dir():
        return False
    return (path / "Phase").is_dir() or (path / "Holograms").is_dir()


def _timelapses() -> list[Path]:
    """The time-lapse folders under `tests/fixtures/`, sorted (empty if absent)."""
    if not _FIXTURES_ROOT.is_dir():
        return []
    return sorted(child for child in _FIXTURES_ROOT.iterdir() if _is_timelapse(child))


@pytest.fixture(params=_timelapses(), ids=lambda timelapse: timelapse.name)
def koala_timelapse(request: pytest.FixtureRequest) -> Path:
    """One real Lyncée Tec Koala time-lapse: its holograms, timing, and reconstructions.

    Opt-in: drop time-lapses under `tests/fixtures/`. With none present (CI, or a
    contributor without the proprietary data) the parameter set is empty, so every test
    requesting this fixture is skipped rather than failing.
    """
    return request.param
