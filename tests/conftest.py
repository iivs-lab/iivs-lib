from __future__ import annotations

from pathlib import Path

import pytest

# Real Koala samples live in the git-ignored `tests/fixtures/` (a sample is ~GB, so
# it never lives in the repo). When the folder is absent or holds no samples, the
# real-data suite skips instead of failing.
_FIXTURES_ROOT = Path(__file__).parent / "fixtures"


def _is_sample(path: Path) -> bool:
    """Test whether `path` is a Koala sample folder (holds `Phase/` or `Holograms/`)."""
    if not path.is_dir():
        return False
    return (path / "Phase").is_dir() or (path / "Holograms").is_dir()


def _samples() -> list[Path]:
    """The Koala sample folders under `tests/fixtures/`, sorted by name (empty if absent)."""
    if not _FIXTURES_ROOT.is_dir():
        return []
    return sorted(child for child in _FIXTURES_ROOT.iterdir() if _is_sample(child))


@pytest.fixture(params=_samples(), ids=lambda sample: sample.name)
def koala_sample(request: pytest.FixtureRequest) -> Path:
    """One real Lyncée Tec Koala sample: its holograms, timing, and reconstructions.

    Opt-in: drop samples under `tests/fixtures/`. With none present (CI, or a
    contributor without the proprietary data) the parameter set is empty, so every
    test requesting this fixture is skipped rather than failing.
    """
    return request.param
