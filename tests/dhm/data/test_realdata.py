"""Invariant checks against a real Lyncée Tec Koala time-lapse.

Opt-in: place time-lapses under `tests/fixtures/` (see `tests/conftest.py`), then
`pytest -m realdata`. Every test is parametrized over each time-lapse found and
skipped when none are present.

These assert ground-truth invariants that must hold for any genuine Koala export:
the `Float/Bin` and `Float/Txt` serializations of the same frames decode alike,
`to_image` reproduces Koala's own uint8 previews, the computed nm bounds match
`phbounds.txt`, holograms re-encode losslessly, and the per-frame modalities share
one frame count. They catch format-reading
bugs the synthetic suite cannot, since that suite feeds our readers our own
writers' output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from iivs.dhm.data.hologram import (
    HologramNpyFolder,
    HologramRawFile,
    HologramTifFolder,
    convert_hologram_sequence,
)
from iivs.dhm.data.intensity import IntensityBinFolder, IntensityTxtFolder
from iivs.dhm.data.phase import (
    PhaseBinFolder,
    PhaseTifFolder,
    PhaseTxtFolder,
    PhaseUnit,
    read_phbounds,
)
from iivs.dhm.data.timestamp import TimestampsTxtFile

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.realdata


def _require(path: Path) -> Path:
    """Return `path`, or skip the test when the time-lapse lacks it (absent or empty).

    A folder that exists but holds no files (e.g. a reconstruction not yet
    regenerated) skips too, so a partially built time-lapse fails softly rather than
    erroring in a reader that rejects an empty folder.
    """
    if not path.exists():
        pytest.skip(f"time-lapse is missing {path.name}")
    if path.is_dir() and not any(path.iterdir()):
        pytest.skip(f"time-lapse folder {path.name} is empty")
    return path


def _sample_indices(n: int) -> tuple[int, ...]:
    """A first / middle / last spread of frame indices over a length-`n` sequence.

    A real time-lapse holds hundreds to thousands of frames; decoding just these keeps
    the suite fast while still exercising both ends of the sequence.
    """
    if n <= 0:
        return ()
    return tuple(sorted({0, n // 2, n - 1}))


def _open_holograms(timelapse: Path) -> HologramRawFile | HologramTifFolder:
    """The time-lapse's holograms: a single `.raw` stack or a per-frame tif folder."""
    holo_dir = _require(timelapse / "Holograms")
    raws = sorted(holo_dir.glob("*.raw"))
    if raws:
        return HologramRawFile(raws[0])
    return HologramTifFolder(holo_dir, validate=None)


# ============================== #
#             phase              #
# ============================== #


def test_phase_bin_folder_loads(koala_timelapse: Path) -> None:
    folder = PhaseBinFolder(
        _require(koala_timelapse / "Phase" / "Float" / "Bin"), validate=None
    )
    assert len(folder) > 0

    height, width = folder.frame_shape
    assert height > 0
    assert width > 0

    header = folder.header
    assert header.pixel_size > 0
    assert np.isfinite(header.height_scale)
    assert header.height_scale != 0

    frame = folder[0]
    assert frame.dtype == np.float32
    assert frame.shape == folder.frame_shape


def test_phase_bin_matches_txt(koala_timelapse: Path) -> None:
    # Koala's Float/Bin and Float/Txt hold the same frames; the two readers must
    # agree to within the text export's ~float32 precision (measured max abs diff
    # ~5e-7 on real data).
    bin_folder = PhaseBinFolder(
        _require(koala_timelapse / "Phase" / "Float" / "Bin"), validate=None
    )
    txt_folder = PhaseTxtFolder(
        _require(koala_timelapse / "Phase" / "Float" / "Txt"), validate=None
    )
    assert len(bin_folder) == len(txt_folder)

    for i in _sample_indices(len(bin_folder)):
        b, t = bin_folder[i], txt_folder[i]
        assert b.shape == t.shape
        assert np.allclose(b, t, rtol=1e-5, atol=1e-5), f"bin/txt disagree at frame {i}"


def test_phase_to_image_reproduces_koala_previews(koala_timelapse: Path) -> None:
    # to_image renders the Float source through phbounds.txt; it must match Koala's
    # own uint8 previews to within one 8-bit code (measured max code diff == 1).
    bin_folder = PhaseBinFolder(
        _require(koala_timelapse / "Phase" / "Float" / "Bin"), validate=None
    )
    previews = PhaseTifFolder(
        _require(koala_timelapse / "Phase" / "Image"), validate=None
    )
    bounds = read_phbounds(_require(koala_timelapse / "phbounds.txt"))
    assert len(bin_folder) == len(previews)

    rendered = bin_folder.to_image(bounds)
    for i in _sample_indices(len(bin_folder)):
        ours = rendered[i].astype(np.int16)
        koala = previews[i].astype(np.int16)
        assert ours.shape == koala.shape
        assert np.abs(ours - koala).max() <= 1, (
            f"preview differs by >1 code at frame {i}"
        )


def test_phase_bounds_match_phbounds(koala_timelapse: Path) -> None:
    # `value_range(NANOMETERS)` reduces the Float source to one global (min, max) in
    # nm; Koala's phbounds.txt is that same global range, so the two agree to within
    # the text file's rounding (measured < 1e-4 nm). Reads every frame, so this stays
    # cheap only because a fixture is a short (~20-frame) clip.
    folder = PhaseBinFolder(
        _require(koala_timelapse / "Phase" / "Float" / "Bin"), validate=None
    )
    stored = read_phbounds(_require(koala_timelapse / "phbounds.txt"))
    low, high = folder.value_range(unit=PhaseUnit.NANOMETERS)
    assert low == pytest.approx(stored.min_nm, rel=1e-4, abs=1e-3)
    assert high == pytest.approx(stored.max_nm, rel=1e-4, abs=1e-3)


# ============================== #
#           intensity            #
# ============================== #


def test_intensity_bin_matches_txt(koala_timelapse: Path) -> None:
    bin_folder = IntensityBinFolder(
        _require(koala_timelapse / "Intensity" / "Float" / "Bin"), validate=None
    )
    txt_folder = IntensityTxtFolder(
        _require(koala_timelapse / "Intensity" / "Float" / "Txt"), validate=None
    )
    assert len(bin_folder) == len(txt_folder)

    for i in _sample_indices(len(bin_folder)):
        b, t = bin_folder[i], txt_folder[i]
        assert b.dtype == np.float32
        assert np.allclose(b, t, rtol=1e-5, atol=1e-5), f"bin/txt disagree at frame {i}"


# ============================== #
#            hologram            #
# ============================== #


def test_hologram_loads(koala_timelapse: Path) -> None:
    holos = _open_holograms(koala_timelapse)
    assert len(holos) > 0

    frame = holos[0]
    assert frame.dtype == np.uint8
    assert frame.ndim == 2
    assert frame.shape == holos.frame_shape


def test_hologram_reencode_is_lossless(koala_timelapse: Path, tmp_path: Path) -> None:
    # Holograms are uint8 with no quantization, so raw/tif -> npy round-trips exactly.
    holos = _open_holograms(koala_timelapse)
    n = min(3, len(holos))
    dest = tmp_path / "holo_npy"

    convert_hologram_sequence(dest, holos[:n], ext="npy")
    reloaded = HologramNpyFolder(dest)

    assert len(reloaded) == n
    for i in range(n):
        np.testing.assert_array_equal(reloaded[i], holos[i])


# ============================== #
#      timestamps / counts       #
# ============================== #


def test_timestamps_load(koala_timelapse: Path) -> None:
    ts = TimestampsTxtFile(_require(koala_timelapse / "timestamps.txt"))
    assert len(ts) > 0
    assert ts.mean_frame_rate > 0
    assert ts.mean_interval_ms > 0
    assert ts[0].interval_ms == 0.0  # no gap before the first frame

    elapsed = [ts[i].elapsed_ms for i in _sample_indices(len(ts))]
    assert elapsed == sorted(elapsed)  # elapsed time is non-decreasing


def test_frame_counts_agree(koala_timelapse: Path) -> None:
    # One time-lapse -> every per-frame modality has the same count.
    counts = {
        "phase_bin": len(
            PhaseBinFolder(
                _require(koala_timelapse / "Phase" / "Float" / "Bin"), validate=None
            )
        ),
        "phase_txt": len(
            PhaseTxtFolder(
                _require(koala_timelapse / "Phase" / "Float" / "Txt"), validate=None
            )
        ),
        "intensity_bin": len(
            IntensityBinFolder(
                _require(koala_timelapse / "Intensity" / "Float" / "Bin"), validate=None
            )
        ),
        "timestamps": len(
            TimestampsTxtFile(_require(koala_timelapse / "timestamps.txt"))
        ),
        "holograms": len(_open_holograms(koala_timelapse)),
    }
    assert len(set(counts.values())) == 1, f"frame counts disagree: {counts}"
