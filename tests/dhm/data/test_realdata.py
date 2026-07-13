"""Invariant checks against a real Lyncée Tec Koala acquisition.

Opt-in: set `IIVS_KOALA_DATA` to a directory of acquisition folders (see
`tests/conftest.py`), then `pytest -m realdata`. Every test is parametrized over
each acquisition found and skipped when the variable is unset.

These assert ground-truth invariants that must hold for any genuine Koala export:
the `Float/Bin` and `Float/Txt` serializations of the same frames decode alike,
`to_image` reproduces Koala's own uint8 previews, holograms re-encode losslessly,
and the per-frame modalities share one frame count. They catch format-reading
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
    read_phbounds,
)
from iivs.dhm.data.timestamp import TimestampsTxtFile

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.realdata

# Decode only a few frames; a real acquisition holds hundreds to thousands.
_SAMPLE = (0, 1, 2)


def _require(path: Path) -> Path:
    """Return `path`, or skip the test when the acquisition lacks it."""
    if not path.exists():
        pytest.skip(f"acquisition is missing {path.name}")
    return path


def _sample_indices(n: int) -> tuple[int, ...]:
    """The `_SAMPLE` indices that fall within a sequence of length `n`."""
    return tuple(i for i in _SAMPLE if i < n)


def _open_holograms(acq: Path) -> HologramRawFile | HologramTifFolder:
    """The acquisition's holograms: a single `.raw` stack or a per-frame tif folder."""
    holo_dir = _require(acq / "Holograms")
    raws = sorted(holo_dir.glob("*.raw"))
    if raws:
        return HologramRawFile(raws[0])
    return HologramTifFolder(holo_dir, validate=None)


# ============================== #
#             phase              #
# ============================== #


def test_phase_bin_folder_loads(koala_acq: Path) -> None:
    folder = PhaseBinFolder(
        _require(koala_acq / "Phase" / "Float" / "Bin"), validate=None
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


def test_phase_bin_matches_txt(koala_acq: Path) -> None:
    # Koala's Float/Bin and Float/Txt hold the same frames; the two readers must
    # agree to within the text export's ~float32 precision (measured max abs diff
    # ~5e-7 on real data).
    bin_folder = PhaseBinFolder(
        _require(koala_acq / "Phase" / "Float" / "Bin"), validate=None
    )
    txt_folder = PhaseTxtFolder(
        _require(koala_acq / "Phase" / "Float" / "Txt"), validate=None
    )
    assert len(bin_folder) == len(txt_folder)

    for i in _sample_indices(len(bin_folder)):
        b, t = bin_folder[i], txt_folder[i]
        assert b.shape == t.shape
        assert np.allclose(b, t, rtol=1e-5, atol=1e-5), f"bin/txt disagree at frame {i}"


def test_phase_to_image_reproduces_koala_previews(koala_acq: Path) -> None:
    # to_image renders the Float source through phbounds.txt; it must match Koala's
    # own uint8 previews to within one 8-bit code (measured max code diff == 1).
    bin_folder = PhaseBinFolder(
        _require(koala_acq / "Phase" / "Float" / "Bin"), validate=None
    )
    previews = PhaseTifFolder(_require(koala_acq / "Phase" / "Image"), validate=None)
    bounds = read_phbounds(_require(koala_acq / "phbounds.txt"))
    assert len(bin_folder) == len(previews)

    rendered = bin_folder.to_image(bounds)
    for i in _sample_indices(len(bin_folder)):
        ours = rendered[i].astype(np.int16)
        koala = previews[i].astype(np.int16)
        assert ours.shape == koala.shape
        assert np.abs(ours - koala).max() <= 1, (
            f"preview differs by >1 code at frame {i}"
        )


# ============================== #
#           intensity            #
# ============================== #


def test_intensity_bin_matches_txt(koala_acq: Path) -> None:
    bin_folder = IntensityBinFolder(
        _require(koala_acq / "Intensity" / "Float" / "Bin"), validate=None
    )
    txt_folder = IntensityTxtFolder(
        _require(koala_acq / "Intensity" / "Float" / "Txt"), validate=None
    )
    assert len(bin_folder) == len(txt_folder)

    for i in _sample_indices(len(bin_folder)):
        b, t = bin_folder[i], txt_folder[i]
        assert b.dtype == np.float32
        assert np.allclose(b, t, rtol=1e-5, atol=1e-5), f"bin/txt disagree at frame {i}"


# ============================== #
#            hologram            #
# ============================== #


def test_hologram_loads(koala_acq: Path) -> None:
    holos = _open_holograms(koala_acq)
    assert len(holos) > 0

    frame = holos[0]
    assert frame.dtype == np.uint8
    assert frame.ndim == 2
    assert frame.shape == holos.frame_shape


def test_hologram_reencode_is_lossless(koala_acq: Path, tmp_path: Path) -> None:
    # Holograms are uint8 with no quantization, so raw/tif -> npy round-trips exactly.
    holos = _open_holograms(koala_acq)
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


def test_timestamps_load(koala_acq: Path) -> None:
    ts = TimestampsTxtFile(_require(koala_acq / "timestamps.txt"))
    assert len(ts) > 0
    assert ts.mean_frame_rate > 0
    assert ts.mean_interval_ms > 0
    assert ts[0].interval_ms == 0.0  # no gap before the first frame

    elapsed = [ts[i].elapsed_ms for i in _sample_indices(len(ts))]
    assert elapsed == sorted(elapsed)  # elapsed time is non-decreasing


def test_frame_counts_agree(koala_acq: Path) -> None:
    # One time-lapse acquisition -> every per-frame modality has the same count.
    counts = {
        "phase_bin": len(
            PhaseBinFolder(
                _require(koala_acq / "Phase" / "Float" / "Bin"), validate=None
            )
        ),
        "phase_txt": len(
            PhaseTxtFolder(
                _require(koala_acq / "Phase" / "Float" / "Txt"), validate=None
            )
        ),
        "intensity_bin": len(
            IntensityBinFolder(
                _require(koala_acq / "Intensity" / "Float" / "Bin"), validate=None
            )
        ),
        "timestamps": len(TimestampsTxtFile(_require(koala_acq / "timestamps.txt"))),
        "holograms": len(_open_holograms(koala_acq)),
    }
    assert len(set(counts.values())) == 1, f"frame counts disagree: {counts}"
