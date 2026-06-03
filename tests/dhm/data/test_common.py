from __future__ import annotations

import numpy as np
import pytest

from iivs.dhm.data.common import FrameShapedMixin, validate_float32_image
from iivs.dhm.data.phase.base import PhaseSequence
from iivs.dhm.data.phase.bin import PhaseBinFolder, PhaseBinList, save_phase_bin

# ========================== #
#  validate_float32_image    #
# ========================== #


def test_rejects_below_2d():
    with pytest.raises(ValueError, match="2-dimensional"):
        validate_float32_image(np.zeros(5, dtype=np.float32))


def test_rejects_non_float32():
    with pytest.raises(ValueError, match="float32"):
        validate_float32_image(np.zeros((2, 2), dtype=np.float64))


def test_rejects_unknown_on_nonfinite():
    with pytest.raises(ValueError, match="on_nonfinite must be"):
        validate_float32_image(np.zeros((2, 2), dtype=np.float32), on_nonfinite="bogus")  # ty: ignore[invalid-argument-type]


def test_ignore_accepts_nonfinite():
    data = np.array([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float32)
    assert validate_float32_image(data, on_nonfinite="ignore") is data


def test_warns_on_nonfinite():
    data = np.array([[np.inf, 1.0], [2.0, 3.0]], dtype=np.float32)
    with pytest.warns(RuntimeWarning, match="finite"):
        validate_float32_image(data, on_nonfinite="warn")


def test_raises_on_nonfinite():
    data = np.array([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="finite"):
        validate_float32_image(data, on_nonfinite="raise")


def test_clean_returns_input():
    data = np.zeros((2, 2), dtype=np.float32)
    assert validate_float32_image(data) is data  # default "warn", finite: as-is


# ========================== #
#       FrameShapedMixin     #
# ========================== #


def _write(root, index, shape=(2, 3)):
    save_phase_bin(
        root / f"{index:05d}_phase.bin",
        np.zeros(shape, dtype=np.float32),
        pixel_size=1e-6,
        height_scale=2e-7,
    )


def test_uniform_sequence_is_role_plus_mixin(tmp_path):
    # "A uniform phase sequence" == PhaseSequence + FrameShapedMixin.
    _write(tmp_path, 0)
    folder = PhaseBinFolder(tmp_path)
    assert isinstance(folder, PhaseSequence)
    assert isinstance(folder, FrameShapedMixin)
    assert folder.frame_shape == (2, 3)


def test_heterogeneous_list_lacks_the_mixin(tmp_path):
    # A plain file list is a PhaseSequence but not FrameShapedMixin.
    _write(tmp_path, 0)
    seq = PhaseBinList([tmp_path / "00000_phase.bin"])
    assert isinstance(seq, PhaseSequence)
    assert not isinstance(seq, FrameShapedMixin)
    assert not hasattr(seq, "frame_shape")
