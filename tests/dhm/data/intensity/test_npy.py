from __future__ import annotations

import numpy as np
import pytest

from iivs.common.data import FrameShapedMixin
from iivs.dhm.data.intensity.base import IntensityFloatSequence
from iivs.dhm.data.intensity.bin import IntensityBinHeader
from iivs.dhm.data.intensity.npy import IntensityNpyFolder, load_intensity_npy


def _write(root, index, value=0.0, shape=(3, 3)):
    np.save(
        root / f"{index:05d}_intensity.npy", np.full(shape, value, dtype=np.float32)
    )


def test_lists_in_order_with_synthesized_header(tmp_path):
    for i in range(2):
        _write(tmp_path, i, i)
    f = IntensityNpyFolder(tmp_path, pixel_size=2e-6)
    assert isinstance(f, IntensityFloatSequence)
    assert isinstance(f, FrameShapedMixin)
    assert f.frame_shape == (3, 3)
    assert isinstance(f.header, IntensityBinHeader)
    assert f.header.pixel_size == pytest.approx(2e-6)
    np.testing.assert_array_equal(f[1], np.full((3, 3), 1.0))
    assert f[0].dtype == np.float32


def test_validate_rejects_shape_mismatch(tmp_path):
    _write(tmp_path, 0, shape=(2, 3))
    _write(tmp_path, 1, shape=(4, 5))
    with pytest.raises(ValueError, match="header"):
        IntensityNpyFolder(tmp_path, pixel_size=1e-6)


def test_validate_data_rejects_non_float32(tmp_path):
    np.save(tmp_path / "00000_intensity.npy", np.zeros((2, 3), dtype=np.float64))
    with pytest.raises(ValueError, match="float32"):
        IntensityNpyFolder(tmp_path, pixel_size=1e-6, validate="data")


# --- load_intensity_npy (standalone, header-less) ---


def test_load_intensity_npy_roundtrip(tmp_path):
    img = np.arange(6, dtype=np.float32).reshape(2, 3)
    path = tmp_path / "x.npy"
    np.save(path, img)
    out = load_intensity_npy(path)
    np.testing.assert_array_equal(out, img)
    assert out.dtype == np.float32


def test_load_intensity_npy_rejects_pickled_object_array(tmp_path):
    path = tmp_path / "x.npy"
    np.save(path, np.array([{"x": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError, match="allow_pickle"):
        load_intensity_npy(path)


def test_load_intensity_npy_rejects_non_2d(tmp_path):
    path = tmp_path / "x.npy"
    np.save(path, np.zeros((2, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="2D image"):
        load_intensity_npy(path)


def test_load_intensity_npy_raises_on_nonfinite(tmp_path):
    path = tmp_path / "x.npy"
    np.save(path, np.array([[np.nan, 1.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        load_intensity_npy(path, on_nonfinite="raise")
