from __future__ import annotations

import numpy as np
import pytest

from iivs.dhm.data.common import FrameShapedMixin
from iivs.dhm.data.hologram.base import HologramSequence
from iivs.dhm.data.hologram.npy import HologramNpyFolder, load_hologram_npy


def _write(root, index, value=1, shape=(4, 4)):
    np.save(root / f"{index:05d}_holo.npy", np.full(shape, value, dtype=np.uint8))


def test_lists_in_order(tmp_path):
    for i in range(2):
        _write(tmp_path, i, i + 1)
    f = HologramNpyFolder(tmp_path)
    assert isinstance(f, HologramSequence)
    assert isinstance(f, FrameShapedMixin)
    assert f.frame_shape == (4, 4)
    assert f.get_meta(0) == tmp_path / "00000_holo.npy"
    np.testing.assert_array_equal(f[0], np.full((4, 4), 1, dtype=np.uint8))
    assert f[1].dtype == np.uint8


def test_rejects_non_uint8(tmp_path):
    np.save(tmp_path / "00000_holo.npy", np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="uint8"):
        HologramNpyFolder(tmp_path, validate="data")


def test_validate_rejects_shape_mismatch(tmp_path):
    _write(tmp_path, 0, shape=(2, 3))
    _write(tmp_path, 1, shape=(4, 5))
    with pytest.raises(ValueError, match="must match"):
        HologramNpyFolder(tmp_path, validate="data")


# --- load_hologram_npy (standalone) ---


def test_load_hologram_npy_roundtrip(tmp_path):
    img = np.arange(12, dtype=np.uint8).reshape(3, 4)
    path = tmp_path / "x.npy"
    np.save(path, img)
    out = load_hologram_npy(path)
    np.testing.assert_array_equal(out, img)
    assert out.dtype == np.uint8


def test_load_hologram_npy_rejects_pickled_object_array(tmp_path):
    path = tmp_path / "x.npy"
    np.save(path, np.array([{"x": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError, match="allow_pickle"):
        load_hologram_npy(path)


def test_load_hologram_npy_rejects_non_uint8(tmp_path):
    path = tmp_path / "x.npy"
    np.save(path, np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="uint8"):
        load_hologram_npy(path)
