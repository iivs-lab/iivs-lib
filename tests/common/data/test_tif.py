from __future__ import annotations

import numpy as np
import pytest
import tifffile

from iivs.common.data import load_tif


def test_load_tif_keeps_uint8(tmp_path):
    path = tmp_path / "a.tif"
    data = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    tifffile.imwrite(path, data)
    got = load_tif(path)
    np.testing.assert_array_equal(got, data)
    assert got.dtype == np.uint8


def test_load_tif_keeps_stored_dtype(tmp_path):
    # The generic reader keeps whatever dtype the tif stores, not just uint8.
    path = tmp_path / "a.tif"
    data = np.array([[1000, 2000], [3000, 4000]], dtype=np.uint16)
    tifffile.imwrite(path, data)
    got = load_tif(path)
    np.testing.assert_array_equal(got, data)
    assert got.dtype == np.uint16


def test_load_tif_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_tif(tmp_path / "nope.tif")
