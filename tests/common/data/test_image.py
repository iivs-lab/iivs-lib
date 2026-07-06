from __future__ import annotations

import numpy as np
import pytest
import tifffile

from iivs.common.data import ImageFileList, load_tif

# --- load_tif ---


def test_load_tif_keeps_uint8(tmp_path):
    path = tmp_path / "a.tif"
    data = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    tifffile.imwrite(path, data)
    got = load_tif(path)
    np.testing.assert_array_equal(got, data)
    assert got.dtype == np.uint8


def test_load_tif_keeps_stored_dtype(tmp_path):
    # The generic reader keeps whatever dtype the tif stores -- not just uint8.
    path = tmp_path / "a.tif"
    data = np.array([[1000, 2000], [3000, 4000]], dtype=np.uint16)
    tifffile.imwrite(path, data)
    got = load_tif(path)
    np.testing.assert_array_equal(got, data)
    assert got.dtype == np.uint16


def test_load_tif_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_tif(tmp_path / "nope.tif")


# --- ImageFileList (via a minimal uint8 concrete) ---


class _U8TifList(ImageFileList[np.uint8]):
    FILE_EXT = "tif"

    def load_file(self, path):
        return load_tif(path).astype(np.uint8, copy=False)


def test_image_file_list_indexes_and_meta_is_path(tmp_path):
    a = tmp_path / "a.tif"
    b = tmp_path / "b.tif"
    tifffile.imwrite(a, np.zeros((2, 3), dtype=np.uint8))
    tifffile.imwrite(b, np.ones((2, 3), dtype=np.uint8))
    seq = _U8TifList([b, a])  # arbitrary order preserved
    assert len(seq) == 2
    assert seq[0].shape == (2, 3)
    assert [seq.get_meta(i) for i in range(2)] == [b, a]


def test_image_file_list_rejects_wrong_extension(tmp_path):
    p = tmp_path / "a.dat"
    p.write_bytes(b"")
    with pytest.raises(ValueError, match="unsupported extension"):
        _U8TifList([p])
