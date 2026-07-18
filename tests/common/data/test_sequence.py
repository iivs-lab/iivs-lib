from __future__ import annotations

import numpy as np
import pytest
import tifffile

from iivs.common.data import ArrayFileList


class _U8TifList(ArrayFileList[np.uint8]):
    FILE_EXT = "tif"

    def load_file(self, path):
        return tifffile.imread(path).astype(np.uint8, copy=False)


def test_array_file_list_indexes_and_meta_is_path(tmp_path):
    a = tmp_path / "a.tif"
    b = tmp_path / "b.tif"
    tifffile.imwrite(a, np.zeros((2, 3), dtype=np.uint8))
    tifffile.imwrite(b, np.ones((2, 3), dtype=np.uint8))
    seq = _U8TifList([b, a])  # arbitrary order preserved
    assert len(seq) == 2
    assert [seq.get_meta(i) for i in range(2)] == [b, a]  # arbitrary order preserved
    assert np.array_equal(seq[0], np.ones((2, 3), dtype=np.uint8))  # index 0 -> b
    assert np.array_equal(seq[1], np.zeros((2, 3), dtype=np.uint8))  # index 1 -> a


def test_array_file_list_rejects_wrong_extension(tmp_path):
    p = tmp_path / "a.dat"
    p.write_bytes(b"")
    with pytest.raises(ValueError, match="unsupported extension"):
        _U8TifList([p])
