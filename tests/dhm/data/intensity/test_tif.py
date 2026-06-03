from __future__ import annotations

import numpy as np
import tifffile

from iivs.dhm.data.common import FrameShapedMixin
from iivs.dhm.data.intensity.base import IntensityImageSequence
from iivs.dhm.data.intensity.tif import IntensityTifFolder, IntensityTifList


def _write(path, data):
    tifffile.imwrite(path, np.asarray(data, dtype=np.uint8))


def test_folder_loads_and_is_image_sequence(tmp_path):
    _write(tmp_path / "00000_intensity.tif", np.full((2, 3), 7, dtype=np.uint8))
    folder = IntensityTifFolder(tmp_path)
    assert isinstance(folder, IntensityImageSequence)
    assert isinstance(folder, FrameShapedMixin)
    assert folder.frame_shape == (2, 3)
    np.testing.assert_array_equal(folder[0], np.full((2, 3), 7, dtype=np.uint8))
    assert folder[0].dtype == np.uint8


def test_list_meta_and_no_frame_shape(tmp_path):
    path = tmp_path / "x.tif"
    _write(path, np.zeros((2, 2), dtype=np.uint8))
    seq = IntensityTifList([path])
    assert seq.get_meta(0) == path
    assert not isinstance(seq, FrameShapedMixin)
