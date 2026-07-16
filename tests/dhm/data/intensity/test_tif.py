from __future__ import annotations

import numpy as np
import tifffile

from iivs.common.data import FrameShapedMixin
from iivs.dhm.data.intensity.base import IntensityImageSequence
from iivs.dhm.data.intensity.tif import (
    IntensityTifFolder,
    IntensityTifList,
    load_intensity_tif,
)


def _write(path, data):
    tifffile.imwrite(path, np.asarray(data, dtype=np.uint8))


def test_load_intensity_tif(tmp_path):
    data = np.arange(20, dtype=np.uint8).reshape(4, 5)
    path = tmp_path / "00000_intensity.tif"
    _write(path, data)

    image = load_intensity_tif(path)

    np.testing.assert_array_equal(image, data)
    assert image.dtype == np.uint8  # the 8-bit preview, not the float intensity


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
