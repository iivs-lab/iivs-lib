from __future__ import annotations

import numpy as np
import pytest
import tifffile

from iivs.dhm.koala.hologram.file import load_tif, save_tif, validate_hologram


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, size=(4, 5), dtype=np.uint8)
    path = tmp_path / "00000_holo.tif"

    save_tif(path, data)
    image = load_tif(path)

    np.testing.assert_array_equal(image, data)
    assert image.dtype == np.uint8


def test_validate_hologram_rejects_non_2d():
    with pytest.raises(ValueError, match="single 2D image"):
        validate_hologram(np.zeros((2, 2, 3), dtype=np.uint8))


def test_validate_hologram_rejects_non_uint8():
    with pytest.raises(ValueError, match="uint8"):
        validate_hologram(np.zeros((2, 2), dtype=np.float32))


def test_save_rejects_non_uint8(tmp_path):
    with pytest.raises(ValueError, match="uint8"):
        save_tif(tmp_path / "bad.tif", np.zeros((2, 2), dtype=np.uint16))


def test_load_rejects_non_uint8_file(tmp_path):
    # A uint16 TIFF decodes but is not a valid hologram.
    path = tmp_path / "00000_holo.tif"
    tifffile.imwrite(path, np.zeros((2, 2), dtype=np.uint16))
    with pytest.raises(ValueError, match="uint8"):
        load_tif(path)


def test_save_overwrite(tmp_path):
    data = np.zeros((2, 2), dtype=np.uint8)
    path = tmp_path / "00000_holo.tif"
    save_tif(path, data)
    with pytest.raises(FileExistsError, match="already exists"):
        save_tif(path, data)
    save_tif(path, data, overwrite=True)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_tif(tmp_path / "nope.tif")
