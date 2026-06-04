from __future__ import annotations

import numpy as np
import pytest

from iivs.dhm.data.intensity.bin import (
    IntensityBinFolder,
    save_intensity_bin,
)
from iivs.dhm.data.intensity.convert import convert_intensity_folder
from iivs.dhm.data.intensity.npy import IntensityNpyFolder, save_intensity_npy
from iivs.dhm.data.intensity.txt import (
    IntensityTxtFolder,
    load_intensity_txt,
    save_intensity_txt,
)


def _bin_folder(root, values):
    root.mkdir()
    for i, value in enumerate(values):
        save_intensity_bin(
            root / f"{i:05d}_intensity.bin",
            np.full((2, 3), float(value), dtype=np.float32),
            pixel_size=1e-6,
        )
    return IntensityBinFolder(root)


# --- single-frame writers ---


def test_save_intensity_txt_roundtrip(tmp_path):
    data = np.array([[0.0, 1.5], [2.0, 3.0]], dtype=np.float32)
    path = tmp_path / "00000_intensity.txt"
    save_intensity_txt(path, data, pixel_size=1e-6)
    loaded, header = load_intensity_txt(path, return_header=True)
    np.testing.assert_allclose(loaded, data, rtol=1e-5)
    assert header.pixel_size == pytest.approx(1e-6)


def test_save_intensity_npy_roundtrip(tmp_path):
    data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    path = tmp_path / "00000_intensity.npy"
    save_intensity_npy(path, data)
    np.testing.assert_array_equal(np.load(path), data)


# --- convert_intensity_folder ---


def test_convert_intensity_to_txt(tmp_path):
    src = _bin_folder(tmp_path / "src", [1.0, 2.0])
    out = tmp_path / "out"
    convert_intensity_folder(out, src, ext="txt")
    dst = IntensityTxtFolder(out)
    assert len(dst) == 2
    np.testing.assert_allclose(dst[0], src[0], rtol=1e-5)
    assert dst.header.pixel_size == pytest.approx(src.header.pixel_size)


def test_convert_intensity_to_npy(tmp_path):
    src = _bin_folder(tmp_path / "src", [1.0, 2.0])
    out = tmp_path / "out"
    convert_intensity_folder(out, src, ext="npy")
    dst = IntensityNpyFolder(out, pixel_size=src.header.pixel_size)
    np.testing.assert_array_equal(dst[0], src[0])


def test_convert_intensity_to_bin(tmp_path):
    src = _bin_folder(tmp_path / "src", [1.0, 2.0])
    out = tmp_path / "out"
    convert_intensity_folder(out, src, ext="bin")
    dst = IntensityBinFolder(out)
    np.testing.assert_array_equal(dst[0], src[0])
    assert dst.header == src.header


def test_convert_intensity_rejects_unknown_format(tmp_path):
    src = _bin_folder(tmp_path / "src", [1.0])
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="ext must be"):
        convert_intensity_folder(out, src, ext="raw")
