from __future__ import annotations

import numpy as np
import pytest

from iivs.dhm.data.intensity.bin import IntensityBinHeader
from iivs.dhm.data.intensity.txt import (
    IntensityTxtFolder,
    IntensityTxtList,
    load_intensity_txt,
    read_intensity_txt_header,
)


def _write_intensity_txt(path, data, *, pixel_size=1e-6):
    h, w = data.shape
    lines = [
        f"h={h} w={w}",
        f"pixel size={pixel_size:.6g} m",
        *(" ".join(f"{v:.6f}" for v in row) for row in data),
    ]
    path.write_text("\n".join(lines) + "\n")


def _write(root, index, value, shape=(2, 3)):
    _write_intensity_txt(
        root / f"{index:05d}_intensity.txt",
        np.full(shape, float(value), dtype=np.float32),
    )


def test_load_roundtrip(tmp_path):
    data = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
    path = tmp_path / "00000_intensity.txt"
    _write_intensity_txt(path, data, pixel_size=2e-6)

    image, header = load_intensity_txt(path, return_header=True)
    np.testing.assert_allclose(image, data, atol=1e-4)
    assert image.dtype == np.float32
    assert isinstance(header, IntensityBinHeader)
    assert header.shape == (2, 3)
    assert header.pixel_size == pytest.approx(2e-6)


def test_header_matches_load(tmp_path):
    data = np.zeros((3, 4), dtype=np.float32)
    path = tmp_path / "00000_intensity.txt"
    _write_intensity_txt(path, data, pixel_size=2e-6)
    assert (
        read_intensity_txt_header(path)
        == load_intensity_txt(path, return_header=True)[1]
    )


def test_rejects_short_header(tmp_path):
    path = tmp_path / "00000_intensity.txt"
    path.write_text("h=2 w=2\n")  # fewer than 2 header lines
    with pytest.raises(ValueError, match="needs 2 lines"):
        load_intensity_txt(path)


def test_rejects_malformed_header(tmp_path):
    path = tmp_path / "00000_intensity.txt"
    path.write_text("h=2 w=2\nno pixel size\n1 2\n3 4\n")
    with pytest.raises(ValueError, match="malformed intensity txt header"):
        load_intensity_txt(path)


def test_load_on_nonfinite_policy(tmp_path):
    data = np.array([[np.inf, 1.0], [2.0, 3.0]], dtype=np.float32)
    path = tmp_path / "00000_intensity.txt"
    _write_intensity_txt(path, data)
    load_intensity_txt(path)  # default "ignore"
    with pytest.raises(ValueError, match="finite"):
        load_intensity_txt(path, on_nonfinite="raise")


def test_folder(tmp_path):
    for i in range(3):
        _write(tmp_path, i, i)
    folder = IntensityTxtFolder(tmp_path)
    assert len(folder) == 3
    assert isinstance(folder.header, IntensityBinHeader)
    assert folder.frame_shape == (2, 3)
    assert folder.get_meta(0) == tmp_path / "00000_intensity.txt"
    np.testing.assert_allclose(folder[2], np.full((2, 3), 2.0), atol=1e-5)


def test_folder_validate(tmp_path):
    _write(tmp_path, 0, 0)
    _write(tmp_path, 2, 2)  # gap at index 1
    with pytest.raises(ValueError, match="non-contiguous"):
        IntensityTxtFolder(tmp_path, validate="names")

    other = tmp_path / "mismatch"
    other.mkdir()
    _write_intensity_txt(other / "00000_intensity.txt", np.zeros((2, 3), np.float32))
    _write_intensity_txt(
        other / "00001_intensity.txt", np.zeros((2, 3), np.float32), pixel_size=9e-6
    )
    with pytest.raises(ValueError, match="header"):
        IntensityTxtFolder(other, validate=None).validate()


def test_folder_validate_data_detects_non_finite(tmp_path):
    nan = np.array([[np.nan, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
    _write_intensity_txt(tmp_path / "00000_intensity.txt", nan)
    IntensityTxtFolder(tmp_path, validate="headers")  # ok
    with pytest.raises(ValueError, match="finite"):
        IntensityTxtFolder(tmp_path, validate="data")


def test_list_sequence(tmp_path):
    a = tmp_path / "a.txt"
    sub = tmp_path / "nested"
    sub.mkdir()
    b = sub / "b.txt"
    _write_intensity_txt(a, np.full((2, 3), 1.0, dtype=np.float32))
    _write_intensity_txt(b, np.full((4, 5), 2.0, dtype=np.float32))

    seq = IntensityTxtList([b, a])
    assert len(seq) == 2
    assert [seq.get_meta(i) for i in range(2)] == [b, a]
    assert not hasattr(seq, "frame_shape")  # heterogeneous
    np.testing.assert_allclose(seq[0], np.full((4, 5), 2.0), atol=1e-5)
