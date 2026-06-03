from __future__ import annotations

import numpy as np
import pytest

from iivs.dhm.data.phase.bin import PhaseBinHeader
from iivs.dhm.data.phase.core import PhaseUnit
from iivs.dhm.data.phase.txt import (
    PhaseTxtFolder,
    PhaseTxtList,
    load_phase_txt,
    read_phase_txt_header,
)


def _write_phase_txt(path, data, *, pixel_size=1e-6, height_scale=2e-7, unit="rad"):
    h, w = data.shape
    lines = [
        f"h={h} w={w}",
        f"pixel size={pixel_size:.6g} m",
        f"data unit={unit}",
        f"height conversion factor (-> m)={height_scale:.6g}",
        *(" ".join(f"{v:.6f}" for v in row) for row in data),
    ]
    path.write_text("\n".join(lines) + "\n")


def _write(root, index, value, shape=(2, 3)):
    _write_phase_txt(
        root / f"{index:05d}_phase.txt", np.full(shape, float(value), dtype=np.float32)
    )


# --- single-file I/O ---


def test_load_roundtrip(tmp_path):
    data = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    path = tmp_path / "00000_phase.txt"
    _write_phase_txt(path, data, pixel_size=2e-6, height_scale=3e-7)

    image, header = load_phase_txt(path, return_header=True)
    np.testing.assert_allclose(image, data, atol=1e-5)
    assert image.dtype == np.float32
    assert isinstance(header, PhaseBinHeader)
    assert header.shape == (2, 3)
    assert header.unit is PhaseUnit.RADIANS
    assert header.pixel_size == pytest.approx(2e-6)
    assert header.height_scale == pytest.approx(3e-7)


def test_unit_meters_mapping(tmp_path):
    data = np.zeros((2, 2), dtype=np.float32)
    path = tmp_path / "00000_phase.txt"
    _write_phase_txt(path, data, unit="m")
    assert read_phase_txt_header(path).unit is PhaseUnit.METERS


def test_header_matches_load(tmp_path):
    data = np.zeros((3, 4), dtype=np.float32)
    path = tmp_path / "00000_phase.txt"
    _write_phase_txt(path, data, pixel_size=2e-6, height_scale=3e-7)
    assert read_phase_txt_header(path) == load_phase_txt(path, return_header=True)[1]


def test_rejects_short_header(tmp_path):
    path = tmp_path / "00000_phase.txt"
    path.write_text("h=2 w=2\n1 2\n3 4\n")  # fewer than 4 header lines
    with pytest.raises(ValueError, match="needs 4 lines"):
        load_phase_txt(path)


def test_rejects_malformed_header(tmp_path):
    path = tmp_path / "00000_phase.txt"
    path.write_text(
        "bad\npixel size=1 m\ndata unit=rad\nheight conversion factor=1\n1 2\n3 4\n"
    )
    with pytest.raises(ValueError, match="malformed phase txt header"):
        load_phase_txt(path)


def test_rejects_missing_height_conversion(tmp_path):
    # h/w and pixel size are valid, but the height-conversion line is missing.
    path = tmp_path / "00000_phase.txt"
    path.write_text("h=2 w=2\npixel size=1 m\ndata unit=rad\n(no factor)\n1 2\n3 4\n")
    with pytest.raises(ValueError, match="malformed phase txt header"):
        load_phase_txt(path)


def test_unit_defaults_to_unknown_when_missing(tmp_path):
    # No parseable `data unit=` line -> the unit falls back to UNKNOWN.
    path = tmp_path / "00000_phase.txt"
    path.write_text(
        "h=2 w=2\npixel size=1 m\n(no unit)\nheight conversion factor=1\n1 2\n3 4\n"
    )
    assert read_phase_txt_header(path).unit is PhaseUnit.UNKNOWN


def test_load_on_nonfinite_policy(tmp_path):
    data = np.array([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float32)
    path = tmp_path / "00000_phase.txt"
    _write_phase_txt(path, data)
    load_phase_txt(path)  # default "ignore"
    with pytest.raises(ValueError, match="finite"):
        load_phase_txt(path, on_nonfinite="raise")


# --- folder sequence ---


def test_folder_lists_items_in_index_order(tmp_path):
    for i in range(3):
        _write(tmp_path, i, i)
    folder = PhaseTxtFolder(tmp_path)
    assert len(folder) == 3
    for i in range(3):
        np.testing.assert_allclose(folder[i], np.full((2, 3), float(i)), atol=1e-5)


def test_folder_header_and_frame_shape(tmp_path):
    _write(tmp_path, 0, 0)
    folder = PhaseTxtFolder(tmp_path)
    assert isinstance(folder.header, PhaseBinHeader)
    assert folder.frame_shape == (2, 3)
    assert folder.get_meta(0) == tmp_path / "00000_phase.txt"
    assert folder.target_unit == PhaseUnit.RADIANS


def test_folder_converts_to_meters(tmp_path):
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    _write_phase_txt(tmp_path / "00000_phase.txt", data, height_scale=2e-7)
    folder = PhaseTxtFolder(tmp_path, target_unit=PhaseUnit.METERS)
    np.testing.assert_allclose(folder[0], data * 2e-7, atol=1e-12)


def test_folder_validate_rejects_gap_and_header_mismatch(tmp_path):
    _write(tmp_path, 0, 0)
    _write(tmp_path, 2, 2)  # gap at index 1
    with pytest.raises(ValueError, match="non-contiguous"):
        PhaseTxtFolder(tmp_path, validate="headers")

    other = tmp_path / "mismatch"
    other.mkdir()
    _write_phase_txt(other / "00000_phase.txt", np.zeros((2, 3), dtype=np.float32))
    _write_phase_txt(
        other / "00001_phase.txt", np.zeros((2, 3), dtype=np.float32), pixel_size=9e-6
    )
    with pytest.raises(ValueError, match="header"):
        PhaseTxtFolder(other, validate=None).validate()


def test_folder_validate_data_detects_non_finite(tmp_path):
    nan = np.array([[np.nan, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
    _write_phase_txt(tmp_path / "00000_phase.txt", nan)
    PhaseTxtFolder(tmp_path, validate="headers")  # pixels not inspected: ok
    with pytest.raises(ValueError, match="finite"):
        PhaseTxtFolder(tmp_path, validate="data")


# --- file list ---


def test_list_sequence(tmp_path):
    a = tmp_path / "a.txt"
    sub = tmp_path / "nested"
    sub.mkdir()
    b = sub / "b.txt"
    _write_phase_txt(a, np.full((2, 3), 1.0, dtype=np.float32))
    _write_phase_txt(b, np.full((4, 5), 2.0, dtype=np.float32), height_scale=2e-7)

    seq = PhaseTxtList([b, a])
    assert len(seq) == 2
    assert [seq.get_meta(i) for i in range(2)] == [b, a]
    assert not hasattr(seq, "frame_shape")  # heterogeneous
    np.testing.assert_allclose(seq[0], np.full((4, 5), 2.0), atol=1e-5)

    converted = PhaseTxtList([b], target_unit=PhaseUnit.METERS)
    assert converted.target_unit is PhaseUnit.METERS
    np.testing.assert_allclose(converted[0], np.full((4, 5), 2.0 * 2e-7), atol=1e-12)
