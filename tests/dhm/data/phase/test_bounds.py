from __future__ import annotations

import numpy as np
import pytest

from iivs.dhm.data.phase.bin import PhaseBinFolder, PhaseBinList, save_phase_bin
from iivs.dhm.data.phase.bounds import PhaseBounds, read_phbounds, write_phbounds
from iivs.dhm.data.phase.core import PhaseUnit


def _save(path, data, *, height_scale=2e-7):
    # height_scale 2e-7 m/rad -> 200 nm per radian.
    save_phase_bin(
        path,
        np.asarray(data, dtype=np.float32),
        pixel_size=1e-6,
        height_scale=height_scale,
    )


# ========================== #
#         PhaseBounds        #
# ========================== #


def test_phase_bounds_rejects_unordered():
    with pytest.raises(ValueError, match="must not exceed max_nm"):
        PhaseBounds(min_nm=1.0, max_nm=0.0)


def test_phase_bounds_allows_equal():
    assert PhaseBounds(min_nm=5.0, max_nm=5.0).min_nm == 5.0


# ========================== #
#      read/write phbounds   #
# ========================== #


def test_phbounds_roundtrip(tmp_path):
    path = tmp_path / "phbounds.txt"
    bounds = PhaseBounds(min_nm=-403.4911, max_nm=635.9849)
    write_phbounds(path, bounds)
    assert read_phbounds(path) == bounds


def test_phbounds_from_file_to_file_roundtrip(tmp_path):
    # The free functions wrap these methods; exercise the methods directly.
    path = tmp_path / "phbounds.txt"
    bounds = PhaseBounds(min_nm=-1.0, max_nm=2.0)
    bounds.to_file(path)
    assert PhaseBounds.from_file(path) == bounds
    assert PhaseBounds.UNIT_TAG == "[nm]"


def test_phbounds_file_format(tmp_path):
    path = tmp_path / "phbounds.txt"
    write_phbounds(path, PhaseBounds(min_nm=-403.4911, max_nm=635.9849))
    assert path.read_text() == "[nm]\n-403.4911 635.9849\n"


def test_write_phbounds_no_overwrite(tmp_path):
    path = tmp_path / "phbounds.txt"
    write_phbounds(path, PhaseBounds(0.0, 1.0))
    with pytest.raises(FileExistsError):
        write_phbounds(path, PhaseBounds(0.0, 1.0))
    write_phbounds(path, PhaseBounds(0.0, 2.0), overwrite=True)
    assert read_phbounds(path).max_nm == pytest.approx(2.0)


def test_read_phbounds_rejects_wrong_line_count(tmp_path):
    path = tmp_path / "phbounds.txt"
    path.write_text("[nm]\n")
    with pytest.raises(ValueError, match="2 non-blank lines"):
        read_phbounds(path)


def test_read_phbounds_rejects_wrong_tag(tmp_path):
    path = tmp_path / "phbounds.txt"
    path.write_text("[rad]\n0 1\n")
    with pytest.raises(ValueError, match=r"\[nm\]"):
        read_phbounds(path)


def test_read_phbounds_rejects_non_pair_bounds(tmp_path):
    path = tmp_path / "phbounds.txt"
    path.write_text("[nm]\n1 2 3\n")
    with pytest.raises(ValueError, match="min max"):
        read_phbounds(path)


def test_read_phbounds_rejects_non_numeric(tmp_path):
    path = tmp_path / "phbounds.txt"
    path.write_text("[nm]\nlo hi\n")
    with pytest.raises(ValueError, match="could not convert"):  # float() rejects them
        read_phbounds(path)


def test_read_phbounds_rejects_wrong_extension(tmp_path):
    path = tmp_path / "phbounds.bin"
    path.write_text("[nm]\n0 1\n")
    with pytest.raises(ValueError, match=r"must have a \.txt extension"):
        read_phbounds(path)


def test_write_phbounds_appends_extension(tmp_path):
    # np.save-style: a suffix-less path gets `.txt` appended.
    write_phbounds(tmp_path / "phbounds", PhaseBounds(0.0, 1.0))
    assert (tmp_path / "phbounds.txt").exists()


def test_write_phbounds_rejects_wrong_extension(tmp_path):
    with pytest.raises(ValueError, match=r"must have a \.txt extension"):
        write_phbounds(tmp_path / "phbounds.bin", PhaseBounds(0.0, 1.0))


# ========================== #
#   preview <-> nm mapping   #
# ========================== #


def test_decode_preview_endpoints_and_dtype():
    bounds = PhaseBounds(min_nm=-100.0, max_nm=300.0)
    preview = np.array([0, 255], dtype=np.uint8)
    nm = bounds.decode_preview(preview)
    assert nm.dtype == np.float32
    assert nm[0] == pytest.approx(-100.0)
    assert nm[1] == pytest.approx(300.0)


def test_encode_preview_endpoints_and_dtype():
    bounds = PhaseBounds(min_nm=-100.0, max_nm=300.0)
    nm = np.array([-100.0, 300.0], dtype=np.float32)
    preview = bounds.encode_preview(nm)
    assert preview.dtype == np.uint8
    assert preview[0] == 0
    assert preview[1] == 255


def test_encode_preview_clamps_out_of_range():
    bounds = PhaseBounds(min_nm=0.0, max_nm=100.0)
    preview = bounds.encode_preview(np.array([-50.0, 150.0], dtype=np.float32))
    assert preview[0] == 0  # below min -> 0
    assert preview[1] == 255  # above max -> 255


def test_preview_roundtrip_within_quantization():
    bounds = PhaseBounds(min_nm=-403.4911, max_nm=635.9849)
    nm = np.linspace(bounds.min_nm, bounds.max_nm, 256, dtype=np.float32)
    recovered = bounds.decode_preview(bounds.encode_preview(nm))
    step = (bounds.max_nm - bounds.min_nm) / 255.0
    assert np.max(np.abs(recovered - nm)) <= step  # 8-bit quantization only


def test_degenerate_bounds_map_to_a_single_value():
    bounds = PhaseBounds(min_nm=42.0, max_nm=42.0)
    # decode: every pixel collapses to the single value (no division by span)
    nm = bounds.decode_preview(np.array([0, 128, 255], dtype=np.uint8))
    assert np.all(nm == np.float32(42.0))
    # encode: zero span maps everything to 0 rather than dividing by zero
    preview = bounds.encode_preview(np.array([42.0, 42.0], dtype=np.float32))
    assert np.all(preview == 0)


# ========================== #
#    PhaseFileList.bounds_nm #
# ========================== #


def test_bounds_nm_over_folder(tmp_path):
    _save(tmp_path / "00000_phase.bin", np.full((2, 3), 1.0))  # 200 nm
    _save(tmp_path / "00001_phase.bin", np.full((2, 3), 3.0))  # 600 nm
    bounds = PhaseBinFolder(tmp_path).bounds_nm()
    assert bounds.min_nm == pytest.approx(200.0)
    assert bounds.max_nm == pytest.approx(600.0)


def test_bounds_nm_within_a_frame(tmp_path):
    path = tmp_path / "00000_phase.bin"
    _save(path, [[0.0, 3.0]])  # 0 .. 600 nm
    bounds = PhaseBinList([path]).bounds_nm()
    assert bounds.min_nm == pytest.approx(0.0)
    assert bounds.max_nm == pytest.approx(600.0)


def test_bounds_nm_converts_each_file_by_its_height_scale(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    _save(a, np.full((1, 2), 1.0), height_scale=2e-7)  # 200 nm
    _save(b, np.full((1, 2), 1.0), height_scale=4e-7)  # 400 nm
    bounds = PhaseBinList([a, b]).bounds_nm()
    assert bounds.min_nm == pytest.approx(200.0)
    assert bounds.max_nm == pytest.approx(400.0)


def test_bounds_nm_is_independent_of_target_unit(tmp_path):
    path = tmp_path / "00000_phase.bin"
    _save(path, np.full((1, 2), 1.0))
    seq = PhaseBinList([path], target_unit=PhaseUnit.RADIANS)
    assert seq.bounds_nm().max_nm == pytest.approx(200.0)  # nm, not the 1.0 radian


def test_bounds_nm_empty_sequence_raises():
    with pytest.raises(ValueError, match="empty"):
        PhaseBinList([]).bounds_nm()


def test_bounds_nm_rejects_unknown_unit(tmp_path):
    path = tmp_path / "00000_phase.bin"
    with pytest.warns(UserWarning, match="UNKNOWN"):  # save_phase_bin warns
        save_phase_bin(
            path,
            np.zeros((1, 2), dtype=np.float32),
            pixel_size=1e-6,
            height_scale=2e-7,
            unit=PhaseUnit.UNKNOWN,
        )
    with pytest.raises(ValueError, match="cannot convert"):
        PhaseBinList([path]).bounds_nm()
