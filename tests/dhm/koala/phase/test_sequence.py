from __future__ import annotations

import numpy as np
import pytest

from iivs.dhm.koala.phase.file import save_phase_bin
from iivs.dhm.koala.phase.header import PhaseBinHeader, PhaseUnit
from iivs.dhm.koala.phase.sequence import PhaseBinSequence


def _write(root, index, value, shape=(2, 3)):
    save_phase_bin(
        root / f"{index:05d}_phase.bin",
        np.full(shape, float(value), dtype=np.float32),
        pixel_size=1e-6,
        height_scale=2e-7,
    )


def test_folder_lists_items_in_index_order(tmp_path):
    for i in range(3):
        _write(tmp_path, i, i)

    folder = PhaseBinSequence(tmp_path)

    assert len(folder) == 3
    for i in range(3):
        np.testing.assert_array_equal(
            folder[i], np.full((2, 3), float(i), dtype=np.float32)
        )


def test_folder_header_attribute(tmp_path):
    _write(tmp_path, 0, 0)
    folder = PhaseBinSequence(tmp_path)
    assert isinstance(folder.header, PhaseBinHeader)
    assert folder.header.shape == (2, 3)
    assert folder.target_unit == PhaseUnit.RADIANS  # defaults to the stored unit


def test_folder_get_meta_is_source_path(tmp_path):
    _write(tmp_path, 0, 0)
    folder = PhaseBinSequence(tmp_path)
    assert folder.get_meta(0) == tmp_path / "00000_phase.bin"


def test_folder_includes_all_matching_files(tmp_path):
    _write(tmp_path, 0, 0)
    _write(tmp_path, 1, 1)
    _write(tmp_path, 3, 3)  # a gap at index 2 does not stop discovery
    assert len(PhaseBinSequence(tmp_path, validate=None)) == 3


def test_folder_ignores_non_matching_names(tmp_path):
    _write(tmp_path, 0, 0)
    blank = np.zeros((2, 3), dtype=np.float32)
    save_phase_bin(
        tmp_path / "0001_phase.bin", blank, pixel_size=1e-6, height_scale=2e-7
    )
    save_phase_bin(
        tmp_path / "00002_amp.bin", blank, pixel_size=1e-6, height_scale=2e-7
    )
    assert len(PhaseBinSequence(tmp_path)) == 1


def test_empty_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no NNNNN_phase"):
        PhaseBinSequence(tmp_path)


def test_init_validate_runs_validation(tmp_path):
    _write(tmp_path, 0, 0)
    _write(tmp_path, 2, 2)  # gap at index 1
    PhaseBinSequence(tmp_path, validate=None)  # constructs despite the gap
    with pytest.raises(ValueError, match="non-contiguous"):
        PhaseBinSequence(tmp_path, validate="headers")


def test_init_validate_data_level_checks_pixels(tmp_path):
    nan = np.array([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float32)
    with pytest.warns(
        RuntimeWarning
    ):  # save_phase_bin validates input (on_nonfinite="warn")
        save_phase_bin(
            tmp_path / "00000_phase.bin", nan, pixel_size=1e-6, height_scale=2e-7
        )
    PhaseBinSequence(tmp_path, validate="headers")  # pixels not inspected: ok
    with pytest.raises(ValueError, match="finite"):
        PhaseBinSequence(tmp_path, validate="data")


def test_validate_passes_on_clean_folder(tmp_path):
    for i in range(3):
        _write(tmp_path, i, i)
    folder = PhaseBinSequence(tmp_path)
    folder.validate()
    folder.validate(level="data")  # finite data: also passes


def test_validate_names_level_skips_header_check(tmp_path):
    blank = np.zeros((2, 3), dtype=np.float32)
    save_phase_bin(
        tmp_path / "00000_phase.bin", blank, pixel_size=1e-6, height_scale=2e-7
    )
    save_phase_bin(
        tmp_path / "00001_phase.bin", blank, pixel_size=9e-6, height_scale=2e-7
    )
    folder = PhaseBinSequence(tmp_path, validate=None)
    folder.validate(level="names")  # header mismatch ignored at "names"
    with pytest.raises(ValueError, match="header"):
        folder.validate()  # default "headers" detects it


def test_validate_file_checks_single_index(tmp_path):
    _write(tmp_path, 0, 0)
    _write(tmp_path, 2, 2)  # gap at index 1
    seq = PhaseBinSequence(tmp_path, validate=None)
    seq.validate_file(0)  # 00000 at index 0: ok
    with pytest.raises(ValueError, match="non-contiguous"):
        seq.validate_file(1)  # 00002 sits at index 1


def test_validate_file_rejects_unknown_level(tmp_path):
    _write(tmp_path, 0, 0)
    seq = PhaseBinSequence(tmp_path, validate=None)
    with pytest.raises(ValueError, match="level must be"):
        seq.validate_file(0, level="bogus")  # ty: ignore[invalid-argument-type]


def test_validate_rejects_gap(tmp_path):
    _write(tmp_path, 0, 0)
    _write(tmp_path, 1, 1)
    _write(tmp_path, 3, 3)  # index 2 missing
    with pytest.raises(ValueError, match="non-contiguous"):
        PhaseBinSequence(tmp_path, validate=None).validate()


def test_validate_rejects_header_mismatch(tmp_path):
    blank = np.zeros((2, 3), dtype=np.float32)
    save_phase_bin(
        tmp_path / "00000_phase.bin", blank, pixel_size=1e-6, height_scale=2e-7
    )
    save_phase_bin(
        tmp_path / "00001_phase.bin", blank, pixel_size=9e-6, height_scale=2e-7
    )
    with pytest.raises(ValueError, match="header"):
        PhaseBinSequence(tmp_path, validate=None).validate()


def test_validate_check_data_detects_non_finite(tmp_path):
    nan = np.array([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float32)
    with pytest.warns(
        RuntimeWarning
    ):  # save_phase_bin validates input (on_nonfinite="warn")
        save_phase_bin(
            tmp_path / "00000_phase.bin", nan, pixel_size=1e-6, height_scale=2e-7
        )
    folder = PhaseBinSequence(tmp_path)
    folder.validate()  # "headers": pixels not inspected, passes
    with pytest.raises(ValueError, match="finite"):
        folder.validate(level="data")


def test_load_converts_radians_to_meters(tmp_path):
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    scale = 2e-7
    save_phase_bin(
        tmp_path / "00000_phase.bin",
        data,
        pixel_size=1e-6,
        height_scale=scale,
        unit=PhaseUnit.RADIANS,
    )
    folder = PhaseBinSequence(tmp_path, target_unit=PhaseUnit.METERS)
    np.testing.assert_array_equal(folder[0], (data * scale).astype(np.float32))


def test_load_converts_meters_to_radians(tmp_path):
    data = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    scale = 2e-7
    save_phase_bin(
        tmp_path / "00000_phase.bin",
        data,
        pixel_size=1e-6,
        height_scale=scale,
        unit=PhaseUnit.METERS,
    )
    folder = PhaseBinSequence(tmp_path, target_unit=PhaseUnit.RADIANS)
    np.testing.assert_array_equal(folder[0], (data / scale).astype(np.float32))


def test_load_no_conversion_by_default_or_same_unit(tmp_path):
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    save_phase_bin(
        tmp_path / "00000_phase.bin",
        data,
        pixel_size=1e-6,
        height_scale=2e-7,
        unit=PhaseUnit.RADIANS,
    )
    np.testing.assert_array_equal(PhaseBinSequence(tmp_path)[0], data)  # unit=None
    np.testing.assert_array_equal(
        PhaseBinSequence(tmp_path, target_unit=PhaseUnit.RADIANS)[0], data
    )


def test_load_rejects_unconvertible_unit(tmp_path):
    _write(tmp_path, 0, 0)  # stored as RADIANS
    folder = PhaseBinSequence(tmp_path, target_unit=PhaseUnit.UNKNOWN)
    with pytest.raises(ValueError, match="cannot convert"):
        _ = folder[0]
