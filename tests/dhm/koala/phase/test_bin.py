from __future__ import annotations

import io
import warnings

import numpy as np
import pytest

from iivs.dhm.koala.phase.bin import (
    PhaseBinHeader,
    PhaseBinSequence,
    load_phase_bin,
    read_phase_bin_header,
    save_phase_bin,
)
from iivs.dhm.koala.phase.core import PhaseUnit


def _valid_header_bytes() -> bytearray:
    header = PhaseBinHeader(
        width=2, height=2, pixel_size=1.0, height_scale=1.0, unit=PhaseUnit.RADIANS
    )
    return bytearray(header.to_dtype().tobytes())


def _write(root, index, value, shape=(2, 3)):
    save_phase_bin(
        root / f"{index:05d}_phase.bin",
        np.full(shape, float(value), dtype=np.float32),
        pixel_size=1e-6,
        height_scale=2e-7,
    )


# ========================== #
#           Header           #
# ========================== #


def test_header_dtype_is_packed_23_bytes():
    assert PhaseBinHeader.DTYPE.itemsize == 23


def test_header_dtype_roundtrip():
    header = PhaseBinHeader(
        width=3,
        height=2,
        pixel_size=0.5,
        height_scale=0.25,
        unit=PhaseUnit.RADIANS,
    )
    assert PhaseBinHeader.from_dtype(header.to_dtype()[0]) == header


def test_header_shape_and_pixel_count():
    header = PhaseBinHeader(
        width=5, height=3, pixel_size=1.0, height_scale=1.0, unit=PhaseUnit.RADIANS
    )
    assert header.shape == (3, 5)
    assert header.pixel_count == 15


def test_header_field_of_view():
    header = PhaseBinHeader(
        width=5,
        height=3,
        pixel_size=2e-6,
        height_scale=1.0,
        unit=PhaseUnit.RADIANS,
    )
    assert header.field_of_view == pytest.approx((3 * 2e-6, 5 * 2e-6))


def test_header_convenience_units():
    header = PhaseBinHeader(
        width=5,
        height=3,
        pixel_size=2e-6,
        height_scale=3e-7,
        unit=PhaseUnit.RADIANS,
    )
    assert header.pixel_size_um == pytest.approx(2.0)
    assert header.field_of_view_um == pytest.approx((3 * 2.0, 5 * 2.0))
    assert header.height_scale_nm == pytest.approx(300.0)


def test_header_unknown_unit_does_not_warn():
    # Constructing/reading an UNKNOWN-unit header is silent; the alert lives
    # at the save boundary (see save_phase_bin) instead of every construction.
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # turn any warning into an error
        PhaseBinHeader(
            width=2,
            height=2,
            pixel_size=1.0,
            height_scale=1.0,
            unit=PhaseUnit.UNKNOWN,
        )


def test_from_stream_rejects_unsupported_version():
    raw = _valid_header_bytes()
    raw[0] = 2  # version byte
    with pytest.raises(ValueError, match="version"):
        PhaseBinHeader.from_stream(io.BytesIO(bytes(raw)))


def test_from_stream_rejects_unsupported_endian():
    raw = _valid_header_bytes()
    raw[1] = 1  # endian byte
    with pytest.raises(ValueError, match="byte order"):
        PhaseBinHeader.from_stream(io.BytesIO(bytes(raw)))


def test_header_version_endian_fixed():
    header = PhaseBinHeader(
        width=2, height=2, pixel_size=1.0, height_scale=1.0, unit=PhaseUnit.RADIANS
    )
    assert header.version == 1
    assert header.endian == 0


def test_header_rejects_nonpositive_dims():
    with pytest.raises(ValueError, match="must be positive"):
        PhaseBinHeader(
            width=0,
            height=2,
            pixel_size=1.0,
            height_scale=1.0,
            unit=PhaseUnit.RADIANS,
        )


def test_header_rejects_nonpositive_pixel_size():
    with pytest.raises(ValueError, match="pixel_size must be positive"):
        PhaseBinHeader(
            width=2,
            height=2,
            pixel_size=0.0,
            height_scale=1.0,
            unit=PhaseUnit.RADIANS,
        )


def test_header_rejects_nonpositive_height_scale():
    with pytest.raises(ValueError, match="height_scale must be positive"):
        PhaseBinHeader(
            width=2,
            height=2,
            pixel_size=1.0,
            height_scale=0.0,
            unit=PhaseUnit.RADIANS,
        )


def test_header_rejects_invalid_unit():
    with pytest.raises(ValueError, match="unit must be one of"):
        PhaseBinHeader(width=2, height=2, pixel_size=1.0, height_scale=1.0, unit=99)


def test_header_rejects_nanometers_unit():
    # NANOMETERS is code-only; it cannot be a stored (header) unit.
    with pytest.raises(ValueError, match="unit must be one of"):
        PhaseBinHeader(
            width=2,
            height=2,
            pixel_size=1.0,
            height_scale=1.0,
            unit=PhaseUnit.NANOMETERS,
        )


def test_header_is_hashable_and_comparable():
    a = PhaseBinHeader(
        width=3, height=2, pixel_size=0.5, height_scale=0.25, unit=PhaseUnit.RADIANS
    )
    b = PhaseBinHeader(
        width=3, height=2, pixel_size=0.5, height_scale=0.25, unit=PhaseUnit.RADIANS
    )
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


# ========================== #
#       Single-file I/O      #
# ========================== #


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 5)).astype(np.float32)
    path = tmp_path / "phase.bin"

    save_phase_bin(
        path, data, unit=PhaseUnit.RADIANS, pixel_size=1e-6, height_scale=2e-7
    )
    image, header = load_phase_bin(path, return_header=True)

    np.testing.assert_array_equal(image, data)
    assert image.dtype == np.float32
    assert image.flags.writeable  # frombuffer view is copied to a writable array
    assert header.shape == (4, 5)
    assert header.unit is PhaseUnit.RADIANS
    assert header.pixel_size == pytest.approx(1e-6)
    assert header.height_scale == pytest.approx(2e-7)


def test_save_load_is_row_major(tmp_path):
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    path = tmp_path / "phase.bin"
    save_phase_bin(path, data, pixel_size=1.0, height_scale=1.0)  # unit defaults
    np.testing.assert_array_equal(load_phase_bin(path), data)


def test_save_with_wavelength_and_refractive_delta(tmp_path):
    data = np.zeros((2, 3), dtype=np.float32)
    path = tmp_path / "phase.bin"
    wavelength, refractive_delta = 632.8e-9, 0.05
    save_phase_bin(
        path,
        data,
        pixel_size=1e-6,
        wavelength=wavelength,
        refractive_delta=refractive_delta,
    )
    header = read_phase_bin_header(path)
    assert header.height_scale == pytest.approx(
        wavelength / (2 * np.pi * refractive_delta)
    )


def test_save_rejects_both_scale_forms(tmp_path):
    with pytest.raises(ValueError, match="not both"):
        save_phase_bin(
            tmp_path / "bad.bin",
            np.zeros((2, 2), dtype=np.float32),
            pixel_size=1.0,
            height_scale=1.0,
            wavelength=1.0,
        )


def test_save_rejects_non_2d(tmp_path):
    with pytest.raises(ValueError, match="single 2D image"):
        save_phase_bin(
            tmp_path / "bad.bin",
            np.zeros(5, dtype=np.float32),
            pixel_size=1.0,
            height_scale=1.0,
        )


def test_save_rejects_3d_stack(tmp_path):
    # A 3D array passes validate_phase (ndim >= 2) but save_phase_bin writes one image.
    with pytest.raises(ValueError, match="single 2D image"):
        save_phase_bin(
            tmp_path / "bad.bin",
            np.zeros((2, 2, 2), dtype=np.float32),
            pixel_size=1.0,
            height_scale=1.0,
        )


def test_save_rejects_non_float32(tmp_path):
    with pytest.raises(ValueError, match="float32"):
        save_phase_bin(
            tmp_path / "bad.bin",
            np.zeros((2, 2), dtype=np.float64),
            pixel_size=1.0,
            height_scale=1.0,
        )


def test_save_warns_on_unknown_unit(tmp_path):
    with pytest.warns(UserWarning, match="UNKNOWN"):
        save_phase_bin(
            tmp_path / "phase.bin",
            np.zeros((2, 2), dtype=np.float32),
            pixel_size=1.0,
            height_scale=1.0,
            unit=PhaseUnit.UNKNOWN,
        )


def test_save_overwrite(tmp_path):
    data = np.zeros((2, 2), dtype=np.float32)
    path = tmp_path / "phase.bin"
    save_phase_bin(path, data, pixel_size=1.0, height_scale=1.0)
    with pytest.raises(FileExistsError, match="already exists"):
        save_phase_bin(path, data, pixel_size=1.0, height_scale=1.0)
    save_phase_bin(path, data, pixel_size=1.0, height_scale=1.0, overwrite=True)


def test_save_nanometers_stores_meters(tmp_path):
    data_nm = np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float32)
    path = tmp_path / "phase.bin"
    save_phase_bin(
        path, data_nm, pixel_size=1e-6, height_scale=2e-7, unit=PhaseUnit.NANOMETERS
    )
    image, header = load_phase_bin(path, return_header=True)
    assert header.unit is PhaseUnit.METERS  # stored as meters, not nanometers
    np.testing.assert_allclose(image, data_nm * 1e-9, rtol=1e-5)


def test_read_header_matches_load(tmp_path):
    data = np.zeros((3, 4), dtype=np.float32)
    path = tmp_path / "phase.bin"
    save_phase_bin(path, data, pixel_size=2e-6, height_scale=3e-7)
    assert read_phase_bin_header(path) == load_phase_bin(path, return_header=True)[1]


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_phase_bin(tmp_path / "nope.bin")


def test_load_rejects_truncated_header(tmp_path):
    path = tmp_path / "trunc.bin"
    path.write_bytes(b"\x00\x01\x02")
    with pytest.raises(ValueError, match="bytes for a header"):
        load_phase_bin(path)


def test_load_rejects_unexpected_header_size(tmp_path):
    data = np.zeros((2, 3), dtype=np.float32)
    path = tmp_path / "phase.bin"
    save_phase_bin(path, data, pixel_size=1.0, height_scale=1.0)
    with path.open("r+b") as fd:
        fd.seek(2)
        fd.write(np.int32(99).tobytes())
    with pytest.raises(ValueError, match="header size"):
        load_phase_bin(path)


def test_load_rejects_pixel_count_mismatch(tmp_path):
    data = np.zeros((2, 3), dtype=np.float32)
    path = tmp_path / "phase.bin"
    save_phase_bin(path, data, pixel_size=1.0, height_scale=1.0)
    with path.open("ab") as fd:
        fd.write(np.float32(1.0).tobytes())
    with pytest.raises(ValueError, match="pixel count"):
        load_phase_bin(path)


def test_load_on_nonfinite_policy(tmp_path):
    data = np.array([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float32)
    path = tmp_path / "phase.bin"
    with pytest.warns(
        RuntimeWarning
    ):  # save_phase_bin validates input (on_nonfinite="warn")
        save_phase_bin(path, data, pixel_size=1.0, height_scale=1.0)

    load_phase_bin(path)  # default on_nonfinite="ignore": accepts silently
    with pytest.raises(ValueError, match="finite"):
        load_phase_bin(path, on_nonfinite="raise")
    with pytest.warns(RuntimeWarning, match="finite"):
        load_phase_bin(path, on_nonfinite="warn")


# ========================== #
#          Sequence          #
# ========================== #


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
    assert folder.frame_shape == (2, 3)
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


def test_rejects_unconvertible_target_unit_at_construction(tmp_path):
    _write(tmp_path, 0, 0)  # stored as RADIANS
    # Fail fast: an unreachable target unit is rejected when constructing,
    # not lazily on first item access.
    with pytest.raises(ValueError, match="cannot convert"):
        PhaseBinSequence(tmp_path, target_unit=PhaseUnit.UNKNOWN)
