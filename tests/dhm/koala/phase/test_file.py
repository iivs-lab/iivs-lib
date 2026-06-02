from __future__ import annotations

import numpy as np
import pytest

from iivs.dhm.koala.phase.file import (
    convert_phase_unit,
    load_bin,
    read_header,
    save_bin,
    validate_phase,
)
from iivs.dhm.koala.phase.header import PhaseUnit


def test_validate_phase_rejects_below_2d():
    with pytest.raises(ValueError, match="2-dimensional"):
        validate_phase(np.zeros(5, dtype=np.float32))


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 5)).astype(np.float32)
    path = tmp_path / "phase.bin"

    save_bin(path, data, unit=PhaseUnit.RADIANS, pixel_size=1e-6, height_scale=2e-7)
    image, header = load_bin(path, return_header=True)

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
    save_bin(path, data, pixel_size=1.0, height_scale=1.0)  # unit defaults
    np.testing.assert_array_equal(load_bin(path), data)


def test_save_with_wavelength_and_refractive_delta(tmp_path):
    data = np.zeros((2, 3), dtype=np.float32)
    path = tmp_path / "phase.bin"
    wavelength, refractive_delta = 632.8e-9, 0.05
    save_bin(
        path,
        data,
        pixel_size=1e-6,
        wavelength=wavelength,
        refractive_delta=refractive_delta,
    )
    header = read_header(path)
    assert header.height_scale == pytest.approx(
        wavelength / (2 * np.pi * refractive_delta)
    )


def test_save_rejects_both_scale_forms(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        save_bin(
            tmp_path / "bad.bin",
            np.zeros((2, 2), dtype=np.float32),
            pixel_size=1.0,
            height_scale=1.0,
            wavelength=1.0,
        )


def test_save_rejects_non_2d(tmp_path):
    with pytest.raises(ValueError, match="single 2D image"):
        save_bin(
            tmp_path / "bad.bin",
            np.zeros(5, dtype=np.float32),
            pixel_size=1.0,
            height_scale=1.0,
        )


def test_save_rejects_3d_stack(tmp_path):
    # A 3D array passes validate_phase (ndim >= 2) but save_bin writes one image.
    with pytest.raises(ValueError, match="single 2D image"):
        save_bin(
            tmp_path / "bad.bin",
            np.zeros((2, 2, 2), dtype=np.float32),
            pixel_size=1.0,
            height_scale=1.0,
        )


def test_save_rejects_non_float32(tmp_path):
    with pytest.raises(ValueError, match="float32"):
        save_bin(
            tmp_path / "bad.bin",
            np.zeros((2, 2), dtype=np.float64),
            pixel_size=1.0,
            height_scale=1.0,
        )


def test_save_warns_on_unknown_unit(tmp_path):
    with pytest.warns(UserWarning, match="UNKNOWN"):
        save_bin(
            tmp_path / "phase.bin",
            np.zeros((2, 2), dtype=np.float32),
            pixel_size=1.0,
            height_scale=1.0,
            unit=PhaseUnit.UNKNOWN,
        )


def test_save_overwrite(tmp_path):
    data = np.zeros((2, 2), dtype=np.float32)
    path = tmp_path / "phase.bin"
    save_bin(path, data, pixel_size=1.0, height_scale=1.0)
    with pytest.raises(FileExistsError, match="already exists"):
        save_bin(path, data, pixel_size=1.0, height_scale=1.0)
    save_bin(path, data, pixel_size=1.0, height_scale=1.0, overwrite=True)


def test_save_nanometers_stores_meters(tmp_path):
    data_nm = np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float32)
    path = tmp_path / "phase.bin"
    save_bin(
        path, data_nm, pixel_size=1e-6, height_scale=2e-7, unit=PhaseUnit.NANOMETERS
    )
    image, header = load_bin(path, return_header=True)
    assert header.unit is PhaseUnit.METERS  # stored as meters, not nanometers
    np.testing.assert_allclose(image, data_nm * 1e-9, rtol=1e-5)


def test_read_header_matches_load(tmp_path):
    data = np.zeros((3, 4), dtype=np.float32)
    path = tmp_path / "phase.bin"
    save_bin(path, data, pixel_size=2e-6, height_scale=3e-7)
    assert read_header(path) == load_bin(path, return_header=True)[1]


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_bin(tmp_path / "nope.bin")


def test_load_rejects_truncated_header(tmp_path):
    path = tmp_path / "trunc.bin"
    path.write_bytes(b"\x00\x01\x02")
    with pytest.raises(ValueError, match="bytes for a header"):
        load_bin(path)


def test_load_rejects_unexpected_header_size(tmp_path):
    data = np.zeros((2, 3), dtype=np.float32)
    path = tmp_path / "phase.bin"
    save_bin(path, data, pixel_size=1.0, height_scale=1.0)
    with path.open("r+b") as fd:
        fd.seek(2)
        fd.write(np.int32(99).tobytes())
    with pytest.raises(ValueError, match="header size"):
        load_bin(path)


def test_load_rejects_pixel_count_mismatch(tmp_path):
    data = np.zeros((2, 3), dtype=np.float32)
    path = tmp_path / "phase.bin"
    save_bin(path, data, pixel_size=1.0, height_scale=1.0)
    with path.open("ab") as fd:
        fd.write(np.float32(1.0).tobytes())
    with pytest.raises(ValueError, match="pixel count"):
        load_bin(path)


def test_load_on_nonfinite_policy(tmp_path):
    data = np.array([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float32)
    path = tmp_path / "phase.bin"
    with pytest.warns(RuntimeWarning):  # save_bin validates input (on_nonfinite="warn")
        save_bin(path, data, pixel_size=1.0, height_scale=1.0)

    load_bin(path)  # default on_nonfinite="ignore": accepts silently
    with pytest.raises(ValueError, match="finite"):
        load_bin(path, on_nonfinite="raise")
    with pytest.warns(RuntimeWarning, match="finite"):
        load_bin(path, on_nonfinite="warn")


def test_convert_phase_unit_radians_to_meters():
    data = np.array([[1.0, 2.0]], dtype=np.float32)
    out = convert_phase_unit(
        data, from_unit=PhaseUnit.RADIANS, to_unit=PhaseUnit.METERS, height_scale=2.0
    )
    np.testing.assert_array_equal(out, (data * 2.0).astype(np.float32))


def test_convert_phase_unit_meters_to_radians():
    data = np.array([[2.0, 4.0]], dtype=np.float32)
    out = convert_phase_unit(
        data, from_unit=PhaseUnit.METERS, to_unit=PhaseUnit.RADIANS, height_scale=2.0
    )
    np.testing.assert_array_equal(out, (data / 2.0).astype(np.float32))


def test_convert_phase_unit_meters_to_nanometers():
    data = np.array([[1e-7, 2e-7]], dtype=np.float32)
    out = convert_phase_unit(
        data, from_unit=PhaseUnit.METERS, to_unit=PhaseUnit.NANOMETERS, height_scale=2.0
    )
    np.testing.assert_allclose(out, data * 1e9, rtol=1e-5)


def test_convert_phase_unit_nanometers_to_meters():
    data = np.array([[100.0, 200.0]], dtype=np.float32)
    out = convert_phase_unit(
        data, from_unit=PhaseUnit.NANOMETERS, to_unit=PhaseUnit.METERS, height_scale=2.0
    )
    np.testing.assert_allclose(out, data * 1e-9, rtol=1e-5)


def test_convert_phase_unit_radians_to_nanometers():
    data = np.array([[1.0, 2.0]], dtype=np.float32)
    out = convert_phase_unit(
        data,
        from_unit=PhaseUnit.RADIANS,
        to_unit=PhaseUnit.NANOMETERS,
        height_scale=3e-7,
    )
    np.testing.assert_allclose(out, data * 3e-7 * 1e9, rtol=1e-5)


def test_convert_phase_unit_nanometers_to_radians():
    data = np.array([[300.0, 600.0]], dtype=np.float32)
    out = convert_phase_unit(
        data,
        from_unit=PhaseUnit.NANOMETERS,
        to_unit=PhaseUnit.RADIANS,
        height_scale=3e-7,
    )
    np.testing.assert_allclose(out, data / 1e9 / 3e-7, rtol=1e-5)


def test_convert_phase_unit_same_unit_returns_input():
    data = np.zeros((2, 2), dtype=np.float32)
    out = convert_phase_unit(
        data, from_unit=PhaseUnit.RADIANS, to_unit=PhaseUnit.RADIANS, height_scale=2.0
    )
    assert out is data


def test_convert_phase_unit_rejects_unknown():
    data = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="cannot convert"):
        convert_phase_unit(
            data,
            from_unit=PhaseUnit.RADIANS,
            to_unit=PhaseUnit.UNKNOWN,
            height_scale=2.0,
        )
