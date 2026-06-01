from __future__ import annotations

import numpy as np
import pytest

from iivs.dhm.lynceetec.phase import (
    PhaseBinHeader,
    PhaseUnit,
    load_bin,
    save_bin,
)


def test_header_dtype_is_packed_23_bytes():
    assert PhaseBinHeader.DTYPE.itemsize == 23


def test_header_dtype_roundtrip():
    header = PhaseBinHeader(
        width=3,
        height=2,
        pixel_size=0.5,
        height_per_radian=0.25,
        unit=PhaseUnit.RADIANS,
    )
    assert PhaseBinHeader.from_dtype(header.to_dtype()[0]) == header


def test_header_shape():
    header = PhaseBinHeader(
        width=5, height=3, pixel_size=1.0, height_per_radian=1.0, unit=PhaseUnit.RADIANS
    )
    assert header.shape == (3, 5)


def test_header_field_of_view():
    header = PhaseBinHeader(
        width=5,
        height=3,
        pixel_size=2e-6,
        height_per_radian=1.0,
        unit=PhaseUnit.RADIANS,
    )
    assert header.field_of_view == pytest.approx((3 * 2e-6, 5 * 2e-6))


def test_header_convenience_units():
    header = PhaseBinHeader(
        width=5,
        height=3,
        pixel_size=2e-6,
        height_per_radian=3e-7,
        unit=PhaseUnit.RADIANS,
    )
    assert header.pixel_size_um == pytest.approx(2.0)
    assert header.field_of_view_um == pytest.approx((3 * 2.0, 5 * 2.0))
    assert header.height_per_radian_nm == pytest.approx(300.0)


def test_header_warns_on_unknown_unit():
    with pytest.warns(UserWarning, match="UNKNOWN"):
        PhaseBinHeader(
            width=2,
            height=2,
            pixel_size=1.0,
            height_per_radian=1.0,
            unit=PhaseUnit.UNKNOWN,
        )


def test_header_version_endian_fixed():
    header = PhaseBinHeader(
        width=2, height=2, pixel_size=1.0, height_per_radian=1.0, unit=PhaseUnit.RADIANS
    )
    assert header.version == 1
    assert header.endian == 0


def test_header_rejects_nonpositive_dims():
    with pytest.raises(ValueError, match="must be positive"):
        PhaseBinHeader(
            width=0,
            height=2,
            pixel_size=1.0,
            height_per_radian=1.0,
            unit=PhaseUnit.RADIANS,
        )


def test_header_rejects_nonpositive_pixel_size():
    with pytest.raises(ValueError, match="pixel_size must be positive"):
        PhaseBinHeader(
            width=2,
            height=2,
            pixel_size=0.0,
            height_per_radian=1.0,
            unit=PhaseUnit.RADIANS,
        )


def test_header_rejects_nonpositive_height_per_radian():
    with pytest.raises(ValueError, match="height_per_radian must be positive"):
        PhaseBinHeader(
            width=2,
            height=2,
            pixel_size=1.0,
            height_per_radian=0.0,
            unit=PhaseUnit.RADIANS,
        )


def test_header_rejects_invalid_unit():
    with pytest.raises(ValueError, match="unit must be one of"):
        PhaseBinHeader(
            width=2, height=2, pixel_size=1.0, height_per_radian=1.0, unit=99
        )


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 5)).astype(np.float32)
    path = tmp_path / "phase.bin"

    save_bin(
        path, data, unit=PhaseUnit.RADIANS, pixel_size=1e-6, height_per_radian=2e-7
    )
    image, header = load_bin(path, return_header=True)

    np.testing.assert_array_equal(image, data)
    assert image.dtype == np.float32
    assert header.shape == (4, 5)
    assert header.unit is PhaseUnit.RADIANS
    assert header.pixel_size == pytest.approx(1e-6)
    assert header.height_per_radian == pytest.approx(2e-7)


def test_save_load_is_row_major(tmp_path):
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    path = tmp_path / "phase.bin"
    save_bin(path, data, pixel_size=1.0, height_per_radian=1.0)  # unit defaults
    np.testing.assert_array_equal(load_bin(path), data)


def test_save_rejects_non_2d(tmp_path):
    with pytest.raises(ValueError, match="2D"):
        save_bin(
            tmp_path / "bad.bin",
            np.zeros(5, dtype=np.float32),
            pixel_size=1.0,
            height_per_radian=1.0,
        )


def test_save_rejects_non_float32(tmp_path):
    with pytest.raises(ValueError, match="float32"):
        save_bin(
            tmp_path / "bad.bin",
            np.zeros((2, 2), dtype=np.float64),
            pixel_size=1.0,
            height_per_radian=1.0,
        )


def test_save_overwrite(tmp_path):
    data = np.zeros((2, 2), dtype=np.float32)
    path = tmp_path / "phase.bin"
    save_bin(path, data, pixel_size=1.0, height_per_radian=1.0)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        save_bin(path, data, pixel_size=1.0, height_per_radian=1.0)
    save_bin(path, data, pixel_size=1.0, height_per_radian=1.0, overwrite=True)


def test_load_rejects_truncated_header(tmp_path):
    path = tmp_path / "trunc.bin"
    path.write_bytes(b"\x00\x01\x02")
    with pytest.raises(ValueError, match="too short"):
        load_bin(path)


def test_load_rejects_unexpected_header_size(tmp_path):
    data = np.zeros((2, 3), dtype=np.float32)
    path = tmp_path / "phase.bin"
    save_bin(path, data, pixel_size=1.0, height_per_radian=1.0)
    with path.open("r+b") as fd:
        fd.seek(2)
        fd.write(np.int32(99).tobytes())
    with pytest.raises(ValueError, match="unexpected header size 99"):
        load_bin(path)


def test_load_rejects_pixel_count_mismatch(tmp_path):
    data = np.zeros((2, 3), dtype=np.float32)
    path = tmp_path / "phase.bin"
    save_bin(path, data, pixel_size=1.0, height_per_radian=1.0)
    with path.open("ab") as fd:
        fd.write(np.float32(1.0).tobytes())
    with pytest.raises(ValueError, match="expected 6 float32 values"):
        load_bin(path)
