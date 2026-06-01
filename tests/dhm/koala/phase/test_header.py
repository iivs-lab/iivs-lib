from __future__ import annotations

import pytest

from iivs.dhm.koala.phase.header import PhaseBinHeader, PhaseUnit


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


def test_header_warns_on_unknown_unit():
    with pytest.warns(UserWarning, match="UNKNOWN"):
        PhaseBinHeader(
            width=2,
            height=2,
            pixel_size=1.0,
            height_scale=1.0,
            unit=PhaseUnit.UNKNOWN,
        )


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
