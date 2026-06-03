from __future__ import annotations

import numpy as np
import pytest

from iivs.dhm.data.phase.core import PhaseUnit, convert_phase_unit, validate_phase


def test_validate_phase_rejects_below_2d():
    with pytest.raises(ValueError, match="2-dimensional"):
        validate_phase(np.zeros(5, dtype=np.float32))


def test_validate_phase_rejects_unknown_on_nonfinite():
    with pytest.raises(ValueError, match="on_nonfinite must be"):
        validate_phase(np.zeros((2, 2), dtype=np.float32), on_nonfinite="bogus")  # ty: ignore[invalid-argument-type]


def test_convert_phase_unit_radians_to_meters():
    data = np.array([[1.0, 2.0]], dtype=np.float32)
    out = convert_phase_unit(
        data, source=PhaseUnit.RADIANS, target=PhaseUnit.METERS, height_scale=2.0
    )
    np.testing.assert_array_equal(out, (data * 2.0).astype(np.float32))


def test_convert_phase_unit_meters_to_radians():
    data = np.array([[2.0, 4.0]], dtype=np.float32)
    out = convert_phase_unit(
        data, source=PhaseUnit.METERS, target=PhaseUnit.RADIANS, height_scale=2.0
    )
    np.testing.assert_array_equal(out, (data / 2.0).astype(np.float32))


def test_convert_phase_unit_meters_to_nanometers():
    data = np.array([[1e-7, 2e-7]], dtype=np.float32)
    out = convert_phase_unit(
        data, source=PhaseUnit.METERS, target=PhaseUnit.NANOMETERS, height_scale=2.0
    )
    np.testing.assert_allclose(out, data * 1e9, rtol=1e-5)


def test_convert_phase_unit_nanometers_to_meters():
    data = np.array([[100.0, 200.0]], dtype=np.float32)
    out = convert_phase_unit(
        data, source=PhaseUnit.NANOMETERS, target=PhaseUnit.METERS, height_scale=2.0
    )
    np.testing.assert_allclose(out, data * 1e-9, rtol=1e-5)


def test_convert_phase_unit_radians_to_nanometers():
    data = np.array([[1.0, 2.0]], dtype=np.float32)
    out = convert_phase_unit(
        data,
        source=PhaseUnit.RADIANS,
        target=PhaseUnit.NANOMETERS,
        height_scale=3e-7,
    )
    np.testing.assert_allclose(out, data * 3e-7 * 1e9, rtol=1e-5)


def test_convert_phase_unit_nanometers_to_radians():
    data = np.array([[300.0, 600.0]], dtype=np.float32)
    out = convert_phase_unit(
        data,
        source=PhaseUnit.NANOMETERS,
        target=PhaseUnit.RADIANS,
        height_scale=3e-7,
    )
    np.testing.assert_allclose(out, data / 1e9 / 3e-7, rtol=1e-5)


def test_convert_phase_unit_same_unit_returns_input():
    data = np.zeros((2, 2), dtype=np.float32)
    out = convert_phase_unit(
        data, source=PhaseUnit.RADIANS, target=PhaseUnit.RADIANS, height_scale=2.0
    )
    assert out is data


def test_convert_phase_unit_rejects_unknown():
    data = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="cannot convert"):
        convert_phase_unit(
            data,
            source=PhaseUnit.RADIANS,
            target=PhaseUnit.UNKNOWN,
            height_scale=2.0,
        )
