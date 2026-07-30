from __future__ import annotations

import math

import numpy as np
import pytest

from iivs.dhm.data.phase.unit import (
    PhaseUnit,
    convert_phase_unit,
    resolve_height_scale,
    resolve_phase_unit,
)

# --- resolve_phase_unit ---


@pytest.mark.parametrize(
    ("name", "expected"),
    (
        ("radians", PhaseUnit.RADIANS),
        ("RADIANS", PhaseUnit.RADIANS),
        ("Meters", PhaseUnit.METERS),
        ("nanometers", PhaseUnit.NANOMETERS),
    ),
)
def test_resolve_phase_unit_is_case_insensitive(name, expected):
    assert resolve_phase_unit(name) is expected


def test_resolve_phase_unit_rejects_unknown_by_name():
    # UNKNOWN is a real member, so the lookup succeeds and the guard must reject
    # it: nothing converts to or from the absence of a unit.
    with pytest.raises(ValueError, match="unsupported unit 'unknown'"):
        resolve_phase_unit("unknown")


def test_resolve_phase_unit_rejects_an_unrecognized_name():
    with pytest.raises(ValueError, match="unsupported unit 'radiuns'") as excinfo:
        resolve_phase_unit("radiuns")

    # the message names the valid set, and never offers UNKNOWN as a choice
    message = str(excinfo.value)
    assert "expected radians, meters, nanometers" in message
    assert "unknown" not in message.removeprefix("unsupported unit 'radiuns'")


# --- resolve_height_scale ---


def test_resolve_height_scale_direct_form_passes_through():
    assert resolve_height_scale(2e-7, None, None) == 2e-7


def test_resolve_height_scale_derives_from_wavelength_pair():
    # height per rad = wavelength / (2*pi * refractive_delta), computed here
    # independently of the implementation.
    got = resolve_height_scale(None, 666e-9, 0.5)
    assert got == pytest.approx(666e-9 / (math.tau * 0.5))


@pytest.mark.parametrize(
    ("height_scale", "wavelength", "refractive_delta"),
    (
        (None, None, None),  # neither form
        (2e-7, 666e-9, 0.5),  # both forms
        (2e-7, 666e-9, None),  # direct form + a stray wavelength
        (2e-7, None, 0.5),  # direct form + a stray refractive_delta
        (None, 666e-9, None),  # half-filled pair (wavelength only)
        (None, None, 0.5),  # half-filled pair (refractive_delta only)
    ),
)
def test_resolve_height_scale_rejects_invalid_forms(
    height_scale, wavelength, refractive_delta
):
    with pytest.raises(ValueError, match="exactly one"):
        resolve_height_scale(height_scale, wavelength, refractive_delta)


@pytest.mark.parametrize("height_scale", (0.0, -1e-7, math.inf, math.nan))
def test_resolve_height_scale_rejects_nonpositive(height_scale):
    with pytest.raises(ValueError, match="height_scale must be positive"):
        resolve_height_scale(height_scale, None, None)


def test_resolve_height_scale_rejects_zero_refractive_delta():
    with pytest.raises(ValueError, match="refractive_delta must be nonzero"):
        resolve_height_scale(None, 666e-9, 0.0)


def test_resolve_height_scale_rejects_negative_wavelength_pair():
    # A negative wavelength derives a negative factor, rejected as non-positive.
    with pytest.raises(ValueError, match="height_scale must be positive"):
        resolve_height_scale(None, -666e-9, 0.5)


# --- convert_phase_unit ---


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


@pytest.mark.parametrize(
    ("source", "target"),
    (
        (PhaseUnit.RADIANS, PhaseUnit.METERS),  # RADIANS <-> METERS branch
        (PhaseUnit.NANOMETERS, PhaseUnit.RADIANS),  # RADIANS <-> NANOMETERS branch
    ),
)
def test_convert_phase_unit_rejects_nonpositive_height_scale(source, target):
    # A zero height_scale would silently zero the image (rad->m) or divide by zero
    # (m->rad); both are rejected up front for the RADIANS-crossing conversions.
    data = np.array([[1.0, 2.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="height_scale must be positive"):
        convert_phase_unit(data, source=source, target=target, height_scale=0.0)
