from __future__ import annotations

import numpy as np
import pytest

from iivs.dhm.analysis.opd import OPDConverter, opd_to_phase, phase_to_opd
from iivs.dhm.data.constants import DEFAULT_WAVELENGTH


def test_phase_to_opd_scales_by_wavelength_over_two_pi():
    phase = np.array([[0.0, np.pi], [2 * np.pi, np.pi / 2]], dtype=np.float32)
    opd = phase_to_opd(phase, wavelength=666e-9)  # OPD in nanometers
    expected = (phase * (666e-9 / (2 * np.pi) * 1e9)).astype(np.float32)
    np.testing.assert_allclose(opd, expected, rtol=1e-6)
    assert opd.dtype == np.float32


def test_phase_to_opd_uses_default_wavelength():
    phase = np.full((2, 2), np.pi, dtype=np.float32)
    np.testing.assert_array_equal(
        phase_to_opd(phase), phase_to_opd(phase, wavelength=DEFAULT_WAVELENGTH)
    )


def test_opd_round_trips_with_phase():
    phase = np.array([[0.1, 1.0], [2.0, 3.0]], dtype=np.float32)
    np.testing.assert_allclose(opd_to_phase(phase_to_opd(phase)), phase, rtol=1e-5)


def test_converter_matches_free_functions():
    phase = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    conv = OPDConverter(wavelength=666e-9)
    np.testing.assert_array_equal(
        conv.convert_to_opd(phase), phase_to_opd(phase, wavelength=666e-9)
    )


def test_converter_round_trips():
    conv = OPDConverter(wavelength=666e-9)
    phase = np.array([[0.1, 1.0], [2.0, 3.0]], dtype=np.float32)
    np.testing.assert_allclose(
        conv.convert_to_phase(conv.convert_to_opd(phase)), phase, rtol=1e-5
    )


def test_converter_defaults_to_default_wavelength():
    assert OPDConverter().wavelength == DEFAULT_WAVELENGTH


def test_converter_rejects_nonpositive_wavelength():
    with pytest.raises(ValueError, match="wavelength must be positive"):
        OPDConverter(wavelength=0.0)


def test_converter_from_wavelength_nm():
    assert OPDConverter.from_wavelength_nm(666).wavelength == pytest.approx(666e-9)


def test_converter_wavelength_nm_property():
    assert OPDConverter(wavelength=666e-9).wavelength_nm == pytest.approx(666.0)


def test_converter_opd_scale_property():
    conv = OPDConverter(wavelength=666e-9)
    assert conv.opd_scale == pytest.approx(666e-9 / (2 * np.pi) * 1e9)  # nm/rad
    # convert_to_opd is exactly phase * opd_scale (OPD already in nanometers)
    phase = np.array([[1.0, 2.0]], dtype=np.float32)
    np.testing.assert_allclose(phase * conv.opd_scale, conv.convert_to_opd(phase))


def test_converter_from_wavelength_nm_rejects_nonpositive():
    with pytest.raises(ValueError, match="wavelength must be positive"):
        OPDConverter.from_wavelength_nm(0.0)
