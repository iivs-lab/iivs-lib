from __future__ import annotations

import numpy as np
import pytest

from iivs.dhm.analysis.height import (
    OpticalHeightConverter,
    height_to_opd,
    opd_to_height,
    phase_to_height,
)
from iivs.dhm.analysis.opd import OPDConverter
from iivs.dhm.data.phase.unit import resolve_height_scale


def test_opd_to_height_divides_by_delta():
    # height = OPD / delta: 100 nm OPD at delta 0.5 is 200 nm of thickness.
    opd = np.full((2, 3), 100.0, dtype=np.float32)
    height = opd_to_height(opd, refractive_delta=0.5)
    np.testing.assert_allclose(height, np.full((2, 3), 200.0, dtype=np.float32))
    assert height.dtype == np.float32


def test_height_to_opd_is_the_inverse():
    height = np.array([[100.0, 250.0], [0.0, -50.0]], dtype=np.float32)
    opd = height_to_opd(height, refractive_delta=0.4)
    np.testing.assert_allclose(opd, height * np.float32(0.4), rtol=1e-6)
    np.testing.assert_allclose(
        opd_to_height(opd, refractive_delta=0.4), height, rtol=1e-6
    )

    converter = OpticalHeightConverter(refractive_delta=0.4)
    np.testing.assert_allclose(
        converter.convert_to_height(converter.convert_to_opd(height)),
        height,
        rtol=1e-6,
    )


def test_phase_to_height_hand_derived():
    # 1 rad at 666 nm, delta 0.5: OPD 666/(2pi) = 105.9957 nm -> height 211.99 nm.
    phase = np.full((2, 2), 1.0, dtype=np.float32)
    height = phase_to_height(phase, wavelength=666e-9, refractive_delta=0.5)
    np.testing.assert_allclose(height, np.full((2, 2), 211.99, np.float32), rtol=1e-4)


def test_height_scale_matches_the_data_layer():
    # The nm twin of the .bin header's height_scale (m/rad): the same factor
    # resolve_height_scale derives for the wavelength + delta form.
    converter = OpticalHeightConverter.from_wavelength(
        wavelength=666e-9, refractive_delta=0.5
    )
    expected_nm = resolve_height_scale(None, 666e-9, 0.5) * 1e9
    assert converter.height_scale == pytest.approx(expected_nm)
    phase = np.full((2, 2), 2.0, dtype=np.float32)
    np.testing.assert_allclose(
        converter.convert_from_phase(phase),
        phase * np.float32(expected_nm),
        rtol=1e-6,
    )


def test_converter_surfaces_the_bound_wavelength():
    converter = OpticalHeightConverter.from_wavelength(wavelength=532e-9)
    assert converter.wavelength == pytest.approx(532e-9)
    assert converter.wavelength_nm == pytest.approx(532.0)


def test_converter_accepts_injected_opd_converter():
    opd_converter = OPDConverter.from_wavelength_nm(532)
    converter = OpticalHeightConverter(
        refractive_delta=0.5, opd_converter=opd_converter
    )
    assert converter.opd_converter is opd_converter
    assert converter.height_scale == pytest.approx(opd_converter.opd_scale / 0.5)


def test_converter_rejects_nonpositive_delta():
    with pytest.raises(ValueError, match="refractive_delta must be positive"):
        OpticalHeightConverter(refractive_delta=0.0)
    with pytest.raises(ValueError, match="refractive_delta must be positive"):
        OpticalHeightConverter(refractive_delta=-0.5)
