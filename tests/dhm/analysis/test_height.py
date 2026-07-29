from __future__ import annotations

import numpy as np
import pytest

from iivs.dhm.analysis.height import (
    OpticalHeightConverter,
    height_to_opd,
    height_to_phase,
    opd_to_height,
    phase_to_height,
)
from iivs.dhm.analysis.opd import OPDConverter
from iivs.dhm.data.phase.unit import resolve_height_scale


def test_opd_to_height_divides_by_delta():
    # height = OPD / delta: 100 nm OPD at delta 0.5 is 200 nm of thickness
    # (routed through phase internally, hence the small float tolerance).
    opd = np.full((2, 3), 100.0, dtype=np.float32)
    height = opd_to_height(opd, refractive_delta=0.5)
    np.testing.assert_allclose(
        height, np.full((2, 3), 200.0, dtype=np.float32), rtol=1e-6
    )
    assert height.dtype == np.float32


def test_opd_entry_is_wavelength_independent():
    # The OPD -> phase -> height composition cancels the bound wavelength, so
    # a converter at any wavelength gives opd / delta.
    opd = np.full((2, 2), 100.0, dtype=np.float32)
    at_532 = OpticalHeightConverter.from_args(wavelength=532e-9, refractive_delta=0.4)
    np.testing.assert_allclose(
        at_532.convert_from_opd(opd),
        np.full((2, 2), 250.0, dtype=np.float32),  # 100 / 0.4, no 532 in sight
        rtol=1e-6,
    )


def test_height_to_opd_is_the_inverse():
    height = np.array([[100.0, 250.0], [0.0, -50.0]], dtype=np.float32)
    opd = height_to_opd(height, refractive_delta=0.4)
    np.testing.assert_allclose(opd, height * np.float32(0.4), rtol=1e-6)
    np.testing.assert_allclose(
        opd_to_height(opd, refractive_delta=0.4), height, rtol=1e-6
    )

    converter = OpticalHeightConverter(refractive_delta=0.4)
    np.testing.assert_allclose(
        converter.convert_from_opd(converter.convert_to_opd(height)),
        height,
        rtol=1e-6,
    )


def test_phase_height_roundtrip():
    phase = np.array([[0.5, 1.0], [2.0, -0.5]], dtype=np.float32)
    converter = OpticalHeightConverter.from_args(
        wavelength=666e-9, refractive_delta=0.5
    )
    back = converter.convert_to_phase(converter.convert_to_height(phase))
    np.testing.assert_allclose(back, phase, rtol=1e-6)

    one_shot = height_to_phase(
        phase_to_height(phase, wavelength=666e-9, refractive_delta=0.5),
        wavelength=666e-9,
        refractive_delta=0.5,
    )
    np.testing.assert_allclose(one_shot, phase, rtol=1e-6)


def test_phase_to_height_hand_derived():
    # 1 rad at 666 nm, delta 0.5: OPD 666/(2pi) = 105.9957 nm -> height 211.99 nm.
    phase = np.full((2, 2), 1.0, dtype=np.float32)
    height = phase_to_height(phase, wavelength=666e-9, refractive_delta=0.5)
    np.testing.assert_allclose(height, np.full((2, 2), 211.99, np.float32), rtol=1e-4)


def test_height_scale_matches_resolve_height_scale():
    # The same factor the shared phase-unit helper derives for the wavelength +
    # delta form, so the analysis and stored-phase conversions stay aligned.
    converter = OpticalHeightConverter.from_args(
        wavelength=666e-9, refractive_delta=0.5
    )
    expected_nm = resolve_height_scale(None, 666e-9, 0.5) * 1e9
    assert converter.height_scale == pytest.approx(expected_nm)
    phase = np.full((2, 2), 2.0, dtype=np.float32)
    np.testing.assert_allclose(
        converter.convert_to_height(phase),
        phase * np.float32(expected_nm),
        rtol=1e-6,
    )


def test_converter_surfaces_the_bound_wavelength():
    converter = OpticalHeightConverter.from_args(
        wavelength=532e-9, refractive_delta=0.5
    )
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
