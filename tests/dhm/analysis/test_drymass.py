from __future__ import annotations

import numpy as np
import pytest

from iivs.dhm.analysis.drymass import (
    DryMassCalculator,
    calc_drymass,
    calc_drymass_from_phase,
)
from iivs.dhm.analysis.opd import OPDConverter, phase_to_opd


def test_calc_drymass_uniform_region():
    # 50 nm OPD over 100 px of 0.1 um pitch; alpha 2e-4 m^3/kg = 0.2 um^3/pg:
    # per px = 0.05 um * (0.1 um)^2 / 0.2 um^3/pg = 2.5e-3 pg; x100 = 0.25 pg.
    opd = np.full((10, 10), 50.0, dtype=np.float32)  # nm
    assert calc_drymass(opd, pixel_size=1e-7, alpha=2.0e-4) == pytest.approx(0.25)


def test_calc_drymass_respects_mask():
    opd = np.full((2, 2), 50.0, dtype=np.float32)  # 4 equal pixels, nm
    mask = np.array([[True, False], [False, False]])  # select 1
    whole = calc_drymass(opd, pixel_size=1e-7, alpha=2.0e-4)
    one = calc_drymass(opd, pixel_size=1e-7, alpha=2.0e-4, mask=mask)
    assert one == pytest.approx(whole / 4)


def test_calc_drymass_scales_with_alpha():
    opd = np.full((4, 4), 30.0, dtype=np.float32)  # nm
    m1 = calc_drymass(opd, pixel_size=1e-7, alpha=2.0e-4)
    m2 = calc_drymass(opd, pixel_size=1e-7, alpha=4.0e-4)
    assert m2 == pytest.approx(m1 / 2)  # mass is inversely proportional to alpha


def test_calc_drymass_from_phase_matches_two_step():
    phase = np.full((5, 5), 1.0, dtype=np.float32)
    direct = calc_drymass_from_phase(
        phase, pixel_size=1e-7, wavelength=666e-9, alpha=2.0e-4
    )
    two_step = calc_drymass(
        phase_to_opd(phase, wavelength=666e-9), pixel_size=1e-7, alpha=2.0e-4
    )
    assert direct == pytest.approx(two_step)


# --- DryMassCalculator ---


def test_calculator_calc_from_opd_matches_free_function():
    opd = np.full((10, 10), 50.0, dtype=np.float32)
    calc = DryMassCalculator(pixel_size=1e-7, alpha=2.0e-4)
    assert calc.calc_from_opd(opd) == pytest.approx(
        calc_drymass(opd, pixel_size=1e-7, alpha=2.0e-4)
    )


def test_calculator_from_wavelength_matches_free_function():
    phase = np.full((5, 5), 1.0, dtype=np.float32)
    calc = DryMassCalculator.from_wavelength(
        pixel_size=1e-7, alpha=2.0e-4, wavelength=666e-9
    )
    assert calc.calc_from_phase(phase) == pytest.approx(
        calc_drymass_from_phase(phase, pixel_size=1e-7, alpha=2.0e-4, wavelength=666e-9)
    )


def test_calculator_accepts_injected_converter():
    phase = np.full((3, 3), 1.0, dtype=np.float32)
    conv = OPDConverter.from_wavelength_nm(666)
    calc = DryMassCalculator(pixel_size=1e-7, alpha=2.0e-4, opd_converter=conv)
    assert calc.opd_converter is conv
    expected = DryMassCalculator.from_wavelength(
        pixel_size=1e-7, alpha=2.0e-4, wavelength=666e-9
    ).calc_from_phase(phase)
    assert calc.calc_from_phase(phase) == pytest.approx(expected)


def test_calculator_wavelength_shortcuts():
    calc = DryMassCalculator.from_wavelength(pixel_size=1e-7, wavelength=666e-9)
    assert calc.wavelength == pytest.approx(666e-9)
    assert calc.wavelength_nm == pytest.approx(666.0)


def test_calculator_drymass_scale():
    calc = DryMassCalculator(pixel_size=1e-7, alpha=2.0e-4)
    assert calc.drymass_scale == pytest.approx(1e-7**2 * 1e6 / 2.0e-4)
    # the scale times the summed OPD (nm) reproduces the dry mass
    opd = np.full((10, 10), 50.0, dtype=np.float32)
    assert calc.drymass_scale * float(opd.sum()) == pytest.approx(
        calc.calc_from_opd(opd)
    )


def test_calculator_respects_mask():
    opd = np.full((2, 2), 50.0, dtype=np.float32)
    mask = np.array([[True, False], [False, False]])
    calc = DryMassCalculator(pixel_size=1e-7, alpha=2.0e-4)
    assert calc.calc_from_opd(opd, mask=mask) == pytest.approx(
        calc.calc_from_opd(opd) / 4
    )


def test_calculator_rejects_nonpositive_pixel_size():
    with pytest.raises(ValueError, match="pixel_size must be positive"):
        DryMassCalculator(pixel_size=0.0)


def test_calculator_rejects_nonpositive_alpha():
    with pytest.raises(ValueError, match="alpha must be positive"):
        DryMassCalculator(pixel_size=1e-7, alpha=0.0)
