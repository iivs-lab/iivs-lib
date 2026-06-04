from __future__ import annotations

import numpy as np
import pytest

from iivs.dhm.data.constants import (
    DEFAULT_REFRACTIVE_DELTA,
    DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    DEFAULT_WAVELENGTH,
    DEFAULT_WAVELENGTH_NM,
    PIXEL_SIZE_10X,
    PIXEL_SIZE_10X_UM,
    PIXEL_SIZE_20X,
    PIXEL_SIZE_20X_UM,
    PIXEL_SIZE_40X,
    PIXEL_SIZE_40X_UM,
)
from iivs.dhm.data.phase import read_phase_bin_header, save_phase_bin


def test_default_optical_parameters():
    assert DEFAULT_WAVELENGTH == 666e-9  # m
    assert DEFAULT_WAVELENGTH_NM == 666.0  # nm
    assert DEFAULT_REFRACTIVE_DELTA == 0.5
    assert DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT == 2.0e-4  # m^3/kg
    # the m and nm forms describe the same wavelength
    assert pytest.approx(DEFAULT_WAVELENGTH * 1e9) == DEFAULT_WAVELENGTH_NM


@pytest.mark.parametrize(
    ("pixel_m", "pixel_um"),
    (
        (PIXEL_SIZE_10X, PIXEL_SIZE_10X_UM),
        (PIXEL_SIZE_20X, PIXEL_SIZE_20X_UM),
        (PIXEL_SIZE_40X, PIXEL_SIZE_40X_UM),
    ),
)
def test_pixel_size_m_and_um_agree(pixel_m, pixel_um):
    # the m and um forms describe the same measured pixel size
    assert pytest.approx(pixel_m * 1e6) == pixel_um


def test_pixel_size_scales_with_magnification():
    # higher magnification -> smaller pixel (10X > 20X > 40X)
    assert PIXEL_SIZE_10X > PIXEL_SIZE_20X > PIXEL_SIZE_40X
    # measured values, ~144 / 285 / 580 nm
    assert PIXEL_SIZE_20X == 2.84871392e-7  # m


def test_defaults_drive_a_valid_height_scale(tmp_path):
    # The defaults are meant to feed save_phase_bin's wavelength/refractive_delta.
    path = tmp_path / "phase.bin"
    save_phase_bin(
        path,
        np.zeros((2, 2), dtype=np.float32),
        pixel_size=1e-6,
        wavelength=DEFAULT_WAVELENGTH,
        refractive_delta=DEFAULT_REFRACTIVE_DELTA,
    )
    expected = DEFAULT_WAVELENGTH / (2 * np.pi * DEFAULT_REFRACTIVE_DELTA)
    assert read_phase_bin_header(path).height_scale == np.float32(expected)
