from __future__ import annotations

import numpy as np
import pytest

from iivs.dhm.data.constants import (
    DEFAULT_REFRACTIVE_DELTA,
    DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    DEFAULT_WAVELENGTH,
    DEFAULT_WAVELENGTH_NM,
    PIXEL_SIZE_20X,
    PIXEL_SIZE_20X_UM,
)
from iivs.dhm.data.phase import read_phase_bin_header, save_phase_bin


def test_default_optical_parameters():
    assert DEFAULT_WAVELENGTH == 666e-9  # m
    assert DEFAULT_WAVELENGTH_NM == 666.0  # nm
    assert DEFAULT_REFRACTIVE_DELTA == 0.5
    assert DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT == 2.0e-4  # m^3/kg
    # the m and nm forms describe the same wavelength
    assert pytest.approx(DEFAULT_WAVELENGTH * 1e9) == DEFAULT_WAVELENGTH_NM


def test_20x_pixel_size():
    # 20X pixel size as recorded by Koala (~285 nm).
    assert PIXEL_SIZE_20X == 2.84871e-7  # m
    # the m and um forms describe the same pixel size
    assert pytest.approx(PIXEL_SIZE_20X * 1e6) == PIXEL_SIZE_20X_UM


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
