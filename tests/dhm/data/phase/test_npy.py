from __future__ import annotations

import numpy as np
import pytest

from iivs.dhm.data.common import FrameShapedMixin
from iivs.dhm.data.phase.base import PhaseFloatSequence
from iivs.dhm.data.phase.bin import PhaseBinHeader
from iivs.dhm.data.phase.core import PhaseUnit
from iivs.dhm.data.phase.npy import PhaseNpyFolder


def _write(root, index, value=0.0, shape=(2, 3)):
    np.save(root / f"{index:05d}_phase.npy", np.full(shape, value, dtype=np.float32))


def test_lists_in_order_with_synthesized_header(tmp_path):
    for i in range(3):
        _write(tmp_path, i, i)
    f = PhaseNpyFolder(
        tmp_path, pixel_size=2e-6, unit=PhaseUnit.RADIANS, height_scale=3e-7
    )
    assert isinstance(f, PhaseFloatSequence)
    assert isinstance(f, FrameShapedMixin)
    assert len(f) == 3
    assert f.frame_shape == (2, 3)
    assert isinstance(f.header, PhaseBinHeader)
    assert f.header.pixel_size == pytest.approx(2e-6)
    assert f.header.height_scale == pytest.approx(3e-7)
    assert f.header.unit is PhaseUnit.RADIANS
    for i in range(3):
        np.testing.assert_array_equal(f[i], np.full((2, 3), float(i)))
    assert f[0].dtype == np.float32


def test_wavelength_form_derives_height_scale(tmp_path):
    _write(tmp_path, 0)
    f = PhaseNpyFolder(
        tmp_path,
        pixel_size=1e-6,
        unit=PhaseUnit.RADIANS,
        wavelength=666e-9,
        refractive_delta=0.5,
    )
    assert f.header.height_scale == pytest.approx(666e-9 / (2 * np.pi * 0.5))


def test_rejects_both_scale_forms(tmp_path):
    _write(tmp_path, 0)
    with pytest.raises(ValueError, match="height_scale, or wavelength"):
        PhaseNpyFolder(
            tmp_path,
            pixel_size=1e-6,
            unit=PhaseUnit.RADIANS,
            height_scale=2e-7,
            wavelength=666e-9,
            refractive_delta=0.5,
        )


def test_rejects_neither_scale_form(tmp_path):
    _write(tmp_path, 0)
    with pytest.raises(ValueError, match="height_scale, or wavelength"):
        PhaseNpyFolder(tmp_path, pixel_size=1e-6, unit=PhaseUnit.RADIANS)


def test_target_unit_converts(tmp_path):
    np.save(tmp_path / "00000_phase.npy", np.full((2, 2), 1.0, dtype=np.float32))
    f = PhaseNpyFolder(
        tmp_path,
        pixel_size=1e-6,
        unit=PhaseUnit.RADIANS,
        height_scale=2e-7,
        target_unit=PhaseUnit.METERS,
    )
    assert f.target_unit is PhaseUnit.METERS
    np.testing.assert_allclose(f[0], np.full((2, 2), 2e-7), atol=1e-12)


def test_validate_rejects_shape_mismatch(tmp_path):
    _write(tmp_path, 0, shape=(2, 3))
    _write(tmp_path, 1, shape=(4, 5))
    with pytest.raises(ValueError, match="header"):
        PhaseNpyFolder(
            tmp_path, pixel_size=1e-6, unit=PhaseUnit.RADIANS, height_scale=2e-7
        )


def test_validate_data_rejects_non_float32(tmp_path):
    np.save(tmp_path / "00000_phase.npy", np.zeros((2, 3), dtype=np.float64))
    with pytest.raises(ValueError, match="float32"):
        PhaseNpyFolder(
            tmp_path,
            pixel_size=1e-6,
            unit=PhaseUnit.RADIANS,
            height_scale=2e-7,
            validate="data",
        )


def test_validate_data_detects_non_finite(tmp_path):
    np.save(tmp_path / "00000_phase.npy", np.array([[np.nan, 1.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        PhaseNpyFolder(
            tmp_path,
            pixel_size=1e-6,
            unit=PhaseUnit.RADIANS,
            height_scale=2e-7,
            validate="data",
        )


def test_rejects_pickled_object_array(tmp_path):
    np.save(
        tmp_path / "00000_phase.npy",
        np.array([{"x": 1}], dtype=object),
        allow_pickle=True,
    )
    with pytest.raises(ValueError, match="memory-mapped"):
        PhaseNpyFolder(
            tmp_path, pixel_size=1e-6, unit=PhaseUnit.RADIANS, height_scale=2e-7
        )
