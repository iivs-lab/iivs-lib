from __future__ import annotations

import numpy as np
import pytest

from iivs.dhm.analysis.area import ProjectedAreaCalculator, calc_projected_area
from iivs.dhm.analysis.height import OpticalHeightConverter, phase_to_height
from iivs.dhm.analysis.volume import (
    OpticalVolumeCalculator,
    calc_volume,
    calc_volume_from_height,
    calc_volume_from_opd,
)


def test_calc_volume_uniform_region():
    # 50 nm OPD over 100 px of 0.1 um pitch at delta 0.5: height 100 nm = 0.1 um;
    # per px = 0.1 um * (0.1 um)^2 = 1e-3 um^3; x100 = 0.1 um^3.
    opd = np.full((10, 10), 50.0, dtype=np.float32)  # nm
    volume = calc_volume_from_opd(opd, pixel_size=1e-7, refractive_delta=0.5)
    assert volume == pytest.approx(0.1)
    assert volume.dtype == np.float32


def test_calc_volume_respects_mask():
    opd = np.full((2, 2), 50.0, dtype=np.float32)  # 4 equal pixels, nm
    mask = np.array([[True, False], [False, False]])  # select 1
    whole = calc_volume_from_opd(opd, pixel_size=1e-7)
    one = calc_volume_from_opd(opd, pixel_size=1e-7, mask=mask)
    assert one == pytest.approx(whole / 4)


def test_calc_volume_batched():
    # (N, H, W) batch -> one volume per image, shape (N,).
    opd = np.stack(
        [np.full((2, 2), 10.0, np.float32), np.full((2, 2), 20.0, np.float32)]
    )
    out = calc_volume_from_opd(opd, pixel_size=1e-7)
    assert out.shape == (2,)
    single = calc_volume_from_opd(opd[0], pixel_size=1e-7)
    assert out[0] == pytest.approx(float(single))
    assert out[1] == pytest.approx(2 * float(single))


def test_calc_volume_label_mask_and_empty_region():
    opd = np.full((2, 2), 50.0, np.float32)
    labels = np.array([[0, 2], [2, 0]])  # label 1 missing, label 2: 2 px
    out = calc_volume_from_opd(opd, pixel_size=1e-7, mask=labels)
    assert out.shape == (2,)
    assert out[0] == 0.0  # empty region -> 0 volume
    assert out[1] == pytest.approx(
        float(calc_volume_from_opd(opd, pixel_size=1e-7)) / 2
    )


def test_calc_volume_reduce_false_returns_map():
    opd = np.full((2, 2), 50.0, np.float32)
    density = calc_volume_from_opd(opd, pixel_size=1e-7, reduce=False)
    assert density.shape == (2, 2)  # the per-pixel map, not summed
    assert density.sum() == pytest.approx(
        float(calc_volume_from_opd(opd, pixel_size=1e-7))
    )


def test_volume_is_area_times_mean_height():
    # The ProjectedArea relation: volume == projected_area * mean(height).
    rng = np.random.default_rng(0)
    height = rng.uniform(0.0, 500.0, size=(4, 4)).astype(np.float32)  # nm
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :3] = True

    calculator = OpticalVolumeCalculator.from_args(
        pixel_size=2e-7, wavelength=666e-9, refractive_delta=0.5
    )
    volume = calculator.calc_from_height(height, mask=mask)

    area = calc_projected_area(height, pixel_size=2e-7, mask=mask)  # um^2
    mean_height_um = float(height[mask].mean()) * 1e-3  # nm -> um
    assert volume == pytest.approx(float(area) * mean_height_um, rel=1e-5)


def test_calc_from_height_matches_calc_from_opd():
    # height = opd / delta, so the two entry points must agree on the same map.
    opd = np.linspace(0.0, 80.0, 16, dtype=np.float32).reshape(4, 4)
    calculator = OpticalVolumeCalculator.from_args(
        pixel_size=1e-7, wavelength=666e-9, refractive_delta=0.4
    )
    height = calculator.height_converter.convert_from_opd(opd)
    assert calculator.calc_from_height(height) == pytest.approx(
        calculator.calc_from_opd(opd), rel=1e-5
    )


def test_calc_volume_from_height_one_shot():
    # height * area_scale * 1e-3 per px, wavelength-free: 100 nm * 0.01 um^2 * 1e-3
    # = 1e-3 um^3 per px; 16 px -> 0.016 um^3 (hand-derived).
    height = np.full((4, 4), 100.0, dtype=np.float32)
    got = calc_volume_from_height(height, pixel_size=1e-7, refractive_delta=0.5)
    assert got == pytest.approx(0.016)


def test_calc_volume_from_phase_matches_two_step():
    phase = np.full((5, 5), 1.0, dtype=np.float32)
    direct = calc_volume(
        phase, pixel_size=1e-7, wavelength=666e-9, refractive_delta=0.5
    )
    height = phase_to_height(phase, wavelength=666e-9, refractive_delta=0.5)
    two_step = OpticalVolumeCalculator.from_args(
        pixel_size=1e-7, wavelength=666e-9, refractive_delta=0.5
    ).calc_from_height(height)
    assert direct == pytest.approx(two_step)


def test_calculator_volume_scale():
    # 0.1 um pixel at delta 0.5, 666 nm: px_area 0.01 um^2 * height_scale (nm/rad)
    # * (1e-3 um/nm) um^3 per summed-rad phase. height_scale = (666/2pi)/0.5.
    calculator = OpticalVolumeCalculator.from_args(
        pixel_size=1e-7, wavelength=666e-9, refractive_delta=0.5
    )
    height_scale = (666.0 / (2 * np.pi)) / 0.5  # nm of height per rad (hand-derived)
    assert calculator.volume_scale == pytest.approx(0.01 * height_scale * 1e-3)
    # the scale times the summed phase (rad) reproduces the volume
    phase = np.full((10, 10), 0.5, dtype=np.float32)
    assert calculator.volume_scale * float(phase.sum()) == pytest.approx(
        calculator.calc_from_phase(phase)
    )


def test_calculator_surfaces_bound_parameters():
    converter = OpticalHeightConverter.from_args(
        wavelength=532e-9, refractive_delta=0.4
    )
    calculator = OpticalVolumeCalculator(
        area_calculator=ProjectedAreaCalculator(pixel_size=2e-7),
        height_converter=converter,
    )
    assert calculator.height_converter is converter
    assert calculator.pixel_size_um == pytest.approx(0.2)
    assert calculator.refractive_delta == pytest.approx(0.4)
    assert calculator.wavelength == pytest.approx(532e-9)
    assert calculator.wavelength_nm == pytest.approx(532.0)


def test_calculator_holds_both_sides_of_the_volume_relation():
    # volume = area * mean(height): the calculator composes a footprint engine
    # and a height converter, and its scale is their product.
    calculator = OpticalVolumeCalculator.from_args(
        pixel_size=2e-7, wavelength=666e-9, refractive_delta=0.4
    )
    area = calculator.area_calculator
    height = calculator.height_converter
    assert isinstance(area, ProjectedAreaCalculator)
    assert area.pixel_size == pytest.approx(calculator.pixel_size)
    # the volume scale is the product of the two engines' factors (per rad phase)
    assert calculator.volume_scale == pytest.approx(
        area.area_scale * height.height_scale * 1e-3
    )


def test_calculator_rejects_bad_inputs():
    with pytest.raises(ValueError, match="pixel_size must be positive"):
        OpticalVolumeCalculator.from_args(
            pixel_size=0.0, wavelength=666e-9, refractive_delta=0.5
        )

    calculator = OpticalVolumeCalculator.from_args(
        pixel_size=1e-7, wavelength=666e-9, refractive_delta=0.5
    )
    with pytest.raises(ValueError, match="phase must be at least 2D"):
        calculator.calc_from_phase(np.zeros(4, dtype=np.float32))
    with pytest.raises(ValueError, match="opd must be at least 2D"):
        calculator.calc_from_opd(np.zeros(4, dtype=np.float32))
    with pytest.raises(ValueError, match="height must be at least 2D"):
        calculator.calc_from_height(np.zeros(4, dtype=np.float32))
