from __future__ import annotations

import numpy as np
import pytest

from iivs.dhm.analysis.drymass import (
    DryMassCalculator,
    calc_drymass,
    calc_drymass_from_phase,
)
from iivs.dhm.analysis.opd import phase_to_opd
from iivs.dhm.analysis.volume import OpticalVolumeCalculator, calc_volume
from iivs.dhm.constants import (
    DEFAULT_REFRACTIVE_DELTA,
    DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    DEFAULT_WAVELENGTH,
    PIXEL_SIZE_20X,
)


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


def test_calc_drymass_batched():
    # (N, H, W) batch -> one mass per image, shape (N,).
    opd = np.stack(
        [np.full((2, 2), 10.0, np.float32), np.full((2, 2), 20.0, np.float32)]
    )
    out = calc_drymass(opd, pixel_size=1e-7, alpha=2.0e-4)
    assert out.shape == (2,)
    single = calc_drymass(opd[0], pixel_size=1e-7, alpha=2.0e-4)
    assert out[0] == pytest.approx(float(single))
    assert out[1] == pytest.approx(2 * float(single))


def test_calc_drymass_channel_mask():
    # (N, H, W) mask -> a trailing axis, shape (..., N).
    opd = np.full((2, 2), 50.0, np.float32)
    masks = np.array(
        [
            [[True, False], [False, False]],  # 1 pixel
            [[True, True], [False, False]],  # 2 pixels
        ]
    )  # (N=2, H=2, W=2)
    out = calc_drymass(opd, pixel_size=1e-7, alpha=2.0e-4, mask=masks)
    assert out.shape == (2,)
    whole = float(calc_drymass(opd, pixel_size=1e-7, alpha=2.0e-4))
    assert out[0] == pytest.approx(whole / 4)
    assert out[1] == pytest.approx(whole / 2)


def test_calc_drymass_reduce_false_returns_map():
    opd = np.full((2, 2), 50.0, np.float32)
    density = calc_drymass(opd, pixel_size=1e-7, alpha=2.0e-4, reduce=False)
    assert density.shape == (2, 2)  # the per-pixel map, not summed
    assert density.sum() == pytest.approx(
        float(calc_drymass(opd, pixel_size=1e-7, alpha=2.0e-4))
    )


def test_calc_drymass_reduce_false_with_mask():
    # reduce=False + mask: per-pixel density with the mask applied (0 elsewhere).
    opd = np.full((2, 2), 50.0, np.float32)
    mask = np.array([[True, False], [False, False]])
    density = calc_drymass(opd, pixel_size=1e-7, alpha=2.0e-4, mask=mask, reduce=False)
    assert density.shape == (2, 2)
    assert density[0, 0] == pytest.approx(
        float(calc_drymass(opd, pixel_size=1e-7, alpha=2.0e-4, mask=mask))
    )
    assert density.sum() == pytest.approx(density[0, 0])  # only the masked pixel


def test_output_is_float32():
    # Accumulate in float64 internally, but every path returns float32.
    opd = np.full((4, 4), 50.0, dtype=np.float32)
    mask2d = np.ones((4, 4), dtype=bool)
    mask3d = np.ones((3, 4, 4), dtype=bool)
    assert calc_drymass(opd, pixel_size=1e-7).dtype == np.float32  # sum, no mask
    assert calc_drymass(opd, pixel_size=1e-7, mask=mask2d).dtype == np.float32
    assert (
        calc_drymass(opd, pixel_size=1e-7, mask=mask3d).dtype == np.float32
    )  # (N,...)
    assert calc_drymass(opd, pixel_size=1e-7, reduce=False).dtype == np.float32  # map


def test_rejects_bad_shapes():
    dmc = DryMassCalculator.from_args(
        pixel_size=1e-7, wavelength=666e-9, refractive_delta=0.5, alpha=2.0e-4
    )
    opd = np.zeros((3, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="at least 2D"):  # opd needs (H, W)
        dmc.calc_from_opd(np.zeros(4, dtype=np.float32))
    with pytest.raises(ValueError, match="mask must be"):  # (T, N, H, W) mask
        dmc.calc_from_opd(opd, mask=np.ones((3, 2, 4, 4), dtype=bool))
    with pytest.raises(ValueError, match=r"\(H, W\) must be"):  # (H, W) mismatch
        dmc.calc_from_opd(opd, mask=np.ones((4, 5), dtype=bool))


def test_calc_drymass_label_mask():
    # an integer label image -> one dry mass per positive label (0 = background)
    opd = np.full((2, 2), 50.0, np.float32)
    labels = np.array([[1, 1], [2, 0]])  # label 1: 2 px, label 2: 1 px
    out = calc_drymass(opd, pixel_size=1e-7, alpha=2.0e-4, mask=labels)
    assert out.shape == (2,)  # kept, unlike a 2D boolean mask
    whole = float(calc_drymass(opd, pixel_size=1e-7, alpha=2.0e-4))
    assert out[0] == pytest.approx(whole / 2)  # 2 of 4 pixels
    assert out[1] == pytest.approx(whole / 4)  # 1 of 4 pixels


def test_calc_drymass_empty_region_is_zero():
    # a label gap (label 1 absent) -> empty region -> 0 pg (matches Sum(empty=0.0))
    opd = np.full((2, 2), 50.0, np.float32)
    labels = np.array([[0, 2], [2, 0]])  # label 1 missing
    out = calc_drymass(opd, pixel_size=1e-7, alpha=2.0e-4, mask=labels)
    assert out[0] == 0.0  # empty region
    assert out[1] > 0


def test_calc_drymass_label_mask_density():
    # reduce=False + labels -> a per-region density stack (..., R, H, W)
    opd = np.full((2, 2), 50.0, np.float32)
    labels = np.array([[1, 1], [2, 0]])
    density = calc_drymass(
        opd, pixel_size=1e-7, alpha=2.0e-4, mask=labels, reduce=False
    )
    assert density.shape == (2, 2, 2)
    per_label = calc_drymass(opd, pixel_size=1e-7, alpha=2.0e-4, mask=labels)
    assert density[0].sum() == pytest.approx(float(per_label[0]))
    assert density[1].sum() == pytest.approx(float(per_label[1]))


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


def test_calculator_calc_from_opd():
    # same 0.25 pg setup as the free-function anchor
    opd = np.full((10, 10), 50.0, dtype=np.float32)
    calc = DryMassCalculator.from_args(
        pixel_size=1e-7, wavelength=666e-9, refractive_delta=0.5, alpha=2.0e-4
    )
    assert calc.calc_from_opd(opd) == pytest.approx(0.25)


def test_calculator_from_wavelength_integrates_phase():
    # 25 px, 1 rad -> OPD 666/(2pi) = 105.9957 nm each; drymass_scale 5e-5 pg/nm:
    # 25 * 105.9957 * 5e-5 = 0.13249 pg.
    phase = np.full((5, 5), 1.0, dtype=np.float32)
    calc = DryMassCalculator.from_args(
        pixel_size=1e-7, wavelength=666e-9, refractive_delta=0.5, alpha=2.0e-4
    )
    assert calc.calc_from_phase(phase) == pytest.approx(0.13249, rel=1e-3)


def test_calculator_accepts_injected_volume_engine():
    phase = np.full((3, 3), 1.0, dtype=np.float32)
    volume = OpticalVolumeCalculator.from_args(
        pixel_size=1e-7, wavelength=666e-9, refractive_delta=0.5
    )
    calc = DryMassCalculator(volume_converter=volume, alpha=2.0e-4)
    assert calc.volume_converter is volume
    expected = DryMassCalculator.from_args(
        pixel_size=1e-7, wavelength=666e-9, refractive_delta=0.5, alpha=2.0e-4
    ).calc_from_phase(phase)
    assert calc.calc_from_phase(phase) == pytest.approx(expected)


def test_calculator_wavelength_shortcuts():
    calc = DryMassCalculator.from_args(
        pixel_size=1e-7, wavelength=666e-9, refractive_delta=0.5, alpha=2.0e-4
    )
    assert calc.wavelength == pytest.approx(666e-9)
    assert calc.wavelength_nm == pytest.approx(666.0)


def test_calculator_drymass_scale():
    # 0.1 um pixel, alpha 2e-4 m^3/kg: px_area 1e-14 m^2; pg per summed-nm OPD =
    # 1e-14 m^2 * (1e-9 m/nm) * (1e15 pg/kg) / 2e-4 = 5e-5 pg/nm (hand-derived).
    calc = DryMassCalculator.from_args(
        pixel_size=1e-7, wavelength=666e-9, refractive_delta=0.5, alpha=2.0e-4
    )
    assert calc.drymass_scale == pytest.approx(5e-5)
    # the scale times the summed OPD (nm) reproduces the dry mass
    opd = np.full((10, 10), 50.0, dtype=np.float32)
    assert calc.drymass_scale * float(opd.sum()) == pytest.approx(
        calc.calc_from_opd(opd)
    )


def test_calculator_respects_mask():
    opd = np.full((2, 2), 50.0, dtype=np.float32)
    mask = np.array([[True, False], [False, False]])
    calc = DryMassCalculator.from_args(
        pixel_size=1e-7, wavelength=666e-9, refractive_delta=0.5, alpha=2.0e-4
    )
    assert calc.calc_from_opd(opd, mask=mask) == pytest.approx(
        calc.calc_from_opd(opd) / 4
    )


def test_calculator_rejects_nonpositive_pixel_size():
    with pytest.raises(ValueError, match="pixel_size must be positive"):
        DryMassCalculator.from_args(
            pixel_size=0.0, wavelength=666e-9, refractive_delta=0.5, alpha=2.0e-4
        )


def test_calculator_rejects_nonpositive_alpha():
    with pytest.raises(ValueError, match="alpha must be positive"):
        DryMassCalculator.from_args(
            pixel_size=1e-7, wavelength=666e-9, refractive_delta=0.5, alpha=0.0
        )


def test_default_construction_uses_lab_constants():
    # every engine in the chain falls back to the lab defaults (20X pixel size)
    calc = DryMassCalculator()
    assert calc.pixel_size == pytest.approx(PIXEL_SIZE_20X)
    assert calc.wavelength == pytest.approx(DEFAULT_WAVELENGTH)
    assert calc.refractive_delta == pytest.approx(DEFAULT_REFRACTIVE_DELTA)
    expected = DryMassCalculator.from_args(
        pixel_size=PIXEL_SIZE_20X,
        wavelength=DEFAULT_WAVELENGTH,
        refractive_delta=DEFAULT_REFRACTIVE_DELTA,
        alpha=DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    )
    assert calc.drymass_scale == pytest.approx(expected.drymass_scale)


def test_calc_from_height_matches_calc_from_opd():
    # height = opd / delta, so converting back must reproduce the OPD mass.
    opd = np.full((4, 4), 50.0, dtype=np.float32)
    calc = DryMassCalculator.from_args(
        pixel_size=1e-7, wavelength=666e-9, refractive_delta=0.5, alpha=2.0e-4
    )
    height = (opd / np.float32(0.5)).astype(np.float32)
    assert calc.calc_from_height(height) == pytest.approx(
        calc.calc_from_opd(opd), rel=1e-6
    )

    with pytest.raises(ValueError, match="height must be at least 2D"):
        calc.calc_from_height(np.zeros(4, dtype=np.float32))


def test_calculator_pixel_size_um():
    assert DryMassCalculator.from_args(
        pixel_size=1e-7, wavelength=666e-9, refractive_delta=0.5, alpha=2.0e-4
    ).pixel_size_um == pytest.approx(0.1)


def test_calc_from_volume_closes_the_chain():
    # The same physics through both routes: 50 nm OPD over 100 px of 0.1 um pitch
    # is 0.25 pg directly (the anchor above) and, at delta 0.5, 0.1 um^3 of volume;
    # mass = volume * delta / alpha * 1e-3 = 0.1 * 0.5 / 2e-4 * 1e-3 = 0.25 pg.
    opd = np.full((10, 10), 50.0, dtype=np.float32)
    calc = DryMassCalculator.from_args(
        pixel_size=1e-7, wavelength=666e-9, refractive_delta=0.5, alpha=2.0e-4
    )
    direct = calc.calc_from_opd(opd)
    volume = calc_volume(opd, pixel_size=1e-7, refractive_delta=0.5)
    via_volume = calc.calc_from_volume(volume, refractive_delta=0.5)
    assert via_volume == pytest.approx(direct, rel=1e-5)
    assert via_volume == pytest.approx(0.25, rel=1e-5)

    # None (the default) falls back to the bound volume engine's delta (0.5 here)
    assert calc.calc_from_volume(volume) == pytest.approx(via_volume)
    assert calc.refractive_delta == pytest.approx(0.5)


def test_calc_from_volume_rejects_nonpositive_delta():
    calc = DryMassCalculator.from_args(
        pixel_size=1e-7, wavelength=666e-9, refractive_delta=0.5, alpha=2.0e-4
    )
    with pytest.raises(ValueError, match="refractive_delta must be positive"):
        calc.calc_from_volume(np.float32(0.1), refractive_delta=0.0)
