from __future__ import annotations

import numpy as np
import pytest

from iivs.dhm.analysis.volume import calc_optical_volume as np_calc_optical_volume
from iivs.dhm.analysis.volume import calc_optical_volume_from_height as np_from_height
from iivs.dhm.analysis.volume import calc_optical_volume_from_opd as np_from_opd

torch = pytest.importorskip("torch")

from iivs.dhm.analysis.pytorch.area import ProjectedArea  # noqa: E402
from iivs.dhm.analysis.pytorch.height import OpticalHeight  # noqa: E402
from iivs.dhm.analysis.pytorch.volume import (  # noqa: E402
    OpticalVolume,
    calc_optical_volume,
    calc_optical_volume_from_height,
    calc_optical_volume_from_opd,
)


def test_forward_is_the_pointwise_phase_density():
    # phase * volume_scale, volume_scale = area_scale * height_scale * 1e-3.
    module = OpticalVolume.from_args(
        pixel_size=1e-7, wavelength=666e-9, refractive_delta=0.5
    )
    height_scale = (666.0 / (2 * np.pi)) / 0.5  # nm/rad, hand-derived
    expected_scale = 0.01 * height_scale * 1e-3  # area_scale 0.01 um^2
    assert module.volume_scale == pytest.approx(expected_scale)
    assert module(torch.tensor(0.5)).item() == pytest.approx(0.5 * expected_scale)


def test_owns_area_and_height_submodules():
    module = OpticalVolume.from_args(
        pixel_size=2e-7, wavelength=532e-9, refractive_delta=0.4
    )
    children = dict(module.named_children())
    assert isinstance(children["area_calculator"], ProjectedArea)
    assert isinstance(children["height_converter"], OpticalHeight)
    # the scale derives from the two submodules, not a copied constant
    assert module.volume_scale == pytest.approx(
        module.area_calculator.area_scale * module.height_converter.height_scale * 1e-3
    )
    # surfaced parameters delegate to the submodules
    assert module.pixel_size == pytest.approx(2e-7)
    assert module.pixel_size_um == pytest.approx(0.2)
    assert module.refractive_delta == pytest.approx(0.4)
    assert module.wavelength == pytest.approx(532e-9)
    assert module.wavelength_nm == pytest.approx(532.0)


def test_construction_binds_the_given_submodules():
    area = ProjectedArea(pixel_size=2e-7)
    height = OpticalHeight.from_args(wavelength=532e-9, refractive_delta=0.4)
    module = OpticalVolume(area_calculator=area, height_converter=height)
    assert module.area_calculator is area
    assert module.height_converter is height
    assert module.volume_scale == pytest.approx(
        area.area_scale * height.height_scale * 1e-3
    )


def test_convert_from_opd_enters_through_phase():
    # opd -> phase -> volume density; the wavelength cancels, so the opd volume is
    # wavelength-free: two modules at different wavelengths must agree.
    opd = torch.linspace(0, 80, 16).reshape(4, 4)
    a = OpticalVolume.from_args(
        pixel_size=1e-7, wavelength=666e-9, refractive_delta=0.5
    )
    b = OpticalVolume.from_args(
        pixel_size=1e-7, wavelength=532e-9, refractive_delta=0.5
    )
    torch.testing.assert_close(a.convert_from_opd(opd), b.convert_from_opd(opd))


def test_convert_from_height_matches_phase_path():
    module = OpticalVolume.from_args(
        pixel_size=1e-7, wavelength=666e-9, refractive_delta=0.5
    )
    phase = torch.rand(3, 3)
    height = module.height_converter.convert_from_phase(phase)
    torch.testing.assert_close(module.convert_from_height(height), module(phase))


def test_calc_optical_volume_matches_numpy():
    phase = torch.rand(3, 3, dtype=torch.float32)
    got = calc_optical_volume(
        phase, pixel_size=1e-7, wavelength=666e-9, refractive_delta=0.5
    )
    expected = np_calc_optical_volume(
        phase.numpy(), pixel_size=1e-7, wavelength=666e-9, refractive_delta=0.5
    )
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-5)


def test_calc_optical_volume_from_opd_matches_numpy():
    opd = torch.rand(4, 4, dtype=torch.float32) * 100
    mask = torch.zeros(4, 4, dtype=torch.bool)
    mask[:2, :3] = True
    got = calc_optical_volume_from_opd(
        opd, pixel_size=1e-7, refractive_delta=0.5, mask=mask
    )
    expected = np_from_opd(
        opd.numpy(), pixel_size=1e-7, refractive_delta=0.5, mask=mask.numpy()
    )
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-5)


def test_calc_optical_volume_from_height_matches_numpy():
    height = torch.rand(3, 3, dtype=torch.float32) * 200
    got = calc_optical_volume_from_height(height, pixel_size=1e-7, refractive_delta=0.5)
    expected = np_from_height(height.numpy(), pixel_size=1e-7, refractive_delta=0.5)
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-5)


def test_calc_optical_volume_from_opd_reduce_false_returns_map():
    opd = torch.full((2, 2), 50.0)
    density = calc_optical_volume_from_opd(opd, pixel_size=1e-7, reduce=False)
    assert density.shape == (2, 2)
    assert density.sum().item() == pytest.approx(
        calc_optical_volume_from_opd(opd, pixel_size=1e-7).item()
    )


def test_preserves_grad_and_device():
    phase = torch.ones(2, 2, requires_grad=True)
    volume = calc_optical_volume(phase, pixel_size=1e-7)
    assert volume.requires_grad
    assert volume.device == phase.device
    volume.backward()
    assert phase.grad is not None
    assert torch.all(phase.grad > 0)  # every pixel contributes positively
