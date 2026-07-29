from __future__ import annotations

import numpy as np
import pytest

from iivs.dhm.analysis.height import phase_to_height as np_phase_to_height

torch = pytest.importorskip("torch")

from iivs.dhm.analysis.pytorch.height import (  # noqa: E402
    OpticalHeight,
    height_to_opd,
    height_to_phase,
    opd_to_height,
    phase_to_height,
)
from iivs.dhm.analysis.pytorch.opd import OpticalPathDifference  # noqa: E402


def test_forward_converts_phase_to_height():
    # height = phase * lambda/(2pi*delta): 1 rad at 666 nm, delta 0.5 -> 211.99 nm.
    module = OpticalHeight.from_args(wavelength=666e-9, refractive_delta=0.5)
    assert module(torch.tensor(1.0)).item() == pytest.approx(211.99, rel=1e-4)


def test_owns_an_injected_opd_submodule():
    opd_converter = OpticalPathDifference(wavelength=532e-9)
    module = OpticalHeight(refractive_delta=0.4, opd_converter=opd_converter)
    assert module.opd_converter is opd_converter
    assert module.wavelength == pytest.approx(532e-9)
    assert module.wavelength_nm == pytest.approx(532.0)
    # registered as a proper child, per the nn.Module composition contract
    assert opd_converter in module.children()


def test_opd_entry_divides_by_delta():
    # OPD enters through phase; the wavelength cancels, leaving opd / delta.
    module = OpticalHeight(refractive_delta=0.5)
    assert module.convert_from_opd(torch.tensor(100.0)).item() == pytest.approx(200.0)


def test_height_to_phase_is_the_phase_inverse():
    phase = torch.rand(3, 3)
    back = height_to_phase(
        phase_to_height(phase, wavelength=666e-9, refractive_delta=0.5),
        wavelength=666e-9,
        refractive_delta=0.5,
    )
    np.testing.assert_allclose(back.numpy(), phase.numpy(), rtol=1e-5)


def test_scales_shared_with_numpy_engine():
    module = OpticalHeight.from_args(wavelength=666e-9, refractive_delta=0.5)
    assert module.refractive_delta == pytest.approx(0.5)
    assert module.wavelength_nm == pytest.approx(666.0)
    phase = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float32)
    got = phase_to_height(phase, wavelength=666e-9, refractive_delta=0.5)
    expected = np_phase_to_height(
        phase.numpy(), wavelength=666e-9, refractive_delta=0.5
    )
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-6)


def test_roundtrip():
    height = torch.rand(4, 4)
    back = opd_to_height(
        height_to_opd(height, refractive_delta=0.4), refractive_delta=0.4
    )
    np.testing.assert_allclose(back.numpy(), height.numpy(), rtol=1e-5)


def test_preserves_grad_and_device():
    opd = torch.ones(2, 2, requires_grad=True)
    height = opd_to_height(opd, refractive_delta=0.5)
    assert height.requires_grad
    assert height.device == opd.device
    height.sum().backward()
    # d(height)/d(opd) = 1 / delta for the linear map
    assert torch.allclose(opd.grad, torch.full_like(opd, 2.0))
