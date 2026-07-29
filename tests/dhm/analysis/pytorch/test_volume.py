from __future__ import annotations

import numpy as np
import pytest

from iivs.dhm.analysis.volume import calc_volume as np_calc_volume
from iivs.dhm.analysis.volume import calc_volume_from_phase as np_from_phase

torch = pytest.importorskip("torch")

from iivs.dhm.analysis.pytorch.volume import (  # noqa: E402
    OpticalVolume,
    calc_volume,
    calc_volume_from_phase,
)


def test_forward_is_the_pointwise_density():
    # 50 nm OPD, 0.1 um pitch, delta 0.5: 1e-3 um^3 per pixel (the NumPy anchor).
    module = OpticalVolume(pixel_size=1e-7, refractive_delta=0.5)
    assert module(torch.tensor(50.0)).item() == pytest.approx(1e-3)


def test_calc_volume_matches_numpy():
    opd = torch.rand(4, 4, dtype=torch.float32) * 100
    mask = torch.zeros(4, 4, dtype=torch.bool)
    mask[:2, :3] = True
    got = calc_volume(opd, pixel_size=1e-7, refractive_delta=0.5, mask=mask)
    expected = np_calc_volume(
        opd.numpy(), pixel_size=1e-7, refractive_delta=0.5, mask=mask.numpy()
    )
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-5)


def test_calc_volume_from_phase_matches_numpy():
    phase = torch.rand(3, 3, dtype=torch.float32)
    got = calc_volume_from_phase(
        phase, pixel_size=1e-7, wavelength=666e-9, refractive_delta=0.5
    )
    expected = np_from_phase(
        phase.numpy(), pixel_size=1e-7, wavelength=666e-9, refractive_delta=0.5
    )
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-5)


def test_calc_volume_reduce_false_returns_map():
    opd = torch.full((2, 2), 50.0)
    density = calc_volume(opd, pixel_size=1e-7, reduce=False)
    assert density.shape == (2, 2)
    assert density.sum().item() == pytest.approx(
        calc_volume(opd, pixel_size=1e-7).item()
    )


def test_preserves_grad_and_device():
    phase = torch.ones(2, 2, requires_grad=True)
    volume = calc_volume_from_phase(phase, pixel_size=1e-7)
    assert volume.requires_grad
    assert volume.device == phase.device
    volume.backward()
    assert phase.grad is not None
    assert torch.all(phase.grad > 0)  # every pixel contributes positively
