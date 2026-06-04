from __future__ import annotations

import pytest

from iivs.dhm.analysis.drymass import calc_drymass as np_calc_drymass
from iivs.dhm.analysis.drymass import calc_drymass_from_phase as np_calc_from_phase

torch = pytest.importorskip("torch")

from iivs.dhm.analysis.pytorch.drymass import (  # noqa: E402
    calc_drymass,
    calc_drymass_from_phase,
)


def test_calc_drymass_matches_numpy():
    opd = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    got = calc_drymass(opd, pixel_size=2.85e-7)
    expected = np_calc_drymass(opd.numpy(), pixel_size=2.85e-7)
    assert got.item() == pytest.approx(expected)


def test_returns_zerodim_tensor_not_float():
    out = calc_drymass(torch.ones(3, 3), pixel_size=1e-6)
    assert isinstance(out, torch.Tensor)
    assert out.ndim == 0


def test_mask_matches_numpy():
    opd = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mask = torch.tensor([True, False, True, False])
    got = calc_drymass(opd, pixel_size=1e-6, mask=mask)
    expected = np_calc_drymass(opd.numpy(), pixel_size=1e-6, mask=mask.numpy())
    assert got.item() == pytest.approx(expected)


def test_from_phase_matches_numpy():
    phase = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    got = calc_drymass_from_phase(phase, pixel_size=2.85e-7, wavelength=666e-9)
    expected = np_calc_from_phase(phase.numpy(), pixel_size=2.85e-7, wavelength=666e-9)
    assert got.item() == pytest.approx(expected)


def test_from_phase_preserves_grad():
    phase = torch.ones(2, 2, requires_grad=True)
    mass = calc_drymass_from_phase(phase, pixel_size=1e-6, wavelength=666e-9)
    assert mass.requires_grad
    mass.backward()
    assert phase.grad is not None
