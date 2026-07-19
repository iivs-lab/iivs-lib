from __future__ import annotations

import pytest

from iivs.dhm.analysis.drymass import calc_drymass as np_calc_drymass
from iivs.dhm.analysis.drymass import calc_drymass_from_phase as np_calc_from_phase

torch = pytest.importorskip("torch")

from torch import fx, nn  # noqa: E402

from iivs.common.data.pytorch import Sum  # noqa: E402
from iivs.dhm.analysis.pytorch.drymass import (  # noqa: E402
    DryMass,
    calc_drymass,
    calc_drymass_from_phase,
)
from iivs.dhm.analysis.pytorch.opd import OpticalPathDifference  # noqa: E402

# --- DryMass: a pure pointwise density layer ---


def test_forward_is_pointwise_density():
    # 50 nm over 100 px at 0.1 um, alpha 2e-4 -> 0.25 pg total (the numpy anchor).
    dm = DryMass(pixel_size=1e-7, alpha=2.0e-4)
    opd = torch.full((10, 10), 50.0)
    density = dm(opd)
    assert density.shape == (10, 10)  # shape-preserving, no reduction
    assert torch.allclose(density, opd * dm.drymass_scale)
    assert density.sum().item() == pytest.approx(0.25)  # summing gives the total


def test_forward_preserves_dtype_and_grad():
    for dt in (torch.float32, torch.float64):
        opd = torch.ones(4, 4, dtype=dt)
        assert DryMass(pixel_size=1e-7)(opd).dtype == dt
    dm = DryMass(pixel_size=1e-7)
    x = torch.ones(4, 4, requires_grad=True)
    dm(x).sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, dm.drymass_scale))  # d/dx = scale


def test_forward_is_traceable_and_composable():
    # the payoff of the pointwise form: a clean tensor-in/tensor-out layer that
    # fx-traces (torch.compile / export build on fx) and drops into nn.Sequential
    # without collapsing; the old keyword-only mask/reduce form did neither.
    dm = DryMass(pixel_size=1e-7)
    opd = torch.ones(4, 4)
    traced = fx.symbolic_trace(dm)
    assert torch.allclose(traced(opd), dm(opd))
    seq = nn.Sequential(dm)
    assert seq(opd).shape == (4, 4)  # density flows on, not reduced to a scalar
    assert torch.allclose(seq(opd), dm(opd))


def test_composes_with_a_separate_reduction():
    opd = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    dm = DryMass(pixel_size=1e-6)
    mask = torch.tensor([[True, False], [True, False]])
    composed = Sum(mask)(dm(opd))  # DryMass density, then a masked reduction
    one_shot = calc_drymass(opd, pixel_size=1e-6, mask=mask)
    assert torch.allclose(composed, one_shot)


# --- calc_drymass / calc_drymass_from_phase: one-shot mass, composed ---


def test_calc_drymass_matches_numpy():
    opd = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    got = calc_drymass(opd, pixel_size=2.85e-7)
    expected = np_calc_drymass(opd.numpy(), pixel_size=2.85e-7)
    assert got.item() == pytest.approx(float(expected))


def test_reduce_returns_zerodim_tensor_not_float():
    out = calc_drymass(torch.ones(3, 3), pixel_size=1e-6)
    assert isinstance(out, torch.Tensor)
    assert out.ndim == 0


def test_mask_matches_numpy():
    opd = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mask = torch.tensor([[True, False], [True, False]])
    got = calc_drymass(opd, pixel_size=1e-6, mask=mask)
    expected = np_calc_drymass(opd.numpy(), pixel_size=1e-6, mask=mask.numpy())
    assert got.item() == pytest.approx(float(expected))


def test_label_mask_matches_numpy():
    opd = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    labels = torch.tensor([[1, 1], [2, 0]])  # one dry mass per positive label
    got = calc_drymass(opd, pixel_size=1e-6, mask=labels)
    expected = np_calc_drymass(opd.numpy(), pixel_size=1e-6, mask=labels.numpy())
    assert got.shape == (2,)
    assert torch.allclose(got, torch.as_tensor(expected))


def test_empty_region_is_zero_mass():
    # a label gap (label 1 absent) -> empty region -> 0 pg, matching numpy
    opd = torch.full((2, 2), 50.0)
    labels = torch.tensor([[0, 2], [2, 0]])  # label 1 missing
    out = calc_drymass(opd, pixel_size=1e-7, alpha=2e-4, mask=labels)
    assert out[0].item() == 0.0
    assert out[1].item() > 0


def test_batched():
    opd = torch.stack([torch.full((2, 2), 10.0), torch.full((2, 2), 20.0)])
    out = calc_drymass(opd, pixel_size=1e-7, alpha=2e-4)
    assert out.shape == (2,)
    single = calc_drymass(opd[0], pixel_size=1e-7, alpha=2e-4).item()
    assert out[0].item() == pytest.approx(single)
    assert out[1].item() == pytest.approx(2 * single)


def test_channel_mask():
    opd = torch.full((2, 2), 50.0)
    masks = torch.tensor(
        [[[True, False], [False, False]], [[True, True], [False, False]]]
    )  # (N=2, H=2, W=2)
    out = calc_drymass(opd, pixel_size=1e-7, alpha=2e-4, mask=masks)
    assert out.shape == (2,)
    whole = calc_drymass(opd, pixel_size=1e-7, alpha=2e-4).item()
    assert out[0].item() == pytest.approx(whole / 4)
    assert out[1].item() == pytest.approx(whole / 2)


def test_preserves_input_dtype():
    mask = torch.ones(4, 4, dtype=torch.bool)
    for dt in (torch.float32, torch.float64):
        opd = torch.ones(4, 4, dtype=dt)
        assert calc_drymass(opd, pixel_size=1e-7).dtype == dt  # reduce, no mask
        assert calc_drymass(opd, pixel_size=1e-7, mask=mask).dtype == dt  # reduce+mask
        assert calc_drymass(opd, pixel_size=1e-7, reduce=False).dtype == dt  # density


def test_rejects_bad_shapes():
    with pytest.raises(ValueError, match="at least 2"):
        calc_drymass(torch.zeros(4), pixel_size=1e-7)
    opd = torch.zeros(3, 4, 4)
    with pytest.raises(ValueError, match="mask must be"):  # (T, N, H, W)
        calc_drymass(
            opd, pixel_size=1e-7, mask=torch.ones(3, 2, 4, 4, dtype=torch.bool)
        )
    with pytest.raises(ValueError, match=r"\(H, W\) must be"):  # (H, W) mismatch
        calc_drymass(opd, pixel_size=1e-7, mask=torch.ones(4, 5, dtype=torch.bool))


def test_reduce_false_with_mask():
    opd = torch.full((2, 2), 50.0)
    m2 = torch.tensor([[True, False], [False, False]])  # (H, W)
    d2 = calc_drymass(opd, pixel_size=1e-7, mask=m2, reduce=False)
    assert d2.shape == (2, 2)
    assert d2.sum().item() == pytest.approx(
        calc_drymass(opd, pixel_size=1e-7, mask=m2).item()
    )
    m3 = torch.tensor(  # (N, H, W) -> per-object density maps
        [[[True, False], [False, False]], [[True, True], [False, False]]]
    )
    assert calc_drymass(opd, pixel_size=1e-7, mask=m3, reduce=False).shape == (2, 2, 2)


def test_reduce_false_returns_map_and_keeps_grad():
    phase = torch.ones(2, 2, requires_grad=True)
    density = calc_drymass_from_phase(
        phase, pixel_size=1e-6, wavelength=666e-9, reduce=False
    )
    assert density.shape == (2, 2)  # per-pixel map, not summed
    assert density.requires_grad
    density.sum().backward()
    slope = _phase_to_mass_slope()  # d(mass)/d(phase)
    assert torch.allclose(phase.grad, torch.full_like(phase, slope))


def test_from_phase_matches_numpy():
    phase = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    got = calc_drymass_from_phase(phase, pixel_size=2.85e-7, wavelength=666e-9)
    expected = np_calc_from_phase(phase.numpy(), pixel_size=2.85e-7, wavelength=666e-9)
    assert got.item() == pytest.approx(float(expected))


def test_from_phase_preserves_grad():
    phase = torch.ones(2, 2, requires_grad=True)
    mass = calc_drymass_from_phase(phase, pixel_size=1e-6, wavelength=666e-9)
    assert mass.requires_grad
    mass.backward()
    slope = _phase_to_mass_slope()  # d(mass)/d(phase)
    assert torch.allclose(phase.grad, torch.full_like(phase, slope))


def _phase_to_mass_slope() -> float:
    # d(mass)/d(phase) = opd_scale (nm/rad) * drymass_scale (pg/nm), the composed
    # OpticalPathDifference + DryMass slope at pixel_size 1e-6, wavelength 666 nm.
    opd_scale = OpticalPathDifference(wavelength=666e-9).opd_scale
    drymass_scale = DryMass(pixel_size=1e-6).drymass_scale
    return opd_scale * drymass_scale
