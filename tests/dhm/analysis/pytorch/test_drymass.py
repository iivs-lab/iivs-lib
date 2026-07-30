from __future__ import annotations

import pytest

from iivs.dhm.analysis.drymass import calc_drymass as np_calc_drymass
from iivs.dhm.analysis.drymass import calc_drymass_from_height as np_from_height
from iivs.dhm.analysis.drymass import calc_drymass_from_opd as np_from_opd
from iivs.dhm.constants import (
    DEFAULT_REFRACTIVE_DELTA,
    DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    DEFAULT_WAVELENGTH,
)

torch = pytest.importorskip("torch")

from torch import fx, nn  # noqa: E402

from iivs.common.data.pytorch import Sum  # noqa: E402
from iivs.dhm.analysis.pytorch.drymass import (  # noqa: E402
    DryMass,
    calc_drymass,
    calc_drymass_from_height,
    calc_drymass_from_opd,
)
from iivs.dhm.analysis.pytorch.volume import OpticalVolume  # noqa: E402


def _drymass(
    *,
    pixel_size: float = 1e-7,
    wavelength: float = DEFAULT_WAVELENGTH,
    refractive_delta: float = DEFAULT_REFRACTIVE_DELTA,
    alpha: float = DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
) -> DryMass:
    return DryMass.from_args(
        pixel_size=pixel_size,
        wavelength=wavelength,
        refractive_delta=refractive_delta,
        alpha=alpha,
    )


# --- DryMass: a pure pointwise density layer over its owned engine ---


def test_forward_is_the_pointwise_phase_density():
    dm = _drymass()
    phase = torch.ones(3, 3)
    density = dm(phase)
    assert density.shape == (3, 3)  # shape-preserving, no reduction
    assert torch.allclose(density, phase * dm.drymass_scale)


def test_owns_volume_submodule():
    dm = _drymass(
        pixel_size=2e-7, wavelength=532e-9, refractive_delta=0.4, alpha=3.0e-4
    )
    children = dict(dm.named_children())
    assert isinstance(children["volume_converter"], OpticalVolume)
    # the scale derives from the volume engine, not a copied constant
    vc = dm.volume_converter
    assert dm.drymass_scale == pytest.approx(
        vc.volume_scale * vc.refractive_delta / 3.0e-4 * 1e-3
    )
    # surfaced parameters delegate to the volume submodule
    assert dm.pixel_size == pytest.approx(2e-7)
    assert dm.pixel_size_um == pytest.approx(0.2)
    assert dm.refractive_delta == pytest.approx(0.4)
    assert dm.wavelength == pytest.approx(532e-9)
    assert dm.wavelength_nm == pytest.approx(532.0)
    assert dm.alpha == pytest.approx(3.0e-4)


def test_forward_preserves_dtype_and_grad():
    for dt in (torch.float32, torch.float64):
        phase = torch.ones(4, 4, dtype=dt)
        assert _drymass()(phase).dtype == dt
    dm = _drymass()
    x = torch.ones(4, 4, requires_grad=True)
    dm(x).sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, dm.drymass_scale))  # d/dx = scale


def test_forward_is_traceable_and_composable():
    dm = _drymass()
    phase = torch.ones(4, 4)
    traced = fx.symbolic_trace(dm)
    assert torch.allclose(traced(phase), dm(phase))
    seq = nn.Sequential(dm)
    assert seq(phase).shape == (4, 4)  # density flows on, not reduced to a scalar
    assert torch.allclose(seq(phase), dm(phase))


def test_calc_from_opd_is_wavelength_free():
    # opd -> phase -> mass; the wavelength cancels, so two wavelengths must agree.
    opd = torch.linspace(0, 80, 16).reshape(4, 4)
    a = _drymass(wavelength=666e-9)
    b = _drymass(wavelength=532e-9)
    torch.testing.assert_close(a.calc_from_opd(opd), b.calc_from_opd(opd))


def test_calc_from_height_matches_phase_path():
    dm = _drymass()
    phase = torch.rand(3, 3)
    height = dm.volume_converter.height_converter.convert_from_phase(phase)
    torch.testing.assert_close(dm.calc_from_height(height), dm(phase))


def test_composes_with_a_separate_reduction():
    phase = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    dm = _drymass(pixel_size=1e-6)
    mask = torch.tensor([[True, False], [True, False]])
    composed = Sum(mask)(dm(phase))  # DryMass density, then a masked reduction
    one_shot = calc_drymass(phase, pixel_size=1e-6, mask=mask)
    assert torch.allclose(composed, one_shot)


# --- one-shot mass, composed with the reductions ---


def test_calc_drymass_matches_numpy():
    phase = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    got = calc_drymass(phase, pixel_size=2.85e-7, wavelength=666e-9)
    expected = np_calc_drymass(phase.numpy(), pixel_size=2.85e-7, wavelength=666e-9)
    assert got.item() == pytest.approx(float(expected))


def test_calc_drymass_from_opd_matches_numpy():
    opd = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    got = calc_drymass_from_opd(opd, pixel_size=2.85e-7)
    expected = np_from_opd(opd.numpy(), pixel_size=2.85e-7)
    assert got.item() == pytest.approx(float(expected))


def test_calc_drymass_from_height_matches_numpy():
    height = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    got = calc_drymass_from_height(height, pixel_size=2.85e-7, refractive_delta=0.5)
    expected = np_from_height(height.numpy(), pixel_size=2.85e-7, refractive_delta=0.5)
    assert got.item() == pytest.approx(float(expected))


def test_reduce_returns_zerodim_tensor_not_float():
    out = calc_drymass(torch.ones(3, 3), pixel_size=1e-6)
    assert isinstance(out, torch.Tensor)
    assert out.ndim == 0


def test_mask_matches_numpy():
    phase = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mask = torch.tensor([[True, False], [True, False]])
    got = calc_drymass(phase, pixel_size=1e-6, mask=mask)
    expected = np_calc_drymass(phase.numpy(), pixel_size=1e-6, mask=mask.numpy())
    assert got.item() == pytest.approx(float(expected))


def test_label_mask_matches_numpy():
    phase = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    labels = torch.tensor([[1, 1], [2, 0]])  # one dry mass per positive label
    got = calc_drymass(phase, pixel_size=1e-6, mask=labels)
    expected = np_calc_drymass(phase.numpy(), pixel_size=1e-6, mask=labels.numpy())
    assert got.shape == (2,)
    assert torch.allclose(got, torch.as_tensor(expected))


def test_empty_region_is_zero_mass():
    phase = torch.full((2, 2), 1.0)
    labels = torch.tensor([[0, 2], [2, 0]])  # label 1 missing
    out = calc_drymass(phase, pixel_size=1e-7, alpha=2e-4, mask=labels)
    assert out[0].item() == 0.0
    assert out[1].item() > 0


def test_batched():
    phase = torch.stack([torch.full((2, 2), 1.0), torch.full((2, 2), 2.0)])
    out = calc_drymass(phase, pixel_size=1e-7, alpha=2e-4)
    assert out.shape == (2,)
    single = calc_drymass(phase[0], pixel_size=1e-7, alpha=2e-4).item()
    assert out[0].item() == pytest.approx(single)
    assert out[1].item() == pytest.approx(2 * single)


def test_channel_mask():
    phase = torch.full((2, 2), 1.0)
    masks = torch.tensor(
        [[[True, False], [False, False]], [[True, True], [False, False]]]
    )  # (N=2, H=2, W=2)
    out = calc_drymass(phase, pixel_size=1e-7, alpha=2e-4, mask=masks)
    assert out.shape == (2,)
    whole = calc_drymass(phase, pixel_size=1e-7, alpha=2e-4).item()
    assert out[0].item() == pytest.approx(whole / 4)
    assert out[1].item() == pytest.approx(whole / 2)


def test_preserves_input_dtype():
    mask = torch.ones(4, 4, dtype=torch.bool)
    for dt in (torch.float32, torch.float64):
        phase = torch.ones(4, 4, dtype=dt)
        assert calc_drymass(phase, pixel_size=1e-7).dtype == dt  # reduce, no mask
        assert (
            calc_drymass(phase, pixel_size=1e-7, mask=mask).dtype == dt
        )  # reduce+mask
        assert calc_drymass(phase, pixel_size=1e-7, reduce=False).dtype == dt  # density


def test_rejects_bad_shapes():
    with pytest.raises(ValueError, match="at least 2"):
        calc_drymass(torch.zeros(4), pixel_size=1e-7)
    phase = torch.zeros(3, 4, 4)
    with pytest.raises(ValueError, match="mask must be"):  # (T, N, H, W)
        calc_drymass(
            phase, pixel_size=1e-7, mask=torch.ones(3, 2, 4, 4, dtype=torch.bool)
        )
    with pytest.raises(ValueError, match=r"\(H, W\) must be"):  # (H, W) mismatch
        calc_drymass(phase, pixel_size=1e-7, mask=torch.ones(4, 5, dtype=torch.bool))


def test_reduce_false_returns_map_and_keeps_grad():
    phase = torch.ones(2, 2, requires_grad=True)
    density = calc_drymass(phase, pixel_size=1e-6, wavelength=666e-9, reduce=False)
    assert density.shape == (2, 2)  # per-pixel map, not summed
    assert density.requires_grad
    density.sum().backward()
    slope = _drymass(
        pixel_size=1e-6, wavelength=666e-9
    ).drymass_scale  # d(mass)/d(phase)
    assert torch.allclose(phase.grad, torch.full_like(phase, slope))
