from __future__ import annotations

import numpy as np
import pytest

from iivs.dhm.analysis.area import calc_projected_area as np_calc_projected_area

torch = pytest.importorskip("torch")

from iivs.dhm.analysis.area import ProjectedAreaCalculator  # noqa: E402
from iivs.dhm.analysis.pytorch.area import (  # noqa: E402
    ProjectedArea,
    calc_projected_area,
)

IMG = torch.zeros(2, 2)


def test_boolean_mask_gives_a_plain_area():
    # 3 pixels of 0.1 um pitch: 0.03 um^2, as the NumPy twin computes.
    mask = torch.tensor([[True, True], [True, False]])
    area = calc_projected_area(IMG, pixel_size=1e-7, mask=mask)
    assert area.shape == ()  # single region keeps the plain shape
    assert area.item() == pytest.approx(0.03)
    assert area.dtype == torch.float32


def test_whole_frame_and_labels_match_numpy():
    got_whole = calc_projected_area(IMG, pixel_size=1e-7)
    assert got_whole.item() == pytest.approx(0.04)  # 4 px * 0.01 um^2

    labels = torch.tensor([[1, 1], [2, 0]])
    got = calc_projected_area(IMG, pixel_size=1e-7, mask=labels)
    expected = np_calc_projected_area(IMG.numpy(), pixel_size=1e-7, mask=labels.numpy())
    assert got.shape == (2,)
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-6)


def test_reduce_false_matches_numpy_density():
    labels = torch.tensor([[1, 1], [2, 0]])
    got = calc_projected_area(IMG, pixel_size=1e-7, mask=labels, reduce=False)
    expected = np_calc_projected_area(
        IMG.numpy(), pixel_size=1e-7, mask=labels.numpy(), reduce=False
    )
    assert got.shape == (2, 2, 2)  # (R, H, W)
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-6)


def test_batched_image_and_bad_shapes():
    batch = torch.zeros(3, 2, 2)
    mask = torch.tensor([[True, False], [False, False]])
    areas = calc_projected_area(batch, pixel_size=1e-7, mask=mask)
    assert areas.shape == (3,)
    np.testing.assert_allclose(areas.numpy(), np.full(3, 0.01), rtol=1e-6)

    with pytest.raises(ValueError, match="image must be at least 2D"):
        calc_projected_area(torch.zeros(4), pixel_size=1e-7)


# --- ProjectedArea (nn.Module) ---


def test_module_binds_the_numpy_scale():
    module = ProjectedArea(pixel_size=1e-7)
    np_engine = ProjectedAreaCalculator(pixel_size=1e-7)
    assert isinstance(module, torch.nn.Module)
    assert module.pixel_size == 1e-7
    assert module.pixel_size_um == pytest.approx(0.1)
    assert module.area_scale == pytest.approx(np_engine.area_scale)  # 0.01 um^2


def test_module_from_pixel_size_um():
    assert ProjectedArea.from_pixel_size_um(0.1).pixel_size == pytest.approx(1e-7)


def test_forward_is_a_constant_density_over_the_grid():
    # forward(image) is area_scale everywhere, whatever the image holds; only the
    # shape and device come from image. `reduce=False` on the one-shot renders the
    # unmasked density, so the two must agree.
    image = torch.arange(6, dtype=torch.float32).reshape(2, 3)  # values must not enter
    density = ProjectedArea(pixel_size=1e-7)(image)
    assert density.shape == (2, 3)
    assert density.dtype == torch.float32
    assert density.device == image.device
    torch.testing.assert_close(density, torch.full((2, 3), 0.01))  # area_scale
    expected = calc_projected_area(image, pixel_size=1e-7, reduce=False)
    torch.testing.assert_close(density, expected)


def test_forward_carries_no_gradient_from_the_image():
    # The density is a fresh constant (torch.full), disconnected from image's graph, so
    # it does not require grad even when image does -- image can get no gradient from it.
    image = torch.zeros(2, 2, requires_grad=True)
    density = ProjectedArea(pixel_size=1e-7)(image)
    assert not density.requires_grad
