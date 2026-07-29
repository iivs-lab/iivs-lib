from __future__ import annotations

import numpy as np
import pytest

from iivs.dhm.analysis.area import calc_projected_area as np_calc_projected_area

torch = pytest.importorskip("torch")

from iivs.dhm.analysis.pytorch.area import calc_projected_area  # noqa: E402


def test_boolean_mask_gives_a_plain_area():
    # 3 pixels of 0.1 um pitch: 0.03 um^2, as the NumPy twin computes.
    mask = torch.tensor([[True, True], [True, False]])
    area = calc_projected_area(mask, pixel_size=1e-7)
    assert area.shape == ()  # single region drops the region axis
    assert area.item() == pytest.approx(0.03)
    assert area.dtype == torch.float32


def test_label_mask_matches_numpy():
    labels = torch.tensor([[1, 1], [2, 0]])
    got = calc_projected_area(labels, pixel_size=1e-7)
    expected = np_calc_projected_area(labels.numpy(), pixel_size=1e-7)
    assert got.shape == (2,)
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-6)


def test_mask_stack_gives_one_area_per_region():
    masks = torch.tensor(
        [
            [[True, False], [False, False]],  # 1 pixel
            [[True, True], [True, False]],  # 3 pixels
        ]
    )
    areas = calc_projected_area(masks, pixel_size=1e-7)
    assert areas.shape == (2,)
    assert areas[0].item() == pytest.approx(0.01)
    assert areas[1].item() == pytest.approx(0.03)
