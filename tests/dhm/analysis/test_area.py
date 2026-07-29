from __future__ import annotations

import numpy as np
import pytest

from iivs.dhm.analysis.area import ProjectedAreaCalculator, calc_projected_area


def test_boolean_mask_gives_a_plain_area():
    # 3 pixels of 0.1 um pitch: 3 * (0.1 um)^2 = 0.03 um^2.
    mask = np.array([[True, True], [True, False]])
    area = calc_projected_area(mask, pixel_size=1e-7)
    assert area.shape == ()  # single region drops the region axis
    assert area == pytest.approx(0.03)
    assert area.dtype == np.float32


def test_mask_stack_gives_one_area_per_region():
    masks = np.array(
        [
            [[True, False], [False, False]],  # 1 pixel
            [[True, True], [True, False]],  # 3 pixels (overlap is fine)
        ]
    )
    areas = calc_projected_area(masks, pixel_size=1e-7)
    assert areas.shape == (2,)
    assert areas[0] == pytest.approx(0.01)
    assert areas[1] == pytest.approx(0.03)


def test_label_mask_gives_one_area_per_label():
    labels = np.array([[1, 1], [2, 0]])  # label 1: 2 px, label 2: 1 px
    areas = calc_projected_area(labels, pixel_size=1e-7)
    assert areas.shape == (2,)
    assert areas[0] == pytest.approx(0.02)
    assert areas[1] == pytest.approx(0.01)


def test_label_gap_is_zero_area():
    labels = np.array([[0, 2], [2, 0]])  # label 1 missing
    areas = calc_projected_area(labels, pixel_size=1e-7)
    assert areas[0] == 0.0
    assert areas[1] == pytest.approx(0.02)


def test_calculator_scale_properties():
    # 0.1 um pixel: 0.01 um^2 per pixel (hand-derived).
    calculator = ProjectedAreaCalculator(pixel_size=1e-7)
    assert calculator.area_scale == pytest.approx(0.01)
    assert calculator.pixel_size_um == pytest.approx(0.1)
    # the scale times the pixel count reproduces the area
    mask = np.ones((4, 4), dtype=bool)
    assert calculator.calc(mask) == pytest.approx(16 * calculator.area_scale)


def test_one_shot_matches_calculator():
    mask = np.array([[True, False], [True, True]])
    one_shot = calc_projected_area(mask, pixel_size=2e-7)
    engine = ProjectedAreaCalculator(pixel_size=2e-7).calc(mask)
    assert one_shot == pytest.approx(engine)


def test_rejects_nonpositive_pixel_size():
    with pytest.raises(ValueError, match="pixel_size must be positive"):
        ProjectedAreaCalculator(pixel_size=0.0)


def test_rejects_malformed_mask():
    calculator = ProjectedAreaCalculator(pixel_size=1e-7)
    with pytest.raises(ValueError, match="mask must be"):
        calculator.calc(np.ones((2, 2), dtype=np.float32))  # neither bool nor labels
