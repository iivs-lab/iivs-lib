from __future__ import annotations

import numpy as np
import pytest

from iivs.dhm.analysis.area import ProjectedAreaCalculator, calc_projected_area

IMG = np.zeros((2, 2), dtype=np.float32)


def test_whole_frame_without_mask():
    # 6 pixels of 0.1 um pitch: 6 * (0.1 um)^2 = 0.06 um^2.
    image = np.zeros((2, 3), dtype=np.float32)
    area = calc_projected_area(image, pixel_size=1e-7)
    assert area.shape == ()
    assert area == pytest.approx(0.06)
    assert area.dtype == np.float32


def test_boolean_mask_gives_a_plain_area():
    mask = np.array([[True, True], [True, False]])
    area = calc_projected_area(IMG, pixel_size=1e-7, mask=mask)
    assert area.shape == ()  # single region keeps the plain shape
    assert area == pytest.approx(0.03)


def test_image_values_and_dtype_never_enter():
    # The image only fixes the grid: any values, of any dtype, give the same
    # footprint (a uint8 preview serves as well as a float32 map).
    rng = np.random.default_rng(0)
    noisy = rng.standard_normal((2, 2)).astype(np.float32)
    preview = rng.integers(0, 256, size=(2, 2), dtype=np.uint8)
    mask = np.array([[True, False], [True, False]])

    expected = float(calc_projected_area(IMG, pixel_size=1e-7, mask=mask))
    assert calc_projected_area(noisy, pixel_size=1e-7, mask=mask) == pytest.approx(
        expected
    )
    assert calc_projected_area(preview, pixel_size=1e-7, mask=mask) == pytest.approx(
        expected
    )


def test_batched_image_gives_one_area_per_image():
    # (N, H, W) image -> shape (N,); the area repeats (it depends on the grid only).
    batch = np.zeros((3, 2, 2), dtype=np.float32)
    mask = np.array([[True, False], [False, False]])
    areas = calc_projected_area(batch, pixel_size=1e-7, mask=mask)
    assert areas.shape == (3,)
    np.testing.assert_allclose(areas, np.full(3, 0.01, dtype=np.float32), rtol=1e-6)


def test_mask_stack_gives_one_area_per_region():
    masks = np.array(
        [
            [[True, False], [False, False]],  # 1 pixel
            [[True, True], [True, False]],  # 3 pixels (overlap is fine)
        ]
    )
    areas = calc_projected_area(IMG, pixel_size=1e-7, mask=masks)
    assert areas.shape == (2,)
    assert areas[0] == pytest.approx(0.01)
    assert areas[1] == pytest.approx(0.03)


def test_label_mask_and_empty_region():
    labels = np.array([[0, 2], [2, 0]])  # label 1 missing, label 2: 2 px
    areas = calc_projected_area(IMG, pixel_size=1e-7, mask=labels)
    assert areas.shape == (2,)
    assert areas[0] == 0.0  # empty region -> 0 area
    assert areas[1] == pytest.approx(0.02)


def test_reduce_false_renders_the_mask_in_area_units():
    # The density map is the mask rendered in um^2: area_scale inside, 0 outside.
    mask = np.array([[True, True], [True, False]])
    calculator = ProjectedAreaCalculator(pixel_size=1e-7)
    density = calculator.calc(IMG, mask=mask, reduce=False)

    assert density.shape == (2, 2)  # single region keeps the plain (H, W)
    assert density.dtype == np.float32
    expected = np.where(mask, np.float32(calculator.area_scale), np.float32(0.0))
    np.testing.assert_array_equal(density, expected)
    assert density.sum() == pytest.approx(float(calculator.calc(IMG, mask=mask)))


def test_reduce_false_label_mask_gives_a_region_stack():
    labels = np.array([[1, 1], [2, 0]])
    density = calc_projected_area(IMG, pixel_size=1e-7, mask=labels, reduce=False)
    assert density.shape == (2, 2, 2)  # (R, H, W)
    per_label = calc_projected_area(IMG, pixel_size=1e-7, mask=labels)
    assert density[0].sum() == pytest.approx(float(per_label[0]))
    assert density[1].sum() == pytest.approx(float(per_label[1]))


def test_calculator_scale_properties():
    # 0.1 um pixel: 0.01 um^2 per pixel (hand-derived).
    calculator = ProjectedAreaCalculator(pixel_size=1e-7)
    assert calculator.area_scale == pytest.approx(0.01)
    assert calculator.pixel_size_um == pytest.approx(0.1)
    # the scale times the pixel count reproduces the area
    image = np.zeros((4, 4), dtype=np.float32)
    assert calculator.calc(image) == pytest.approx(16 * calculator.area_scale)


def test_one_shot_matches_calculator():
    mask = np.array([[True, False], [True, True]])
    one_shot = calc_projected_area(IMG, pixel_size=2e-7, mask=mask)
    engine = ProjectedAreaCalculator(pixel_size=2e-7).calc(IMG, mask=mask)
    assert one_shot == pytest.approx(engine)


def test_rejects_nonpositive_pixel_size():
    with pytest.raises(ValueError, match="pixel_size must be positive"):
        ProjectedAreaCalculator(pixel_size=0.0)


def test_rejects_bad_shapes():
    calculator = ProjectedAreaCalculator(pixel_size=1e-7)
    with pytest.raises(ValueError, match="image must be at least 2D"):
        calculator.calc(np.zeros(4, dtype=np.float32))
    with pytest.raises(ValueError, match="mask must be"):
        calculator.calc(IMG, mask=np.ones((2, 2), dtype=np.float32))  # bad dtype
