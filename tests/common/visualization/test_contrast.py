from __future__ import annotations

import numpy as np
import pytest

from iivs.common.visualization import auto_rescale


def test_uint8_fills_dtype_range_and_rounds():
    # saturated=0 -> percentiles (0, 100) = (0, 200); out_range=None -> (0, 255).
    # 100 -> 100/200 * 255 = 127.5 -> 128 (round half to even).
    out = auto_rescale(np.array([0, 100, 200], dtype=np.uint8), saturated=0)
    assert out.dtype == np.uint8
    assert out.tolist() == [0, 128, 255]


def test_float_dtype_fills_unit_range():
    out = auto_rescale(np.array([0.0, 5.0, 10.0]), saturated=0)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0])


def test_explicit_out_range():
    out = auto_rescale(np.array([0.0, 10.0]), saturated=0, out_range=(0.0, 100.0))
    np.testing.assert_allclose(out, [0.0, 100.0])


def test_saturated_clips_both_tails():
    # 0..100; saturated=2 -> half 1 -> percentiles (1, 99) -> in_range (1, 99).
    data = np.arange(0, 101, dtype=np.float64)
    out = auto_rescale(data, saturated=2, out_range=(0.0, 100.0))
    assert out[0] == 0.0  # value 0 clips to the low end
    assert out[100] == 100.0  # value 100 clips to the high end
    assert out[50] == pytest.approx((50 - 1) / (99 - 1) * 100)  # 50.0


def test_constant_image_maps_to_low():
    out = auto_rescale(np.full(6, 5.0), out_range=(0.0, 255.0))
    assert np.all(out == 0.0)


def test_flat_percentile_window_falls_back_to_full_range():
    # A flat median with two outliers: the 49.5/50.5 percentiles both land on 5, so the
    # (min, max) fallback recovers the (0, 10) span.
    data = np.array([0.0, 5, 5, 5, 5, 5, 5, 5, 5, 10.0])
    out = auto_rescale(data, saturated=99, out_range=(0.0, 10.0))
    assert out[0] == 0.0
    assert out[-1] == 10.0
    assert out[1] == 5.0


def test_ignores_and_preserves_nan():
    data = np.array([0.0, 5.0, 10.0, np.nan])
    out = auto_rescale(data, saturated=0, out_range=(0.0, 100.0))
    np.testing.assert_allclose(out[:3], [0.0, 50.0, 100.0])
    assert np.isnan(out[3])


def test_stack_pools_pixels_into_one_range():
    # 4 pixels across two frames span [0, 20]; the shared range rescales every frame.
    stack = np.array([[[0.0, 10.0]], [[5.0, 20.0]]])  # (2, 1, 2)
    out = auto_rescale(stack, saturated=0, out_range=(0.0, 100.0))
    np.testing.assert_allclose(out, [[[0.0, 50.0]], [[25.0, 100.0]]])


def test_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        auto_rescale(np.array([], dtype=np.float64))


def test_rejects_saturated_too_high():
    with pytest.raises(ValueError, match="saturated must be"):
        auto_rescale(np.zeros((2, 2)), saturated=100)


def test_rejects_saturated_negative():
    with pytest.raises(ValueError, match="saturated must be"):
        auto_rescale(np.zeros((2, 2)), saturated=-1)
