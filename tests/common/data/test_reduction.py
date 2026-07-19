from __future__ import annotations

import warnings

import numpy as np
import pytest

from iivs.common.data.reduction import (
    MaskedReduction,
    Mean,
    Norm,
    Std,
    Sum,
    Variance,
    apply_mask,
    region_stack,
)

# A 2x2 map used across the reductions; values chosen so the moments are exact.
V = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)


class TestRegionStack:
    def test_none_is_one_whole_region(self):
        stack = region_stack(None, (2, 3))
        assert stack.shape == (1, 2, 3)
        assert stack.dtype == np.bool_
        assert stack.all()

    def test_bool_2d_becomes_one_region(self):
        mask = np.array([[True, False], [False, True]])
        stack = region_stack(mask, (2, 2))
        assert stack.shape == (1, 2, 2)
        assert np.array_equal(stack[0], mask)

    def test_bool_3d_passthrough_allows_overlap(self):
        mask = np.array(
            [[[True, True], [False, False]], [[True, False], [False, False]]]
        )
        stack = region_stack(mask, (2, 2))
        assert stack.shape == (2, 2, 2)
        assert np.array_equal(stack, mask)  # overlap at (0, 0) preserved, not merged

    def test_labels_expand_one_layer_per_positive_label(self):
        labels = np.array([[0, 1], [2, 2]])
        stack = region_stack(labels, (2, 2))
        assert stack.shape == (2, 2, 2)
        assert np.array_equal(stack[0], labels == 1)  # background 0 excluded
        assert np.array_equal(stack[1], labels == 2)

    def test_labels_max_sets_region_count_even_with_gaps(self):
        # label 1 is absent, but max()==2 still yields R==2; layer 1 is empty.
        labels = np.array([[0, 2], [2, 0]])
        stack = region_stack(labels, (2, 2))
        assert stack.shape == (2, 2, 2)
        assert not stack[0].any()  # the missing label 1 -> an empty region
        assert np.array_equal(stack[1], labels == 2)

    def test_all_background_labels_give_zero_regions(self):
        stack = region_stack(np.zeros((2, 2), dtype=int), (2, 2))
        assert stack.shape == (0, 2, 2)  # no positive label -> R == 0
        assert Sum()(V, np.zeros((2, 2), dtype=int)).shape == (0,)

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match=r"must be \(2, 2\)"):
            region_stack(np.ones((3, 3), dtype=bool), (2, 2))

    def test_rejects_bool_4d(self):
        with pytest.raises(ValueError, match="boolean mask must be"):
            region_stack(np.ones((1, 1, 2, 2), dtype=bool), (2, 2))

    def test_rejects_label_3d(self):
        with pytest.raises(ValueError, match="label mask must be"):
            region_stack(np.ones((1, 2, 2), dtype=int), (2, 2))

    def test_rejects_negative_label(self):
        with pytest.raises(ValueError, match="non-negative"):
            region_stack(np.array([[-1, 0], [0, 0]]), (2, 2))

    def test_rejects_float_dtype(self):
        with pytest.raises(ValueError, match="boolean or integer"):
            region_stack(np.zeros((2, 2), dtype=np.float32), (2, 2))


class TestSum:
    def test_sums_whole_frame_to_scalar(self):
        out = Sum()(V)
        assert out.shape == ()  # single region -> no region axis
        assert out == pytest.approx(10.0)

    def test_per_region_counts_overlap_independently(self):
        masks = np.array(
            [[[True, True], [False, False]], [[True, False], [False, False]]]
        )
        # region 0 -> {1, 2} = 3; region 1 -> {1} = 1; the shared pixel is in both.
        assert Sum()(V, masks) == pytest.approx([3.0, 1.0])

    def test_batched_single_region_keeps_only_leading_axes(self):
        batch = np.stack([V, 2.0 * V])  # (T=2, H, W)
        out = Sum()(batch)  # single region -> (T,), not (T, 1)
        assert out.shape == (2,)
        assert out == pytest.approx([10.0, 20.0])

    def test_nonfinite_background_does_not_poison(self):
        # `where=` skips out-of-region pixels, so a NaN in the background does not
        # poison a region's sum.
        v = np.ones((3, 3), dtype=np.float32)
        v[0, 0] = np.nan  # outside the region
        region = np.zeros((3, 3), dtype=bool)
        region[1, 1] = region[2, 2] = True
        assert Sum()(v, region) == pytest.approx(2.0)


class TestMean:
    def test_mean_whole_frame(self):
        assert Mean()(V) == pytest.approx(2.5)

    def test_mean_per_region_matches_numpy(self):
        mask = np.array([[True, True], [True, False]])  # {1, 2, 3}, one region
        assert Mean()(V, mask) == pytest.approx(np.mean([1.0, 2.0, 3.0]))


class TestNorm:
    def test_default_is_l2(self):
        assert Norm()(V) == pytest.approx(np.sqrt(1 + 4 + 9 + 16))

    def test_p1_is_sum_of_absolute_values(self):
        w = np.array([[-3.0, 4.0]], dtype=np.float32)  # negatives -> abs
        assert Norm(1)(w) == pytest.approx(7.0)

    def test_l2_uses_absolute_values(self):
        w = np.array([[-3.0, 4.0]], dtype=np.float32)
        assert Norm()(w) == pytest.approx(5.0)

    def test_arbitrary_p(self):
        # (1^3 + 2^3 + 3^3 + 4^3)^(1/3) = 100^(1/3)
        assert Norm(3)(V) == pytest.approx(100.0 ** (1 / 3))

    def test_rejects_nonpositive_p(self):
        with pytest.raises(ValueError, match="p must be positive"):
            Norm(0)
        with pytest.raises(ValueError, match="p must be positive"):
            Norm(-2)

    def test_rejects_nan_p(self):
        with pytest.raises(ValueError, match="p must be positive"):
            Norm(float("nan"))


class TestVarianceAndStd:
    def test_population_variance_matches_numpy(self):
        assert Variance()(V) == pytest.approx(np.var(V))

    def test_sample_variance_matches_numpy(self):
        assert Variance(1)(V) == pytest.approx(np.var(V, ddof=1))

    def test_variance_per_region_matches_numpy(self):
        mask = np.array([[True, True], [True, False]])  # {1, 2, 3}
        assert Variance()(V, mask) == pytest.approx(np.var([1.0, 2.0, 3.0]))

    def test_std_is_sqrt_of_variance(self):
        assert Std()(V) == pytest.approx(np.std(V))
        assert Std(1)(V) == pytest.approx(np.std(V, ddof=1))

    def test_constant_region_is_zero_not_negative(self):
        const = np.full((3, 3), 5.0, dtype=np.float32)
        assert Variance()(const) == pytest.approx(0.0)
        assert Std()(const) == pytest.approx(0.0)  # sqrt guard, never NaN

    def test_single_pixel_sample_variance_is_nan(self):
        one = np.array([[True, False], [False, False]])  # n == 1, one region
        assert np.isnan(Variance(1)(V, one))  # undefined, not filled by `empty`
        assert Variance()(V, one) == pytest.approx(0.0)  # population is defined

    def test_rejects_bad_ddof(self):
        with pytest.raises(ValueError, match="ddof must be 0 or 1"):
            Variance(2)
        with pytest.raises(ValueError, match="ddof must be 0 or 1"):
            Std(-1)

    def test_float32_variance_over_large_offset(self):
        # |x|**p is formed in float64, so a small spread on a large DC offset is
        # not lost to float32 squaring; matches numpy's float64 var on the data.
        x = (1e4 + np.array([[0.02, -0.02], [-0.02, 0.02]])).astype(np.float32)
        oracle = float(np.var(x.astype(np.float64)))
        assert Variance()(x) == pytest.approx(oracle, rel=1e-6)


class TestEmptyPolicy:
    empty_masks = np.array(
        [[[False, False], [False, False]], [[True, True], [True, True]]]
    )

    def test_empty_region_is_nan_by_default(self):
        # Uniform across reductions: the first (empty) region -> NaN, not 0.
        for reducer in (Sum(), Mean(), Norm(), Variance(), Std()):
            out = reducer(V, self.empty_masks)
            assert np.isnan(out[0]), type(reducer).__name__
            assert np.isfinite(out[1]), type(reducer).__name__

    def test_empty_fill_override(self):
        out = Sum(empty=0.0)(V, self.empty_masks)
        assert out == pytest.approx([0.0, 10.0])
        assert Mean(empty=-1.0)(V, self.empty_masks) == pytest.approx([-1.0, 2.5])

    def test_empty_region_emits_no_warning(self):
        # errstate must silence the internal 0/0; the fill is deterministic.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = Mean()(V, self.empty_masks)
        assert np.isnan(out[0])
        assert out[1] == pytest.approx(2.5)


class TestMaskBinding:
    def test_bound_mask_used_when_call_omits_one(self):
        bound = Sum(np.array([[True, False], [False, False]]))  # selects {1}
        assert bound(V) == pytest.approx(1.0)

    def test_call_mask_overrides_bound(self):
        bound = Sum(np.array([[True, False], [False, False]]))
        assert bound(V, np.ones((2, 2), dtype=bool)) == pytest.approx(10.0)

    def test_output_is_float64(self):
        for reducer in (Sum(), Mean(), Norm(), Variance(), Std()):
            assert reducer(V).dtype == np.float64, type(reducer).__name__

    def test_rejects_values_below_2d(self):
        with pytest.raises(ValueError, match="at least 2"):
            Sum()(np.zeros(4, dtype=np.float32))


class TestApplyMask:
    def test_none_is_whole_frame(self):
        out = apply_mask(V)  # single region -> no region axis
        assert out.shape == (2, 2)
        assert np.array_equal(out, V)

    def test_splits_into_per_region_maps_with_overlap(self):
        masks = np.array(
            [[[True, True], [False, False]], [[True, False], [False, False]]]
        )
        out = apply_mask(V, masks)
        assert out.shape == (2, 2, 2)
        assert np.array_equal(out[0], [[1.0, 2.0], [0.0, 0.0]])
        assert np.array_equal(out[1], [[1.0, 0.0], [0.0, 0.0]])  # shared (0,0) kept

    def test_labels_give_one_map_per_label(self):
        labels = np.array([[0, 1], [2, 2]])
        out = apply_mask(V, labels)
        assert out.shape == (2, 2, 2)  # a label image keeps the region axis
        assert np.array_equal(out[0], [[0.0, 2.0], [0.0, 0.0]])  # label 1
        assert np.array_equal(out[1], [[0.0, 0.0], [3.0, 4.0]])  # label 2

    def test_bool_2d_drops_region_axis(self):
        mask = np.array([[True, False], [False, False]])
        out = apply_mask(V, mask)  # single region -> (H, W)
        assert out.shape == (2, 2)
        assert np.array_equal(out, [[1.0, 0.0], [0.0, 0.0]])

    def test_batched_stack_keeps_leading_and_region_axes(self):
        batch = np.stack([V, 2.0 * V])  # (T=2, H, W)
        masks = np.array([[[True, False], [False, False]]])  # (R=1, H, W) stack
        out = apply_mask(batch, masks)
        assert out.shape == (2, 1, 2, 2)  # (T, R, H, W); a 3D mask keeps R

    def test_sum_of_maps_matches_the_reduction(self):
        masks = np.array(
            [[[True, True], [False, False]], [[False, False], [True, True]]]
        )
        maps = apply_mask(V, masks)  # (R, H, W)
        assert maps.sum(axis=(-2, -1)) == pytest.approx(Sum()(V, masks))

    def test_rejects_values_below_2d(self):
        with pytest.raises(ValueError, match="at least 2"):
            apply_mask(np.zeros(4, dtype=np.float32))


def test_masked_reduction_is_abstract():
    with pytest.raises(TypeError):
        MaskedReduction()  # cannot instantiate without a concrete `_reduce`
