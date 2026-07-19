from __future__ import annotations

import pytest

import iivs.common.data.reduction as npr

torch = pytest.importorskip("torch")

from iivs.common.data.pytorch.reduction import (  # noqa: E402
    MaskedReduction,
    Mean,
    Norm,
    Std,
    Sum,
    Variance,
    apply_mask,
    region_stack,
)

V = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # float32


class TestRegionStack:
    def test_none_is_one_whole_region(self):
        stack = region_stack(None, (2, 3))
        assert stack.shape == (1, 2, 3)
        assert stack.dtype == torch.bool
        assert stack.all()

    def test_bool_2d_becomes_one_region(self):
        mask = torch.tensor([[True, False], [False, True]])
        assert region_stack(mask, (2, 2)).shape == (1, 2, 2)

    def test_bool_3d_passthrough_allows_overlap(self):
        mask = torch.tensor(
            [[[True, True], [False, False]], [[True, False], [False, False]]]
        )
        assert torch.equal(region_stack(mask, (2, 2)), mask)

    def test_labels_expand_one_layer_per_positive_label(self):
        labels = torch.tensor([[0, 1], [2, 2]])
        stack = region_stack(labels, (2, 2))
        assert stack.shape == (2, 2, 2)
        assert torch.equal(stack[0], labels == 1)
        assert torch.equal(stack[1], labels == 2)

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match=r"\(H, W\) must be"):
            region_stack(torch.ones(3, 3, dtype=torch.bool), (2, 2))

    def test_rejects_bool_4d(self):
        with pytest.raises(ValueError, match="boolean mask must be"):
            region_stack(torch.ones(1, 1, 2, 2, dtype=torch.bool), (2, 2))

    def test_rejects_label_3d(self):
        with pytest.raises(ValueError, match="label mask must be"):
            region_stack(torch.ones(1, 2, 2, dtype=torch.int64), (2, 2))

    def test_rejects_negative_label(self):
        with pytest.raises(ValueError, match="non-negative"):
            region_stack(torch.tensor([[-1, 0], [0, 0]]), (2, 2))

    def test_rejects_float_dtype(self):
        with pytest.raises(ValueError, match="boolean or integer"):
            region_stack(torch.zeros(2, 2), (2, 2))


class TestReductions:
    def test_sum_whole_frame_is_scalar(self):
        out = Sum()(V)
        assert out.ndim == 0  # single region -> no region axis
        assert out.item() == pytest.approx(10.0)

    def test_matches_numpy_engine(self):
        vn = V.numpy()
        assert Sum()(V).item() == pytest.approx(float(npr.Sum()(vn)))
        assert Mean()(V).item() == pytest.approx(float(npr.Mean()(vn)))
        assert Norm()(V).item() == pytest.approx(float(npr.Norm()(vn)))
        assert Norm(3)(V).item() == pytest.approx(float(npr.Norm(3)(vn)))
        assert Variance()(V).item() == pytest.approx(float(npr.Variance()(vn)))
        assert Variance(1)(V).item() == pytest.approx(float(npr.Variance(1)(vn)))
        assert Std()(V).item() == pytest.approx(float(npr.Std()(vn)))

    def test_norm_uses_absolute_values(self):
        w = torch.tensor([[-3.0, 4.0]])
        assert Norm(1)(w).item() == pytest.approx(7.0)
        assert Norm()(w).item() == pytest.approx(5.0)  # sqrt(9 + 16)

    def test_overlap_counted_independently(self):
        masks = torch.tensor(
            [[[True, True], [False, False]], [[True, False], [False, False]]]
        )
        assert torch.allclose(Sum()(V, masks), torch.tensor([3.0, 1.0]))

    def test_label_mask_one_value_per_label(self):
        labels = torch.tensor([[0, 1], [2, 2]])  # label 1 -> {2}; label 2 -> {3, 4}
        out = Sum()(V, labels)
        assert out.shape == (2,)  # a label image keeps the region axis
        assert torch.allclose(out, torch.tensor([2.0, 7.0]))

    def test_batched_single_region(self):
        batch = torch.stack([V, 2.0 * V])  # (T=2, H, W)
        out = Sum()(batch)
        assert out.shape == (2,)
        assert torch.allclose(out, torch.tensor([10.0, 20.0]))

    def test_constant_region_is_zero_not_negative(self):
        const = torch.full((3, 3), 5.0)
        assert Variance()(const).item() == pytest.approx(0.0)
        assert Std()(const).item() == pytest.approx(0.0)  # never NaN

    def test_single_pixel_sample_variance_is_nan(self):
        one = torch.tensor([[True, False], [False, False]])  # n == 1
        assert torch.isnan(Variance(1)(V, one))
        assert Variance()(V, one).item() == pytest.approx(0.0)

    def test_all_background_labels_give_zero_regions(self):
        labels = torch.zeros((2, 2), dtype=torch.int64)  # no positive label
        out = Sum()(V, labels)
        assert out.shape == (0,)  # R == 0, no crash on the empty region stack


class TestEmptyPolicy:
    empty_masks = torch.tensor(
        [[[False, False], [False, False]], [[True, True], [True, True]]]
    )

    def test_empty_region_is_nan_by_default(self):
        for reducer in (Sum(), Mean(), Norm(), Variance(), Std()):
            out = reducer(V, self.empty_masks)
            assert torch.isnan(out[0]), type(reducer).__name__
            assert torch.isfinite(out[1]), type(reducer).__name__

    def test_empty_fill_override(self):
        out = Sum(empty=0.0)(V, self.empty_masks)
        assert torch.allclose(out, torch.tensor([0.0, 10.0]))


class TestTorchSemantics:
    def test_preserves_input_dtype(self):
        for dt in (torch.float32, torch.float64):
            values = torch.ones(3, 3, dtype=dt)
            for reducer in (Sum(), Mean(), Norm(), Variance(), Std()):
                assert reducer(values).dtype == dt, (type(reducer).__name__, dt)

    def test_sum_gradient_flows(self):
        values = V.clone().requires_grad_(True)
        Sum()(values).backward()
        assert torch.allclose(values.grad, torch.ones_like(values))  # d(sum)/dx = 1

    def test_mean_gradient_flows(self):
        values = V.clone().requires_grad_(True)
        Mean()(values).backward()
        assert torch.allclose(values.grad, torch.full_like(values, 0.25))  # 1/n

    def test_is_nn_module(self):
        assert isinstance(Sum(), torch.nn.Module)

    def test_masked_reduction_is_abstract(self):
        with pytest.raises(TypeError):
            MaskedReduction()


class TestMaskBinding:
    def test_bound_mask_used_and_overridden(self):
        bound = Sum(torch.tensor([[True, False], [False, False]]))  # selects {1}
        assert bound(V).item() == pytest.approx(1.0)
        full = torch.ones(2, 2, dtype=torch.bool)
        assert bound(V, full).item() == pytest.approx(10.0)  # per-call override

    def test_bound_mask_is_a_buffer(self):
        bound = Sum(torch.tensor([[True, False], [False, False]]))
        assert "mask" in dict(bound.named_buffers())  # moves with `.to(device)`

    def test_rejects_bad_args(self):
        with pytest.raises(ValueError, match="at least 2"):
            Sum()(torch.zeros(4))
        with pytest.raises(ValueError, match="p must be positive"):
            Norm(0)
        with pytest.raises(ValueError, match="ddof must be 0 or 1"):
            Variance(2)


class TestApplyMask:
    def test_none_is_whole_frame(self):
        out = apply_mask(V)
        assert out.shape == (2, 2)
        assert torch.equal(out, V)

    def test_bool_2d_drops_region_axis(self):
        mask = torch.tensor([[True, False], [False, False]])
        out = apply_mask(V, mask)
        assert out.shape == (2, 2)
        assert torch.equal(out, torch.tensor([[1.0, 0.0], [0.0, 0.0]]))

    def test_labels_keep_region_axis(self):
        out = apply_mask(V, torch.tensor([[0, 1], [2, 2]]))
        assert out.shape == (2, 2, 2)

    def test_sum_of_maps_matches_reduction(self):
        masks = torch.tensor(
            [[[True, True], [False, False]], [[False, False], [True, True]]]
        )
        maps = apply_mask(V, masks)
        assert torch.allclose(maps.sum(dim=(-2, -1)), Sum()(V, masks))

    def test_gradient_flows(self):
        values = torch.ones(2, 2, requires_grad=True)
        apply_mask(values).sum().backward()
        assert torch.allclose(values.grad, torch.ones_like(values))
