from __future__ import annotations

__all__ = (
    "MaskedReduction",
    "Mean",
    "MomentReduction",
    "Norm",
    "Std",
    "Sum",
    "Variance",
    "apply_mask",
    "region_stack",
)

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, override

import torch
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor


def _validate_ndim(values: Tensor) -> None:
    if values.ndim < 2:
        msg = f"values must be at least 2D (..., H, W) (got {values.ndim}D)"
        raise ValueError(msg)


def region_stack(
    mask: Tensor | None, shape: tuple[int, int], *, device: torch.device | None = None
) -> Tensor:
    """Normalize any mask to a `(R, H, W)` boolean region stack.

    The single form every reduction works on. `None` is one region spanning the
    whole `shape` (created on `device`); a boolean `(H, W)` is one region and a
    boolean `(N, H, W)` is `N` regions (which may overlap); an integer label image
    `(H, W)` (0 = background) becomes one layer per positive label, so `R ==
    labels.max()` (0 if the image is all background). A given mask keeps its own
    device; `device` only places the whole-frame region for `None`.

    Raises:
        ValueError: If the mask's `(H, W)` differs from `shape`, a boolean mask
            is not 2D or 3D, a label mask is not 2D or holds a negative label, or
            the dtype is neither boolean nor integer.
    """
    if mask is None:
        return torch.ones((1, *shape), dtype=torch.bool, device=device)

    if tuple(mask.shape[-2:]) != shape:
        got = tuple(mask.shape[-2:])
        msg = f"mask (H, W) must be {shape} (got {got})"
        raise ValueError(msg)

    if mask.dtype == torch.bool:
        if mask.ndim == 2:
            return mask.unsqueeze(0)
        if mask.ndim == 3:
            return mask
        msg = f"a boolean mask must be (H, W) or (N, H, W) (got {mask.ndim}D)"
        raise ValueError(msg)

    if not mask.is_floating_point() and not mask.is_complex():  # integer labels
        if mask.ndim != 2:
            msg = f"a label mask must be (H, W) (got {mask.ndim}D)"
            raise ValueError(msg)
        if mask.min() < 0:
            msg = f"label values must be non-negative (got {int(mask.min())})"
            raise ValueError(msg)
        labels = torch.arange(1, int(mask.max()) + 1, device=mask.device)
        return labels[:, None, None] == mask.unsqueeze(0)

    msg = f"mask must be boolean or integer (got {mask.dtype})"
    raise ValueError(msg)


def _single_region(mask: Tensor | None) -> bool:
    """Test whether `mask` denotes one region (None or a boolean 2D image)."""
    return mask is None or (mask.ndim == 2 and mask.dtype == torch.bool)


def apply_mask(values: Tensor, mask: Tensor | None = None) -> Tensor:
    """Split `values` into per-region masked maps.

    In each region's layer every pixel outside that region is zeroed. The pointwise
    companion to the reductions (which instead collapse each region to a scalar). A
    single-region mask (None or a boolean 2D image) gives `(..., H, W)`; a stack or
    label image adds a region axis, giving `(..., R, H, W)`. Overlapping regions
    each keep the shared pixels. Preserves the input's dtype, device, and autograd
    graph.

    Raises:
        ValueError: If `values` has fewer than two axes, or the mask is malformed
            (see `region_stack`).
    """
    _validate_ndim(values)
    regions = region_stack(mask, tuple(values.shape[-2:]), device=values.device)
    maps = values.unsqueeze(-3) * regions
    return maps[..., 0, :, :] if _single_region(mask) else maps


def _masked_sum(summand: Tensor, regions: Tensor) -> Tensor:
    """Sum `summand` over each region in float64, stacked as `(..., R)`.

    Out-of-region pixels are replaced by 0 (not multiplied by the mask), so a
    non-finite background value does not poison a region's sum.
    """
    if len(regions) == 0:
        return summand.new_empty((*summand.shape[:-2], 0), dtype=torch.float64)
    sums = [
        torch.where(region, summand, 0.0).sum(dim=(-2, -1), dtype=torch.float64)
        for region in regions
    ]
    return torch.stack(sums, dim=-1)


class MaskedReduction(nn.Module, ABC):
    """An abstract reduction of a map over masked regions of its last two axes.

    The torch twin of `iivs.common.data.reduction.MaskedReduction`, an `nn.Module`
    so a bound mask registers as a buffer and moves with `.to(device)`. A
    `(..., H, W)` map reduces over the `R` regions of a mask (see `region_stack`;
    regions may overlap). A single-region mask (None or a boolean 2D image) drops
    the region axis, giving `(...)`; a stack or label image keeps it, giving
    `(..., R)`. A mask bound at construction is the default; a mask passed to the
    call overrides it. A region with no pixels reduces to `empty` (NaN by default;
    pass `empty=0.0` for a benign fill). Accumulates in float64 and returns the
    input's dtype, preserving its device and autograd graph. Its region structure
    is data-dependent, so it is an eager reduction head: backprop flows through it
    (including mid-pipeline), but it is not `torch.fx` / `torch.jit.script`
    traceable (`torch.compile` falls back to eager) the way a pointwise layer is.
    Subclasses implement `_reduce`; `MomentReduction` supplies the per-region power
    sums.

    Args:
        mask: Default mask (a registered buffer); the call's `mask` overrides it.
            `None` treats the whole frame as one region.
        empty: The value a region with no pixels reduces to.
    """

    mask: Tensor | None  # a registered buffer; typed so `self.mask` narrows
    empty: float

    def __init__(self, mask: Tensor | None = None, *, empty: float = math.nan) -> None:
        super().__init__()
        self.register_buffer("mask", mask)
        self.empty = float(empty)

    def forward(self, values: Tensor, mask: Tensor | None = None) -> Tensor:
        """Reduce `values` over each region of `mask` (or the bound mask).

        The `nn.Module` call form. Returns the per-region reduction: `(...)` for a
        single-region mask (None or a boolean 2D image), else `(..., R)`, in the
        input's dtype. Empty regions hold `empty`.

        Raises:
            ValueError: If `values` has fewer than two axes, or the mask is
                malformed (see `region_stack`).
        """
        _validate_ndim(values)
        mask = self.mask if mask is None else mask
        regions = region_stack(mask, tuple(values.shape[-2:]), device=values.device)
        result = self._reduce(values, regions)

        empties = regions.sum(dim=(-2, -1)) == 0
        result = torch.where(empties, self.empty, result)
        if _single_region(mask):
            result = result[..., 0]
        return result.to(values.dtype)

    @abstractmethod
    def _reduce(self, values: Tensor, regions: Tensor) -> Tensor:
        """Combine `values` over each region of `regions` `(R, H, W)`, in float64.

        Returns `(..., R)`; the empty-region fill and the cast back to the input
        dtype are applied by `forward`, so an implementation need not special-case
        a zero-pixel region.
        """
        raise NotImplementedError


class MomentReduction(MaskedReduction):
    """A `MaskedReduction` expressed through per-region power sums.

    The signed sum `sum(x)` masked-sums each region in float64 (out-of-region
    pixels replaced by 0, so a non-finite background does not poison it); the
    absolute power sum `sum(|x| ** p)` forms `|x| ** p` in float64 first, so
    `Variance` / `Std` stay accurate for a small spread over a large offset. Also
    supplies the region pixel count; `Sum`, `Mean`, `Norm`, `Variance`, and `Std`
    compose from these.
    """

    @staticmethod
    def _count(regions: Tensor) -> Tensor:
        return regions.sum(dim=(-2, -1), dtype=torch.float64)

    @staticmethod
    def _sum(values: Tensor, regions: Tensor) -> Tensor:
        return _masked_sum(values, regions)

    @staticmethod
    def _abs_power_sum(values: Tensor, regions: Tensor, p: float) -> Tensor:
        return _masked_sum(values.double().abs() ** p, regions)


class Sum(MomentReduction):
    """Sum of the values in each region."""

    @override
    def _reduce(self, values: Tensor, regions: Tensor) -> Tensor:
        return self._sum(values, regions)


class Mean(MomentReduction):
    """Mean of the values in each region."""

    @override
    def _reduce(self, values: Tensor, regions: Tensor) -> Tensor:
        return self._sum(values, regions) / self._count(regions)


class Norm(MomentReduction):
    """The p-norm of the values in each region (`p` positive, default 2 = L2)."""

    _p: float

    def __init__(
        self, p: float = 2.0, *, mask: Tensor | None = None, empty: float = math.nan
    ) -> None:
        super().__init__(mask, empty=empty)
        if not p > 0:
            msg = f"p must be positive (got {p})"
            raise ValueError(msg)
        self._p = float(p)

    @override
    def _reduce(self, values: Tensor, regions: Tensor) -> Tensor:
        return self._abs_power_sum(values, regions, self._p) ** (1.0 / self._p)


class Variance(MomentReduction):
    """Variance of the values in each region (`ddof` 0 = population, 1 = sample)."""

    _ddof: int

    def __init__(
        self, ddof: int = 0, *, mask: Tensor | None = None, empty: float = math.nan
    ) -> None:
        super().__init__(mask, empty=empty)
        if ddof not in (0, 1):
            msg = f"ddof must be 0 or 1 (got {ddof})"
            raise ValueError(msg)
        self._ddof = ddof

    @override
    def _reduce(self, values: Tensor, regions: Tensor) -> Tensor:
        n = self._count(regions)
        mean = self._sum(values, regions) / n
        var = self._abs_power_sum(values, regions, 2.0) / n - mean * mean
        if self._ddof:
            var = var * (n / (n - self._ddof))
        # clamp a hair-negative one-pass result to 0, but keep NaN (var < 0 is
        # False for NaN, so an empty / single-pixel-sample region stays NaN)
        return torch.where(var < 0, 0.0, var)


class Std(Variance):
    """Standard deviation of the values in each region."""

    @override
    def _reduce(self, values: Tensor, regions: Tensor) -> Tensor:
        return super()._reduce(values, regions).sqrt()
