from __future__ import annotations

__all__ = (
    "MaskLike",
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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast, override

import numpy as np

from iivs.common.data.validation import validate_ndim

if TYPE_CHECKING:
    from numpy.typing import NDArray

type MaskLike = NDArray[np.bool_] | NDArray[np.integer]


def region_stack(
    mask: MaskLike | None,
    shape: tuple[int, int],
) -> NDArray[np.bool_]:
    """Normalize any mask to a `(R, H, W)` boolean region stack.

    The single form every reduction works on. `None` is one region spanning the
    whole `shape`; a boolean `(H, W)` is one region and a boolean `(N, H, W)` is
    `N` regions (which may overlap); an integer label image `(H, W)` (0 =
    background) becomes one layer per positive label, so `R == labels.max()` (0
    if the image is all background).

    Args:
        mask: The mask to normalize, or `None` for the whole frame.
        shape: The `(H, W)` the mask must cover.

    Raises:
        ValueError: If the mask's `(H, W)` differs from `shape`, a boolean mask
            is not 2D or 3D, a label mask is not 2D or holds a negative label, or
            the dtype is neither boolean nor integer.
    """
    if mask is None:
        return np.ones((1, *shape), dtype=np.bool_)

    if mask.shape[-2:] != shape:
        got = tuple(mask.shape[-2:])
        msg = f"mask (H, W) must be {shape} (got {got})"
        raise ValueError(msg)

    if np.issubdtype(mask.dtype, np.bool_):
        bmask = cast("NDArray[np.bool_]", mask)
        if bmask.ndim == 2:
            return bmask[np.newaxis]
        if bmask.ndim == 3:
            return bmask
        msg = f"a boolean mask must be (H, W) or (N, H, W) (got {bmask.ndim}D)"
        raise ValueError(msg)

    if np.issubdtype(mask.dtype, np.integer):
        if mask.ndim != 2:
            msg = f"a label mask must be (H, W) (got {mask.ndim}D)"
            raise ValueError(msg)
        if mask.min() < 0:
            msg = f"label values must be non-negative (got {int(mask.min())})"
            raise ValueError(msg)
        labels = np.arange(1, int(mask.max()) + 1)
        return labels[:, np.newaxis, np.newaxis] == mask[np.newaxis]

    msg = f"mask must be boolean or integer (got {mask.dtype})"
    raise ValueError(msg)


def _single_region(mask: MaskLike | None) -> bool:
    """Test whether `mask` denotes one region (None or a boolean 2D image)."""
    return mask is None or (mask.ndim == 2 and np.issubdtype(mask.dtype, np.bool_))


def apply_mask(
    values: NDArray[np.floating],
    mask: MaskLike | None = None,
) -> NDArray[np.floating]:
    """Split `values` into per-region masked maps.

    In each region's layer every pixel outside that region is zeroed. The pointwise
    companion to the reductions (which instead collapse each region to a scalar). A
    single-region mask (None or a boolean 2D image) gives `(..., H, W)`; a stack or
    label image adds a region axis, giving `(..., R, H, W)`. Overlapping regions
    each keep the shared pixels.

    Raises:
        ValueError: If `values` has fewer than two axes, or the mask is malformed
            (see `region_stack`).
    """
    values = validate_ndim(np.asarray(values), ndim=2)
    regions = region_stack(mask, values.shape[-2:])
    maps = values[..., np.newaxis, :, :] * regions
    return maps[..., 0, :, :] if _single_region(mask) else maps


class MaskedReduction(ABC):
    """An abstract reduction of a map over masked regions of its last two axes.

    A `(..., H, W)` map reduces over the `R` regions of a mask (see `region_stack`
    for the accepted forms; regions may overlap). A single-region mask (None or a
    boolean 2D image) drops the region axis, giving `(...)`; a stack or label image
    keeps it, giving `(..., R)`. A mask bound at construction is the default; a mask
    passed to the call overrides it. A region with no pixels reduces to `empty` (NaN
    by default; pass `empty=0.0` for a benign fill). Subclasses implement `_reduce`;
    `MomentReduction` supplies the per-region power sums the statistical reductions
    share.

    Args:
        mask: Default mask; the call's `mask` overrides it. `None` treats the
            whole frame as one region.
        empty: The value a region with no pixels reduces to.
    """

    def __init__(
        self,
        mask: MaskLike | None = None,
        *,
        empty: float = np.nan,
    ) -> None:
        self._mask = mask
        self._empty = float(empty)

    def __call__(
        self,
        values: NDArray[np.floating],
        mask: MaskLike | None = None,
    ) -> NDArray[np.float64]:
        """Reduce `values` over each region of `mask` (or the bound mask).

        Args:
            values: The map(s) to reduce, shape `(..., H, W)`.
            mask: Mask for this call, overriding the bound one; `None` uses the
                bound mask (or the whole frame if none was bound).

        Returns:
            The per-region reduction in float64: `(...)` for a single-region mask
            (None or a boolean 2D image), else `(..., R)`. Empty regions hold
            `empty`.

        Raises:
            ValueError: If `values` has fewer than two axes, or the mask is
                malformed (see `region_stack`).
        """
        values = validate_ndim(np.asarray(values), ndim=2)
        mask = self._mask if mask is None else mask
        regions = region_stack(mask, values.shape[-2:])
        result = self._reduce(values, regions)

        empties = regions.sum(axis=(-2, -1)) == 0
        result = np.where(empties, self._empty, result)
        return result[..., 0] if _single_region(mask) else result

    @abstractmethod
    def _reduce(
        self, values: NDArray[np.floating], regions: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        """Combine `values` over each region of `regions` `(R, H, W)`.

        Returns `(..., R)`; the empty-region fill is applied by `__call__`, so an
        implementation need not special-case a zero-pixel region.
        """
        raise NotImplementedError


class MomentReduction(MaskedReduction):
    """A `MaskedReduction` expressed through per-region power sums.

    The signed sum `sum(x)` masked-sums each region in float64 without copying the
    map; the absolute power sum `sum(|x| ** p)` forms `|x| ** p` in float64 first,
    so `Variance` / `Std` stay accurate for a small spread over a large offset.
    Also supplies the region pixel count. `Sum`, `Mean`, `Norm`, `Variance`, and
    `Std` compose from these.
    """

    @staticmethod
    def _count(regions: NDArray[np.bool_]) -> NDArray[np.float64]:
        return regions.sum(axis=(-2, -1), dtype=np.float64)

    @staticmethod
    def _sum(
        values: NDArray[np.floating], regions: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        out = np.empty((*values.shape[:-2], len(regions)), dtype=np.float64)
        for r, region in enumerate(regions):
            out[..., r] = np.sum(values, axis=(-2, -1), where=region, dtype=np.float64)
        return out

    @staticmethod
    def _abs_power_sum(
        values: NDArray[np.floating], regions: NDArray[np.bool_], p: float
    ) -> NDArray[np.float64]:
        powered = np.abs(values, dtype=np.float64) ** p  # float64 before the power
        out = np.empty((*values.shape[:-2], len(regions)), dtype=np.float64)
        for r, region in enumerate(regions):
            out[..., r] = np.sum(powered, axis=(-2, -1), where=region, dtype=np.float64)
        return out


class Sum(MomentReduction):
    """Sum of the values in each region."""

    @override
    def _reduce(
        self, values: NDArray[np.floating], regions: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        return self._sum(values, regions)


class Mean(MomentReduction):
    """Mean of the values in each region."""

    @override
    def _reduce(
        self, values: NDArray[np.floating], regions: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        with np.errstate(invalid="ignore"):
            return self._sum(values, regions) / self._count(regions)


class Norm(MomentReduction):
    """The p-norm of the values in each region (`p` positive, default 2 = L2)."""

    def __init__(
        self,
        p: float = 2.0,
        *,
        mask: MaskLike | None = None,
        empty: float = np.nan,
    ) -> None:
        super().__init__(mask, empty=empty)
        if not p > 0:
            msg = f"p must be positive (got {p})"
            raise ValueError(msg)
        self._p = float(p)

    @override
    def _reduce(
        self, values: NDArray[np.floating], regions: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        return self._abs_power_sum(values, regions, self._p) ** (1.0 / self._p)


class Variance(MomentReduction):
    """Variance of the values in each region (`ddof` 0 = population, 1 = sample)."""

    def __init__(
        self,
        ddof: int = 0,
        *,
        mask: MaskLike | None = None,
        empty: float = np.nan,
    ) -> None:
        super().__init__(mask, empty=empty)
        if ddof not in (0, 1):
            msg = f"ddof must be 0 or 1 (got {ddof})"
            raise ValueError(msg)
        self._ddof = ddof

    @override
    def _reduce(
        self, values: NDArray[np.floating], regions: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        n = self._count(regions)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = self._sum(values, regions) / n
            var = self._abs_power_sum(values, regions, 2.0) / n - mean * mean
            if self._ddof:
                var = var * (n / (n - self._ddof))
            # the one-pass formula can dip a hair below 0 on a near-constant region
            return np.maximum(var, 0.0)


class Std(Variance):
    """Standard deviation of the values in each region."""

    @override
    def _reduce(
        self, values: NDArray[np.floating], regions: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        return np.sqrt(super()._reduce(values, regions))
