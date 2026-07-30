from __future__ import annotations

__all__ = ("MaskedRegionCalculator",)

from typing import TYPE_CHECKING, Any

from iivs.common.data.reduction import apply_mask

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from iivs.common.data.reduction import MaskLike, Sum


class MaskedRegionCalculator:
    """Shared base for the analysis calculators that reduce a map over regions.

    `ProjectedAreaCalculator`, `OpticalVolumeCalculator`, and `DryMassCalculator`
    each measure a quantity over the last two axes (H, W) of a batched map,
    optionally restricted to mask region(s). This base supplies the common
    ``(..., H, W)`` shape guard and the `mask` / `reduce` dispatch; each subclass
    binds its own per-pixel scale and `Sum` engine (`_sum`). It holds no fields, so
    a frozen slotted dataclass can inherit it without disturbing its own slots.
    """

    __slots__ = ()

    _sum: Sum  # each subclass binds a Sum(empty=0.0) engine in __post_init__

    @staticmethod
    def _require_2d(array: NDArray[np.generic], name: str) -> None:
        """Raise if `array` is not at least 2-D ``(..., H, W)``."""
        if array.ndim < 2:
            msg = f"{name} must be at least 2D (..., H, W) (got {array.ndim}D)"
            raise ValueError(msg)

    def _reduce(
        self, values: NDArray[np.float32], mask: MaskLike | None, *, reduce: bool
    ) -> NDArray[np.floating[Any]]:
        """Sum `values` over each `mask` region, or mask them when not `reduce`.

        `reduce` True sums each region (empty region -> 0, via the bound `_sum`);
        False returns the per-region masked map. Both normalize the mask and drop
        the region axis for a single region. The caller casts back to float32.
        """
        return self._sum(values, mask) if reduce else apply_mask(values, mask)
