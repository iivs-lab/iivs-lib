from __future__ import annotations

__all__ = ("ProjectedAreaCalculator", "calc_projected_area")

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from iivs.common.data.reduction import region_stack

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from iivs.common.data.reduction import MaskLike


@dataclass(frozen=True, slots=True)
class ProjectedAreaCalculator:
    """A mask-to-projected-area (um^2) calculator at a fixed pixel size.

    Bind the pixel size once, then measure region masks repeatedly::

        pac = ProjectedAreaCalculator(pixel_size=px)
        area = pac.calc(cell_mask)  # um^2

    Projected area is ``pixel_count * pixel_size**2``: the footprint a segmented
    region covers in the image plane. The one quantity here computed from the mask
    alone; it enters the volume relation as ``volume = area * mean(height)``
    (`OpticalVolumeCalculator`).

    Attributes:
        pixel_size: Physical size of one (square) pixel, in m.
    """

    pixel_size: float
    _scale: float = field(init=False, repr=False, compare=False)  # um^2 per pixel

    def __post_init__(self) -> None:
        """Validate the pixel size and cache the per-pixel area (um^2)."""
        if self.pixel_size <= 0:
            msg = f"pixel_size must be positive (got {self.pixel_size})"
            raise ValueError(msg)
        object.__setattr__(self, "_scale", (self.pixel_size * 1e6) ** 2)

    @property
    def pixel_size_um(self) -> float:
        """The pixel size in um."""
        return self.pixel_size * 1e6

    @property
    def area_scale(self) -> float:
        """um^2 of projected area per pixel (``pixel_size**2`` in um^2).

        Projected area's canonical unit here is um^2, so this needs no suffix (cf.
        `pixel_size` vs `pixel_size_um`).
        """
        return self._scale

    def calc(self, mask: MaskLike) -> NDArray[np.float32]:
        """Measure the projected area [um^2] of each region of `mask`.

        Args:
            mask: The region(s) to measure: a boolean ``(H, W)`` (one region) or
                ``(N, H, W)`` (`N` regions, which may overlap), or an integer label
                image ``(H, W)`` (0 = background, one region per positive label). A
                boolean ``(H, W)`` gives a plain scalar; the multi-region forms give
                one area per region, shape ``(R,)``.

        Raises:
            ValueError: If the mask is malformed (see `region_stack`: a boolean mask
                not 2-D or 3-D, a label mask not 2-D or holding a negative label, or
                a non-boolean / non-integer dtype).
        """
        regions = region_stack(mask, mask.shape[-2:])
        counts = regions.sum(axis=(-2, -1), dtype=np.int64)

        areas = (counts * self._scale).astype(np.float32, copy=False)
        if mask.ndim == 2 and mask.dtype == np.bool_:
            return areas.squeeze(0)  # single-region form drops the region axis
        return areas


def calc_projected_area(
    mask: MaskLike,
    *,
    pixel_size: float,
) -> NDArray[np.float32]:
    """Measure the projected area [um^2] of each region of `mask`.

    A one-shot `ProjectedAreaCalculator`; for repeated masks at one pixel size,
    reuse the calculator.

    Args:
        mask: Region mask (boolean or integer labels); see
            `ProjectedAreaCalculator.calc`.
        pixel_size: Physical size of one (square) pixel, in m.

    Returns:
        The projected area in um^2: a scalar for a boolean ``(H, W)`` mask, else
        one area per region, shape ``(R,)``.
    """
    return ProjectedAreaCalculator(pixel_size=pixel_size).calc(mask)
