from __future__ import annotations

__all__ = ("ProjectedAreaCalculator", "calc_projected_area")

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from iivs.common.data.reduction import Sum, apply_mask
from iivs.dhm.constants import PIXEL_SIZE_20X

if TYPE_CHECKING:
    from typing import Self

    from numpy.typing import NDArray

    from iivs.common.data.reduction import MaskLike


@dataclass(frozen=True, slots=True)
class ProjectedAreaCalculator:
    """An image-and-mask projected-area (um^2) calculator at a fixed pixel size.

    Bind the pixel size once, then measure images repeatedly::

        pac = ProjectedAreaCalculator(pixel_size=px)
        area = pac.calc(image, mask=cell)  # um^2 of the masked footprint

    Projected area is ``pixel_count * pixel_size**2``: the footprint the selected
    region(s) cover in the image plane. It enters the volume relation as ``volume =
    area * mean(height)`` (`OpticalVolumeCalculator`). The call shape matches the
    volume / dry-mass engines (`image`, `mask`, `reduce`), but unlike their
    integrals the image's *values* never enter the area: `image` fixes the pixel
    grid (and any leading batch axes), and each selected pixel contributes the
    constant `area_scale`.

    Attributes:
        pixel_size: Physical size of one (square) pixel, in m. Defaults to the
            lab's 20X objective (`PIXEL_SIZE_20X`).
    """

    pixel_size: float = PIXEL_SIZE_20X
    _scale: float = field(init=False, repr=False, compare=False)  # um^2 per pixel
    # the region summation engine; empty region -> 0 area:
    _sum: Sum = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the pixel size and cache the per-pixel area (um^2)."""
        if self.pixel_size <= 0:
            msg = f"pixel_size must be positive (got {self.pixel_size})"
            raise ValueError(msg)
        object.__setattr__(self, "_scale", (self.pixel_size * 1e6) ** 2)
        object.__setattr__(self, "_sum", Sum(empty=0.0))

    @classmethod
    def from_pixel_size_um(cls, pixel_size_um: float) -> Self:
        """Build a calculator from a pixel size given in um."""
        return cls(pixel_size=pixel_size_um * 1e-6)

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

    def calc(
        self,
        image: NDArray[np.generic],
        *,
        mask: MaskLike | None = None,
        reduce: bool = True,
    ) -> NDArray[np.float32]:
        """Measure the projected area [um^2] over `image`'s last two axes (H, W).

        Args:
            image: The map(s) fixing the pixel grid, shape ``(..., H, W)``, of any
                dtype (a uint8 preview serves as well as a float32 map): the values
                never enter the area, so only the shape (and the mask) matter.
            mask: Optional mask selecting the region(s) to measure: a boolean
                ``(H, W)`` (one region) or ``(N, H, W)`` (`N` regions, which may
                overlap), or an integer label image ``(H, W)`` (0 = background, one
                region per positive label); None (default) measures the whole
                frame. A boolean ``(H, W)`` (or None) keeps the plain shape; the
                multi-region forms add a trailing region axis.
            reduce: If True (default), count each region up into its area, shape
                ``(...)`` (or ``(..., R)`` for a multi-region mask). If False,
                return the per-pixel area-density map instead (`area_scale` inside
                a region, 0 outside; summing back to the area), shape
                ``(..., H, W)`` (or ``(..., R, H, W)``).

        Raises:
            ValueError: If `image` is not at least 2-D ``(..., H, W)``, or the mask
                is malformed (see `region_stack`: a wrong ``(H, W)``, a boolean
                mask not 2-D or 3-D, a label mask not 2-D or holding a negative
                label, or a non-boolean / non-integer dtype).
        """
        if image.ndim < 2:
            msg = f"image must be at least 2D (..., H, W) (got {image.ndim}D)"
            raise ValueError(msg)

        # each pixel contributes the constant per-pixel area, whatever its value
        density = np.broadcast_to(np.float32(self._scale), image.shape)
        region_op = self._sum if reduce else apply_mask
        result = region_op(density, mask)

        return result.astype(np.float32, copy=False)


def calc_projected_area(
    image: NDArray[np.generic],
    *,
    pixel_size: float = PIXEL_SIZE_20X,
    mask: MaskLike | None = None,
    reduce: bool = True,
) -> NDArray[np.float32]:
    """Measure the projected area [um^2] over `image`'s last two axes (H, W).

    A one-shot `ProjectedAreaCalculator`; for repeated measurements at one pixel
    size, reuse the calculator.

    Args:
        image: The map(s) fixing the pixel grid, shape ``(..., H, W)``, of any
            dtype; its values never enter the area.
        pixel_size: Physical size of one (square) pixel, in m. Defaults to the
            lab's 20X objective.
        mask: Optional region mask (boolean or integer labels); see
            `ProjectedAreaCalculator.calc`.
        reduce: Count each region up into its area (True), or return the per-pixel
            area-density map (False). See `ProjectedAreaCalculator.calc`.

    Returns:
        The projected area in um^2, shape ``(...)`` (or ``(..., R)``); or the
        unreduced density map when `reduce` is False.
    """
    return ProjectedAreaCalculator(pixel_size=pixel_size).calc(
        image, mask=mask, reduce=reduce
    )
