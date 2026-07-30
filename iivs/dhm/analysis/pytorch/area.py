from __future__ import annotations

__all__ = ("ProjectedArea", "calc_projected_area")

from typing import TYPE_CHECKING

import torch
from torch import nn

from iivs.common.data.pytorch.reduction import Sum, apply_mask
from iivs.dhm.analysis.area import ProjectedAreaCalculator

if TYPE_CHECKING:
    from typing import Self

    from torch import Tensor


class ProjectedArea(nn.Module):
    """Torch `nn.Module` for the per-pixel projected-area density (um^2).

    The torch twin of `iivs.dhm.analysis.area.ProjectedAreaCalculator`. `forward(image)`
    is the per-pixel area density: the constant `area_scale` (um^2 per pixel) over
    `image`'s grid, whatever `image`'s values. Unlike `OpticalVolume` / `OpticalHeight`,
    the density does not depend on `image`'s values, so no autograd graph flows from
    `image` to the result; it exists as a module so `OpticalVolume` can own it as a
    submodule (``volume = area * mean(height)``), mirroring the NumPy composition.
    Masking into regions and reducing to a total area are a separate concern: compose
    with the `iivs.common.data.pytorch` reductions, e.g.
    ``Sum(mask=cell)(ProjectedArea(pixel_size=px)(image))``, or use the
    `calc_projected_area` one-shot. `area_scale` (``pixel_size**2`` in um^2) is reused
    from the NumPy `ProjectedAreaCalculator`.

    Attributes:
        pixel_size: Physical size of one (square) pixel, in m.
        area_scale: um^2 of projected area per pixel.
    """

    def __init__(self, *, pixel_size: float) -> None:
        """Bind the pixel size and cache the per-pixel area (reused from NumPy)."""
        super().__init__()

        calculator = ProjectedAreaCalculator(pixel_size=pixel_size)
        self.pixel_size = calculator.pixel_size
        self.area_scale = calculator.area_scale

    @classmethod
    def from_pixel_size_um(cls, pixel_size_um: float) -> Self:
        """Build a module from a pixel size given in um."""
        return cls(pixel_size=pixel_size_um * 1e-6)

    @property
    def pixel_size_um(self) -> float:
        """The pixel size in um."""
        return self.pixel_size * 1e6

    def forward(self, image: Tensor) -> Tensor:
        """The per-pixel area density (um^2): `area_scale` over `image`'s grid.

        `image` fixes only the grid shape and device; its values never enter, so the
        result carries no gradient back to `image`.
        """
        return torch.full(
            image.shape, self.area_scale, dtype=torch.float32, device=image.device
        )


def calc_projected_area(
    image: Tensor,
    *,
    pixel_size: float,
    mask: Tensor | None = None,
    reduce: bool = True,
) -> Tensor:
    """Measure the projected area [um^2] over `image`'s last two axes (H, W).

    Composes `ProjectedArea` (per-pixel density) with the `iivs.common.data.pytorch`
    reductions. The area is not a pointwise map of the image's values (each selected
    pixel contributes the constant per-pixel scale), so no autograd graph flows from
    `image` to the result; the output is a float32 tensor on `image`'s device.

    Args:
        image: The map(s) fixing the pixel grid, shape ``(..., H, W)``, on any
            device; its values never enter the area.
        pixel_size: Physical size of one (square) pixel, in m.
        mask: Optional region mask (boolean or integer labels); None (default)
            measures the whole frame. See `iivs.common.data.pytorch.region_stack`.
        reduce: If True (default), count each region up into its area; if False,
            return the per-pixel area-density map (the per-pixel scale inside a
            region, 0 outside).

    Raises:
        ValueError: If `image` is not at least 2-D ``(..., H, W)``, or the mask is
            malformed (see `region_stack`).
    """
    if image.ndim < 2:
        msg = f"image must be at least 2D (..., H, W) (got {image.ndim}D)"
        raise ValueError(msg)

    density = ProjectedArea(pixel_size=pixel_size)(image)
    # empty region -> 0 area, matching the NumPy `ProjectedAreaCalculator`
    return Sum(empty=0.0)(density, mask) if reduce else apply_mask(density, mask)
