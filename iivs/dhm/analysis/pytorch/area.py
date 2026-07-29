from __future__ import annotations

__all__ = ("calc_projected_area",)

from typing import TYPE_CHECKING

import torch

from iivs.common.data.pytorch.reduction import Sum, apply_mask
from iivs.dhm.analysis.area import ProjectedAreaCalculator

if TYPE_CHECKING:
    from torch import Tensor


def calc_projected_area(
    image: Tensor,
    *,
    pixel_size: float,
    mask: Tensor | None = None,
    reduce: bool = True,
) -> Tensor:
    """Measure the projected area [um^2] over `image`'s last two axes (H, W).

    The torch twin of `iivs.dhm.analysis.calc_projected_area`. The area is not a
    pointwise map of the image's values (each selected pixel contributes the
    constant per-pixel scale, reused from the NumPy `ProjectedAreaCalculator`), so
    there is no `nn.Module` here (cf. `OpticalHeight` / `OpticalVolume`) and no
    autograd graph flows from `image` to the result; the output is a float32
    tensor on `image`'s device.

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

    scale = ProjectedAreaCalculator(pixel_size=pixel_size).area_scale
    density = torch.full(image.shape, scale, dtype=torch.float32, device=image.device)

    # empty region -> 0 area, matching the NumPy `ProjectedAreaCalculator`
    return Sum(empty=0.0)(density, mask) if reduce else apply_mask(density, mask)
