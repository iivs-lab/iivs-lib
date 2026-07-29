from __future__ import annotations

__all__ = ("calc_projected_area",)

from typing import TYPE_CHECKING

import torch

from iivs.common.data.pytorch.reduction import region_stack
from iivs.dhm.analysis.area import ProjectedAreaCalculator

if TYPE_CHECKING:
    from torch import Tensor


def calc_projected_area(mask: Tensor, *, pixel_size: float) -> Tensor:
    """Measure the projected area [um^2] of each region of `mask`.

    The torch twin of `iivs.dhm.analysis.calc_projected_area`. Counting mask pixels
    is not a pointwise map, so there is no `nn.Module` here (cf. `OpticalHeight` /
    `OpticalVolume`); the per-pixel scale is reused from the NumPy
    `ProjectedAreaCalculator`. The result is a float32 tensor on `mask`'s device (a
    mask carries no autograd graph to preserve).

    Args:
        mask: The region(s) to measure, on any device: a boolean ``(H, W)`` (one
            region) or ``(N, H, W)`` (`N` regions, which may overlap), or an integer
            label image ``(H, W)`` (0 = background, one region per positive label).
            A boolean ``(H, W)`` gives a 0-D tensor; the multi-region forms give one
            area per region, shape ``(R,)``.
        pixel_size: Physical size of one (square) pixel, in m.

    Raises:
        ValueError: If the mask is malformed (see `region_stack`).
    """
    scale = ProjectedAreaCalculator(pixel_size=pixel_size).area_scale
    regions = region_stack(mask, (int(mask.shape[-2]), int(mask.shape[-1])))
    counts = regions.sum(dim=(-2, -1), dtype=torch.int64)

    areas = counts.to(torch.float32) * scale
    if mask.ndim == 2 and mask.dtype == torch.bool:
        return areas.squeeze(0)  # single-region form drops the region axis
    return areas
