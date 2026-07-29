from __future__ import annotations

__all__ = ("OpticalVolume", "calc_volume", "calc_volume_from_phase")

from typing import TYPE_CHECKING

from torch import nn

from iivs.common.data.pytorch.reduction import Sum, apply_mask
from iivs.dhm.analysis.pytorch.height import OpticalHeight
from iivs.dhm.analysis.volume import OpticalVolumeCalculator
from iivs.dhm.constants import DEFAULT_REFRACTIVE_DELTA, DEFAULT_WAVELENGTH

if TYPE_CHECKING:
    from torch import Tensor


class OpticalVolume(nn.Module):
    """Torch `nn.Module` for the per-pixel volume density (um^3) from OPD.

    A pure pointwise layer: `forward(opd) = opd * volume_scale`, the volume density
    (um^3 per pixel) of a background-corrected OPD map (nm). It preserves shape,
    dtype, device, and the autograd graph, so it drops cleanly into
    `nn.Sequential`, forward hooks, `torch.jit.script`, and `torch.compile`.
    Masking into regions and reducing to a total volume are a separate concern:
    compose with the `iivs.common.data.pytorch` reductions, e.g.
    `Sum(mask=cell)(OpticalVolume(pixel_size=px)(opd))`, or use the `calc_volume`
    one-shot. The scale (``pixel_area / refractive_delta``, no wavelength) is
    reused from the NumPy `OpticalVolumeCalculator`; for the phase path, precede it
    with an `OpticalPathDifference`.

    Attributes:
        pixel_size: Physical size of one (square) pixel, in m.
        refractive_delta: Refractive-index difference ``n_object - n_medium``.
        volume_scale: um^3 of volume per nm of OPD, per pixel.
    """

    def __init__(
        self,
        *,
        pixel_size: float,
        refractive_delta: float = DEFAULT_REFRACTIVE_DELTA,
    ) -> None:
        super().__init__()

        calculator = OpticalVolumeCalculator.from_wavelength(
            pixel_size=pixel_size, refractive_delta=refractive_delta
        )
        self.pixel_size = calculator.pixel_size
        self.refractive_delta = calculator.refractive_delta
        self.volume_scale = calculator.volume_scale

    def forward(self, opd: Tensor) -> Tensor:
        """Map an OPD map (nm) to its volume density (um^3 per pixel)."""
        return opd * self.volume_scale


def calc_volume(
    opd: Tensor,
    *,
    pixel_size: float,
    refractive_delta: float = DEFAULT_REFRACTIVE_DELTA,
    mask: Tensor | None = None,
    reduce: bool = True,
) -> Tensor:
    """Integrate an OPD map (nm) into optical volume [um^3].

    Composes `OpticalVolume` (per-pixel density) with the
    `iivs.common.data.pytorch` reductions, keeping `opd`'s device and autograd
    graph.

    Args:
        opd: OPD map(s), in nm, shape ``(..., H, W)``, already background-corrected.
        pixel_size: Physical size of one (square) pixel, in m.
        refractive_delta: Refractive-index difference ``n_object - n_medium``.
        mask: Optional region mask (boolean or integer labels); see
            `iivs.common.data.pytorch.region_stack`.
        reduce: If True (default), sum each masked region to a volume; if False,
            return the masked per-pixel density map.
    """
    density = OpticalVolume(pixel_size=pixel_size, refractive_delta=refractive_delta)(
        opd
    )
    # empty region -> 0 volume, matching the NumPy `OpticalVolumeCalculator`
    return Sum(empty=0.0)(density, mask) if reduce else apply_mask(density, mask)


def calc_volume_from_phase(
    phase: Tensor,
    *,
    pixel_size: float,
    wavelength: float = DEFAULT_WAVELENGTH,
    refractive_delta: float = DEFAULT_REFRACTIVE_DELTA,
    mask: Tensor | None = None,
    reduce: bool = True,
) -> Tensor:
    """Integrate a phase map (rad) into optical volume [um^3] at `wavelength`.

    Converts `phase` to height (via `OpticalHeight`), then integrates its
    ``pixel_area`` multiple as `calc_volume` would; keeps the input's device and
    autograd graph.

    Args:
        phase: Phase map(s), in rad, shape ``(..., H, W)``, already
            background-corrected.
        pixel_size: Physical size of one (square) pixel, in m.
        wavelength: Illumination wavelength, in m.
        refractive_delta: Refractive-index difference ``n_object - n_medium``.
        mask: Optional region mask (boolean or integer labels).
        reduce: Sum each masked region to a volume (True), or return the masked
            per-pixel density map (False).
    """
    height = OpticalHeight(refractive_delta, wavelength=wavelength).convert_from_phase(
        phase
    )
    density = height * ((pixel_size * 1e6) ** 2 * 1e-3)  # nm of height -> um^3

    return Sum(empty=0.0)(density, mask) if reduce else apply_mask(density, mask)
