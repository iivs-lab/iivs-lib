from __future__ import annotations

__all__ = ("DryMass", "calc_drymass", "calc_drymass_from_phase")

from typing import TYPE_CHECKING

from torch import nn

from iivs.common.data.pytorch.reduction import Sum, apply_mask
from iivs.dhm.analysis.drymass import DryMassCalculator
from iivs.dhm.analysis.pytorch.opd import OpticalPathDifference
from iivs.dhm.constants import (
    DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    DEFAULT_WAVELENGTH,
)

if TYPE_CHECKING:
    from torch import Tensor


class DryMass(nn.Module):
    """Torch `nn.Module` for the per-pixel dry-mass density (pg) from OPD.

    A pure pointwise layer: `forward(opd) = opd * drymass_scale`, the dry-mass
    density (pg per pixel) of a background-corrected OPD map (nm). It preserves
    shape, dtype, device, and the autograd graph, so it drops cleanly into
    `nn.Sequential`, forward hooks, `torch.jit.script`, and `torch.compile`.
    Masking into regions and reducing to a total dry mass are a separate concern:
    compose with the `iivs.common.data.pytorch` reductions, e.g.
    `Sum(mask=cell)(DryMass(pixel_size=px)(opd))`, or use the `calc_drymass`
    one-shot. The scale (`pixel_size**2 / alpha`, no wavelength) is reused from the
    NumPy `DryMassCalculator`; for the phase path, precede it with an
    `OpticalPathDifference`.

    Attributes:
        pixel_size: Physical size of one (square) pixel, in m.
        alpha: Specific refractive increment, in m^3/kg.
        drymass_scale: pg of dry mass per nm of OPD, per pixel (a plain float).
    """

    def __init__(
        self,
        *,
        pixel_size: float,
        alpha: float = DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    ) -> None:
        super().__init__()
        calculator = DryMassCalculator(pixel_size=pixel_size, alpha=alpha)
        self.pixel_size = calculator.pixel_size
        self.alpha = calculator.alpha
        self.drymass_scale = calculator.drymass_scale

    def forward(self, opd: Tensor) -> Tensor:
        """Map an OPD map (nm) to its dry-mass density (pg per pixel): `opd * scale`."""
        return opd * self.drymass_scale


def calc_drymass(
    opd: Tensor,
    *,
    pixel_size: float,
    alpha: float = DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    mask: Tensor | None = None,
    reduce: bool = True,
) -> Tensor:
    """Integrate an OPD map (nm) into dry mass [pg].

    Composes `DryMass` (per-pixel density) with the `iivs.common.data.pytorch`
    reductions, keeping `opd`'s device and autograd graph.

    Args:
        opd: OPD map(s), in nm, shape ``(..., H, W)``, already background-corrected.
        pixel_size: Physical size of one (square) pixel, in m.
        alpha: Specific refractive increment, in m^3/kg.
        mask: Optional region mask (boolean or integer labels); see
            `iivs.common.data.pytorch.region_stack`.
        reduce: If True (default), sum each masked region to a dry mass; if False,
            return the masked per-pixel density map.
    """
    density = DryMass(pixel_size=pixel_size, alpha=alpha)(opd)
    return Sum()(density, mask) if reduce else apply_mask(density, mask)


def calc_drymass_from_phase(
    phase: Tensor,
    *,
    pixel_size: float,
    wavelength: float = DEFAULT_WAVELENGTH,
    alpha: float = DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    mask: Tensor | None = None,
    reduce: bool = True,
) -> Tensor:
    """Integrate a phase map (rad) into dry mass [pg] at `wavelength`.

    Converts `phase` to OPD (via `OpticalPathDifference`), then integrates as
    `calc_drymass`; keeps the input's device and autograd graph.

    Args:
        phase: Phase map(s), in rad, shape ``(..., H, W)``, already
            background-corrected.
        pixel_size: Physical size of one (square) pixel, in m.
        wavelength: Illumination wavelength, in m.
        alpha: Specific refractive increment, in m^3/kg.
        mask: Optional region mask (boolean or integer labels).
        reduce: Sum each masked region to a dry mass (True), or return the masked
            per-pixel density map (False).
    """
    opd = OpticalPathDifference(wavelength=wavelength)(phase)
    return calc_drymass(
        opd, pixel_size=pixel_size, alpha=alpha, mask=mask, reduce=reduce
    )
