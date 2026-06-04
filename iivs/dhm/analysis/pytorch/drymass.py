from __future__ import annotations

__all__ = ("calc_drymass", "calc_drymass_from_phase")

import torch

from iivs.dhm.analysis.drymass import DryMassCalculator
from iivs.dhm.data.constants import (
    DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    DEFAULT_WAVELENGTH,
)


def calc_drymass(
    opd: torch.Tensor,
    *,
    pixel_size: float,
    alpha: float = DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dry mass (pg) from an OPD map (nm) as a 0-dim `torch.Tensor`.

    The torch twin of `iivs.dhm.analysis.drymass.calc_drymass`: sums in float64
    and scales by the same cached `DryMassCalculator.drymass_scale`. Returns a
    0-dim tensor -- never a Python `float` -- so it keeps `opd`'s device and the
    autograd graph (a `float()` cast would sync off-device and drop gradients).

    Args:
        opd: OPD map (or batch), in nm, already background-corrected.
        pixel_size: Physical size of one (square) pixel, in m.
        alpha: Specific refractive increment, in m^3/kg.
        mask: Optional boolean tensor selecting the object's pixels.
    """
    scale = DryMassCalculator(pixel_size=pixel_size, alpha=alpha).drymass_scale
    selected = opd if mask is None else opd[mask]
    return selected.sum(dtype=torch.float64) * scale


def calc_drymass_from_phase(
    phase: torch.Tensor,
    *,
    pixel_size: float,
    wavelength: float = DEFAULT_WAVELENGTH,
    alpha: float = DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dry mass (pg) from a phase map (rad) as a 0-dim `torch.Tensor`.

    Converts `phase` to OPD at `wavelength`, then integrates as `calc_drymass`;
    preserves the input tensor's device and autograd graph.

    Args:
        phase: Phase map (or batch), in rad, already background-corrected.
        pixel_size: Physical size of one (square) pixel, in m.
        wavelength: Illumination wavelength, in m.
        alpha: Specific refractive increment, in m^3/kg.
        mask: Optional boolean tensor selecting the object's pixels.
    """
    calc = DryMassCalculator.from_wavelength(
        pixel_size=pixel_size, wavelength=wavelength, alpha=alpha
    )
    opd = phase * calc.opd_converter.opd_scale
    selected = opd if mask is None else opd[mask]
    return selected.sum(dtype=torch.float64) * calc.drymass_scale
