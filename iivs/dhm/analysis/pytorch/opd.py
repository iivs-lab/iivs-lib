from __future__ import annotations

__all__ = ("opd_to_phase", "phase_to_opd")

from typing import TYPE_CHECKING

from iivs.dhm.analysis.opd import OPDConverter
from iivs.dhm.data.constants import DEFAULT_WAVELENGTH

if TYPE_CHECKING:
    import torch


def phase_to_opd(
    phase: torch.Tensor, *, wavelength: float = DEFAULT_WAVELENGTH
) -> torch.Tensor:
    """Convert phase (rad) to OPD (nm) as a `torch.Tensor`.

    The torch twin of `iivs.dhm.analysis.opd.phase_to_opd`: multiplies by the
    same cached `OPDConverter.opd_scale` (a plain float), so the result keeps the
    input tensor's dtype and device and stays in the autograd graph. For repeated
    use, read `OPDConverter(wavelength=...).opd_scale` once and multiply directly.

    Args:
        phase: Phase image (or batch), in rad.
        wavelength: Illumination wavelength, in m.
    """
    return phase * OPDConverter(wavelength=wavelength).opd_scale


def opd_to_phase(
    opd: torch.Tensor, *, wavelength: float = DEFAULT_WAVELENGTH
) -> torch.Tensor:
    """Convert OPD (nm) to phase (rad) as a `torch.Tensor`.

    The inverse of `phase_to_opd`; preserves the input tensor's dtype, device,
    and autograd graph.

    Args:
        opd: OPD image (or batch), in nm.
        wavelength: Illumination wavelength, in m.
    """
    return opd / OPDConverter(wavelength=wavelength).opd_scale
