from __future__ import annotations

__all__ = ("OPDConverter", "opd_to_phase", "phase_to_opd")

from typing import TYPE_CHECKING

from torch import nn

from iivs.dhm.analysis.opd import OPDConverter as _NpOPDConverter
from iivs.dhm.data.constants import DEFAULT_WAVELENGTH

if TYPE_CHECKING:
    from torch import Tensor


class OPDConverter(nn.Module):
    """Torch `nn.Module` twin of `iivs.dhm.analysis.opd.OPDConverter`.

    Binds a wavelength and the cached `opd_scale` (nm of OPD per rad, a plain
    float reused from the NumPy engine, so the physics is shared). The
    `convert_*` / `forward` methods are pure scalar multiplies, so they preserve
    the input tensor's dtype, device, and autograd graph.

    Tensors are expected to hold a real floating dtype (e.g. ``float32``); the
    output keeps the input's dtype (torch's `Tensor` type does not carry it).

    Attributes:
        wavelength: Illumination wavelength, in m.
        opd_scale: nm of OPD per rad of phase (a plain float).
    """

    def __init__(self, wavelength: float = DEFAULT_WAVELENGTH) -> None:
        super().__init__()
        self.wavelength = wavelength
        self.opd_scale = _NpOPDConverter(wavelength=wavelength).opd_scale

    def convert_to_opd(self, phase: Tensor) -> Tensor:
        """Convert `phase` (rad) to OPD (nm)."""
        return phase * self.opd_scale

    def convert_to_phase(self, opd: Tensor) -> Tensor:
        """Convert `opd` (nm) to phase (rad)."""
        return opd / self.opd_scale

    def forward(self, phase: Tensor) -> Tensor:
        """Alias of `convert_to_opd` (phase -> OPD), so the module is callable."""
        return self.convert_to_opd(phase)


def phase_to_opd(phase: Tensor, *, wavelength: float = DEFAULT_WAVELENGTH) -> Tensor:
    """Convert phase (rad) to OPD (nm); a one-shot `OPDConverter.convert_to_opd`.

    Preserves the input tensor's dtype, device, and autograd graph. For repeated
    use, build an `OPDConverter` (or read its `opd_scale`) once.

    Args:
        phase: Phase image (or batch), in rad.
        wavelength: Illumination wavelength, in m.
    """
    return OPDConverter(wavelength=wavelength).convert_to_opd(phase)


def opd_to_phase(opd: Tensor, *, wavelength: float = DEFAULT_WAVELENGTH) -> Tensor:
    """Convert OPD (nm) to phase (rad); a one-shot `OPDConverter.convert_to_phase`.

    The inverse of `phase_to_opd`; preserves dtype, device, and the autograd
    graph.

    Args:
        opd: OPD image (or batch), in nm.
        wavelength: Illumination wavelength, in m.
    """
    return OPDConverter(wavelength=wavelength).convert_to_phase(opd)
