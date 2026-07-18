from __future__ import annotations

__all__ = ("OpticalPathDifference", "opd_to_phase", "phase_to_opd")

from typing import TYPE_CHECKING

from torch import nn

from iivs.dhm.analysis.opd import OPDConverter
from iivs.dhm.constants import DEFAULT_WAVELENGTH

if TYPE_CHECKING:
    from typing import Self

    from torch import Tensor


class OpticalPathDifference(nn.Module):
    """Torch `nn.Module` for the OPD <-> phase relation at a fixed wavelength.

    The torch twin of `iivs.dhm.analysis.opd.OPDConverter` (named for the quantity, per
    the `nn.Module` convention). Binds a wavelength and the cached `opd_scale` (nm of
    OPD per rad, a plain float reused from the NumPy engine, so the physics is shared);
    `from_wavelength_nm` / `wavelength_nm` give its nm form. The `convert_*` / `forward`
    methods are pure scalar multiplies, so they preserve the input tensor's dtype,
    device, and autograd graph.

    Tensors are expected to hold a real floating dtype (e.g. ``float32``); the output
    keeps the input's dtype (torch's `Tensor` type does not carry it).

    Attributes:
        wavelength: Illumination wavelength, in m.
        opd_scale: nm of OPD per rad of phase (a plain float).
    """

    def __init__(self, wavelength: float = DEFAULT_WAVELENGTH) -> None:
        super().__init__()

        converter = OPDConverter(wavelength=wavelength)
        self.wavelength = converter.wavelength
        self.opd_scale = converter.opd_scale

    @classmethod
    def from_wavelength_nm(cls, wavelength_nm: float) -> Self:
        """Build an `OpticalPathDifference` from a wavelength given in nm."""
        return cls(wavelength=wavelength_nm * 1e-9)

    @property
    def wavelength_nm(self) -> float:
        """The wavelength in nm."""
        return self.wavelength * 1e9

    def convert_to_opd(self, phase: Tensor) -> Tensor:
        """Convert `phase` (rad) to OPD (nm)."""
        return phase * self.opd_scale

    def convert_to_phase(self, opd: Tensor) -> Tensor:
        """Convert `opd` (nm) to phase (rad)."""
        return opd / self.opd_scale

    def forward(self, phase: Tensor) -> Tensor:
        """Convert `phase` (rad) to OPD (nm); the `nn.Module` call form."""
        return self.convert_to_opd(phase)


def phase_to_opd(phase: Tensor, *, wavelength: float = DEFAULT_WAVELENGTH) -> Tensor:
    """Convert phase (rad) to OPD (nm); a one-shot `OpticalPathDifference`.

    Preserves the input tensor's dtype, device, and autograd graph. For repeated use,
    build an `OpticalPathDifference` (or read its `opd_scale`) once.

    Args:
        phase: Phase image (or batch), in rad.
        wavelength: Illumination wavelength, in m.
    """
    return OpticalPathDifference(wavelength=wavelength).convert_to_opd(phase)


def opd_to_phase(opd: Tensor, *, wavelength: float = DEFAULT_WAVELENGTH) -> Tensor:
    """Convert OPD (nm) to phase (rad); a one-shot `OpticalPathDifference`.

    The inverse of `phase_to_opd`; preserves dtype, device, and the autograd graph. For
    repeated use, build an `OpticalPathDifference` (or read its `opd_scale`) once.

    Args:
        opd: OPD image (or batch), in nm.
        wavelength: Illumination wavelength, in m.
    """
    return OpticalPathDifference(wavelength=wavelength).convert_to_phase(opd)
