from __future__ import annotations

__all__ = ("OpticalHeight", "height_to_opd", "opd_to_height", "phase_to_height")

from typing import TYPE_CHECKING

from torch import nn

from iivs.dhm.analysis.height import OpticalHeightConverter
from iivs.dhm.constants import DEFAULT_REFRACTIVE_DELTA, DEFAULT_WAVELENGTH

if TYPE_CHECKING:
    from torch import Tensor


class OpticalHeight(nn.Module):
    """Torch `nn.Module` for the OPD <-> optical height relation at a fixed delta.

    The torch twin of `iivs.dhm.analysis.height.OpticalHeightConverter` (named for
    the quantity, per the `nn.Module` convention). Binds a refractive-index
    difference and, for the phase path, the cached `height_scale` (nm of height per
    rad, a plain float reused from the NumPy engine, so the physics is shared). The
    `convert_*` / `forward` methods are pure scalar multiplies, so they preserve the
    input tensor's dtype, device, and autograd graph.

    Attributes:
        refractive_delta: Refractive-index difference ``n_object - n_medium``.
        wavelength: Illumination wavelength for the phase path, in m.
        height_scale: nm of height per rad of phase (a plain float).
    """

    def __init__(
        self,
        refractive_delta: float = DEFAULT_REFRACTIVE_DELTA,
        *,
        wavelength: float = DEFAULT_WAVELENGTH,
    ) -> None:
        super().__init__()

        converter = OpticalHeightConverter.from_args(
            wavelength=wavelength, refractive_delta=refractive_delta
        )
        self.refractive_delta = converter.refractive_delta
        self.wavelength = converter.wavelength
        self.height_scale = converter.height_scale

    @property
    def wavelength_nm(self) -> float:
        """The wavelength in nm."""
        return self.wavelength * 1e9

    def convert_to_height(self, opd: Tensor) -> Tensor:
        """Convert `opd` (nm) to optical height (nm)."""
        return opd / self.refractive_delta

    def convert_to_opd(self, height: Tensor) -> Tensor:
        """Convert optical `height` (nm) to OPD (nm)."""
        return height * self.refractive_delta

    def convert_from_phase(self, phase: Tensor) -> Tensor:
        """Convert `phase` (rad) to optical height (nm) via the bound wavelength."""
        return phase * self.height_scale

    def forward(self, opd: Tensor) -> Tensor:
        """Convert `opd` (nm) to optical height (nm); the `nn.Module` call form."""
        return self.convert_to_height(opd)


def opd_to_height(
    opd: Tensor, *, refractive_delta: float = DEFAULT_REFRACTIVE_DELTA
) -> Tensor:
    """Convert OPD (nm) to optical height (nm); a one-shot `OpticalHeight`.

    Preserves the input tensor's dtype, device, and autograd graph. For repeated
    use, build an `OpticalHeight` once.

    Args:
        opd: OPD image (or batch), in nm.
        refractive_delta: Refractive-index difference ``n_object - n_medium``.
    """
    return OpticalHeight(refractive_delta).convert_to_height(opd)


def height_to_opd(
    height: Tensor, *, refractive_delta: float = DEFAULT_REFRACTIVE_DELTA
) -> Tensor:
    """Convert optical height (nm) to OPD (nm); a one-shot `OpticalHeight`.

    The inverse of `opd_to_height`; preserves dtype, device, and the autograd graph.

    Args:
        height: Optical height image (or batch), in nm.
        refractive_delta: Refractive-index difference ``n_object - n_medium``.
    """
    return OpticalHeight(refractive_delta).convert_to_opd(height)


def phase_to_height(
    phase: Tensor,
    *,
    wavelength: float = DEFAULT_WAVELENGTH,
    refractive_delta: float = DEFAULT_REFRACTIVE_DELTA,
) -> Tensor:
    """Convert phase (rad) to optical height (nm); a one-shot `OpticalHeight`.

    Preserves the input tensor's dtype, device, and autograd graph. For repeated
    use, build an `OpticalHeight` (or read its `height_scale`) once.

    Args:
        phase: Phase image (or batch), in rad.
        wavelength: Illumination wavelength, in m.
        refractive_delta: Refractive-index difference ``n_object - n_medium``.
    """
    module = OpticalHeight(refractive_delta, wavelength=wavelength)
    return module.convert_from_phase(phase)
