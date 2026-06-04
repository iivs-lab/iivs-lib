from __future__ import annotations

__all__ = ("DryMassCalculator", "calc_drymass", "calc_drymass_from_phase")

from typing import TYPE_CHECKING

from torch import float64, nn

from iivs.dhm.analysis.drymass import DryMassCalculator as _NpDryMassCalculator
from iivs.dhm.analysis.pytorch.opd import OPDConverter
from iivs.dhm.data.constants import (
    DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    DEFAULT_WAVELENGTH,
)

if TYPE_CHECKING:
    from typing import Self

    from torch import Tensor


class DryMassCalculator(nn.Module):
    """Torch `nn.Module` twin of `iivs.dhm.analysis.drymass.DryMassCalculator`.

    Binds the pixel size, specific refractive increment, and an `OPDConverter`
    (for the phase path) once; the per-pixel `drymass_scale` (a plain float) is
    reused from the NumPy engine. `calc_from_opd` sums in ``float64`` and scales,
    returning a 0-dim tensor -- never a Python `float` -- so it stays on the
    input's device and in the autograd graph (a `float()` cast would sync
    off-device and drop gradients). The OPD must already be background-corrected;
    pass `mask` to restrict the sum to one segmented object.

    Attributes:
        pixel_size: Physical size of one (square) pixel, in m.
        alpha: Specific refractive increment, in m^3/kg.
        opd_converter: The `OPDConverter` used by `calc_from_phase` (a registered
            submodule).
        drymass_scale: pg of dry mass per nm of summed OPD (a plain float).
    """

    def __init__(
        self,
        *,
        pixel_size: float,
        alpha: float = DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
        opd_converter: OPDConverter | None = None,
    ) -> None:
        super().__init__()
        self.pixel_size = pixel_size
        self.alpha = alpha
        self.opd_converter = (
            opd_converter if opd_converter is not None else OPDConverter()
        )
        self.drymass_scale = _NpDryMassCalculator(
            pixel_size=pixel_size, alpha=alpha
        ).drymass_scale

    @classmethod
    def from_wavelength(
        cls,
        *,
        pixel_size: float,
        alpha: float = DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
        wavelength: float = DEFAULT_WAVELENGTH,
    ) -> Self:
        """Build a calculator whose phase path uses `wavelength` (in m)."""
        return cls(
            pixel_size=pixel_size,
            alpha=alpha,
            opd_converter=OPDConverter(wavelength=wavelength),
        )

    def calc_from_opd(self, opd: Tensor, *, mask: Tensor | None = None) -> Tensor:
        """Dry mass [pg] from an OPD map (nm), optionally masked, as a 0-dim tensor."""
        selected = opd if mask is None else opd[mask]
        return selected.sum(dtype=float64) * self.drymass_scale

    def calc_from_phase(self, phase: Tensor, *, mask: Tensor | None = None) -> Tensor:
        """Dry mass [pg] from a phase map (rad): to OPD, then `calc_from_opd`."""
        return self.calc_from_opd(self.opd_converter.convert_to_opd(phase), mask=mask)

    def forward(self, opd: Tensor, *, mask: Tensor | None = None) -> Tensor:
        """Alias of `calc_from_opd`, so the module is callable."""
        return self.calc_from_opd(opd, mask=mask)


def calc_drymass(
    opd: Tensor,
    *,
    pixel_size: float,
    alpha: float = DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    mask: Tensor | None = None,
) -> Tensor:
    """Dry mass [pg] from an OPD map (nm); a one-shot `DryMassCalculator`.

    Returns a 0-dim tensor on `opd`'s device, keeping the autograd graph.

    Args:
        opd: OPD map (or batch), in nm, already background-corrected.
        pixel_size: Physical size of one (square) pixel, in m.
        alpha: Specific refractive increment, in m^3/kg.
        mask: Optional boolean tensor selecting the object's pixels.
    """
    return DryMassCalculator(pixel_size=pixel_size, alpha=alpha).calc_from_opd(
        opd, mask=mask
    )


def calc_drymass_from_phase(
    phase: Tensor,
    *,
    pixel_size: float,
    wavelength: float = DEFAULT_WAVELENGTH,
    alpha: float = DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    mask: Tensor | None = None,
) -> Tensor:
    """Dry mass [pg] from a phase map (rad); a one-shot `DryMassCalculator`.

    Converts `phase` to OPD at `wavelength`, then integrates as `calc_drymass`;
    returns a 0-dim tensor on the input's device, keeping the autograd graph.

    Args:
        phase: Phase map (or batch), in rad, already background-corrected.
        pixel_size: Physical size of one (square) pixel, in m.
        wavelength: Illumination wavelength, in m.
        alpha: Specific refractive increment, in m^3/kg.
        mask: Optional boolean tensor selecting the object's pixels.
    """
    return DryMassCalculator.from_wavelength(
        pixel_size=pixel_size, alpha=alpha, wavelength=wavelength
    ).calc_from_phase(phase, mask=mask)
