from __future__ import annotations

__all__ = ("DryMass", "calc_drymass", "calc_drymass_from_phase")

from typing import TYPE_CHECKING

from torch import float64, nn

from iivs.dhm.analysis.drymass import DryMassCalculator as _NpDryMassCalculator
from iivs.dhm.analysis.pytorch.opd import OpticalPathDifference
from iivs.dhm.data.constants import (
    DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    DEFAULT_WAVELENGTH,
)

if TYPE_CHECKING:
    from typing import Self

    from torch import Tensor


class DryMass(nn.Module):
    """Torch `nn.Module` for dry mass (pg) from OPD or phase.

    The torch twin of `iivs.dhm.analysis.drymass.DryMassCalculator` (named for
    the quantity, per the `nn.Module` convention). Binds the pixel size, specific
    refractive increment, and an `OpticalPathDifference` (for the phase path)
    once; the per-pixel `drymass_scale` (a plain float) is reused from the NumPy
    engine. `calc_from_opd` sums the masked OPD over the last two axes (H, W) in
    ``float64`` and scales, returning a tensor -- never a Python `float` -- so it
    stays on the input's device and in the autograd graph (a `float()` cast would
    sync off-device and drop gradients). Inputs are batched (``(..., H, W)``); a
    ``(C, H, W)`` mask adds a trailing channel axis (``(..., C)``);
    ``reduce=False`` returns the per-pixel mass-density map instead of the sum.
    The OPD must already be background-corrected.

    Attributes:
        pixel_size: Physical size of one (square) pixel, in m.
        alpha: Specific refractive increment, in m^3/kg.
        opd_converter: The `OpticalPathDifference` used by `calc_from_phase` (a
            registered submodule).
        drymass_scale: pg of dry mass per nm of summed OPD (a plain float).
    """

    def __init__(
        self,
        *,
        pixel_size: float,
        alpha: float = DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
        opd_converter: OpticalPathDifference | None = None,
    ) -> None:
        super().__init__()
        self.pixel_size = pixel_size
        self.alpha = alpha
        self.opd_converter = (
            opd_converter if opd_converter is not None else OpticalPathDifference()
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
            opd_converter=OpticalPathDifference(wavelength=wavelength),
        )

    def calc_from_opd(
        self, opd: Tensor, *, mask: Tensor | None = None, reduce: bool = True
    ) -> Tensor:
        """Dry mass [pg] from an OPD map (nm), summed over the last two axes (H, W).

        Args:
            opd: OPD map(s), in nm, shape ``(..., H, W)``.
            mask: Optional boolean mask, shape ``(H, W)`` or ``(C, H, W)`` for
                `C` objects; multiplied in (broadcast), the 3-D form adding a
                trailing channel axis.
            reduce: If True (default), sum over (H, W) and return the dry mass,
                shape ``(...)`` (or ``(..., C)`` with a ``(C, H, W)`` mask), as a
                tensor (0-dim for a single image). If False, return the per-pixel
                mass-density map (``opd * scale``, masked) without summing.
        """
        if mask is not None:
            index = (..., *((None,) * (mask.ndim - 2)), slice(None), slice(None))
            opd = opd[index] * mask
        if not reduce:
            return opd * self.drymass_scale
        return opd.sum(dim=(-2, -1), dtype=float64) * self.drymass_scale

    def calc_from_phase(
        self, phase: Tensor, *, mask: Tensor | None = None, reduce: bool = True
    ) -> Tensor:
        """Dry mass [pg] from a phase map (rad): to OPD, then `calc_from_opd`."""
        opd = self.opd_converter.convert_to_opd(phase)
        return self.calc_from_opd(opd, mask=mask, reduce=reduce)

    def forward(
        self, opd: Tensor, *, mask: Tensor | None = None, reduce: bool = True
    ) -> Tensor:
        """Alias of `calc_from_opd`, so the module is callable."""
        return self.calc_from_opd(opd, mask=mask, reduce=reduce)


def calc_drymass(
    opd: Tensor,
    *,
    pixel_size: float,
    alpha: float = DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    mask: Tensor | None = None,
    reduce: bool = True,
) -> Tensor:
    """Dry mass [pg] from an OPD map (nm); a one-shot `DryMass`.

    Keeps `opd`'s device and the autograd graph.

    Args:
        opd: OPD map(s), in nm, shape ``(..., H, W)``, already
            background-corrected.
        pixel_size: Physical size of one (square) pixel, in m.
        alpha: Specific refractive increment, in m^3/kg.
        mask: Optional boolean mask, shape ``(H, W)`` or ``(C, H, W)``.
        reduce: Sum over (H, W) to a dry mass (True), or return the per-pixel
            mass-density map (False). See `DryMass.calc_from_opd`.
    """
    return DryMass(pixel_size=pixel_size, alpha=alpha).calc_from_opd(
        opd, mask=mask, reduce=reduce
    )


def calc_drymass_from_phase(
    phase: Tensor,
    *,
    pixel_size: float,
    wavelength: float = DEFAULT_WAVELENGTH,
    alpha: float = DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    mask: Tensor | None = None,
    reduce: bool = True,
) -> Tensor:
    """Dry mass [pg] from a phase map (rad); a one-shot `DryMass`.

    Converts `phase` to OPD at `wavelength`, then integrates as `calc_drymass`;
    keeps the input's device and the autograd graph.

    Args:
        phase: Phase map(s), in rad, shape ``(..., H, W)``, already
            background-corrected.
        pixel_size: Physical size of one (square) pixel, in m.
        wavelength: Illumination wavelength, in m.
        alpha: Specific refractive increment, in m^3/kg.
        mask: Optional boolean mask, shape ``(H, W)`` or ``(C, H, W)``.
        reduce: Sum over (H, W) to a dry mass (True), or return the per-pixel
            mass-density map (False).
    """
    return DryMass.from_wavelength(
        pixel_size=pixel_size, alpha=alpha, wavelength=wavelength
    ).calc_from_phase(phase, mask=mask, reduce=reduce)
