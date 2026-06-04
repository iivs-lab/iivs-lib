from __future__ import annotations

__all__ = ("DryMass", "calc_drymass", "calc_drymass_from_phase")

from typing import TYPE_CHECKING

from kaparoo.utils.optional import factory_if_none
from torch import nn, tensordot

from iivs.dhm.analysis.drymass import DryMassCalculator
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
    ``float64`` (for precision) and scales, returning a tensor in the input's
    dtype -- never a Python `float` -- so it stays on the input's device, dtype,
    and autograd graph (a `float()` cast would sync off-device and drop
    gradients). Inputs are batched (``(..., H, W)``); a
    ``(N, H, W)`` mask adds a trailing channel axis (``(..., N)``);
    ``reduce=False`` returns the per-pixel mass-density map instead of the sum.
    The OPD must already be background-corrected.

    Attributes:
        pixel_size: Physical size of one (square) pixel, in m.
        alpha: Specific refractive increment, in m^3/kg.
        opd_module: The `OpticalPathDifference` used by `calc_from_phase` (a
            registered submodule).
        drymass_scale: pg of dry mass per nm of summed OPD (a plain float).
    """

    def __init__(
        self,
        *,
        pixel_size: float,
        alpha: float = DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
        opd_module: OpticalPathDifference | None = None,
    ) -> None:
        super().__init__()

        calculator = DryMassCalculator(pixel_size=pixel_size, alpha=alpha)
        self.pixel_size = calculator.pixel_size
        self.alpha = calculator.alpha
        self.drymass_scale = calculator.drymass_scale

        self.opd_module = factory_if_none(opd_module, OpticalPathDifference)

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
            opd_module=OpticalPathDifference(wavelength=wavelength),
        )

    def calc_from_opd(
        self,
        opd: Tensor,
        *,
        mask: Tensor | None = None,
        reduce: bool = True,
    ) -> Tensor:
        """Dry mass [pg] from an OPD map (nm), summed over the last two axes (H, W).

        Args:
            opd: OPD map(s), in nm, shape ``(..., H, W)``.
            mask: Optional boolean mask, shape ``(H, W)`` or ``(N, H, W)`` for
                `N` objects; multiplied in (broadcast), the 3-D form adding a
                trailing channel axis.
            reduce: If True (default), sum over (H, W) and return the dry mass,
                shape ``(...)`` (or ``(..., N)`` with a ``(N, H, W)`` mask), as a
                tensor (0-dim for a single image). If False, return the per-pixel
                mass-density map (``opd * scale``, masked) without summing.

        Raises:
            ValueError: If `opd` is not at least 2-D ``(..., H, W)``; if `mask`
                is not 2-D ``(H, W)`` or 3-D ``(N, H, W)`` (a per-frame /
                higher-rank mask like ``(T, N, H, W)`` is unsupported -- loop
                over its leading axes); or if `mask`'s ``(H, W)`` does not match
                `opd`'s.
        """
        if opd.ndim < 2:
            msg = f"opd must be at least 2D (..., H, W) (got {opd.ndim}D)"
            raise ValueError(msg)

        use_mask = mask is not None

        if use_mask:
            if mask.ndim not in (2, 3):
                msg = f"mask must be (H, W) or (N, H, W) (got {mask.ndim}D); loop over the extra (e.g. time) axis"
                raise ValueError(msg)
            if mask.shape[-2:] != opd.shape[-2:]:
                msg = f"opd and mask (H, W) must match (got {tuple(opd.shape[-2:])} vs {tuple(mask.shape[-2:])})"
                raise ValueError(msg)

        out_dtype = opd.dtype

        if reduce:
            opd = opd.double()  # accumulate the sum in float64
            if use_mask:
                # contract (H, W) without building the (..., N, H, W) product
                result = tensordot(opd, mask.double(), dims=([-2, -1], [-2, -1]))
            else:
                result = opd.sum(dim=(-2, -1))
        elif use_mask:
            if mask.ndim == 3:  # (N, H, W): object axis before (H, W)
                opd = opd[..., None, :, :]
            result = opd * mask
        else:
            result = opd

        # OPD (nm) -> dry mass (pg); cast the float64 accumulation back to the
        # input dtype (keeps device + autograd; preserves f16 / f32 / f64).
        return (result * self.drymass_scale).to(out_dtype)

    def calc_from_phase(
        self,
        phase: Tensor,
        *,
        mask: Tensor | None = None,
        reduce: bool = True,
    ) -> Tensor:
        """Dry mass [pg] from a phase map (rad): to OPD, then `calc_from_opd`."""
        opd = self.opd_module(phase)
        return self.calc_from_opd(opd, mask=mask, reduce=reduce)

    def forward(
        self,
        opd: Tensor,
        *,
        mask: Tensor | None = None,
        reduce: bool = True,
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
        mask: Optional boolean mask, shape ``(H, W)`` or ``(N, H, W)``.
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
        mask: Optional boolean mask, shape ``(H, W)`` or ``(N, H, W)``.
        reduce: Sum over (H, W) to a dry mass (True), or return the per-pixel
            mass-density map (False).
    """
    return DryMass.from_wavelength(
        pixel_size=pixel_size, alpha=alpha, wavelength=wavelength
    ).calc_from_phase(phase, mask=mask, reduce=reduce)
