from __future__ import annotations

__all__ = ("DryMassCalculator", "calc_drymass", "calc_drymass_from_phase")

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from iivs.dhm.analysis.opd import OPDConverter
from iivs.dhm.data.constants import (
    DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    DEFAULT_WAVELENGTH,
)

if TYPE_CHECKING:
    from typing import Self

    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class DryMassCalculator:
    """Integrate OPD (or phase) into dry mass (pg) at a fixed pixel size and alpha.

    Bind the pixel size, specific refractive increment, and (for the phase path) an
    `OPDConverter` once; the per-pixel mass factor is precomputed::

        dmc = DryMassCalculator(pixel_size=px)  # alpha, wavelength default
        dmc = DryMassCalculator.from_wavelength(pixel_size=px, wavelength=666e-9)
        mass = dmc.calc_from_opd(opd, mask=cell)  # opd in nm
        mass = dmc.calc_from_phase(phase, mask=cell)  # phase in rad

    Dry mass is ``(1 / alpha) * sum(OPD * pixel_area)`` (Barer), in pg, summed in
    float64 over the last two axes (H, W) and returned as float32. Inputs are batched
    (``(..., H, W)``) and a ``(N, H, W)`` mask adds a trailing channel axis. See
    `calc_from_opd` for the shape / `reduce` details. The OPD must already be
    background-corrected (≈ 0 outside the object); segmentation and background
    estimation stay the caller's responsibility.

    The free `calc_drymass` / `calc_drymass_from_phase` are one-shot conveniences over
    this class. For PyTorch, use `DryMass` from `iivs.dhm.analysis.pytorch`.

    Attributes:
        pixel_size: Physical size of one (square) pixel, in m.
        alpha: Specific refractive increment, in m^3/kg.
        opd_converter: Phase-to-OPD converter used by `calc_from_phase`. Defaults to one
            at the default wavelength; inject your own or use `from_wavelength`.
    """

    pixel_size: float
    alpha: float = DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT
    opd_converter: OPDConverter = field(default_factory=OPDConverter)

    # pg of dry mass per nm of OPD summed over pixels:
    _scale: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate inputs and precompute the per-pixel mass factor."""
        if self.pixel_size <= 0:
            msg = f"pixel_size must be positive (got {self.pixel_size})"
            raise ValueError(msg)

        if self.alpha <= 0:
            msg = f"alpha must be positive (got {self.alpha})"
            raise ValueError(msg)

        # pg per summed-nm OPD: px_area(m^2) * (nm->m 1e-9) * (kg->pg 1e15) / alpha.
        object.__setattr__(self, "_scale", self.pixel_size**2 * 1e6 / self.alpha)

    @classmethod
    def from_wavelength(
        cls,
        *,
        pixel_size: float,
        wavelength: float = DEFAULT_WAVELENGTH,
        alpha: float = DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    ) -> Self:
        """Build a calculator whose phase path uses `wavelength` (in m)."""
        opd_converter = OPDConverter(wavelength=wavelength)
        return cls(pixel_size=pixel_size, alpha=alpha, opd_converter=opd_converter)

    @property
    def wavelength(self) -> float:
        """The bound OPD converter's wavelength, in m."""
        return self.opd_converter.wavelength

    @property
    def wavelength_nm(self) -> float:
        """The bound OPD converter's wavelength, in nm."""
        return self.opd_converter.wavelength_nm

    @property
    def drymass_scale(self) -> float:
        """pg of dry mass per nm of OPD summed over pixels.

        The cached ``pixel_area / alpha`` factor: ``mass == drymass_scale *
        sum(opd_nm)``. The OPD analogue is `OPDConverter.opd_scale`.
        """
        return self._scale

    def calc_from_opd(
        self,
        opd: NDArray[np.float32],
        *,
        mask: NDArray[np.bool_] | None = None,
        reduce: bool = True,
    ) -> NDArray[np.float32]:
        """Dry mass [pg] from an OPD map (nm), summed over the last two axes (H, W).

        Args:
            opd: OPD map(s), in nm, shape ``(..., H, W)``.
            mask: Optional boolean mask, shape ``(H, W)`` or ``(N, H, W)`` for `N`
                objects; multiplied in (broadcast), the 3-D form adding a trailing
                channel axis.
            reduce: If True (default), sum the per-pixel mass over (H, W) and return the
                dry mass, shape ``(...)`` (or ``(..., N)`` with a ``(N, H, W)`` mask).
                If False, return the per-pixel mass-density map (``opd * scale``,
                masked) without summing, shape ``(..., H, W)`` (or ``(..., N, H, W)``).

        Raises:
            ValueError: If `opd` is not at least 2-D ``(..., H, W)``; if `mask` is not
                2-D ``(H, W)`` or 3-D ``(N, H, W)`` (a per-frame / higher-rank mask like
                ``(T, N, H, W)`` is unsupported; loop over its leading axes); or if
                `mask`'s ``(H, W)`` does not match `opd`'s.
        """
        if opd.ndim < 2:
            msg = f"opd must be at least 2D (..., H, W) (got {opd.ndim}D)"
            raise ValueError(msg)

        use_mask = mask is not None

        if use_mask:
            if mask.ndim not in (2, 3):
                msg = f"mask must be (H, W) or (N, H, W) (got {mask.ndim}D)"
                raise ValueError(msg)
            if mask.shape[-2:] != opd.shape[-2:]:
                opd_hw = tuple(opd.shape[-2:])
                mask_hw = tuple(mask.shape[-2:])
                msg = f"opd and mask (H, W) must match (got {opd_hw} vs {mask_hw})"
                raise ValueError(msg)

        if reduce:
            opd = opd.astype(np.float64, copy=False)
            if use_mask:
                result = np.tensordot(opd, mask, axes=([-2, -1], [-2, -1]))
            else:
                result = np.sum(opd, axis=(-2, -1))
        elif use_mask:
            if mask.ndim == 3:  # (N, H, W): object axis before (H, W)
                opd = opd[..., None, :, :]
            result = opd * mask
        else:
            result = opd

        # OPD (nm) -> dry mass (pg); accumulated in float64, returned as float32.
        return (result * self._scale).astype(np.float32, copy=False)

    def calc_from_phase(
        self,
        phase: NDArray[np.float32],
        *,
        mask: NDArray[np.bool_] | None = None,
        reduce: bool = True,
    ) -> NDArray[np.float32]:
        """Dry mass [pg] from a phase map (rad): to OPD, then `calc_from_opd`."""
        opd = self.opd_converter.convert_to_opd(phase)
        return self.calc_from_opd(opd, mask=mask, reduce=reduce)


def calc_drymass(
    opd: NDArray[np.float32],
    *,
    pixel_size: float,
    alpha: float = DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    mask: NDArray[np.bool_] | None = None,
    reduce: bool = True,
) -> NDArray[np.float32]:
    """Dry mass [pg] of an OPD map (nm); one-shot `DryMassCalculator.calc_from_opd`.

    Args:
        opd: OPD map(s), in nm (e.g. from `phase_to_opd`), shape ``(..., H, W)``,
            already background-corrected.
        pixel_size: Physical size of one (square) pixel, in m.
        alpha: Specific refractive increment, in m^3/kg.
        mask: Optional boolean mask, shape ``(H, W)`` or ``(N, H, W)``.
        reduce: Sum over (H, W) to a dry mass (True), or return the per-pixel
            mass-density map (False). See `DryMassCalculator.calc_from_opd`.

    Returns:
        Dry mass in pg, shape ``(...)`` (or ``(..., N)``); or the unreduced
        density map when `reduce` is False.
    """
    return DryMassCalculator(pixel_size=pixel_size, alpha=alpha).calc_from_opd(
        opd, mask=mask, reduce=reduce
    )


def calc_drymass_from_phase(
    phase: NDArray[np.float32],
    *,
    pixel_size: float,
    wavelength: float = DEFAULT_WAVELENGTH,
    alpha: float = DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    mask: NDArray[np.bool_] | None = None,
    reduce: bool = True,
) -> NDArray[np.float32]:
    """Dry mass [pg] from a phase map (rad); one-shot `DryMassCalculator`.

    Converts `phase` to OPD at `wavelength`, then integrates as `calc_drymass`.

    Args:
        phase: Phase map(s), in rad, shape ``(..., H, W)``, already
            background-corrected.
        pixel_size: Physical size of one (square) pixel, in m.
        wavelength: Illumination wavelength, in m.
        alpha: Specific refractive increment, in m^3/kg.
        mask: Optional boolean mask, shape ``(H, W)`` or ``(N, H, W)``.
        reduce: Sum over (H, W) to a dry mass (True), or return the per-pixel
            mass-density map (False).

    Returns:
        Dry mass in pg, shape ``(...)`` (or ``(..., N)``); or the unreduced
        density map when `reduce` is False.
    """
    return DryMassCalculator.from_wavelength(
        pixel_size=pixel_size, alpha=alpha, wavelength=wavelength
    ).calc_from_phase(phase, mask=mask, reduce=reduce)
