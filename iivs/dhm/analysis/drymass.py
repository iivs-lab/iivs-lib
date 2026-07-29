from __future__ import annotations

__all__ = ("DryMassCalculator", "calc_drymass", "calc_drymass_from_phase")

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from iivs.common.data.reduction import Sum, apply_mask
from iivs.dhm.analysis.opd import OPDConverter
from iivs.dhm.constants import (
    DEFAULT_REFRACTIVE_DELTA,
    DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    DEFAULT_WAVELENGTH,
)

if TYPE_CHECKING:
    from typing import Self

    from numpy.typing import NDArray

    from iivs.common.data.reduction import MaskLike


@dataclass(frozen=True, slots=True)
class DryMassCalculator:
    """An OPD-to-dry-mass (pg) integrator at a fixed pixel size and alpha.

    Bind the pixel size, specific refractive increment, and (for the phase path) an
    `OPDConverter` once; the per-pixel mass factor is precomputed::

        dmc = DryMassCalculator(pixel_size=px)  # alpha, wavelength default
        dmc = DryMassCalculator.from_wavelength(pixel_size=px, wavelength=666e-9)
        mass = dmc.calc_from_opd(opd, mask=cell)  # opd in nm
        mass = dmc.calc_from_phase(phase, mask=cell)  # phase in rad

    Dry mass is ``(1 / alpha) * sum(OPD * pixel_area)`` (Barer), in pg, summed in
    float64 over the last two axes (H, W) and returned as float32. Inputs are batched
    (``(..., H, W)``) and a multi-region mask adds a trailing region axis. See
    `calc_from_opd` for the shape / `mask` / `reduce` details. The OPD must already be
    background-corrected (≈ 0 outside the object); segmentation and background
    estimation stay the caller's responsibility.

    For PyTorch, use `DryMass` from `iivs.dhm.analysis.pytorch`.

    Attributes:
        pixel_size: Physical size of one (square) pixel, in m.
        alpha: Specific refractive increment, in m^3/kg.
        opd_converter: Phase-to-OPD converter used by `calc_from_phase`. Defaults to one
            at the default wavelength; inject your own or use `from_wavelength`.
    """

    pixel_size: float
    alpha: float = DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT
    opd_converter: OPDConverter = field(default_factory=OPDConverter)

    _scale: float = field(init=False, repr=False, compare=False)
    # the region summation engine; empty region -> 0 mass:
    _sum: Sum = field(init=False, repr=False, compare=False)

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
        object.__setattr__(self, "_sum", Sum(empty=0.0))

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
    def pixel_size_um(self) -> float:
        """The pixel size in um."""
        return self.pixel_size * 1e6

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
        mask: MaskLike | None = None,
        reduce: bool = True,
    ) -> NDArray[np.float32]:
        """Integrate an OPD map (nm) into dry mass [pg] over the last two axes (H, W).

        Args:
            opd: OPD map(s), in nm, shape ``(..., H, W)``.
            mask: Optional mask selecting the region(s) to integrate: a boolean
                ``(H, W)`` (one region) or ``(N, H, W)`` (`N` regions, which may
                overlap), or an integer label image ``(H, W)`` (0 = background, one
                region per positive label). A boolean ``(H, W)`` keeps the plain
                shape; the multi-region forms add a trailing region axis.
            reduce: If True (default), sum the per-pixel mass over each region and
                return the dry mass, shape ``(...)`` (or ``(..., R)`` for a
                multi-region mask). If False, return the per-pixel mass-density map
                (``opd * scale``, masked) without summing, shape ``(..., H, W)`` (or
                ``(..., R, H, W)``).

        Raises:
            ValueError: If `opd` is not at least 2-D ``(..., H, W)``, or the mask is
                malformed (see `region_stack`: a wrong ``(H, W)``, a boolean mask not
                2-D or 3-D, a label mask not 2-D or holding a negative label, or a
                non-boolean / non-integer dtype).
        """
        if opd.ndim < 2:
            msg = f"opd must be at least 2D (..., H, W) (got {opd.ndim}D)"
            raise ValueError(msg)

        # reduce to a dry mass per region, or keep the per-pixel density map; both
        # normalize the mask, drop the region axis for a single region, and handle
        # None (the whole frame) themselves
        region_op = self._sum if reduce else apply_mask
        result = region_op(opd, mask)

        # OPD (nm) -> dry mass (pg); the summed path accumulates in float64, and
        # every path returns float32.
        return (result * self._scale).astype(np.float32, copy=False)

    def calc_from_phase(
        self,
        phase: NDArray[np.float32],
        *,
        mask: MaskLike | None = None,
        reduce: bool = True,
    ) -> NDArray[np.float32]:
        """Integrate a phase map (rad) into dry mass [pg]."""
        opd = self.opd_converter.convert_to_opd(phase)
        return self.calc_from_opd(opd, mask=mask, reduce=reduce)

    def calc_from_volume(
        self,
        volume: NDArray[np.float32],
        *,
        refractive_delta: float = DEFAULT_REFRACTIVE_DELTA,
    ) -> NDArray[np.float32]:
        """Convert an already-integrated optical volume [um^3] into dry mass [pg].

        ``mass = volume * refractive_delta / alpha`` (Barer), closing the OPD ->
        height -> volume -> dry-mass chain: this is `calc_from_opd`'s result when
        the volume came from the same map (`OpticalVolumeCalculator`). Unlike the
        map paths there is nothing left to mask or reduce. `refractive_delta` is a
        parameter rather than an attribute because it belongs to the volume side of
        the bridge; this calculator's own scale never involves it.

        Args:
            volume: Optical volume(s), in um^3.
            refractive_delta: The refractive-index difference the volume was
                computed with.

        Raises:
            ValueError: If `refractive_delta` is not positive.
        """
        if refractive_delta <= 0:
            msg = f"refractive_delta must be positive (got {refractive_delta})"
            raise ValueError(msg)

        scale = refractive_delta / self.alpha * 1e-3  # pg per um^3

        return (volume * scale).astype(np.float32, copy=False)


def calc_drymass(
    opd: NDArray[np.float32],
    *,
    pixel_size: float,
    alpha: float = DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    mask: MaskLike | None = None,
    reduce: bool = True,
) -> NDArray[np.float32]:
    """Integrate an OPD map (nm) into dry mass [pg].

    Args:
        opd: OPD map(s), in nm (e.g. from `phase_to_opd`), shape ``(..., H, W)``,
            already background-corrected.
        pixel_size: Physical size of one (square) pixel, in m.
        alpha: Specific refractive increment, in m^3/kg.
        mask: Optional region mask (boolean or integer labels); see
            `DryMassCalculator.calc_from_opd`.
        reduce: Sum over each region to a dry mass (True), or return the per-pixel
            mass-density map (False). See `DryMassCalculator.calc_from_opd`.

    Returns:
        Dry mass in pg, shape ``(...)`` (or ``(..., R)``); or the unreduced
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
    mask: MaskLike | None = None,
    reduce: bool = True,
) -> NDArray[np.float32]:
    """Integrate a phase map (rad) into dry mass [pg] at `wavelength`.

    Converts `phase` to OPD at `wavelength`, then integrates as `calc_drymass`.

    Args:
        phase: Phase map(s), in rad, shape ``(..., H, W)``, already
            background-corrected.
        pixel_size: Physical size of one (square) pixel, in m.
        wavelength: Illumination wavelength, in m.
        alpha: Specific refractive increment, in m^3/kg.
        mask: Optional region mask (boolean or integer labels); see
            `DryMassCalculator.calc_from_opd`.
        reduce: Sum over each region to a dry mass (True), or return the per-pixel
            mass-density map (False).

    Returns:
        Dry mass in pg, shape ``(...)`` (or ``(..., R)``); or the unreduced
        density map when `reduce` is False.
    """
    return DryMassCalculator.from_wavelength(
        pixel_size=pixel_size, alpha=alpha, wavelength=wavelength
    ).calc_from_phase(phase, mask=mask, reduce=reduce)
