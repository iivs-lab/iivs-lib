from __future__ import annotations

__all__ = (
    "DryMassCalculator",
    "calc_drymass",
    "calc_drymass_from_height",
    "calc_drymass_from_opd",
)

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from kaparoo.utils import replace_if_none

from iivs.common.data.reduction import Sum
from iivs.dhm.analysis.calculator import MaskedRegionCalculator
from iivs.dhm.analysis.volume import OpticalVolumeCalculator
from iivs.dhm.constants import (
    DEFAULT_REFRACTIVE_DELTA,
    DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    DEFAULT_WAVELENGTH,
    PIXEL_SIZE_20X,
)

if TYPE_CHECKING:
    from typing import Self

    from numpy.typing import NDArray

    from iivs.common.data.reduction import MaskLike


@dataclass(frozen=True, slots=True)
class DryMassCalculator(MaskedRegionCalculator):
    """A phase-to-dry-mass (pg) integrator over a volume engine and an alpha.

    The last link of the OPD -> height -> volume -> dry-mass chain: bind an
    `OpticalVolumeCalculator` (which itself binds the area and height engines) and
    the specific refractive increment once; the per-pixel mass factor is
    precomputed::

        dmc = DryMassCalculator()  # lab defaults throughout (20X pixel size)
        dmc = DryMassCalculator(volume_converter=ovc)  # or bind a prebuilt engine
        dmc = DryMassCalculator.from_args(
            pixel_size=px, wavelength=wl, refractive_delta=dn, alpha=a
        )  # or fully explicit plain parameters
        mass = dmc.calc(phase, mask=cell)  # phase in rad
        mass = dmc.calc_from_opd(opd, mask=cell)  # opd in nm

    Dry mass is ``(1 / alpha) * sum(OPD * pixel_area)`` (Barer), in pg, summed in
    float64 over the last two axes (H, W) and returned as float32; equivalently
    ``volume * refractive_delta / alpha`` over the bound engine's volume, which is
    how `drymass_scale` is derived (the delta cancels, so the mass never depends
    on it). Inputs are batched (``(..., H, W)``) and a multi-region mask adds a
    trailing region axis. See `calc` for the shape / `mask` / `reduce` details.
    The map must already be background-corrected (≈ 0 outside the object);
    segmentation and background estimation stay the caller's responsibility.

    For PyTorch, use `DryMass` from `iivs.dhm.analysis.pytorch`.

    Attributes:
        volume_converter: The bound volume engine; carries the pixel size (via its
            area calculator) and the wavelength / delta (via its height converter).
        alpha: Specific refractive increment, in m^3/kg.
    """

    alpha: float = DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT
    volume_converter: OpticalVolumeCalculator = field(
        default_factory=OpticalVolumeCalculator
    )

    _scale: float = field(init=False, repr=False, compare=False)
    # the region summation engine; empty region -> 0 mass:
    _sum: Sum = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate inputs and precompute the per-pixel mass factor."""
        if self.alpha <= 0:
            msg = f"alpha must be positive (got {self.alpha})"
            raise ValueError(msg)

        # pg per summed-rad phase, via the volume engine: mass = volume *
        # delta / alpha (Barer). The delta cancels volume_scale's own, so an OPD
        # map's mass is delta-free; the phase scale carries the wavelength.
        volume = self.volume_converter
        scale = volume.volume_scale * volume.refractive_delta / self.alpha * 1e-3
        object.__setattr__(self, "_scale", scale)
        object.__setattr__(self, "_sum", Sum(empty=0.0))

    @classmethod
    def from_args(
        cls,
        *,
        pixel_size: float,
        wavelength: float,
        refractive_delta: float,
        alpha: float,
    ) -> Self:
        """Build a calculator from plain parameters, constructing the engine chain."""
        volume_converter = OpticalVolumeCalculator.from_args(
            pixel_size=pixel_size,
            wavelength=wavelength,
            refractive_delta=refractive_delta,
        )
        return cls(volume_converter=volume_converter, alpha=alpha)

    @property
    def wavelength(self) -> float:
        """The bound volume engine's wavelength, in m."""
        return self.volume_converter.wavelength

    @property
    def wavelength_nm(self) -> float:
        """The bound volume engine's wavelength, in nm."""
        return self.volume_converter.wavelength_nm

    @property
    def refractive_delta(self) -> float:
        """The bound volume engine's refractive-index difference."""
        return self.volume_converter.refractive_delta

    @property
    def pixel_size(self) -> float:
        """The bound volume engine's pixel size, in m."""
        return self.volume_converter.pixel_size

    @property
    def pixel_size_um(self) -> float:
        """The bound volume engine's pixel size, in um."""
        return self.volume_converter.pixel_size_um

    @property
    def drymass_scale(self) -> float:
        """pg of dry mass per rad of phase summed over pixels.

        The cached ``volume_scale * refractive_delta / alpha * 1e-3`` factor:
        ``mass == drymass_scale * sum(phase)``. Dry mass's canonical unit here is
        pg, so this needs no suffix.
        """
        return self._scale

    def calc(
        self,
        phase: NDArray[np.float32],
        *,
        mask: MaskLike | None = None,
        reduce: bool = True,
    ) -> NDArray[np.float32]:
        """Integrate a phase map (rad) into dry mass [pg] over the last two axes.

        The canonical entry point; `calc_from_opd` and `calc_from_height` convert
        their input back to phase and funnel through here.

        Args:
            phase: Phase map(s), in rad, shape ``(..., H, W)``, already
                background-corrected.
            mask: Optional mask selecting the region(s) to integrate: a boolean
                ``(H, W)`` (one region) or ``(N, H, W)`` (`N` regions, which may
                overlap), or an integer label image ``(H, W)`` (0 = background, one
                region per positive label). A boolean ``(H, W)`` keeps the plain
                shape; the multi-region forms add a trailing region axis.
            reduce: If True (default), sum the per-pixel mass over each region and
                return the dry mass, shape ``(...)`` (or ``(..., R)`` for a
                multi-region mask). If False, return the per-pixel mass-density map
                (``phase * scale``, masked) without summing, shape ``(..., H, W)``
                (or ``(..., R, H, W)``).

        Raises:
            ValueError: If `phase` is not at least 2-D ``(..., H, W)``, or the mask
                is malformed (see `region_stack`).
        """
        self._require_2d(phase, "phase")

        result = self._reduce(phase, mask, reduce=reduce)

        # phase (rad) -> dry mass (pg); the summed path accumulates in float64,
        # and every path returns float32.
        return (result * self._scale).astype(np.float32, copy=False)

    def calc_from_opd(
        self,
        opd: NDArray[np.float32],
        *,
        mask: MaskLike | None = None,
        reduce: bool = True,
    ) -> NDArray[np.float32]:
        """Integrate an OPD map (nm) into dry mass [pg], entering through phase.

        Maps `opd` back to phase (via the bound `opd_converter`), then integrates
        as `calc`; shapes, `mask`, and `reduce` behave identically.

        Raises:
            ValueError: If `opd` is not at least 2-D ``(..., H, W)``, or the mask
                is malformed (see `region_stack`).
        """
        self._require_2d(opd, "opd")

        opd_converter = self.volume_converter.height_converter.opd_converter
        phase = opd_converter.convert_to_phase(opd)

        return self.calc(phase, mask=mask, reduce=reduce)

    def calc_from_height(
        self,
        height: NDArray[np.float32],
        *,
        mask: MaskLike | None = None,
        reduce: bool = True,
    ) -> NDArray[np.float32]:
        """Integrate an optical height map (nm) into dry mass [pg] through phase.

        Converts `height` back to phase (via the bound height converter), then
        integrates as `calc`; shapes, `mask`, and `reduce` behave identically.

        Raises:
            ValueError: If `height` is not at least 2-D ``(..., H, W)``, or the
                mask is malformed (see `region_stack`).
        """
        self._require_2d(height, "height")

        phase = self.volume_converter.height_converter.convert_to_phase(height)

        return self.calc(phase, mask=mask, reduce=reduce)

    def calc_from_volume(
        self,
        volume: NDArray[np.float32],
        *,
        refractive_delta: float | None = None,
    ) -> NDArray[np.float32]:
        """Convert an already-integrated optical volume [um^3] into dry mass [pg].

        ``mass = volume * refractive_delta / alpha`` (Barer), closing the OPD ->
        height -> volume -> dry-mass chain: this is `calc_from_opd`'s result when
        the volume came from the bound engine (`volume_converter`). Unlike the map
        paths there is nothing left to mask or reduce. `refractive_delta` defaults
        to the bound engine's; pass it only for a volume computed with a different
        one.

        Args:
            volume: Optical volume(s), in um^3.
            refractive_delta: The refractive-index difference the volume was
                computed with, or None (default) for the bound engine's.

        Raises:
            ValueError: If `refractive_delta` is given and not positive.
        """
        delta = replace_if_none(refractive_delta, self.refractive_delta)
        if delta <= 0:
            msg = f"refractive_delta must be positive (got {delta})"
            raise ValueError(msg)

        scale = delta / self.alpha * 1e-3  # pg per um^3

        return (volume * scale).astype(np.float32, copy=False)


def calc_drymass(
    phase: NDArray[np.float32],
    *,
    pixel_size: float = PIXEL_SIZE_20X,
    wavelength: float = DEFAULT_WAVELENGTH,
    refractive_delta: float = DEFAULT_REFRACTIVE_DELTA,
    alpha: float = DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    mask: MaskLike | None = None,
    reduce: bool = True,
) -> NDArray[np.float32]:
    """Integrate a phase map (rad) into dry mass [pg] at `wavelength`.

    The canonical one-shot `DryMassCalculator`; `calc_drymass_from_opd` and
    `calc_drymass_from_height` are the OPD / height entry points.

    Args:
        phase: Phase map(s), in rad, shape ``(..., H, W)``, already
            background-corrected.
        pixel_size: Physical size of one (square) pixel, in m. Defaults to the
            lab's 20X objective.
        wavelength: Illumination wavelength, in m.
        refractive_delta: Refractive-index difference ``n_object - n_medium``.
            Enters only the engine chain; it cancels out of the mass.
        alpha: Specific refractive increment, in m^3/kg.
        mask: Optional region mask (boolean or integer labels); see
            `DryMassCalculator.calc`.
        reduce: Sum over each region to a dry mass (True), or return the per-pixel
            mass-density map (False).

    Returns:
        Dry mass in pg, shape ``(...)`` (or ``(..., R)``); or the unreduced
        density map when `reduce` is False.
    """
    return DryMassCalculator.from_args(
        pixel_size=pixel_size,
        wavelength=wavelength,
        refractive_delta=refractive_delta,
        alpha=alpha,
    ).calc(phase, mask=mask, reduce=reduce)


def calc_drymass_from_opd(
    opd: NDArray[np.float32],
    *,
    pixel_size: float = PIXEL_SIZE_20X,
    wavelength: float = DEFAULT_WAVELENGTH,
    refractive_delta: float = DEFAULT_REFRACTIVE_DELTA,
    alpha: float = DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    mask: MaskLike | None = None,
    reduce: bool = True,
) -> NDArray[np.float32]:
    """Integrate an OPD map (nm) into dry mass [pg].

    Args:
        opd: OPD map(s), in nm (e.g. from `phase_to_opd`), shape ``(..., H, W)``,
            already background-corrected.
        pixel_size: Physical size of one (square) pixel, in m. Defaults to the
            lab's 20X objective.
        wavelength: Illumination wavelength, in m. Enters only the engine chain;
            the mass of an OPD map never depends on it.
        refractive_delta: Refractive-index difference ``n_object - n_medium``.
            Enters only the engine chain; it cancels out of the mass.
        alpha: Specific refractive increment, in m^3/kg.
        mask: Optional region mask (boolean or integer labels); see
            `DryMassCalculator.calc`.
        reduce: Sum over each region to a dry mass (True), or return the per-pixel
            mass-density map (False).

    Returns:
        Dry mass in pg, shape ``(...)`` (or ``(..., R)``); or the unreduced
        density map when `reduce` is False.
    """
    return DryMassCalculator.from_args(
        pixel_size=pixel_size,
        wavelength=wavelength,
        refractive_delta=refractive_delta,
        alpha=alpha,
    ).calc_from_opd(opd, mask=mask, reduce=reduce)


def calc_drymass_from_height(
    height: NDArray[np.float32],
    *,
    pixel_size: float = PIXEL_SIZE_20X,
    wavelength: float = DEFAULT_WAVELENGTH,
    refractive_delta: float = DEFAULT_REFRACTIVE_DELTA,
    alpha: float = DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    mask: MaskLike | None = None,
    reduce: bool = True,
) -> NDArray[np.float32]:
    """Integrate an optical height map (nm) into dry mass [pg].

    Args:
        height: Optical height map(s), in nm, shape ``(..., H, W)``, already
            background-corrected.
        pixel_size: Physical size of one (square) pixel, in m. Defaults to the
            lab's 20X objective.
        wavelength: Illumination wavelength, in m.
        refractive_delta: Refractive-index difference ``n_object - n_medium``.
            Enters only the engine chain; it cancels out of the mass.
        alpha: Specific refractive increment, in m^3/kg.
        mask: Optional region mask (boolean or integer labels); see
            `DryMassCalculator.calc`.
        reduce: Sum over each region to a dry mass (True), or return the per-pixel
            mass-density map (False).

    Returns:
        Dry mass in pg, shape ``(...)`` (or ``(..., R)``); or the unreduced
        density map when `reduce` is False.
    """
    return DryMassCalculator.from_args(
        pixel_size=pixel_size,
        wavelength=wavelength,
        refractive_delta=refractive_delta,
        alpha=alpha,
    ).calc_from_height(height, mask=mask, reduce=reduce)
