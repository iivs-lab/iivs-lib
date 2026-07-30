from __future__ import annotations

__all__ = (
    "OpticalVolumeCalculator",
    "calc_optical_volume",
    "calc_optical_volume_from_height",
    "calc_optical_volume_from_opd",
)

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from iivs.common.data.reduction import Sum
from iivs.dhm.analysis.area import ProjectedAreaCalculator
from iivs.dhm.analysis.calculator import MaskedRegionCalculator
from iivs.dhm.analysis.height import OpticalHeightConverter
from iivs.dhm.constants import (
    DEFAULT_REFRACTIVE_DELTA,
    DEFAULT_WAVELENGTH,
    PIXEL_SIZE_20X,
)

if TYPE_CHECKING:
    from typing import Self

    from numpy.typing import NDArray

    from iivs.common.data.reduction import MaskLike


@dataclass(frozen=True, slots=True)
class OpticalVolumeCalculator(MaskedRegionCalculator):
    """A phase-to-optical-volume (um^3) integrator over an area and a height engine.

    Built from the two sides of the volume relation, bound at construction; the
    per-pixel volume factors are precomputed::

        ovc = OpticalVolumeCalculator.from_args(
            pixel_size=px, wavelength=wl, refractive_delta=dn
        )  # plain parameters; or bind prebuilt engines:
        ovc = OpticalVolumeCalculator(area_calculator=..., height_converter=...)
        volume = ovc.calc(phase, mask=cell)  # phase in rad
        volume = ovc.calc_from_opd(opd, mask=cell)  # opd in nm

    Optical volume is ``sum(height * pixel_area)`` with ``height = OPD /
    refractive_delta``, in um^3 (1 um^3 = 1 fL); equivalently ``projected_area *
    mean(height)``. The bound `ProjectedAreaCalculator` supplies the um^2
    footprint factor and the `OpticalHeightConverter` the phase-to-height factor,
    so `volume_scale` is their product ``area_scale * height_scale * 1e-3`` (um^3
    of volume per rad of phase). Phase is the canonical input: an OPD or height map
    is first converted back to phase (via the bound converters), which the volume
    scale then integrates. Summation is in float64 over the last two axes (H, W),
    returned as float32. Inputs are batched (``(..., H, W)``) and a multi-region
    mask adds a trailing region axis; see `calc` for the shape / `mask` /
    `reduce` details.
    The map must already be background-corrected (~ 0 outside the object). Dry
    mass is the ``refractive_delta / alpha`` multiple of this volume
    (`DryMassCalculator`, which binds a volume calculator the same way).

    For PyTorch, use `OpticalVolume` from `iivs.dhm.analysis.pytorch`.

    Attributes:
        area_calculator: The footprint engine (the `area` in ``area *
            mean(height)``); carries the pixel size.
        height_converter: OPD-to-height converter used by the `opd` / `phase`
            paths; carries the refractive-index difference and wavelength.
    """

    area_calculator: ProjectedAreaCalculator = field(
        default_factory=ProjectedAreaCalculator
    )

    height_converter: OpticalHeightConverter = field(
        default_factory=OpticalHeightConverter
    )

    _scale: float = field(init=False, repr=False, compare=False)
    _sum: Sum = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Precompute the per-pixel volume factor from the bound engines."""
        # px_area(um^2) * height(nm/rad) * (nm -> um 1e-3): the phase form every
        # calc path funnels into (opd and height convert back to phase first)
        area_scale = self.area_calculator.area_scale  # um^2 per pixel
        scale = area_scale * self.height_converter.height_scale * 1e-3
        object.__setattr__(self, "_scale", scale)
        object.__setattr__(self, "_sum", Sum(empty=0.0))

    @classmethod
    def from_args(
        cls,
        *,
        pixel_size: float,
        wavelength: float,
        refractive_delta: float,
    ) -> Self:
        """Build a calculator from plain parameters, constructing both engines."""
        area_calculator = ProjectedAreaCalculator(pixel_size=pixel_size)
        height_converter = OpticalHeightConverter.from_args(
            wavelength=wavelength, refractive_delta=refractive_delta
        )
        return cls(area_calculator=area_calculator, height_converter=height_converter)

    @property
    def pixel_size(self) -> float:
        """The bound area calculator's pixel size, in m."""
        return self.area_calculator.pixel_size

    @property
    def pixel_size_um(self) -> float:
        """The bound area calculator's pixel size, in um."""
        return self.area_calculator.pixel_size_um

    @property
    def refractive_delta(self) -> float:
        """The bound height converter's refractive-index difference."""
        return self.height_converter.refractive_delta

    @property
    def wavelength(self) -> float:
        """The bound height converter's wavelength, in m."""
        return self.height_converter.wavelength

    @property
    def wavelength_nm(self) -> float:
        """The bound height converter's wavelength, in nm."""
        return self.height_converter.wavelength_nm

    @property
    def volume_scale(self) -> float:
        """um^3 of optical volume per rad of phase summed over pixels.

        The cached ``area_scale * height_scale * 1e-3`` factor: ``volume ==
        volume_scale * sum(phase)``. Volume's canonical unit here is um^3, so this
        needs no suffix.
        """
        return self._scale

    def calc(
        self,
        phase: NDArray[np.float32],
        *,
        mask: MaskLike | None = None,
        reduce: bool = True,
    ) -> NDArray[np.float32]:
        """Integrate a phase map (rad) into volume [um^3] over the last two axes.

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
            reduce: If True (default), sum the per-pixel volume over each region
                and return the volume, shape ``(...)`` (or ``(..., R)`` for a
                multi-region mask). If False, return the per-pixel volume-density
                map (``phase * scale``, masked) without summing, shape
                ``(..., H, W)`` (or ``(..., R, H, W)``).

        Raises:
            ValueError: If `phase` is not at least 2-D ``(..., H, W)``, or the mask
                is malformed (see `region_stack`).
        """
        self._require_2d(phase, "phase")

        result = self._reduce(phase, mask, reduce=reduce)

        return (result * self._scale).astype(np.float32, copy=False)

    def calc_from_height(
        self,
        height: NDArray[np.float32],
        *,
        mask: MaskLike | None = None,
        reduce: bool = True,
    ) -> NDArray[np.float32]:
        """Integrate an optical height map (nm) into volume [um^3] through phase.

        Converts `height` back to phase (via the bound height converter), then
        integrates as `calc`; shapes, `mask`, and `reduce` behave
        identically.

        Raises:
            ValueError: If `height` is not at least 2-D ``(..., H, W)``, or the
                mask is malformed (see `region_stack`).
        """
        self._require_2d(height, "height")

        phase = self.height_converter.convert_to_phase(height)

        return self.calc(phase, mask=mask, reduce=reduce)

    def calc_from_opd(
        self,
        opd: NDArray[np.float32],
        *,
        mask: MaskLike | None = None,
        reduce: bool = True,
    ) -> NDArray[np.float32]:
        """Integrate an OPD map (nm) into volume [um^3], entering through phase.

        Maps `opd` back to phase (via the bound `opd_converter`, whose wavelength
        then cancels), then integrates as `calc`; shapes, `mask`, and
        `reduce` behave identically.

        Raises:
            ValueError: If `opd` is not at least 2-D ``(..., H, W)``, or the mask
                is malformed (see `region_stack`).
        """
        self._require_2d(opd, "opd")

        opd_converter = self.height_converter.opd_converter
        phase = opd_converter.convert_to_phase(opd)

        return self.calc(phase, mask=mask, reduce=reduce)


def calc_optical_volume(
    phase: NDArray[np.float32],
    *,
    pixel_size: float = PIXEL_SIZE_20X,
    wavelength: float = DEFAULT_WAVELENGTH,
    refractive_delta: float = DEFAULT_REFRACTIVE_DELTA,
    mask: MaskLike | None = None,
    reduce: bool = True,
) -> NDArray[np.float32]:
    """Integrate a phase map (rad) into optical volume [um^3] at `wavelength`.

    The canonical one-shot `OpticalVolumeCalculator`; `calc_optical_volume_from_opd` and
    `calc_optical_volume_from_height` are the OPD / height entry points.

    Args:
        phase: Phase map(s), in rad, shape ``(..., H, W)``, already
            background-corrected.
        pixel_size: Physical size of one (square) pixel, in m.
        wavelength: Illumination wavelength, in m.
        refractive_delta: Refractive-index difference ``n_object - n_medium``.
        mask: Optional region mask (boolean or integer labels); see
            `OpticalVolumeCalculator.calc`.
        reduce: Sum over each region to a volume (True), or return the per-pixel
            volume-density map (False).

    Returns:
        Optical volume in um^3, shape ``(...)`` (or ``(..., R)``); or the unreduced
        density map when `reduce` is False.
    """
    return OpticalVolumeCalculator.from_args(
        pixel_size=pixel_size,
        wavelength=wavelength,
        refractive_delta=refractive_delta,
    ).calc(phase, mask=mask, reduce=reduce)


def calc_optical_volume_from_height(
    height: NDArray[np.float32],
    *,
    pixel_size: float = PIXEL_SIZE_20X,
    wavelength: float = DEFAULT_WAVELENGTH,
    refractive_delta: float = DEFAULT_REFRACTIVE_DELTA,
    mask: MaskLike | None = None,
    reduce: bool = True,
) -> NDArray[np.float32]:
    """Integrate an optical height map (nm) into optical volume [um^3].

    Args:
        height: Optical height map(s), in nm, shape ``(..., H, W)``, already
            background-corrected.
        pixel_size: Physical size of one (square) pixel, in m.
        wavelength: Illumination wavelength, in m.
        refractive_delta: Refractive-index difference ``n_object - n_medium``.
        mask: Optional region mask (boolean or integer labels); see
            `OpticalVolumeCalculator.calc`.
        reduce: Sum over each region to a volume (True), or return the per-pixel
            volume-density map (False). See
            `OpticalVolumeCalculator.calc`.

    Returns:
        Optical volume in um^3, shape ``(...)`` (or ``(..., R)``); or the unreduced
        density map when `reduce` is False.
    """
    return OpticalVolumeCalculator.from_args(
        pixel_size=pixel_size,
        wavelength=wavelength,
        refractive_delta=refractive_delta,
    ).calc_from_height(height, mask=mask, reduce=reduce)


def calc_optical_volume_from_opd(
    opd: NDArray[np.float32],
    *,
    pixel_size: float = PIXEL_SIZE_20X,
    wavelength: float = DEFAULT_WAVELENGTH,
    refractive_delta: float = DEFAULT_REFRACTIVE_DELTA,
    mask: MaskLike | None = None,
    reduce: bool = True,
) -> NDArray[np.float32]:
    """Integrate an OPD map (nm) into optical volume [um^3].

    Args:
        opd: OPD map(s), in nm (e.g. from `phase_to_opd`), shape ``(..., H, W)``,
            already background-corrected.
        pixel_size: Physical size of one (square) pixel, in m.
        refractive_delta: Refractive-index difference ``n_object - n_medium``.
        mask: Optional region mask (boolean or integer labels); see
            `OpticalVolumeCalculator.calc`.
        reduce: Sum over each region to a volume (True), or return the per-pixel
            volume-density map (False). See
            `OpticalVolumeCalculator.calc`.

    Returns:
        Optical volume in um^3, shape ``(...)`` (or ``(..., R)``); or the unreduced
        density map when `reduce` is False.
    """
    return OpticalVolumeCalculator.from_args(
        pixel_size=pixel_size,
        wavelength=wavelength,
        refractive_delta=refractive_delta,
    ).calc_from_opd(opd, mask=mask, reduce=reduce)
