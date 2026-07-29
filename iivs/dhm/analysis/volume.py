from __future__ import annotations

__all__ = ("OpticalVolumeCalculator", "calc_volume", "calc_volume_from_phase")

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from iivs.common.data.reduction import Sum, apply_mask
from iivs.dhm.analysis.area import ProjectedAreaCalculator
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
class OpticalVolumeCalculator:
    """An OPD-to-optical-volume (um^3) integrator over an area and a height engine.

    Built from the two sides of the volume relation, bound at construction; the
    per-pixel volume factors are precomputed::

        ovc = OpticalVolumeCalculator.from_args(
            pixel_size=px, wavelength=wl, refractive_delta=dn
        )  # plain parameters; or bind prebuilt engines:
        ovc = OpticalVolumeCalculator(area_calculator=..., height_converter=...)
        volume = ovc.calc_from_opd(opd, mask=cell)  # opd in nm
        volume = ovc.calc_from_phase(phase, mask=cell)  # phase in rad

    Optical volume is ``sum(height * pixel_area)`` with ``height = OPD /
    refractive_delta``, in um^3 (1 um^3 = 1 fL); equivalently ``projected_area *
    mean(height)``. The bound `ProjectedAreaCalculator` supplies the um^2
    footprint factor and the `OpticalHeightConverter` the OPD-to-height factor, so
    `volume_scale` is their product ``area_scale * 1e-3 / refractive_delta``.
    Summation is in float64 over the last two axes (H, W), returned as float32.
    Inputs are batched (``(..., H, W)``) and a multi-region mask adds a trailing
    region axis; see `calc_from_opd` for the shape / `mask` / `reduce` details.
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

    height_converter: OpticalHeightConverter = field(
        default_factory=OpticalHeightConverter
    )

    area_calculator: ProjectedAreaCalculator = field(
        default_factory=ProjectedAreaCalculator
    )

    # um^3 of volume per summed nm of OPD:
    _scale: float = field(init=False, repr=False, compare=False)
    # the region summation engine; empty region -> 0 volume:
    _sum: Sum = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Precompute the per-pixel volume factor from the bound engines."""
        # px_area(um^2) * (nm -> um 1e-3) / delta: the OPD form every calc path
        # funnels into (height and phase convert to OPD first)
        area_scale = self.area_calculator.area_scale  # um^2 per pixel
        scale = area_scale * 1e-3 / self.height_converter.refractive_delta
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
        height_converter = OpticalHeightConverter.from_args(
            wavelength=wavelength, refractive_delta=refractive_delta
        )
        area_calculator = ProjectedAreaCalculator(pixel_size=pixel_size)
        return cls(height_converter=height_converter, area_calculator=area_calculator)

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
        """um^3 of optical volume per nm of OPD summed over pixels.

        The cached ``pixel_area / refractive_delta`` factor: ``volume ==
        volume_scale * sum(opd_nm)``. The dry-mass analogue is
        `DryMassCalculator.drymass_scale`; volume's canonical unit here is um^3, so
        this needs no suffix.
        """
        return self._scale

    @property
    def pixel_size(self) -> float:
        """The bound area calculator's pixel size, in m."""
        return self.area_calculator.pixel_size

    @property
    def pixel_size_um(self) -> float:
        """The bound area calculator's pixel size, in um."""
        return self.area_calculator.pixel_size_um

    def calc_from_opd(
        self,
        opd: NDArray[np.float32],
        *,
        mask: MaskLike | None = None,
        reduce: bool = True,
    ) -> NDArray[np.float32]:
        if opd.ndim < 2:
            msg = f"opd must be at least 2D (..., H, W) (got {opd.ndim}D)"
            raise ValueError(msg)

        region_op = self._sum if reduce else apply_mask
        result = region_op(opd, mask)

        return (result * self._scale).astype(np.float32, copy=False)

    def calc_from_height(
        self,
        height: NDArray[np.float32],
        *,
        mask: MaskLike | None = None,
        reduce: bool = True,
    ) -> NDArray[np.float32]:
        """Integrate an optical height map (nm) into volume [um^3].

        Converts `height` back to OPD (``height * refractive_delta``, via the
        bound height converter), then integrates as `calc_from_opd`; shapes,
        `mask`, and `reduce` behave identically.

        Raises:
            ValueError: If `height` is not at least 2-D ``(..., H, W)``, or the
                mask is malformed (see `region_stack`).
        """
        if height.ndim < 2:
            msg = f"height must be at least 2D (..., H, W) (got {height.ndim}D)"
            raise ValueError(msg)

        opd = self.height_converter.convert_to_opd(height)

        return self.calc_from_opd(opd, mask=mask, reduce=reduce)

    def calc_from_phase(
        self,
        phase: NDArray[np.float32],
        *,
        mask: MaskLike | None = None,
        reduce: bool = True,
    ) -> NDArray[np.float32]:
        """Integrate a phase map (rad) into volume [um^3]."""
        height = self.height_converter.convert_from_phase(phase)
        return self.calc_from_height(height, mask=mask, reduce=reduce)


def calc_volume(
    opd: NDArray[np.float32],
    *,
    pixel_size: float = PIXEL_SIZE_20X,
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
            `OpticalVolumeCalculator.calc_from_opd`.
        reduce: Sum over each region to a volume (True), or return the per-pixel
            volume-density map (False). See `OpticalVolumeCalculator.calc_from_opd`.

    Returns:
        Optical volume in um^3, shape ``(...)`` (or ``(..., R)``); or the unreduced
        density map when `reduce` is False.
    """
    return OpticalVolumeCalculator.from_args(
        pixel_size=pixel_size,
        wavelength=DEFAULT_WAVELENGTH,
        refractive_delta=refractive_delta,
    ).calc_from_opd(opd, mask=mask, reduce=reduce)


def calc_volume_from_phase(
    phase: NDArray[np.float32],
    *,
    pixel_size: float = PIXEL_SIZE_20X,
    wavelength: float = DEFAULT_WAVELENGTH,
    refractive_delta: float = DEFAULT_REFRACTIVE_DELTA,
    mask: MaskLike | None = None,
    reduce: bool = True,
) -> NDArray[np.float32]:
    """Integrate a phase map (rad) into optical volume [um^3] at `wavelength`.

    Converts `phase` to height at `wavelength` and `refractive_delta`, then
    integrates as `calc_volume`.

    Args:
        phase: Phase map(s), in rad, shape ``(..., H, W)``, already
            background-corrected.
        pixel_size: Physical size of one (square) pixel, in m.
        wavelength: Illumination wavelength, in m.
        refractive_delta: Refractive-index difference ``n_object - n_medium``.
        mask: Optional region mask (boolean or integer labels); see
            `OpticalVolumeCalculator.calc_from_opd`.
        reduce: Sum over each region to a volume (True), or return the per-pixel
            volume-density map (False).

    Returns:
        Optical volume in um^3, shape ``(...)`` (or ``(..., R)``); or the unreduced
        density map when `reduce` is False.
    """
    return OpticalVolumeCalculator.from_args(
        pixel_size=pixel_size, wavelength=wavelength, refractive_delta=refractive_delta
    ).calc_from_phase(phase, mask=mask, reduce=reduce)
