from __future__ import annotations

__all__ = ("OpticalVolumeCalculator", "calc_volume", "calc_volume_from_phase")

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from iivs.common.data.reduction import Sum, apply_mask
from iivs.dhm.analysis.area import ProjectedAreaCalculator
from iivs.dhm.analysis.height import OpticalHeightConverter
from iivs.dhm.constants import DEFAULT_REFRACTIVE_DELTA, DEFAULT_WAVELENGTH

if TYPE_CHECKING:
    from typing import Self

    from numpy.typing import NDArray

    from iivs.common.data.reduction import MaskLike


@dataclass(frozen=True, slots=True)
class OpticalVolumeCalculator:
    """An OPD-to-optical-volume (um^3) integrator at a fixed pixel size and delta.

    Bind the pixel size and (via the height converter) the refractive-index
    difference once; the per-pixel volume factor is precomputed::

        ovc = OpticalVolumeCalculator(pixel_size=px)  # delta, wavelength default
        volume = ovc.calc_from_opd(opd, mask=cell)  # opd in nm
        volume = ovc.calc_from_phase(phase, mask=cell)  # phase in rad

    Optical volume is ``sum(height * pixel_area)`` with ``height = OPD /
    refractive_delta``, in um^3 (1 um^3 = 1 fL); equivalently ``projected_area *
    mean(height)``. The calculator holds both sides of that relation: a
    `ProjectedAreaCalculator` (the um^2 footprint factor, built from `pixel_size`)
    and an `OpticalHeightConverter` (the OPD-to-height factor), so `volume_scale`
    is their product ``area_scale * 1e-3 / refractive_delta``. Summation is
    in float64 over the last two axes (H, W), returned as float32. Inputs are
    batched (``(..., H, W)``) and a multi-region mask adds a trailing region axis;
    see `calc_from_opd` for the shape / `mask` / `reduce` details. The map must
    already be background-corrected (~ 0 outside the object). Dry mass is the
    ``refractive_delta / alpha`` multiple of this volume
    (`DryMassCalculator.calc_from_volume`).

    For PyTorch, use `OpticalVolume` from `iivs.dhm.analysis.pytorch`.

    Attributes:
        pixel_size: Physical size of one (square) pixel, in m.
        height_converter: OPD-to-height converter used by the `opd` / `phase` paths.
            Defaults to one at the default wavelength and delta; inject your own or
            use `from_wavelength`.
    """

    pixel_size: float
    height_converter: OpticalHeightConverter = field(
        default_factory=OpticalHeightConverter
    )

    # the footprint engine (validates pixel_size and carries the um^2 factor):
    _area: ProjectedAreaCalculator = field(init=False, repr=False, compare=False)
    # um^3 per summed nm of OPD / of height:
    _opd_scale: float = field(init=False, repr=False, compare=False)
    _height_scale: float = field(init=False, repr=False, compare=False)
    # the region summation engine; empty region -> 0 volume:
    _sum: Sum = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Build the footprint engine and precompute the per-pixel volume factors."""
        area = ProjectedAreaCalculator(pixel_size=self.pixel_size)
        object.__setattr__(self, "_area", area)

        # um^3 per summed nm of height: px_area(um^2) * (nm -> um 1e-3).
        height_scale = area.area_scale * 1e-3
        object.__setattr__(self, "_height_scale", height_scale)
        object.__setattr__(
            self, "_opd_scale", height_scale / self.height_converter.refractive_delta
        )
        object.__setattr__(self, "_sum", Sum(empty=0.0))

    @classmethod
    def from_wavelength(
        cls,
        *,
        pixel_size: float,
        wavelength: float = DEFAULT_WAVELENGTH,
        refractive_delta: float = DEFAULT_REFRACTIVE_DELTA,
    ) -> Self:
        """Build a calculator whose phase path uses `wavelength` (in m)."""
        height_converter = OpticalHeightConverter.from_wavelength(
            wavelength=wavelength, refractive_delta=refractive_delta
        )
        return cls(pixel_size=pixel_size, height_converter=height_converter)

    @property
    def area_calculator(self) -> ProjectedAreaCalculator:
        """The bound footprint engine (the `area` in ``area * mean(height)``)."""
        return self._area

    @property
    def pixel_size_um(self) -> float:
        """The pixel size in um."""
        return self._area.pixel_size_um

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
        return self._opd_scale

    def calc_from_opd(
        self,
        opd: NDArray[np.float32],
        *,
        mask: MaskLike | None = None,
        reduce: bool = True,
    ) -> NDArray[np.float32]:
        """Integrate an OPD map (nm) into volume [um^3] over the last two axes (H, W).

        Args:
            opd: OPD map(s), in nm, shape ``(..., H, W)``.
            mask: Optional mask selecting the region(s) to integrate: a boolean
                ``(H, W)`` (one region) or ``(N, H, W)`` (`N` regions, which may
                overlap), or an integer label image ``(H, W)`` (0 = background, one
                region per positive label). A boolean ``(H, W)`` keeps the plain
                shape; the multi-region forms add a trailing region axis.
            reduce: If True (default), sum the per-pixel volume over each region and
                return the volume, shape ``(...)`` (or ``(..., R)`` for a
                multi-region mask). If False, return the per-pixel volume-density
                map (``opd * scale``, masked) without summing, shape ``(..., H, W)``
                (or ``(..., R, H, W)``).

        Raises:
            ValueError: If `opd` is not at least 2-D ``(..., H, W)``, or the mask is
                malformed (see `region_stack`).
        """
        if opd.ndim < 2:
            msg = f"opd must be at least 2D (..., H, W) (got {opd.ndim}D)"
            raise ValueError(msg)

        region_op = self._sum if reduce else apply_mask
        result = region_op(opd, mask)

        return (result * self._opd_scale).astype(np.float32, copy=False)

    def calc_from_height(
        self,
        height: NDArray[np.float32],
        *,
        mask: MaskLike | None = None,
        reduce: bool = True,
    ) -> NDArray[np.float32]:
        """Integrate an optical height map (nm) into volume [um^3].

        The direct ``sum(height * pixel_area)`` form of `calc_from_opd` (which
        divides by `refractive_delta` first); shapes, `mask`, and `reduce` behave
        identically.

        Raises:
            ValueError: If `height` is not at least 2-D ``(..., H, W)``, or the mask
                is malformed (see `region_stack`).
        """
        if height.ndim < 2:
            msg = f"height must be at least 2D (..., H, W) (got {height.ndim}D)"
            raise ValueError(msg)

        region_op = self._sum if reduce else apply_mask
        result = region_op(height, mask)

        return (result * self._height_scale).astype(np.float32, copy=False)

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
    pixel_size: float,
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
    converter = OpticalHeightConverter(refractive_delta=refractive_delta)
    calculator = OpticalVolumeCalculator(
        pixel_size=pixel_size, height_converter=converter
    )
    return calculator.calc_from_opd(opd, mask=mask, reduce=reduce)


def calc_volume_from_phase(
    phase: NDArray[np.float32],
    *,
    pixel_size: float,
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
    return OpticalVolumeCalculator.from_wavelength(
        pixel_size=pixel_size, wavelength=wavelength, refractive_delta=refractive_delta
    ).calc_from_phase(phase, mask=mask, reduce=reduce)
