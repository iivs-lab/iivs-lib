from __future__ import annotations

__all__ = (
    "DryMass",
    "calc_drymass",
    "calc_drymass_from_height",
    "calc_drymass_from_opd",
)

from typing import TYPE_CHECKING

from torch import nn

from iivs.common.data.pytorch.reduction import reduce_regions
from iivs.dhm.analysis.pytorch.volume import OpticalVolume
from iivs.dhm.constants import (
    DEFAULT_REFRACTIVE_DELTA,
    DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    DEFAULT_WAVELENGTH,
)

if TYPE_CHECKING:
    from typing import Self

    from torch import Tensor


class DryMass(nn.Module):
    """Torch `nn.Module` for the per-pixel dry-mass density (pg) from phase.

    The torch twin of `iivs.dhm.analysis.drymass.DryMassCalculator`, mirroring its
    composition: the module owns an `OpticalVolume` submodule (`volume_converter`,
    itself owning the area and height submodules) and the specific refractive
    increment, and binds the cached `drymass_scale` (``volume_scale *
    refractive_delta / alpha * 1e-3``, pg of dry mass per rad of phase, a plain
    float derived from the volume engine so the physics is shared). Phase is the
    canonical input: `forward(phase) = phase * drymass_scale` is the per-pixel
    dry-mass density (pg) of a background-corrected phase map (rad); an OPD or
    height map enters through phase via `convert_from_opd` / `convert_from_height`.
    The `convert_*` / `forward` methods are pure scalar multiplies, so they preserve
    the input tensor's dtype, device, and autograd graph. Masking into regions and
    reducing to a total dry mass are a separate concern: compose with the
    `iivs.common.data.pytorch` reductions, or use the `calc_drymass` one-shot.

    Attributes:
        volume_converter: The owned volume submodule; carries the pixel size,
            wavelength, and refractive-index difference.
        alpha: Specific refractive increment, in m^3/kg.
        drymass_scale: pg of dry mass per rad of phase, per pixel.
    """

    def __init__(
        self,
        *,
        volume_converter: OpticalVolume,
        alpha: float = DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    ) -> None:
        """Bind the volume submodule and the specific refractive increment."""
        super().__init__()

        self.volume_converter = volume_converter
        self.alpha = alpha
        # pg per rad of phase, via the volume engine: mass = volume * delta / alpha
        delta = volume_converter.refractive_delta
        self.drymass_scale = volume_converter.volume_scale * delta / alpha * 1e-3

    @classmethod
    def from_args(
        cls,
        *,
        pixel_size: float,
        wavelength: float,
        refractive_delta: float,
        alpha: float,
    ) -> Self:
        """Build a module from plain parameters, constructing the engine chain."""
        volume_converter = OpticalVolume.from_args(
            pixel_size=pixel_size,
            wavelength=wavelength,
            refractive_delta=refractive_delta,
        )
        return cls(volume_converter=volume_converter, alpha=alpha)

    @property
    def pixel_size(self) -> float:
        """The bound volume engine's pixel size, in m."""
        return self.volume_converter.pixel_size

    @property
    def pixel_size_um(self) -> float:
        """The bound volume engine's pixel size, in um."""
        return self.volume_converter.pixel_size_um

    @property
    def refractive_delta(self) -> float:
        """The bound volume engine's refractive-index difference."""
        return self.volume_converter.refractive_delta

    @property
    def wavelength(self) -> float:
        """The bound volume engine's wavelength, in m."""
        return self.volume_converter.wavelength

    @property
    def wavelength_nm(self) -> float:
        """The bound volume engine's wavelength, in nm."""
        return self.volume_converter.wavelength_nm

    def forward(self, phase: Tensor) -> Tensor:
        """Map a phase map (rad) to its per-pixel dry-mass density: `phase * scale`."""
        return phase * self.drymass_scale

    def convert_from_opd(self, opd: Tensor) -> Tensor:
        """Map an OPD map (nm) to its dry-mass density (pg per pixel) through phase.

        The owned volume engine's `opd_converter` first maps `opd` back to phase
        (its wavelength then cancels), which `forward` integrates.
        """
        opd_converter = self.volume_converter.height_converter.opd_converter
        return self.forward(opd_converter.convert_to_phase(opd))

    def convert_from_height(self, height: Tensor) -> Tensor:
        """Map an optical height (nm) to its dry-mass density (pg), through phase."""
        phase = self.volume_converter.height_converter.convert_to_phase(height)
        return self.forward(phase)


def calc_drymass(
    phase: Tensor,
    *,
    pixel_size: float,
    wavelength: float = DEFAULT_WAVELENGTH,
    refractive_delta: float = DEFAULT_REFRACTIVE_DELTA,
    alpha: float = DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    mask: Tensor | None = None,
    reduce: bool = True,
) -> Tensor:
    """Integrate a phase map (rad) into dry mass [pg] at `wavelength`.

    The canonical one-shot, composing `DryMass` (per-pixel density) with the
    `iivs.common.data.pytorch` reductions and keeping the input's device and
    autograd graph. `calc_drymass_from_opd` / `calc_drymass_from_height` are the
    OPD / height entry points.

    Args:
        phase: Phase map(s), in rad, shape ``(..., H, W)``, already
            background-corrected.
        pixel_size: Physical size of one (square) pixel, in m.
        wavelength: Illumination wavelength, in m.
        refractive_delta: Refractive-index difference ``n_object - n_medium``;
            enters only the engine chain and cancels out of the mass.
        alpha: Specific refractive increment, in m^3/kg.
        mask: Optional region mask (boolean or integer labels); see
            `iivs.common.data.pytorch.region_stack`.
        reduce: If True (default), sum each masked region to a dry mass; if False,
            return the masked per-pixel density map.
    """
    module = DryMass.from_args(
        pixel_size=pixel_size,
        wavelength=wavelength,
        refractive_delta=refractive_delta,
        alpha=alpha,
    )
    return reduce_regions(module(phase), mask, reduce=reduce)


def calc_drymass_from_opd(
    opd: Tensor,
    *,
    pixel_size: float,
    wavelength: float = DEFAULT_WAVELENGTH,
    refractive_delta: float = DEFAULT_REFRACTIVE_DELTA,
    alpha: float = DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    mask: Tensor | None = None,
    reduce: bool = True,
) -> Tensor:
    """Integrate an OPD map (nm) into dry mass [pg], entering through phase.

    Composes `DryMass.convert_from_opd` (the wavelength cancels) with the
    `iivs.common.data.pytorch` reductions, keeping `opd`'s device and autograd
    graph.

    Args:
        opd: OPD map(s), in nm, shape ``(..., H, W)``, already background-corrected.
        pixel_size: Physical size of one (square) pixel, in m.
        wavelength: Illumination wavelength, in m; the OPD mass never depends on it
            (it cancels), so the default serves.
        refractive_delta: Refractive-index difference ``n_object - n_medium``.
        alpha: Specific refractive increment, in m^3/kg.
        mask: Optional region mask (boolean or integer labels).
        reduce: Sum each masked region to a dry mass (True), or return the masked
            per-pixel density map (False).
    """
    module = DryMass.from_args(
        pixel_size=pixel_size,
        wavelength=wavelength,
        refractive_delta=refractive_delta,
        alpha=alpha,
    )
    return reduce_regions(module.convert_from_opd(opd), mask, reduce=reduce)


def calc_drymass_from_height(
    height: Tensor,
    *,
    pixel_size: float,
    wavelength: float = DEFAULT_WAVELENGTH,
    refractive_delta: float = DEFAULT_REFRACTIVE_DELTA,
    alpha: float = DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    mask: Tensor | None = None,
    reduce: bool = True,
) -> Tensor:
    """Integrate an optical height map (nm) into dry mass [pg] through phase.

    Composes `DryMass.convert_from_height` with the `iivs.common.data.pytorch`
    reductions, keeping the input's device and autograd graph.

    Args:
        height: Optical height map(s), in nm, shape ``(..., H, W)``, already
            background-corrected.
        pixel_size: Physical size of one (square) pixel, in m.
        wavelength: Illumination wavelength, in m.
        refractive_delta: Refractive-index difference ``n_object - n_medium``.
        alpha: Specific refractive increment, in m^3/kg.
        mask: Optional region mask (boolean or integer labels).
        reduce: Sum each masked region to a dry mass (True), or return the masked
            per-pixel density map (False).
    """
    module = DryMass.from_args(
        pixel_size=pixel_size,
        wavelength=wavelength,
        refractive_delta=refractive_delta,
        alpha=alpha,
    )
    return reduce_regions(module.convert_from_height(height), mask, reduce=reduce)
