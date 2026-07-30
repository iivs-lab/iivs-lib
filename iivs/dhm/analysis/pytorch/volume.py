from __future__ import annotations

__all__ = (
    "OpticalVolume",
    "calc_volume",
    "calc_volume_from_height",
    "calc_volume_from_opd",
)

from typing import TYPE_CHECKING

from torch import nn

from iivs.common.data.pytorch.reduction import Sum, apply_mask
from iivs.dhm.analysis.pytorch.area import ProjectedArea
from iivs.dhm.analysis.pytorch.height import OpticalHeight
from iivs.dhm.constants import DEFAULT_REFRACTIVE_DELTA, DEFAULT_WAVELENGTH

if TYPE_CHECKING:
    from typing import Self

    from torch import Tensor


class OpticalVolume(nn.Module):
    """Torch `nn.Module` for the per-pixel volume density (um^3) from phase.

    The torch twin of `iivs.dhm.analysis.volume.OpticalVolumeCalculator`, mirroring
    its composition: the module owns a `ProjectedArea` submodule (`area_calculator`,
    carrying the pixel size, which has no lab default here) and an `OpticalHeight`
    submodule (`height_converter`), and binds the cached
    `volume_scale` (``area_scale * height_scale * 1e-3``, um^3 of volume per rad of
    phase, a plain float derived from the two submodules' factors, so the physics is
    shared). Phase is the canonical input: `forward(phase) = phase * volume_scale`
    is the per-pixel volume density (um^3) of a background-corrected phase map (rad);
    an OPD or height map enters through phase via `convert_from_opd` /
    `convert_from_height`. The `convert_*` / `forward` methods are pure scalar
    multiplies, so they preserve the input tensor's dtype, device, and autograd
    graph, dropping cleanly into `nn.Sequential`, forward hooks, `torch.jit.script`,
    and `torch.compile`. Masking into regions and reducing to a total volume are a
    separate concern: compose with the `iivs.common.data.pytorch` reductions, e.g.
    `Sum(mask=cell)(OpticalVolume.from_args(...)(phase))`, or use the `calc_volume`
    one-shot.

    Attributes:
        area_calculator: The owned footprint submodule (the `area` in ``area *
            mean(height)``); carries the pixel size.
        height_converter: The owned phase <-> height submodule; carries the
            refractive-index difference and wavelength.
        volume_scale: um^3 of volume per rad of phase, per pixel.
    """

    def __init__(
        self,
        *,
        area_calculator: ProjectedArea,
        height_converter: OpticalHeight,
    ) -> None:
        """Bind the area and phase <-> height submodules (see `from_args`)."""
        super().__init__()

        self.area_calculator = area_calculator
        self.height_converter = height_converter
        self.volume_scale = (
            self.area_calculator.area_scale * self.height_converter.height_scale * 1e-3
        )

    @classmethod
    def from_args(
        cls, *, pixel_size: float, wavelength: float, refractive_delta: float
    ) -> Self:
        """Build a module from plain parameters, constructing both submodules."""
        area_calculator = ProjectedArea(pixel_size=pixel_size)
        height_converter = OpticalHeight.from_args(
            wavelength=wavelength, refractive_delta=refractive_delta
        )
        return cls(area_calculator=area_calculator, height_converter=height_converter)

    @property
    def pixel_size(self) -> float:
        """The owned area submodule's pixel size, in m."""
        return self.area_calculator.pixel_size

    @property
    def pixel_size_um(self) -> float:
        """The owned area submodule's pixel size, in um."""
        return self.area_calculator.pixel_size_um

    @property
    def refractive_delta(self) -> float:
        """The owned height submodule's refractive-index difference."""
        return self.height_converter.refractive_delta

    @property
    def wavelength(self) -> float:
        """The owned height submodule's wavelength, in m."""
        return self.height_converter.wavelength

    @property
    def wavelength_nm(self) -> float:
        """The owned height submodule's wavelength, in nm."""
        return self.height_converter.wavelength_nm

    def convert_from_opd(self, opd: Tensor) -> Tensor:
        """Map an OPD map (nm) to its volume density (um^3 per pixel) through phase.

        The owned height submodule's `opd_converter` first maps `opd` back to phase
        (its wavelength then cancels), which `forward` integrates.
        """
        phase = self.height_converter.opd_converter.convert_to_phase(opd)
        return self.forward(phase)

    def convert_from_height(self, height: Tensor) -> Tensor:
        """Map an optical height (nm) to its volume density (um^3), through phase."""
        phase = self.height_converter.convert_to_phase(height)
        return self.forward(phase)

    def forward(self, phase: Tensor) -> Tensor:
        """Map a phase map (rad) to its per-pixel volume density: `phase * scale`."""
        return phase * self.volume_scale


def calc_volume(
    phase: Tensor,
    *,
    pixel_size: float,
    wavelength: float = DEFAULT_WAVELENGTH,
    refractive_delta: float = DEFAULT_REFRACTIVE_DELTA,
    mask: Tensor | None = None,
    reduce: bool = True,
) -> Tensor:
    """Integrate a phase map (rad) into optical volume [um^3] at `wavelength`.

    The canonical one-shot, composing `OpticalVolume` (per-pixel density) with the
    `iivs.common.data.pytorch` reductions and keeping the input's device and
    autograd graph. `calc_volume_from_opd` / `calc_volume_from_height` are the OPD /
    height entry points.

    Args:
        phase: Phase map(s), in rad, shape ``(..., H, W)``, already
            background-corrected.
        pixel_size: Physical size of one (square) pixel, in m.
        wavelength: Illumination wavelength, in m.
        refractive_delta: Refractive-index difference ``n_object - n_medium``.
        mask: Optional region mask (boolean or integer labels); see
            `iivs.common.data.pytorch.region_stack`.
        reduce: If True (default), sum each masked region to a volume; if False,
            return the masked per-pixel density map.
    """
    module = OpticalVolume.from_args(
        pixel_size=pixel_size, wavelength=wavelength, refractive_delta=refractive_delta
    )
    density = module(phase)
    # empty region -> 0 volume, matching the NumPy `OpticalVolumeCalculator`
    return Sum(empty=0.0)(density, mask) if reduce else apply_mask(density, mask)


def calc_volume_from_opd(
    opd: Tensor,
    *,
    pixel_size: float,
    wavelength: float = DEFAULT_WAVELENGTH,
    refractive_delta: float = DEFAULT_REFRACTIVE_DELTA,
    mask: Tensor | None = None,
    reduce: bool = True,
) -> Tensor:
    """Integrate an OPD map (nm) into optical volume [um^3], entering through phase.

    Composes `OpticalVolume.convert_from_opd` (the wavelength cancels) with the
    `iivs.common.data.pytorch` reductions, keeping `opd`'s device and autograd
    graph.

    Args:
        opd: OPD map(s), in nm, shape ``(..., H, W)``, already background-corrected.
        pixel_size: Physical size of one (square) pixel, in m.
        wavelength: Illumination wavelength, in m; the OPD volume never depends on
            it (it cancels), so the default serves.
        refractive_delta: Refractive-index difference ``n_object - n_medium``.
        mask: Optional region mask (boolean or integer labels).
        reduce: Sum each masked region to a volume (True), or return the masked
            per-pixel density map (False).
    """
    module = OpticalVolume.from_args(
        pixel_size=pixel_size, wavelength=wavelength, refractive_delta=refractive_delta
    )
    density = module.convert_from_opd(opd)
    return Sum(empty=0.0)(density, mask) if reduce else apply_mask(density, mask)


def calc_volume_from_height(
    height: Tensor,
    *,
    pixel_size: float,
    wavelength: float = DEFAULT_WAVELENGTH,
    refractive_delta: float = DEFAULT_REFRACTIVE_DELTA,
    mask: Tensor | None = None,
    reduce: bool = True,
) -> Tensor:
    """Integrate an optical height map (nm) into optical volume [um^3] through phase.

    Composes `OpticalVolume.convert_from_height` with the `iivs.common.data.pytorch`
    reductions, keeping the input's device and autograd graph.

    Args:
        height: Optical height map(s), in nm, shape ``(..., H, W)``, already
            background-corrected.
        pixel_size: Physical size of one (square) pixel, in m.
        wavelength: Illumination wavelength, in m.
        refractive_delta: Refractive-index difference ``n_object - n_medium``.
        mask: Optional region mask (boolean or integer labels).
        reduce: Sum each masked region to a volume (True), or return the masked
            per-pixel density map (False).
    """
    module = OpticalVolume.from_args(
        pixel_size=pixel_size, wavelength=wavelength, refractive_delta=refractive_delta
    )
    density = module.convert_from_height(height)
    return Sum(empty=0.0)(density, mask) if reduce else apply_mask(density, mask)
