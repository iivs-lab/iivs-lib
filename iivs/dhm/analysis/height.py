from __future__ import annotations

__all__ = (
    "OpticalHeightConverter",
    "height_to_opd",
    "opd_to_height",
    "phase_to_height",
)

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from iivs.dhm.analysis.opd import OPDConverter
from iivs.dhm.constants import DEFAULT_REFRACTIVE_DELTA, DEFAULT_WAVELENGTH

if TYPE_CHECKING:
    from typing import Self

    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class OpticalHeightConverter:
    """An OPD <-> optical height converter bound to one refractive-index difference.

    Bind the refractive-index difference once, then convert repeatedly::

        conv = OpticalHeightConverter()  # lab defaults
        height = conv.convert_to_height(opd)  # opd in nm -> height in nm
        height = conv.convert_from_phase(phase)  # phase in rad -> height in nm

    ``height = OPD / refractive_delta``, both in nm: the physical thickness that
    produces the measured path difference. Transmission QPI literature usually calls
    this quantity the sample *thickness* (``phase = 2*pi * delta * t / wavelength``);
    "height" here follows Koala's and this library's data-layer naming (the `.bin`
    header's `height_scale`, `PhaseUnit.METERS`). It is the same height the data
    layer's `PhaseUnit.NANOMETERS` represents, so `convert_from_phase` agrees
    numerically with `convert_phase_unit(..., target=NANOMETERS)` when the file's
    stored `height_scale` was built from the same wavelength and delta. For PyTorch,
    use `OpticalHeight` from `iivs.dhm.analysis.pytorch`.

    Attributes:
        refractive_delta: Refractive-index difference ``n_object - n_medium``.
        opd_converter: Phase-to-OPD converter used by `convert_from_phase`. Defaults
            to one at the default wavelength.
    """

    refractive_delta: float = DEFAULT_REFRACTIVE_DELTA
    opd_converter: OPDConverter = field(default_factory=OPDConverter)

    _scale: float = field(init=False, repr=False, compare=False)  # nm of height / rad

    def __post_init__(self) -> None:
        """Validate the delta and cache the phase-to-height scale (nm/rad)."""
        if self.refractive_delta <= 0:
            msg = f"refractive_delta must be positive (got {self.refractive_delta})"
            raise ValueError(msg)
        scale = self.opd_converter.opd_scale / self.refractive_delta
        object.__setattr__(self, "_scale", scale)

    @classmethod
    def from_wavelength(
        cls,
        *,
        wavelength: float = DEFAULT_WAVELENGTH,
        refractive_delta: float = DEFAULT_REFRACTIVE_DELTA,
    ) -> Self:
        """Build a converter whose phase path uses `wavelength` (in m)."""
        opd_converter = OPDConverter(wavelength=wavelength)
        return cls(refractive_delta=refractive_delta, opd_converter=opd_converter)

    @property
    def wavelength(self) -> float:
        """The bound OPD converter's wavelength, in m."""
        return self.opd_converter.wavelength

    @property
    def wavelength_nm(self) -> float:
        """The bound OPD converter's wavelength, in nm."""
        return self.opd_converter.wavelength_nm

    @property
    def height_scale(self) -> float:
        """nm of height per rad of phase (``wavelength / (2 * pi * delta)`` in nm).

        The nm twin of the `.bin` header's `height_scale` (m per rad); the same
        conversion the data layer applies for `PhaseUnit.NANOMETERS`.
        """
        return self._scale

    def convert_to_height(self, opd: NDArray[np.float32]) -> NDArray[np.float32]:
        """Convert `opd` (nm) to optical height (nm) at this delta."""
        return (opd / self.refractive_delta).astype(np.float32, copy=False)

    def convert_to_opd(self, height: NDArray[np.float32]) -> NDArray[np.float32]:
        """Convert optical `height` (nm) to OPD (nm) at this delta."""
        return (height * self.refractive_delta).astype(np.float32, copy=False)

    def convert_from_phase(self, phase: NDArray[np.float32]) -> NDArray[np.float32]:
        """Convert `phase` (rad) to optical height (nm) via the bound wavelength."""
        return (phase * self._scale).astype(np.float32, copy=False)


def opd_to_height(
    opd: NDArray[np.float32],
    *,
    refractive_delta: float = DEFAULT_REFRACTIVE_DELTA,
) -> NDArray[np.float32]:
    """Convert OPD (nm) to optical height (nm); a one-shot `OpticalHeightConverter`.

    For repeated conversions at one delta, reuse an `OpticalHeightConverter`.

    Args:
        opd: OPD image or stack, in nm.
        refractive_delta: Refractive-index difference ``n_object - n_medium``.

    Returns:
        Optical height as a float32 array of the same shape, in nm.
    """
    converter = OpticalHeightConverter(refractive_delta=refractive_delta)
    return converter.convert_to_height(opd)


def height_to_opd(
    height: NDArray[np.float32],
    *,
    refractive_delta: float = DEFAULT_REFRACTIVE_DELTA,
) -> NDArray[np.float32]:
    """Convert optical height (nm) to OPD (nm); a one-shot `OpticalHeightConverter`.

    The inverse of `opd_to_height`. For repeated conversions at one delta, reuse an
    `OpticalHeightConverter`.

    Args:
        height: Optical height image or stack, in nm.
        refractive_delta: Refractive-index difference ``n_object - n_medium``.

    Returns:
        OPD as a float32 array of the same shape, in nm.
    """
    converter = OpticalHeightConverter(refractive_delta=refractive_delta)
    return converter.convert_to_opd(height)


def phase_to_height(
    phase: NDArray[np.float32],
    *,
    wavelength: float = DEFAULT_WAVELENGTH,
    refractive_delta: float = DEFAULT_REFRACTIVE_DELTA,
) -> NDArray[np.float32]:
    """Convert phase (rad) to optical height (nm); a one-shot converter.

    For repeated conversions at one wavelength and delta, reuse an
    `OpticalHeightConverter`.

    Args:
        phase: Phase image or stack, in rad.
        wavelength: Illumination wavelength, in m.
        refractive_delta: Refractive-index difference ``n_object - n_medium``.

    Returns:
        Optical height as a float32 array of the same shape, in nm.
    """
    converter = OpticalHeightConverter.from_wavelength(
        wavelength=wavelength, refractive_delta=refractive_delta
    )
    return converter.convert_from_phase(phase)
