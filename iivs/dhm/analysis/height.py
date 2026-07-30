from __future__ import annotations

__all__ = (
    "OpticalHeightConverter",
    "height_to_opd",
    "height_to_phase",
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
    """A phase <-> optical height converter bound to one wavelength and delta.

    Bind the refractive-index difference (and, via the OPD converter, the
    wavelength) once, then convert repeatedly::

        conv = OpticalHeightConverter()  # lab defaults
        height = conv.convert_from_phase(phase)  # phase in rad -> height in nm
        height = conv.convert_from_opd(opd)  # opd in nm -> height in nm

    ``height = phase * wavelength / (2 * pi * refractive_delta)``, in nm: the
    physical thickness that produces the measured phase. Transmission QPI
    literature usually calls this quantity the sample *thickness* (``phase =
    2*pi * delta * t / wavelength``); "height" keeps this library's established
    name for it. Phase is the preferred representation: an OPD input is first
    mapped back to phase by the bound `opd_converter`, whose wavelength then
    cancels out of the composition (``height == opd / refractive_delta``).
    For PyTorch, use `OpticalHeight` from `iivs.dhm.analysis.pytorch`.

    Attributes:
        refractive_delta: Refractive-index difference ``n_object - n_medium``.
        opd_converter: Phase <-> OPD converter backing the phase path's scale and
            the `opd` entry / exit. Defaults to one at the default wavelength.
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
    def from_args(cls, *, wavelength: float, refractive_delta: float) -> Self:
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

        Height's canonical unit here is nm, so this needs no suffix (cf.
        `wavelength` vs `wavelength_nm`).
        """
        return self._scale

    def convert_from_phase(self, phase: NDArray[np.float32]) -> NDArray[np.float32]:
        """Convert `phase` (rad) to optical height (nm) at this wavelength and delta."""
        return (phase * self._scale).astype(np.float32, copy=False)

    def convert_to_phase(self, height: NDArray[np.float32]) -> NDArray[np.float32]:
        """Convert optical `height` (nm) to phase (rad) at this wavelength and delta."""
        return (height / self._scale).astype(np.float32, copy=False)

    def convert_from_opd(self, opd: NDArray[np.float32]) -> NDArray[np.float32]:
        """Convert `opd` (nm) to optical height (nm), entering through phase.

        The bound `opd_converter` first maps `opd` back to phase, then the
        phase-to-height scale applies. Its wavelength cancels out of that
        composition, so the result is ``opd / refractive_delta`` regardless of it.
        """
        phase = self.opd_converter.convert_to_phase(opd)
        return self.convert_from_phase(phase)

    def convert_to_opd(self, height: NDArray[np.float32]) -> NDArray[np.float32]:
        """Convert optical `height` (nm) to OPD (nm), exiting through phase.

        The inverse of `convert_from_opd`; the bound wavelength cancels the same
        way, leaving ``height * refractive_delta``.
        """
        phase = self.convert_to_phase(height)
        return self.opd_converter.convert_from_phase(phase)


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
    converter = OpticalHeightConverter.from_args(
        wavelength=wavelength, refractive_delta=refractive_delta
    )
    return converter.convert_from_phase(phase)


def height_to_phase(
    height: NDArray[np.float32],
    *,
    wavelength: float = DEFAULT_WAVELENGTH,
    refractive_delta: float = DEFAULT_REFRACTIVE_DELTA,
) -> NDArray[np.float32]:
    """Convert optical height (nm) to phase (rad); a one-shot converter.

    The inverse of `phase_to_height`. Unlike the OPD one-shots, this conversion
    genuinely depends on the wavelength. For repeated conversions at one
    wavelength and delta, reuse an `OpticalHeightConverter`.

    Args:
        height: Optical height image or stack, in nm.
        wavelength: Illumination wavelength, in m.
        refractive_delta: Refractive-index difference ``n_object - n_medium``.

    Returns:
        Phase as a float32 array of the same shape, in rad.
    """
    converter = OpticalHeightConverter.from_args(
        wavelength=wavelength, refractive_delta=refractive_delta
    )
    return converter.convert_to_phase(height)


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
    return converter.convert_from_opd(opd)


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
