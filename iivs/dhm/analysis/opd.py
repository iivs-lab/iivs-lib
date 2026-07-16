from __future__ import annotations

__all__ = ("OPDConverter", "opd_to_phase", "phase_to_opd")

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from iivs.dhm.constants import DEFAULT_WAVELENGTH

if TYPE_CHECKING:
    from typing import Self

    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class OPDConverter:
    """Convert between phase and optical path difference (OPD) at a wavelength.

    Bind the wavelength once, then convert repeatedly in either direction::

        conv = OPDConverter.from_wavelength_nm(666)  # or OPDConverter(666e-9)
        opd = conv.convert_to_opd(phase)
        phase = conv.convert_to_phase(opd)

    ``OPD = phase * wavelength / (2 * pi)``, independent of refractive index (distinct
    from the height `PhaseUnit.METERS` represents, which additionally divides by the
    refractive-index difference). OPD is in **nm** (the QPI convention) while
    `wavelength` is SI (m); the per-rad scale is cached as `opd_scale`. For PyTorch, use
    `OpticalPathDifference` from `iivs.dhm.analysis.pytorch`.

    Attributes:
        wavelength: Illumination wavelength, in m.
    """

    wavelength: float = DEFAULT_WAVELENGTH
    _scale: float = field(init=False, repr=False, compare=False)  # nm of OPD / rad

    def __post_init__(self) -> None:
        """Validate the wavelength and cache the phase-to-OPD scale (nm/rad)."""
        if self.wavelength <= 0:
            msg = f"wavelength must be positive (got {self.wavelength})"
            raise ValueError(msg)
        object.__setattr__(self, "_scale", self.wavelength / math.tau * 1e9)

    @classmethod
    def from_wavelength_nm(cls, wavelength_nm: float) -> Self:
        """Build a converter from a wavelength given in nm."""
        return cls(wavelength=wavelength_nm * 1e-9)

    @property
    def wavelength_nm(self) -> float:
        """The wavelength in nm."""
        return self.wavelength * 1e9

    @property
    def opd_scale(self) -> float:
        """nm of OPD per rad of phase (``wavelength / (2 * pi)`` in nm).

        OPD's canonical unit here is nm, so this needs no suffix (cf. `wavelength` vs
        `wavelength_nm`).
        """
        return self._scale

    def convert_to_opd(self, phase: NDArray[np.float32]) -> NDArray[np.float32]:
        """Convert `phase` (rad) to OPD (nm) at this wavelength."""
        return (phase * self._scale).astype(np.float32, copy=False)

    def convert_to_phase(self, opd: NDArray[np.float32]) -> NDArray[np.float32]:
        """Convert `opd` (nm) to phase (rad) at this wavelength."""
        return (opd / self._scale).astype(np.float32, copy=False)


def phase_to_opd(
    phase: NDArray[np.float32], *, wavelength: float = DEFAULT_WAVELENGTH
) -> NDArray[np.float32]:
    """Convert phase (rad) to OPD (nm); a one-shot `OPDConverter`.

    For repeated conversions at one wavelength, reuse an `OPDConverter`.

    Args:
        phase: Phase image or stack, in rad.
        wavelength: Illumination wavelength, in m.

    Returns:
        OPD as a float32 array of the same shape, in nm.
    """
    return OPDConverter(wavelength=wavelength).convert_to_opd(phase)


def opd_to_phase(
    opd: NDArray[np.float32], *, wavelength: float = DEFAULT_WAVELENGTH
) -> NDArray[np.float32]:
    """Convert OPD (nm) to phase (rad); a one-shot `OPDConverter`.

    The inverse of `phase_to_opd`. For repeated conversions at one wavelength, reuse an
    `OPDConverter`.

    Args:
        opd: OPD image or stack, in nm.
        wavelength: Illumination wavelength, in m.

    Returns:
        Phase as a float32 array of the same shape, in rad.
    """
    return OPDConverter(wavelength=wavelength).convert_to_phase(opd)
