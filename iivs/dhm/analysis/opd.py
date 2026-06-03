from __future__ import annotations

__all__ = ("OPDConverter", "opd_to_phase", "phase_to_opd")

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from iivs.dhm.data.constants import DEFAULT_WAVELENGTH

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

    ``OPD = phase * wavelength / (2 * pi)`` -- independent of any refractive
    index, and distinct from the physical height `PhaseUnit.METERS` represents
    (height additionally divides by the refractive-index difference).

    OPD is expressed in **nanometers** (the quantitative-phase-imaging
    convention), while the input `wavelength` is SI (meters) like the rest of
    the package -- use `from_wavelength_nm` / `wavelength_nm` for its nanometer
    form. The scale (in nm of OPD per radian) is cached once and exposed as
    `opd_scale`.

    The module-level `phase_to_opd` / `opd_to_phase` are one-shot conveniences
    over this class (as `json.dumps` is over `json.JSONEncoder`).

    Attributes:
        wavelength: Illumination wavelength, in meters.
    """

    wavelength: float = DEFAULT_WAVELENGTH
    _scale: float = field(init=False, repr=False, compare=False)  # nm of OPD / rad

    def __post_init__(self) -> None:
        """Validate the wavelength and cache the phase-to-OPD scale (nm/rad)."""
        if self.wavelength <= 0:
            msg = f"wavelength must be positive (got {self.wavelength})"
            raise ValueError(msg)
        object.__setattr__(self, "_scale", self.wavelength / (2.0 * np.pi) * 1e9)

    @classmethod
    def from_wavelength_nm(cls, wavelength_nm: float) -> Self:
        """Build a converter from a wavelength given in nanometers."""
        return cls(wavelength=wavelength_nm * 1e-9)

    @property
    def wavelength_nm(self) -> float:
        """The wavelength in nanometers."""
        return self.wavelength * 1e9

    @property
    def opd_scale(self) -> float:
        """Nanometers of OPD per radian of phase (``wavelength / (2 * pi)`` in nm).

        OPD's canonical unit here is nanometers, so this needs no suffix (cf.
        `wavelength` vs `wavelength_nm`). It is the nm analogue of
        `PhaseBinHeader.height_scale_nm`: ``height_scale_nm = opd_scale / delta_n``.
        """
        return self._scale

    def convert_to_opd(self, phase: NDArray[np.float32]) -> NDArray[np.float32]:
        """Convert `phase` (radians) to OPD (nanometers) at this wavelength."""
        return (phase * self._scale).astype(np.float32, copy=False)

    def convert_to_phase(self, opd: NDArray[np.float32]) -> NDArray[np.float32]:
        """Convert `opd` (nanometers) to phase (radians) at this wavelength."""
        return (opd / self._scale).astype(np.float32, copy=False)


def phase_to_opd(
    phase: NDArray[np.float32], *, wavelength: float = DEFAULT_WAVELENGTH
) -> NDArray[np.float32]:
    """Convert phase (radians) to OPD (nanometers); a one-shot `OPDConverter`.

    ``OPD = phase * wavelength / (2 * pi)``, in nanometers, independent of any
    refractive index. For repeated conversions at one wavelength, reuse an
    `OPDConverter`.

    Args:
        phase: Phase image or stack, in radians.
        wavelength: Illumination wavelength, in meters.

    Returns:
        OPD as a float32 array of the same shape, in nanometers.
    """
    return OPDConverter(wavelength=wavelength).convert_to_opd(phase)


def opd_to_phase(
    opd: NDArray[np.float32], *, wavelength: float = DEFAULT_WAVELENGTH
) -> NDArray[np.float32]:
    """Inverse of `phase_to_opd`; one-shot `OPDConverter.convert_to_phase` (nm)."""
    return OPDConverter(wavelength=wavelength).convert_to_phase(opd)
