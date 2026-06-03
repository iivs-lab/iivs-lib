from __future__ import annotations

__all__ = ("DryMassCalculator", "calc_drymass", "calc_drymass_from_phase")

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from iivs.dhm.analysis.opd import OPDConverter
from iivs.dhm.data.constants import (
    DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    DEFAULT_WAVELENGTH,
)

if TYPE_CHECKING:
    from typing import Self

    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class DryMassCalculator:
    """Integrate OPD (or phase) into dry mass (pg) at a fixed pixel size and alpha.

    Bind the pixel size, specific refractive increment, and -- for the phase
    path -- an `OPDConverter` once; the per-pixel mass factor is precomputed::

        dmc = DryMassCalculator(pixel_size=header.pixel_size)  # defaults
        dmc = DryMassCalculator.from_wavelength(pixel_size=px, wavelength=666e-9)
        dmc = DryMassCalculator(
            pixel_size=px, opd_converter=OPDConverter.from_wavelength_nm(666)
        )
        mass = dmc.calc_from_opd(opd, mask=cell)  # opd in nanometers
        mass = dmc.calc_from_phase(phase, mask=cell)  # phase in radians

    Dry mass is ``(1 / alpha) * sum(OPD * pixel_area)`` (Barer), in picograms.
    The OPD must already be background-corrected (≈ 0 outside the object); pass
    `mask` to restrict the sum to one segmented object -- segmentation and
    background estimation stay the caller's responsibility. The sum is
    accumulated in float64.

    The module-level `calc_drymass` / `calc_drymass_from_phase` are one-shot
    conveniences over this class (as `json.dumps` is over `json.JSONEncoder`).

    Attributes:
        pixel_size: Physical size of one (square) pixel, in meters.
        alpha: Specific refractive increment, in mL/g (= um^3/pg).
        opd_converter: Phase-to-OPD converter used by `calc_from_phase`.
            Defaults to one at the default wavelength; inject your own or use
            `from_wavelength`.
    """

    pixel_size: float
    alpha: float = DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT
    opd_converter: OPDConverter = field(default_factory=OPDConverter)
    # pg of dry mass per nanometer of OPD summed over pixels:
    _scale: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate inputs and precompute the per-pixel mass factor."""
        if self.pixel_size <= 0:
            msg = f"pixel_size must be positive (got {self.pixel_size})"
            raise ValueError(msg)
        if self.alpha <= 0:
            msg = f"alpha must be positive (got {self.alpha})"
            raise ValueError(msg)
        pixel_area_um2 = (self.pixel_size * 1e6) ** 2  # m^2 -> um^2; nm -> um is 1e-3
        object.__setattr__(self, "_scale", pixel_area_um2 * 1e-3 / self.alpha)

    @classmethod
    def from_wavelength(
        cls,
        *,
        pixel_size: float,
        alpha: float = DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
        wavelength: float = DEFAULT_WAVELENGTH,
    ) -> Self:
        """Build a calculator whose phase path uses `wavelength` (in meters)."""
        return cls(
            pixel_size=pixel_size,
            alpha=alpha,
            opd_converter=OPDConverter(wavelength=wavelength),
        )

    @property
    def wavelength(self) -> float:
        """The bound OPD converter's wavelength, in meters (shortcut)."""
        return self.opd_converter.wavelength

    @property
    def wavelength_nm(self) -> float:
        """The bound OPD converter's wavelength, in nanometers (shortcut)."""
        return self.opd_converter.wavelength_nm

    @property
    def drymass_scale(self) -> float:
        """Picograms of dry mass per nanometer of OPD summed over pixels.

        The cached ``pixel_area / alpha`` factor (with unit bookkeeping):
        ``mass == drymass_scale * sum(opd_nm)``. The OPD analogue is
        `OPDConverter.opd_scale`.
        """
        return self._scale

    def calc_from_opd(
        self, opd: NDArray[np.float32], *, mask: NDArray[np.bool_] | None = None
    ) -> float:
        """Dry mass [pg] from an OPD map (nanometers), optionally masked."""
        selected = opd if mask is None else opd[mask]
        return float(np.sum(selected, dtype=np.float64)) * self._scale

    def calc_from_phase(
        self, phase: NDArray[np.float32], *, mask: NDArray[np.bool_] | None = None
    ) -> float:
        """Dry mass [pg] from a phase map (radians): to OPD, then `calc_from_opd`."""
        return self.calc_from_opd(self.opd_converter.convert_to_opd(phase), mask=mask)


def calc_drymass(
    opd: NDArray[np.float32],
    *,
    pixel_size: float,
    alpha: float = DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    mask: NDArray[np.bool_] | None = None,
) -> float:
    """Dry mass [pg] of an OPD map (nm); one-shot `DryMassCalculator`.

    Args:
        opd: Optical path difference, in nanometers (e.g. from `phase_to_opd`),
            already background-corrected.
        pixel_size: Physical size of one (square) pixel, in meters.
        alpha: Specific refractive increment, in mL/g (= um^3/pg).
        mask: Optional boolean array selecting the object's pixels.

    Returns:
        Dry mass in picograms (pg).
    """
    return DryMassCalculator(pixel_size=pixel_size, alpha=alpha).calc_from_opd(
        opd, mask=mask
    )


def calc_drymass_from_phase(
    phase: NDArray[np.float32],
    *,
    pixel_size: float,
    wavelength: float = DEFAULT_WAVELENGTH,
    alpha: float = DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    mask: NDArray[np.bool_] | None = None,
) -> float:
    """Dry mass [pg] from phase; one-shot `DryMassCalculator` over the phase path."""
    return DryMassCalculator.from_wavelength(
        pixel_size=pixel_size, alpha=alpha, wavelength=wavelength
    ).calc_from_phase(phase, mask=mask)
