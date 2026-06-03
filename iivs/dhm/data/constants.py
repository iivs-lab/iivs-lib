"""Typical optical and biophysical parameters for the lab's transmission DHM.

Override these per experiment when a setup differs. Only the refractive-index
*difference* is kept: `save_phase_bin` and the phase-to-height conversion need
just ``n_object - n_medium``, so the individual indices are not defined here.
"""

from __future__ import annotations

__all__ = (
    "DEFAULT_REFRACTIVE_DELTA",
    "DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT",
    "DEFAULT_WAVELENGTH",
    "DEFAULT_WAVELENGTH_NM",
)

DEFAULT_WAVELENGTH = 666e-9  # meters (666 nm)
DEFAULT_WAVELENGTH_NM = 666.0  # nanometers
DEFAULT_REFRACTIVE_DELTA = 0.5  # n_object (~1.5) - n_medium (~1.0), transmission

# Specific refractive increment for dry mass (Barer: n = n_medium + alpha * C).
# Nearly universal across biomolecules; typical 0.18-0.21. Unit: mL/g (= um^3/pg).
DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT = 0.2
