"""Typical optical, biophysical, and geometric parameters for the lab's
transmission DHM.

Override these per experiment when a setup differs. Only the refractive-index
*difference* is kept: `save_phase_bin` and the phase-to-height conversion need
just ``n_object - n_medium``, so the individual indices are not defined here.
The pixel size is normally read from the `.bin` header (`PhaseBinHeader`); the
20X default is a header-less fallback and is magnification-specific.
"""

from __future__ import annotations

__all__ = (
    "DEFAULT_REFRACTIVE_DELTA",
    "DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT",
    "DEFAULT_WAVELENGTH",
    "DEFAULT_WAVELENGTH_NM",
    "PIXEL_SIZE_20X",
    "PIXEL_SIZE_20X_UM",
)

DEFAULT_WAVELENGTH = 666e-9  # m (666 nm)
DEFAULT_WAVELENGTH_NM = 666.0  # nm
DEFAULT_REFRACTIVE_DELTA = 0.5  # n_object (~1.5) - n_medium (~1.0), transmission

# Specific refractive increment (Barer dry mass); typical 1.8-2.1e-4 m^3/kg.
DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT = 2.0e-4  # m^3/kg

# 20X pixel size as recorded by Koala (~285 nm); header-less fallback for
# PhaseBinHeader.pixel_size.
PIXEL_SIZE_20X = 2.84871e-7  # m
PIXEL_SIZE_20X_UM = 0.284871  # um
