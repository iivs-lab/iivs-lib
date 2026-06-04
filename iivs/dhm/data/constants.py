"""Default optical and geometric parameters for the lab's transmission DHM.

Override per experiment when the setup differs. The phase helpers
(`save_phase_bin`, the phase-to-height conversion) take the refractive-index
*difference* ``n_object - n_medium``, so only that delta is defined, not the
individual indices. Pixel size normally comes from the `.bin` header
(`PhaseBinHeader`); the 20X constants are a magnification-specific, header-less
fallback.
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

DEFAULT_WAVELENGTH = 666e-9  # m
DEFAULT_WAVELENGTH_NM = 666.0  # nm
DEFAULT_REFRACTIVE_DELTA = 0.5  # n_object (~1.5) - n_medium (~1.0)

# Barer specific refractive increment for dry mass; typically 1.8-2.1e-4.
DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT = 2.0e-4  # m^3/kg

# 20X Koala pixel size (~285 nm); header-less fallback for PhaseBinHeader.
PIXEL_SIZE_20X = 2.84871e-7  # m
PIXEL_SIZE_20X_UM = 0.284871  # um
