"""Default optical and geometric parameters for the lab's transmission DHM.

Override per experiment when the setup differs. The phase helpers
(`save_phase_bin`, the phase-to-height conversion) take the refractive-index
*difference* ``n_object - n_medium``, so only that delta is defined, not the
individual indices. Pixel size normally comes from the `.bin` header
(`PhaseBinHeader`); the per-magnification ``PIXEL_SIZE_*`` constants are a
header-less fallback, measured on this setup's 10X / 20X / 40X objectives.
"""

from __future__ import annotations

__all__ = (
    "DEFAULT_REFRACTIVE_DELTA",
    "DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT",
    "DEFAULT_WAVELENGTH",
    "DEFAULT_WAVELENGTH_NM",
    "PIXEL_SIZE_10X",
    "PIXEL_SIZE_10X_UM",
    "PIXEL_SIZE_20X",
    "PIXEL_SIZE_20X_UM",
    "PIXEL_SIZE_40X",
    "PIXEL_SIZE_40X_UM",
)

DEFAULT_WAVELENGTH = 666e-9  # m
DEFAULT_WAVELENGTH_NM = 666.0  # nm
DEFAULT_REFRACTIVE_DELTA = 0.5  # n_object (~1.5) - n_medium (~1.0)

# Barer specific refractive increment for dry mass; typically 1.8-2.1e-4.
DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT = 2.0e-4  # m^3/kg

# Measured Koala pixel sizes per objective; header-less fallback for
# PhaseBinHeader. Pixel size scales ~1/M, so 10X / 20X / 40X run large to small.
PIXEL_SIZE_10X = 5.799421218e-7  # m (~580 nm)
PIXEL_SIZE_10X_UM = 0.5799421218  # um
PIXEL_SIZE_20X = 2.84871392e-7  # m (~285 nm)
PIXEL_SIZE_20X_UM = 0.284871392  # um
PIXEL_SIZE_40X = 1.440906909e-7  # m (~144 nm)
PIXEL_SIZE_40X_UM = 0.1440906909  # um
