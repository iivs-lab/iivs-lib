"""Koala's fixed acquisition-layout vocabulary: folder / file names and their paths."""

from __future__ import annotations

__all__ = (
    "BIN",
    "FLOAT",
    "HOLOGRAMS",
    "IMAGE",
    "INTENSITY",
    "INTENSITY_FLOAT_BIN",
    "INTENSITY_FLOAT_TXT",
    "INTENSITY_IMAGE",
    "PHASE",
    "PHASE_FLOAT_BIN",
    "PHASE_FLOAT_TXT",
    "PHASE_IMAGE",
    "PHBOUNDS",
    "TIMESTAMPS",
    "TXT",
)


# ============================================================ #
#                        atomic names                          #
# ============================================================ #

# Koala's fixed layout vocabulary: the modality folders (`PHASE`, `INTENSITY`,
# `HOLOGRAMS`), the float32 subfolders (`FLOAT`, `BIN`, `TXT`) and the preview folder
# (`IMAGE`), plus the root files (`TIMESTAMPS`, `PHBOUNDS`).
HOLOGRAMS = "Holograms"
INTENSITY = "Intensity"
PHASE = "Phase"
FLOAT = "Float"
BIN = "Bin"
TXT = "Txt"
IMAGE = "Image"
TIMESTAMPS = "timestamps.txt"
PHBOUNDS = "phbounds.txt"


# ============================================================ #
#                       composed paths                         #
# ============================================================ #

# Each reconstruction modality's fixed subfolder paths, relative to its time-lapse root.
INTENSITY_FLOAT_BIN = f"{INTENSITY}/{FLOAT}/{BIN}"
"""The `Intensity/Float/Bin` folder's time-lapse-relative path."""
INTENSITY_FLOAT_TXT = f"{INTENSITY}/{FLOAT}/{TXT}"
"""The `Intensity/Float/Txt` folder's time-lapse-relative path."""
INTENSITY_IMAGE = f"{INTENSITY}/{IMAGE}"
"""The `Intensity/Image` preview folder's time-lapse-relative path."""

PHASE_FLOAT_BIN = f"{PHASE}/{FLOAT}/{BIN}"
"""The `Phase/Float/Bin` folder's time-lapse-relative path."""
PHASE_FLOAT_TXT = f"{PHASE}/{FLOAT}/{TXT}"
"""The `Phase/Float/Txt` folder's time-lapse-relative path."""
PHASE_IMAGE = f"{PHASE}/{IMAGE}"
"""The `Phase/Image` preview folder's time-lapse-relative path."""
