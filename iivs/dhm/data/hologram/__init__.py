"""Hologram sequences and I/O.

Uint8 holograms in Koala's `.raw` stack, `.tif`, and `.npy` -- their
readers/writers and lazy sequences, plus `convert_hologram_sequence` format
conversion.
"""

from __future__ import annotations

__all__ = (
    "HologramNpyFolder",
    "HologramRawFile",
    "HologramRawHeader",
    "HologramSequence",
    "HologramTifFolder",
    "HologramTifList",
    "convert_hologram_sequence",
    "load_hologram_tif",
    "read_hologram_raw_header",
    "save_hologram_npy",
    "save_hologram_raw",
    "save_hologram_tif",
)

from iivs.dhm.data.hologram.base import HologramSequence
from iivs.dhm.data.hologram.convert import convert_hologram_sequence
from iivs.dhm.data.hologram.npy import HologramNpyFolder, save_hologram_npy
from iivs.dhm.data.hologram.raw import (
    HologramRawFile,
    HologramRawHeader,
    read_hologram_raw_header,
    save_hologram_raw,
)
from iivs.dhm.data.hologram.tif import (
    HologramTifFolder,
    HologramTifList,
    load_hologram_tif,
    save_hologram_tif,
)
