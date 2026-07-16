"""Hologram sequences and I/O.

Uint8 holograms in Koala's `.raw` stack, `.tif`, and `.npy`: their readers/writers and
lazy sequences, plus `convert_hologram_sequence` format conversion.
"""

__all__ = (
    "HOLOGRAM_FORMATS",
    "HOLOGRAM_TREE",
    "AmbiguousHologramsError",
    "HologramFormat",
    "HologramNpyFolder",
    "HologramRawFile",
    "HologramRawHeader",
    "HologramSequence",
    "HologramTifFolder",
    "HologramTifList",
    "convert_hologram_sequence",
    "load_hologram_npy",
    "load_hologram_tif",
    "open_holograms",
    "read_hologram_raw_header",
    "save_hologram_npy",
    "save_hologram_raw",
    "save_hologram_tif",
    "search_ambiguous_holograms",
    "search_holograms",
)

from iivs.dhm.data.hologram.base import HologramSequence
from iivs.dhm.data.hologram.dispatch import (
    HOLOGRAM_FORMATS,
    HologramFormat,
    convert_hologram_sequence,
)
from iivs.dhm.data.hologram.layout import (
    HOLOGRAM_TREE,
    AmbiguousHologramsError,
    open_holograms,
    search_ambiguous_holograms,
    search_holograms,
)
from iivs.dhm.data.hologram.npy import (
    HologramNpyFolder,
    load_hologram_npy,
    save_hologram_npy,
)
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
