"""Readers, writers, and sequences for Lyncée Tec Koala acquisition data.

The imaging modalities are each their own subpackage: `phase` and `intensity`
(quantitative float32 ``.bin`` / ``.txt`` / ``.npy``, plus uint8 ``.tif`` previews) and
`hologram` (uint8 ``.raw`` / ``.tif`` / ``.npy``). A `timestamp` module handles
``timestamps.txt``, and a `timelapse` module opens a whole acquisition
(`KoalaTimelapse`, wiring every modality over the standard Koala layout). Within a
modality, each file format is a codec module exposing a loader, a saver, and lazy
sequence types, and a `convert` helper re-encodes a sequence from one format to another.
Blocks shared across modalities live in `koala`; `constants` holds the lab's default
optical parameters.

What is re-exported here is the acquisition level: the whole-acquisition reader, the
timing file, and the optical defaults. The modalities stay in their own subpackages, so
reach for `iivs.dhm.data.phase` and its siblings by name.

The ``.bin``, ``.tif`` / ``.raw``, and ``timestamps.txt`` formats are `Lyncée Tec
<https://www.lynceetec.com/>`_'s proprietary Koala formats. The ``.bin`` container
(float32, shared by phase and intensity) was verified against their reference
implementation, `pyKoalaUtils <https://github.com/lynceetec/pyKoalaUtils>`_ (MIT); this
package is an independent reimplementation and contains no code from it.
"""

__all__ = (
    "DEFAULT_REFRACTIVE_DELTA",
    "DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT",
    "DEFAULT_WAVELENGTH",
    "DEFAULT_WAVELENGTH_NM",
    "KOALA_TIMELAPSE_TREE",
    "PIXEL_SIZE_10X",
    "PIXEL_SIZE_10X_UM",
    "PIXEL_SIZE_20X",
    "PIXEL_SIZE_20X_UM",
    "PIXEL_SIZE_40X",
    "PIXEL_SIZE_40X_UM",
    "KoalaTimelapse",
    "TimestampsTxtFile",
    "search_timelapses",
)

from iivs.dhm.data.constants import (
    DEFAULT_REFRACTIVE_DELTA,
    DEFAULT_SPECIFIC_REFRACTIVE_INCREMENT,
    DEFAULT_WAVELENGTH,
    DEFAULT_WAVELENGTH_NM,
    PIXEL_SIZE_10X,
    PIXEL_SIZE_10X_UM,
    PIXEL_SIZE_20X,
    PIXEL_SIZE_20X_UM,
    PIXEL_SIZE_40X,
    PIXEL_SIZE_40X_UM,
)
from iivs.dhm.data.timelapse import (
    KOALA_TIMELAPSE_TREE,
    KoalaTimelapse,
    search_timelapses,
)
from iivs.dhm.data.timestamp import TimestampsTxtFile
