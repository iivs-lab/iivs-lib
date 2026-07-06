from __future__ import annotations

__all__ = ("IntensityTifFolder", "IntensityTifList")

from pathlib import Path
from typing import ClassVar

from iivs.dhm.data.intensity.base import IntensityImageSequence
from iivs.dhm.data.koala import ImageTifFolder, ImageTifList


class IntensityTifList(ImageTifList, IntensityImageSequence[Path]):
    """An intensity preview sequence over an arbitrary list of `Image/*.tif` files.

    The uint8 display-preview twin of `IntensityBinList`: each file is decoded
    independently as a uint8 image, with no naming/contiguity constraint. This
    is an `IntensityImageSequence`, *not* a quantitative
    `IntensityFloatSequence`; the pixels are the 8-bit preview, not the float
    intensity. `IntensityTifFolder` is the auto-discovered, same-shape special
    case.

    Args:
        files: The `.tif` files to expose, in the given order.
    """


class IntensityTifFolder(ImageTifFolder, IntensityTifList):
    """An ordered sequence of Koala `{index:05d}_intensity.tif` uint8 previews.

    The uint8 display-preview twin of `IntensityBinFolder`, and the
    auto-discovered, same-shape special case of `IntensityTifList`: lists
    `{index:05d}_intensity.tif`, sharing one (lazily read) `frame_shape`.
    Construction and validation are inherited from `ImageTifFolder`; this
    supplies only the file stem.

    The LZW-compressed Koala previews decode via the core `imagecodecs`
    dependency (no extra needed).

    Args:
        root: The folder to scan.
        validate: Validation level at construction, or None to skip.
    """

    FILE_STEM: ClassVar[str] = "intensity"
