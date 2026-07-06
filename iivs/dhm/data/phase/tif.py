from __future__ import annotations

__all__ = ("PhaseTifFolder", "PhaseTifList")

from pathlib import Path
from typing import ClassVar

from iivs.dhm.data.koala import ImageTifFolder, ImageTifList
from iivs.dhm.data.phase.base import PhaseImageSequence


class PhaseTifList(ImageTifList, PhaseImageSequence[Path]):
    """A phase preview sequence over an arbitrary list of `Image/*.tif` files.

    The uint8 display-preview twin of `PhaseBinList`: each file is decoded
    independently as a uint8 image, with no naming/contiguity constraint. This
    is a `PhaseImageSequence`, *not* a quantitative `PhaseFloatSequence`; the
    pixels are the 8-bit preview Koala renders, not the float phase.
    `PhaseTifFolder` is the auto-discovered, same-shape special case.

    Args:
        files: The `.tif` files to expose, in the given order.
    """


class PhaseTifFolder(ImageTifFolder, PhaseTifList):
    """An ordered sequence of Koala `{index:05d}_phase.tif` uint8 previews.

    The uint8 display-preview twin of `PhaseBinFolder`, and the auto-discovered,
    same-shape special case of `PhaseTifList`: lists `{index:05d}_phase.tif`,
    sharing one (lazily read) `frame_shape`. Construction and validation are
    inherited from `ImageTifFolder`; this supplies only the file stem.

    The LZW-compressed Koala previews decode via the core `imagecodecs`
    dependency (no extra needed).

    Args:
        root: The folder to scan.
        validate: Validation level at construction, or None to skip.
    """

    FILE_STEM: ClassVar[str] = "phase"
