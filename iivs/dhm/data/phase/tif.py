from __future__ import annotations

__all__ = ("PhaseTifFolder", "PhaseTifList", "load_phase_tif")

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from iivs.dhm.data.koala import ImageTifFolder, ImageTifList, load_uint8_tif
from iivs.dhm.data.phase.base import PhaseImageSequence

if TYPE_CHECKING:
    import numpy as np
    from kaparoo.filesystem.types import StrPath
    from numpy.typing import NDArray


def load_phase_tif(path: StrPath) -> NDArray[np.uint8]:
    """Load a Koala uint8 phase preview from a `.tif` file.

    The pixels are the 8-bit preview Koala rendered, not the float phase: recovering
    nanometres from them needs the acquisition's `phbounds.txt` (`PhaseBounds`), and
    only to the precision 8 bits survived. Read `load_phase_bin` / `load_phase_txt` for
    the quantitative source.

    Raises:
        FileNotFoundError: If `path` does not exist.
        NotAFileError: If `path` exists but is not a regular file.
        ValueError: If the decoded image is not a 2D uint8 array.
    """
    return load_uint8_tif(path)


class PhaseTifList(ImageTifList, PhaseImageSequence[Path]):
    """A phase preview sequence over an arbitrary list of `Image/*.tif` files.

    The uint8 display-preview twin of `PhaseBinList`: each file is decoded independently
    as a uint8 image, with no naming/contiguity constraint. This is a
    `PhaseImageSequence`, *not* a quantitative `PhaseFloatSequence`; the pixels are the
    8-bit preview Koala renders, not the float phase. `PhaseTifFolder` is the
    auto-discovered, same-shape special case.

    Args:
        files: The `.tif` files to expose, in the given order.
    """


class PhaseTifFolder(ImageTifFolder, PhaseTifList):
    """An ordered sequence of Koala `{index:05d}_phase.tif` uint8 previews.

    The uint8 display-preview twin of `PhaseBinFolder`, and the auto-discovered,
    same-shape special case of `PhaseTifList`: lists `{index:05d}_phase.tif`, sharing
    one (lazily read) `frame_shape`.

    The LZW-compressed Koala previews decode via the core `imagecodecs` dependency (no
    extra needed).

    Args:
        root: The folder to scan.
        validate: Validation level at construction, or None to skip.
    """

    FILE_STEM: ClassVar[str] = "phase"
