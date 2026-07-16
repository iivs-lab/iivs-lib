from __future__ import annotations

__all__ = ("IntensityTifFolder", "IntensityTifList", "load_intensity_tif")

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from iivs.dhm.data.intensity.base import IntensityImageSequence
from iivs.dhm.data.koala import ImageTifFolder, ImageTifList, load_uint8_tif

if TYPE_CHECKING:
    import numpy as np
    from kaparoo.filesystem.types import StrPath
    from numpy.typing import NDArray


def load_intensity_tif(path: StrPath) -> NDArray[np.uint8]:
    """Load a Koala uint8 intensity preview from a `.tif` file.

    The pixels are the 8-bit preview Koala rendered, not the float intensity. Read
    `load_intensity_bin` / `load_intensity_txt` for the quantitative source.

    Raises:
        FileNotFoundError: If `path` does not exist.
        NotAFileError: If `path` exists but is not a regular file.
        ValueError: If the decoded image is not a 2D uint8 array.
    """
    return load_uint8_tif(path)


class IntensityTifList(ImageTifList, IntensityImageSequence[Path]):
    """An intensity preview sequence over an arbitrary list of `Image/*.tif` files.

    The uint8 display-preview twin of `IntensityBinList`: each file is decoded
    independently as a uint8 image, with no naming/contiguity constraint. This is an
    `IntensityImageSequence`, *not* a quantitative `IntensityFloatSequence`; the pixels
    are the 8-bit preview, not the float intensity. `IntensityTifFolder` is the
    auto-discovered, same-shape special case.

    Args:
        files: The `.tif` files to expose, in the given order.
    """


class IntensityTifFolder(ImageTifFolder, IntensityTifList):
    """An ordered sequence of Koala `{index:05d}_intensity.tif` uint8 previews.

    The uint8 display-preview twin of `IntensityBinFolder`, and the auto-discovered,
    same-shape special case of `IntensityTifList`: lists `{index:05d}_intensity.tif`,
    sharing one (lazily read) `frame_shape`.

    The LZW-compressed Koala previews decode via the core `imagecodecs` dependency (no
    extra needed).

    Args:
        root: The folder to scan.
        validate: Validation level at construction, or None to skip.
    """

    FILE_STEM: ClassVar[str] = "intensity"
