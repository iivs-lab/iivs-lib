from __future__ import annotations

__all__ = ("IntensityFloatSequence", "IntensityImageSequence", "IntensitySequence")

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from kaparoo.data.sequences import DataSequence
from numpy.typing import NDArray

from iivs.dhm.data.koala import KoalaFloatFileFolder, KoalaFloatFileList

if TYPE_CHECKING:
    # Binds `H` via the string forward-ref in the class bases below. ruff does
    # not count a class-base string subscript as a use, but `ty` resolves it;
    # a runtime import would cycle (`bin` imports the list/folder from here).
    from iivs.dhm.data.intensity.bin import IntensityBinHeader  # noqa: F401


class IntensitySequence[T, M](DataSequence[T, M]):
    """A read-only sequence of intensity images, from any source.

    The modality-level base over both representations Koala exports:
    quantitative float32 (`IntensityFloatSequence`, from `Float/{Bin,Txt}`) and
    the uint8 display preview (`IntensityImageSequence`, from `Image/*.tif`).
    Annotate with it to accept any intensity sequence regardless of pixel type;
    annotate with the `Float` / `Image` subtype when the dtype matters.

    Type Parameters:
        T: The item (image) array type (`NDArray[np.float32]` for quantitative,
            `NDArray[np.uint8]` for preview).
        M: The per-item metadata type chosen by the concrete sequence (e.g. the
            source `Path`).
    """


class IntensityImageSequence[M](IntensitySequence[NDArray[np.uint8], M]):
    """A read-only sequence of uint8 intensity preview images.

    The display-only 8-bit preview Koala renders under `Image/*.tif`, distinct
    from, and not a substitute for, the quantitative `IntensityFloatSequence`.
    Same-shape sources also expose `frame_shape`.
    """


class IntensityFloatSequence[M](IntensitySequence[NDArray[np.float32], M]):
    """A read-only sequence of quantitative float32 intensity images.

    The intensity reconstruction Koala exports as `Float/{Bin,Txt}`; annotate
    parameters with it to accept any float32 intensity source: one acquisition
    (`IntensityBinFolder`) or an arbitrary `IntensityBinList` of unrelated
    files, and their `.txt` twins. Same-shape sources additionally expose
    `frame_shape`.
    """


class IntensityFileList(
    KoalaFloatFileList["IntensityBinHeader"], IntensityFloatSequence[Path]
):
    """Format-agnostic intensity file list over a ``(read_header, decode)`` codec.

    An arbitrary list of float32 intensity files, each read independently.
    Concrete subclasses (`IntensityBinList`, `IntensityTxtList`) bind a format;
    `IntensityFileFolder` is the auto-discovered, same-shape case.
    """


class IntensityFileFolder(
    KoalaFloatFileFolder["IntensityBinHeader"], IntensityFileList
):
    """Format-agnostic intensity folder: numbered discovery + one shared header.

    The auto-discovered, same-shape case of `IntensityFileList`: one
    acquisition's numbered files sharing a single `header`. Concrete folders
    (`IntensityBinFolder`, `IntensityTxtFolder`) bind a format.
    """

    FILE_STEM: ClassVar[str] = "intensity"
