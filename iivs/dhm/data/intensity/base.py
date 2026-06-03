from __future__ import annotations

__all__ = (
    "IntensityFloatSequence",
    "IntensityImageSequence",
    "IntensitySequence",
)

from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, override

import numpy as np
from kaparoo.data.sequences import DataSequence, FileListSequence
from numpy.typing import NDArray

from iivs.dhm.data.common import SequentialFileFolder, ensure_file_extension

if TYPE_CHECKING:
    from typing import Literal

    from kaparoo.filesystem.types import StrPath, StrPaths

    from iivs.dhm.data.intensity.bin import IntensityBinHeader


class IntensitySequence[T, M](DataSequence[T, M]):
    """A read-only sequence of intensity images, from any source.

    The modality-level base over both representations Koala exports:
    quantitative float32 (`IntensityFloatSequence`, from `Float/{Bin,Txt}`) and
    the uint8 display preview (`IntensityImageSequence`, from `Image/*.tif`).
    Annotate with it to accept any intensity sequence regardless of pixel type;
    annotate with the `Float` / `Image` subtype when the dtype matters.

    Type Parameters:
        T: The item (image) array type -- `NDArray[np.float32]` (quantitative)
            or `NDArray[np.uint8]` (preview).
        M: The per-item metadata type chosen by the concrete sequence (e.g. the
            source `Path`).
    """


class IntensityFloatSequence[M](IntensitySequence[NDArray[np.float32], M]):
    """A read-only sequence of quantitative float32 intensity images.

    The intensity reconstruction Koala exports as `Float/{Bin,Txt}`; annotate
    parameters with it to accept any float32 intensity source -- one acquisition
    (`IntensityBinFolder`) or an arbitrary `IntensityBinList` of unrelated
    files, and their `.txt` twins. Same-shape sources additionally mix in
    `data.common.FrameShapedMixin` to expose `frame_shape`.
    """


class IntensityImageSequence[M](IntensitySequence[NDArray[np.uint8], M]):
    """A read-only sequence of uint8 intensity preview images.

    The display-only 8-bit preview Koala renders under `Image/*.tif` -- distinct
    from, and not a substitute for, the quantitative `IntensityFloatSequence`.
    Same-shape sources mix in `data.common.FrameShapedMixin` to expose
    `frame_shape`.
    """


class IntensityFileList(
    FileListSequence[NDArray[np.float32], Path], IntensityFloatSequence[Path]
):
    """Format-agnostic intensity file list over a ``(read_header, decode)`` codec.

    Holds the list machinery (and the `.<FILE_EXT>` check) once; a concrete
    subclass (`IntensityBinList`, `IntensityTxtList`) supplies only `FILE_EXT`
    and the `_read_header` / `_decode` codec for its on-disk format. Intensity
    carries no unit, so `load_file` is the bare decode. `IntensityFileFolder` is
    the auto-discovered, same-shape specialization.

    Args:
        files: The files to expose, in the given order.

    Raises:
        ValueError: If any path does not have the subclass `.<FILE_EXT>` suffix.
    """

    FILE_EXT: ClassVar[str]

    def __init__(self, files: StrPaths) -> None:
        super().__init__([ensure_file_extension(f, self.FILE_EXT) for f in files])

    @override
    def get_meta(self, index: int) -> Path:
        """Return the source path of the file at `index`."""
        return self.get_file(index)

    @override
    def load_file(self, path: Path) -> NDArray[np.float32]:
        """Load the image at `path`."""
        return self._decode(path)

    @abstractmethod
    def _read_header(self, path: StrPath) -> IntensityBinHeader:
        """Read the format's header (subclass codec)."""
        raise NotImplementedError

    @abstractmethod
    def _decode(
        self,
        path: StrPath,
        *,
        on_nonfinite: Literal["ignore", "warn", "raise"] = "ignore",
    ) -> NDArray[np.float32]:
        """Decode the format's image (subclass codec)."""
        raise NotImplementedError


class IntensityFileFolder(SequentialFileFolder[NDArray[np.float32]], IntensityFileList):
    """Format-agnostic intensity folder: numbered discovery + one shared header.

    The auto-discovered, same-shape specialization of `IntensityFileList`; it
    reuses that list's `load_file` codec. Concrete folders set only `FILE_EXT`.

    Args:
        root: The folder to scan.
        validate: Validation level at construction, or None to skip.
    """

    LEVELS: ClassVar[tuple[str, ...]] = ("names", "headers", "data")
    DEFAULT_LEVEL: ClassVar[str] = "headers"
    FILE_STEM: ClassVar[str] = "intensity"

    def __init__(
        self,
        root: StrPath,
        *,
        validate: Literal["names", "headers", "data"] | None = "headers",
    ) -> None:
        super().__init__(root)  # discovers the files; rejects an empty folder

        self._header = self._read_header(self.get_file(0))

        if validate is not None:
            self.validate(level=validate)

    @property
    def header(self) -> IntensityBinHeader:
        """The shared acquisition header, read from the first file."""
        return self._header

    @property
    @override
    def frame_shape(self) -> tuple[int, int]:
        """The (height, width) of each image, from the shared header."""
        return self._header.shape

    @override
    def _validate_content(self, path: Path, *, level: str) -> None:
        """Check `path`'s header matches the reference; at "data", decode too."""
        if self._read_header(path) != self.header:
            msg = f"header of {path.name} differs from the first file"
            raise ValueError(msg)

        if level == "data":
            self._decode(path, on_nonfinite="raise")
