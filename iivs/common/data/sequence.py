from __future__ import annotations

__all__ = ("ArrayFileList",)

from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, override

from kaparoo.data.sequences import FileListSequence
from kaparoo.filesystem import ensure_file_extension
from numpy.typing import NDArray

if TYPE_CHECKING:
    import numpy as np
    from kaparoo.filesystem.types import StrPaths


class ArrayFileList[U: np.generic](FileListSequence[NDArray[U], Path]):
    """An ndarray sequence over an arbitrary list of files, via a `load_file` codec.

    The format-agnostic body for header-less array sources (`.tif`, `.npy`, ...): each
    file is decoded independently, so the files may live anywhere and differ in shape.
    Generic in the array dtype `U`; a concrete subclass binds `U` and supplies
    `load_file` (and `FILE_EXT`) for its on-disk format. Item metadata is the source
    path.

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
    @abstractmethod
    def load_file(self, path: Path) -> NDArray[U]:
        """Decode the array at `path` (subclass codec)."""
        raise NotImplementedError
