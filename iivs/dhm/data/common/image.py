from __future__ import annotations

__all__ = (
    "ImageFileFolder",
    "ImageFileList",
    "ImageTifFolder",
    "ImageTifList",
    "load_uint8_tif",
)

from abc import abstractmethod
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, override

import numpy as np
import tifffile
from kaparoo.data.sequences import FileListSequence
from kaparoo.filesystem import ensure_file_exists, ensure_file_extension
from numpy.typing import NDArray

from iivs.common.data import validate_uint8_array
from iivs.dhm.data.common.sequence import SequentialFileFolder

if TYPE_CHECKING:
    from typing import Literal

    from kaparoo.filesystem.types import StrPath, StrPaths


def load_uint8_tif(path: StrPath) -> NDArray[np.uint8]:
    """Load a single uint8 raster from a `.tif` file.

    Modality-agnostic: the Koala uint8 previews (`Image/*.tif`) and any other
    single-page 8-bit tif decode through this. The Koala previews are
    LZW-compressed; `imagecodecs` is a core dependency, so they decode without
    any extra.

    Raises:
        FileNotFoundError: If `path` does not exist.
        NotAFileError: If `path` exists but is not a regular file.
        ValueError: If the decoded image is not a 2D uint8 array.
    """
    path = ensure_file_exists(path)
    return validate_uint8_array(tifffile.imread(path), allow_stack=False)


class ImageFileList(FileListSequence[NDArray[np.uint8], Path]):
    """A uint8 image sequence over an arbitrary list of files, via a `load_file` codec.

    The modality- and format-agnostic body for header-less uint8 image sources
    (`.tif` previews, `.npy` arrays): each file is decoded independently, so the
    files may live anywhere and differ in shape. A concrete subclass supplies
    only `load_file` (and `FILE_EXT`) for its on-disk format (`ImageTifList`); a
    modality adds its role by also inheriting its `<Modality>ImageSequence` --
    e.g. `PhaseTifList(ImageTifList, PhaseImageSequence[Path])`. `ImageFileFolder`
    is the auto-discovered, same-shape specialization.

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
    def load_file(self, path: Path) -> NDArray[np.uint8]:
        """Decode the uint8 image at `path` (subclass codec)."""
        raise NotImplementedError


class ImageFileFolder(SequentialFileFolder[NDArray[np.uint8]], ImageFileList):
    """A uint8 image folder: numbered discovery + one shared (lazily read) shape.

    The auto-discovered, same-shape specialization of `ImageFileList`. A
    header-less image file carries no shape metadata, so `frame_shape` is read
    lazily from the first file (via the subclass `load_file`) and every image is
    required to match it. Concrete folders set `FILE_EXT` / `FILE_STEM`, supply
    a `load_file`, and add their image role -- e.g. `ImageTifFolder` (tif) or
    `HologramNpyFolder` (npy).

    Args:
        root: The folder to scan.
        validate: Validation level at construction, or None to skip.
    """

    LEVELS: ClassVar[tuple[str, ...]] = ("names", "data")
    DEFAULT_LEVEL: ClassVar[str] = "names"

    def __init__(
        self,
        root: StrPath,
        *,
        validate: Literal["names", "data"] | None = "names",
    ) -> None:
        super().__init__(root)  # discovers the files; rejects an empty folder

        if validate is not None:
            self.validate(level=validate)

    @cached_property
    @override
    def frame_shape(self) -> tuple[int, int]:
        """The (height, width) of the first image, loaded lazily and cached.

        A header-less image folder carries no shape metadata, so this reads the
        first file and assumes every image shares its shape.
        """
        shape = self.load_file(self.get_file(0)).shape
        return (shape[0], shape[1])

    @override
    def _validate_content(self, path: Path, *, level: str) -> None:
        """Decode `path` and require its shape to match the first file.

        `level` is fixed by the hook contract but unused: a header-less image
        folder carries no header, so "data" is its only level past "names".
        """
        image = self.load_file(path)
        if image.shape != self.frame_shape:
            msg = f"shape of {path.name} must match the first file {self.frame_shape} (got {image.shape})"
            raise ValueError(msg)


class ImageTifList(ImageFileList):
    """A uint8 `.tif` image sequence over an arbitrary list of files.

    Supplies the `.tif` codec (`load_uint8_tif`) over `ImageFileList`. A modality
    adds its role by also inheriting its `<Modality>ImageSequence` (e.g.
    `PhaseTifList(ImageTifList, PhaseImageSequence[Path])`). `ImageTifFolder` is
    the auto-discovered, same-shape specialization.

    Args:
        files: The `.tif` files to expose, in the given order.
    """

    FILE_EXT: ClassVar[str] = "tif"

    @override
    def load_file(self, path: Path) -> NDArray[np.uint8]:
        """Load and decode the uint8 tif image at `path`."""
        return load_uint8_tif(path)


class ImageTifFolder(ImageFileFolder, ImageTifList):
    """A uint8 `.tif` folder: `ImageFileFolder` over the `.tif` codec.

    The auto-discovered, same-shape specialization of `ImageTifList`. Concrete
    folders set `FILE_STEM` (and inherit `FILE_EXT = "tif"`) and add their image
    role -- e.g. `PhaseTifFolder(ImageTifFolder, PhaseTifList)`.

    Args:
        root: The folder to scan.
        validate: Validation level at construction, or None to skip.
    """
