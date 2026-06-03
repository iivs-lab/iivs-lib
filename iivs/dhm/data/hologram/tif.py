from __future__ import annotations

__all__ = (
    "HologramTifFolder",
    "HologramTifList",
    "load_hologram_tif",
    "save_hologram_tif",
)

import io
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, override

import numpy as np
import tifffile
from kaparoo.data.sequences import FileListSequence
from kaparoo.filesystem import StagedFile, ensure_file_exists
from numpy.typing import NDArray

from iivs.dhm.data.common import (
    FrameShapedMixin,
    SequentialFileFolder,
    validate_uint8_image,
)
from iivs.dhm.data.hologram.base import HologramSequence

if TYPE_CHECKING:
    from typing import Literal

    from kaparoo.filesystem.types import StrPath


def load_hologram_tif(path: StrPath) -> NDArray[np.uint8]:
    """Load a Lyncée Tec Koala uint8 hologram from a `.tif` file.

    Raises:
        FileNotFoundError: If `path` does not exist.
        NotAFileError: If `path` exists but is not a regular file.
        ValueError: If the decoded image is not a 2D uint8 array.
    """
    path = ensure_file_exists(path)
    data = tifffile.imread(path)
    return validate_uint8_image(data, allow_stack=False)


def save_hologram_tif(
    path: StrPath, data: NDArray[np.uint8], *, overwrite: bool = False
) -> None:
    """Save a 2D uint8 hologram as a `.tif` file.

    The file is written atomically: content is staged to a temp file in the
    destination's directory and moved into place on success.

    Args:
        path: The `.tif` file to write.
        data: The hologram image to save, of shape (H, W).
        overwrite: Whether to replace `path` if it already exists. Defaults
            to False.

    Raises:
        ValueError: If `data` is not a 2D uint8 array.
        FileExistsError: If `path` exists and `overwrite` is False.
        FileNotFoundError: If the parent directory of `path` does not exist.
    """
    data = validate_uint8_image(data, allow_stack=False)

    # tifffile needs a named target, so encode in memory and stage the bytes.
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, data)

    with StagedFile(path, binary=True, overwrite=overwrite) as staged:
        staged.write(buffer.getvalue())


class HologramTifFolder(
    SequentialFileFolder[NDArray[np.uint8]],
    HologramSequence[Path],
    FrameShapedMixin,
):
    """An ordered sequence of Lyncée Tec Koala `NNNNN_holo.tif` uint8 hologram images.

    Lists the direct children matching `{index:05d}_holo.tif` (exactly five
    digits, case-sensitive), in index order. Each item is the decoded uint8
    image and its metadata is the source path.

    Args:
        root: The folder to scan. Must exist, be a directory, and contain at
            least one matching file.
        validate: Run `validate` to this level ("names" or "data") at
            construction, or None to skip. Defaults to "names".

    Raises:
        DirectoryNotFoundError: If `root` does not exist.
        NotADirectoryError: If `root` exists but is not a directory.
        FileNotFoundError: If no `NNNNN_holo.tif` files are found in `root`.
        ValueError: If `validate` is set and the sequence fails validation.
    """

    FILE_STEM: ClassVar[str] = "holo"
    FILE_EXT: ClassVar[str] = "tif"
    LEVELS: ClassVar[tuple[str, ...]] = ("names", "data")
    DEFAULT_LEVEL: ClassVar[str] = "names"

    def __init__(
        self,
        root: StrPath,
        *,
        validate: Literal["names", "data"] | None = "names",
    ) -> None:
        super().__init__(root)

        if validate is not None:
            self.validate(level=validate)

    @cached_property
    @override
    def frame_shape(self) -> tuple[int, int]:
        """The (height, width) of the first image, loaded lazily and cached.

        The `.tif` folder carries no header, so this reads the first file and
        assumes every image shares its shape.
        """
        shape = load_hologram_tif(self.get_file(0)).shape
        return (shape[0], shape[1])

    @override
    def load_file(self, path: Path) -> NDArray[np.uint8]:
        """Load and decode the hologram at `path`."""
        return load_hologram_tif(path)

    @override
    def _validate_content(self, path: Path, *, level: str) -> None:
        """Decode `path` and require its shape to match the first file.

        `level` is fixed by the hook contract but unused here: the `.tif`
        folder carries no header, so "data" is its only level past "names".
        """
        image = load_hologram_tif(path)
        if image.shape != self.frame_shape:
            msg = f"shape of {path.name} must match the first file {self.frame_shape} (got {image.shape})"
            raise ValueError(msg)


class HologramTifList(
    FileListSequence[NDArray[np.uint8], Path], HologramSequence[Path]
):
    """A hologram sequence over an explicit, arbitrary list of `.tif` files.

    Unlike `HologramTifFolder`, imposes no naming, contiguity, or
    single-folder constraint: the files may live anywhere and each is decoded
    independently. The images may therefore differ in shape, so this is a
    plain `HologramSequence` (no `frame_shape`). Each item is the decoded uint8
    image and its metadata is the source path.

    Args:
        files: The `.tif` files to expose, in the given order.
    """

    @override
    def get_meta(self, index: int) -> Path:
        """Return the source path of the file at `index`."""
        return self.get_file(index)

    @override
    def load_file(self, path: Path) -> NDArray[np.uint8]:
        """Load and decode the hologram at `path`."""
        return load_hologram_tif(path)
