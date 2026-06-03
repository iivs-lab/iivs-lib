from __future__ import annotations

__all__ = ("HologramTifSequence", "load_hologram_tif", "save_hologram_tif")

import io
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tifffile
from kaparoo.data.sequences import FileFolderSequence
from kaparoo.filesystem import StagedFile, ensure_file_exists
from kaparoo.filesystem.search import search_files
from kaparoo.filesystem.search.filters import Regex
from natsort import natsorted
from numpy.typing import NDArray

from iivs.dhm.koala.hologram.base import HologramSequence
from iivs.dhm.koala.hologram.core import validate_hologram

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
    return validate_hologram(data)


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
    data = validate_hologram(data)

    # tifffile needs a named target, so encode in memory and stage the bytes.
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, data)

    with StagedFile(path, binary=True, overwrite=overwrite) as staged:
        staged.write(buffer.getvalue())


class HologramTifSequence(
    FileFolderSequence[NDArray[np.uint8], Path], HologramSequence[Path]
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
    def frame_shape(self) -> tuple[int, int]:
        """The (height, width) of the first image, loaded lazily and cached.

        The `.tif` folder carries no header, so this reads the first file and
        assumes every image shares its shape.
        """
        shape = load_hologram_tif(self.get_file(0)).shape
        return (shape[0], shape[1])

    def get_meta(self, index: int) -> Path:
        """Return the source path of the file at `index`."""
        return self.get_file(index)

    def list_files(self, root: Path) -> list[Path]:
        """List the `NNNNN_holo.tif` files under `root`, in index order."""
        files = search_files(root, name_filter=Regex(r"\d{5}_holo\.tif"), max_depth=1)
        if not files:
            msg = f"no NNNNN_holo.tif files found in {root}"
            raise FileNotFoundError(msg)
        return natsorted(files)

    def load_file(self, path: Path) -> NDArray[np.uint8]:
        """Load and decode the hologram at `path`."""
        return load_hologram_tif(path)

    def validate(self, *, level: Literal["names", "data"] = "names") -> None:
        """Validate the sequence to the given `level`.

        Args:
            level: How deep to check, cumulatively: "names" (default) that
                files are numbered contiguously from 0; "data" also that
                every image decodes as a 2D uint8 array sharing the first
                image's shape (expensive -- it reads every image).

        Raises:
            ValueError: If the numbering has gaps, or (at "data") an image
                fails to load or its shape differs from the first image.
        """
        for index in range(len(self)):
            self.validate_file(index, level=level)

    def validate_file(
        self, index: int, *, level: Literal["names", "data"] = "names"
    ) -> None:
        """Validate the file at `index` to `level` (see `validate`)."""
        if level not in ("names", "data"):
            msg = f"level must be 'names' or 'data' (got {level!r})"
            raise ValueError(msg)

        path = self.get_file(index)
        expected = f"{index:05d}_holo.tif"
        if path.name != expected:
            msg = f"non-contiguous numbering: index {index} must be {expected} (got {path.name})"
            raise ValueError(msg)

        if level == "data":
            image = load_hologram_tif(path)
            if image.shape != self.frame_shape:
                msg = f"shape of {path.name} must match the first file {self.frame_shape} (got {image.shape})"
                raise ValueError(msg)
