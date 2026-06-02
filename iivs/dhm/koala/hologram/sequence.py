from __future__ import annotations

__all__ = ("HologramTifSequence",)

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from kaparoo.data.sequences import FileFolderSequence
from kaparoo.filesystem.search import search_files
from kaparoo.filesystem.search.filters import Regex
from natsort import natsorted
from numpy.typing import NDArray

from iivs.dhm.koala.hologram.file import load_hologram_tif

if TYPE_CHECKING:
    from typing import Literal

    from kaparoo.filesystem.types import StrPath


class HologramTifSequence(FileFolderSequence[NDArray[np.uint8], Path]):
    """An ordered sequence of Koala `NNNNN_holo.tif` uint8 hologram images.

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

    def validate(self, *, level: Literal["names", "data"] = "names") -> None:
        """Validate the sequence to the given `level`.

        Args:
            level: How deep to check, cumulatively: "names" (default) that
                files are numbered contiguously from 0; "data" also that
                every image decodes as a 2D uint8 array (expensive -- it
                reads every image).

        Raises:
            ValueError: If the numbering has gaps or (at "data") an image
                fails to load.
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
            load_hologram_tif(path)

    def list_files(self, root: Path) -> list[Path]:
        files = search_files(root, name_filter=Regex(r"\d{5}_holo\.tif"), max_depth=1)
        if not files:
            msg = f"no NNNNN_holo.tif files found in {root}"
            raise FileNotFoundError(msg)
        return natsorted(files)

    def get_meta(self, index: int) -> Path:
        return self.get_file(index)

    def load_file(self, path: Path) -> NDArray[np.uint8]:
        return load_hologram_tif(path)
