from __future__ import annotations

__all__ = ("load_tif",)

from typing import TYPE_CHECKING

import tifffile
from kaparoo.filesystem import ensure_file_exists

if TYPE_CHECKING:
    from typing import Any

    from kaparoo.filesystem.types import StrPath
    from numpy.typing import NDArray


def load_tif(path: StrPath) -> NDArray[Any]:
    """Load a single raster from a `.tif` file, keeping its stored dtype.

    Decodes any single-page tif via `tifffile` (LZW and other codecs handled by the core
    `imagecodecs` dependency) and returns the array as stored. A caller needing a
    specific dtype validates the result (e.g. `validate_uint_array`).

    Raises:
        FileNotFoundError: If `path` does not exist.
        NotAFileError: If `path` exists but is not a regular file.
    """
    path = ensure_file_exists(path)
    return tifffile.imread(path)
