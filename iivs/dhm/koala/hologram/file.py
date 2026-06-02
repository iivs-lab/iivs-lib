from __future__ import annotations

__all__ = ("load_hologram_tif", "save_hologram_tif", "validate_hologram")

import io
from typing import TYPE_CHECKING

import numpy as np
import tifffile
from kaparoo.filesystem import StagedFile, ensure_file_exists

if TYPE_CHECKING:
    from kaparoo.filesystem.types import StrPath
    from numpy.typing import NDArray


def validate_hologram(data: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Validate a single uint8 hologram image and return it.

    `data` is never modified.

    Raises:
        ValueError: If `data` is not a 2D uint8 array.
    """
    if data.ndim != 2:
        msg = f"hologram must be a single 2D image (got shape {data.shape})"
        raise ValueError(msg)

    if data.dtype != np.uint8:
        msg = f"hologram must be uint8 (got {data.dtype})"
        raise ValueError(msg)

    return data


def load_hologram_tif(path: StrPath) -> NDArray[np.uint8]:
    """Load a Koala uint8 hologram from a `.tif` file.

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
