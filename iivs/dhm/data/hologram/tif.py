from __future__ import annotations

__all__ = (
    "HologramTifFolder",
    "HologramTifList",
    "load_hologram_tif",
    "save_hologram_tif",
)

import io
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import tifffile
from kaparoo.filesystem import StagedFile

from iivs.dhm.data.common import (
    ImageTifFolder,
    ImageTifList,
    load_uint8_tif,
    validate_uint8_image,
    with_file_extension,
)
from iivs.dhm.data.hologram.base import HologramSequence

if TYPE_CHECKING:
    import numpy as np
    from kaparoo.filesystem.types import StrPath
    from numpy.typing import NDArray


def load_hologram_tif(path: StrPath) -> NDArray[np.uint8]:
    """Load a Lyncée Tec Koala uint8 hologram from a `.tif` file.

    A thin alias over `common.load_uint8_tif`.

    Raises:
        FileNotFoundError: If `path` does not exist.
        NotAFileError: If `path` exists but is not a regular file.
        ValueError: If the decoded image is not a 2D uint8 array.
    """
    return load_uint8_tif(path)


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
        ValueError: If `path` has a non-`.tif` extension, or `data` is not a 2D
            uint8 array.
        FileExistsError: If `path` exists and `overwrite` is False.
        FileNotFoundError: If the parent directory of `path` does not exist.
    """
    path = with_file_extension(path, "tif")
    data = validate_uint8_image(data, allow_stack=False)

    # tifffile needs a named target, so encode in memory and stage the bytes.
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, data)

    with StagedFile(path, binary=True, overwrite=overwrite) as staged:
        staged.write(buffer.getvalue())


class HologramTifList(ImageTifList, HologramSequence[Path]):
    """A hologram sequence over an explicit, arbitrary list of `.tif` files.

    The uint8 tif body comes from `common.ImageTifList`; this adds the hologram
    role. Imposes no naming, contiguity, or single-folder constraint: the files
    may live anywhere and each is decoded independently, so they may differ in
    shape (hence a plain `HologramSequence`, no `frame_shape`). Each item is the
    decoded uint8 image and its metadata is the source path. `HologramTifFolder`
    is the auto-discovered, same-shape special case.

    Args:
        files: The `.tif` files to expose, in the given order.
    """


class HologramTifFolder(ImageTifFolder, HologramTifList):
    """An ordered sequence of Lyncée Tec Koala `NNNNN_holo.tif` uint8 holograms.

    The auto-discovered, same-shape special case of `HologramTifList`: lists the
    direct children matching `{index:05d}_holo.tif` (exactly five digits,
    case-sensitive) in index order, sharing one (lazily read) `frame_shape`.
    Construction and validation are inherited from `ImageTifFolder`; this
    supplies only the file stem.

    Args:
        root: The folder to scan.
        validate: Validation level at construction ("names" or "data"), or None
            to skip. Defaults to "names".
    """

    FILE_STEM: ClassVar[str] = "holo"
