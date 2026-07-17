from __future__ import annotations

__all__ = ("HologramNpyFolder", "load_hologram_npy", "save_hologram_npy")

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, override

from iivs.common.data import load_uint8_npy, save_uint8_npy
from iivs.dhm.data.hologram.base import HologramSequence
from iivs.dhm.data.koala import ImageFileFolder

if TYPE_CHECKING:
    import numpy as np
    from kaparoo.filesystem.types import StrPath
    from numpy.typing import NDArray


def load_hologram_npy(path: StrPath) -> NDArray[np.uint8]:
    """Load a header-less `.npy` uint8 hologram image.

    The `.npy` twin of `load_hologram_tif`: the lossless, codec-free single-frame
    reader.

    Raises:
        FileNotFoundError: If `path` does not exist.
        NotAFileError: If `path` exists but is not a regular file.
        ValueError: If the array is pickled or is not a 2D uint8 image.
    """
    return load_uint8_npy(path)


def save_hologram_npy(
    path: StrPath, data: NDArray[np.uint8], *, overwrite: bool = False
) -> None:
    """Save a 2D uint8 hologram as an uncompressed `.npy` file.

    The header-less, lossless twin of `save_hologram_tif` (no codec / LZW). Written
    atomically.

    Args:
        path: The `.npy` file to write.
        data: The hologram image to save, of shape (H, W).
        overwrite: Whether to replace `path` if it already exists. Defaults to False.

    Raises:
        ValueError: If `path` has a non-`.npy` extension, or `data` is not a 2D uint8
            array.
        FileExistsError: If `path` exists and `overwrite` is False.
        FileNotFoundError: If the parent directory of `path` does not exist.
    """
    save_uint8_npy(path, data, overwrite=overwrite)


class HologramNpyFolder(ImageFileFolder, HologramSequence[Path]):
    """An ordered sequence of header-less `{index:05d}_holo.npy` uint8 holograms.

    Numbered discovery + one shared (lazily read) `frame_shape`; a pickled object array
    is rejected. Holograms carry no physical metadata, so (unlike phase / intensity)
    nothing extra is needed at construction. Create the files with `numpy.save`
    (uncompressed `.npy`, one 2-D uint8 frame per file).

    Args:
        root: The folder to scan.
        validate: Run `validate` to this level (`"names"` or `"data"`) at
            construction, or None to skip. Defaults to `"names"`.
    """

    FILE_STEM: ClassVar[str] = "holo"
    FILE_EXT: ClassVar[str] = "npy"

    @override
    def load_file(self, path: Path) -> NDArray[np.uint8]:
        """Load the `.npy` uint8 image at `path` (pickle disabled)."""
        return load_hologram_npy(path)
