from __future__ import annotations

__all__ = ("HologramNpyFolder",)

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, override

import numpy as np

from iivs.dhm.data.common import ImageFileFolder, validate_uint8_image
from iivs.dhm.data.hologram.base import HologramSequence

if TYPE_CHECKING:
    from numpy.typing import NDArray


class HologramNpyFolder(ImageFileFolder, HologramSequence[Path]):
    """An ordered sequence of header-less `{index:05d}_holo.npy` uint8 holograms.

    The `.npy` codec over `common.ImageFileFolder`: numbered discovery + one
    shared (lazily read) `frame_shape`, with each frame loaded by
    `numpy.load(allow_pickle=False)` (a pickled object array is rejected).
    Holograms carry no physical metadata, so -- unlike phase / intensity --
    nothing extra is needed at construction. Create the files with `numpy.save`
    (uncompressed `.npy`, one 2-D uint8 frame per file).

    Args:
        root: The folder to scan.
        validate: Run `validate` to this level ("names" or "data") at
            construction, or None to skip. Defaults to "names".
    """

    FILE_STEM: ClassVar[str] = "holo"
    FILE_EXT: ClassVar[str] = "npy"

    @override
    def load_file(self, path: Path) -> NDArray[np.uint8]:
        """Load the `.npy` uint8 image at `path` (pickle disabled)."""
        return validate_uint8_image(
            np.load(path, allow_pickle=False), allow_stack=False
        )
