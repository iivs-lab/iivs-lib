from __future__ import annotations

__all__ = ("IntensityNpyFolder",)

from typing import TYPE_CHECKING, ClassVar, override

import numpy as np

from iivs.dhm.data.common import read_npy_shape, validate_float32_image
from iivs.dhm.data.intensity.base import IntensityFileFolder
from iivs.dhm.data.intensity.bin import IntensityBinHeader

if TYPE_CHECKING:
    from typing import Literal

    from kaparoo.filesystem.types import StrPath
    from numpy.typing import NDArray


class IntensityNpyFolder(IntensityFileFolder):
    """An ordered sequence of header-less `{index:05d}_intensity.npy` float32 images.

    A `.npy` array carries no Koala header, so the only metadata the `.bin` /
    `.txt` formats embed -- `pixel_size` -- is supplied to the constructor
    instead and shared by every frame (intensity has no unit or height scale).
    Files are loaded with `numpy.load(allow_pickle=False)`, so a pickled object
    array is rejected; create them with `numpy.save` (uncompressed `.npy`, one
    2-D float32 frame per file).

    Args:
        root: The folder to scan.
        pixel_size: Physical size of one (square) pixel, in m.
        validate: Run `validate` to this level at construction, or None to skip.
            Defaults to "headers".

    Raises:
        ValueError: If `validate` is set and the sequence fails validation.
    """

    FILE_EXT: ClassVar[str] = "npy"

    def __init__(
        self,
        root: StrPath,
        *,
        pixel_size: float,
        validate: Literal["names", "headers", "data"] | None = "headers",
    ) -> None:
        # Set the synthesized-header metadata before super().__init__, which
        # reads the first file's header via _read_header (uses it).
        self._pixel_size = pixel_size
        super().__init__(root, validate=validate)

    @override
    def _read_header(self, path: StrPath) -> IntensityBinHeader:
        """Build the header from the `.npy` shape and the shared `pixel_size`."""
        height, width = read_npy_shape(path)
        return IntensityBinHeader(
            width=width, height=height, pixel_size=self._pixel_size
        )

    @override
    def _decode(
        self,
        path: StrPath,
        *,
        on_nonfinite: Literal["ignore", "warn", "raise"] = "ignore",
    ) -> NDArray[np.float32]:
        """Load the `.npy` float32 image (pickle disabled)."""
        data = np.load(path, allow_pickle=False)
        return validate_float32_image(data, on_nonfinite=on_nonfinite, allow_stack=False)
