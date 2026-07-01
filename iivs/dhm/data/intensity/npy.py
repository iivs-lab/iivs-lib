from __future__ import annotations

__all__ = ("IntensityNpyFolder", "load_intensity_npy", "save_intensity_npy")

from typing import TYPE_CHECKING, ClassVar, override

import numpy as np
from kaparoo.filesystem import ensure_file_exists, ensure_file_extension

from iivs.common.data import read_npy_shape, validate_float32_array, write_npy
from iivs.dhm.data.intensity.base import IntensityFileFolder
from iivs.dhm.data.intensity.bin import IntensityBinHeader

if TYPE_CHECKING:
    from kaparoo.filesystem.types import StrPath
    from numpy.typing import NDArray

    from iivs.common.data import OnNonFinite
    from iivs.dhm.data.common import ValidationLevel


def load_intensity_npy(
    path: StrPath,
    *,
    on_nonfinite: OnNonFinite = "ignore",
) -> NDArray[np.float32]:
    """Load a header-less `.npy` float32 intensity image.

    The `.npy` twin of `load_intensity_bin` / `load_intensity_txt`, but **image
    only**: a `.npy` carries no Koala header, so there is no `return_header`
    form and no `read_intensity_npy_header` -- the `pixel_size` metadata must be
    supplied separately (e.g. via `IntensityNpyFolder`). Loaded with
    `numpy.load(allow_pickle=False)`, so a pickled object array is rejected.

    Args:
        path: The `.npy` file to read.
        on_nonfinite: How to handle non-finite values (NaN, +inf, -inf) in the
            decoded data: "ignore" (default) accepts silently, "warn" emits a
            RuntimeWarning, "raise" raises a ValueError.

    Raises:
        FileNotFoundError: If `path` does not exist.
        NotAFileError: If `path` exists but is not a regular file.
        ValueError: If the array is pickled, not a single 2D float32 image, or
            holds non-finite values while `on_nonfinite` is "raise".
    """
    path = ensure_file_exists(path)
    data = np.load(path, allow_pickle=False)
    return validate_float32_array(data, on_nonfinite=on_nonfinite, allow_stack=False)


def save_intensity_npy(
    path: StrPath,
    data: NDArray[np.float32],
    *,
    overwrite: bool = False,
    on_nonfinite: OnNonFinite = "warn",
) -> None:
    """Save a 2D float32 intensity image as an uncompressed `.npy` file.

    Header-less: only the array is stored, so the `pixel_size` the `.bin` /
    `.txt` formats carry is dropped (supply it via `IntensityNpyFolder` on
    read). Written atomically.

    Args:
        path: The `.npy` file to write.
        data: The intensity image to save, of shape (H, W).
        overwrite: Whether to replace `path` if it already exists. Defaults to
            False.
        on_nonfinite: How to handle non-finite values (NaN, +inf, -inf) in
            `data`: "ignore" accepts silently, "warn" (default) emits a
            RuntimeWarning, "raise" rejects with a ValueError.

    Raises:
        ValueError: If `path` has a non-`.npy` extension, `data` is not a single
            2D float32 image, or it holds non-finite values while `on_nonfinite`
            is "raise".
        FileExistsError: If `path` exists and `overwrite` is False.
        FileNotFoundError: If the parent directory of `path` does not exist.
    """
    path = ensure_file_extension(path, "npy", add=True)
    data = validate_float32_array(data, on_nonfinite=on_nonfinite, allow_stack=False)
    write_npy(path, data, overwrite=overwrite)


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
        validate: ValidationLevel | None = "headers",
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
        on_nonfinite: OnNonFinite = "ignore",
    ) -> tuple[NDArray[np.float32], IntensityBinHeader]:
        """Load the `.npy` float32 image and synthesize its header from the shape."""
        data = load_intensity_npy(path, on_nonfinite=on_nonfinite)
        header = IntensityBinHeader(
            width=int(data.shape[1]),
            height=int(data.shape[0]),
            pixel_size=self._pixel_size,
        )
        return data, header
