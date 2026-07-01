from __future__ import annotations

__all__ = ("PhaseNpyFolder", "load_phase_npy", "save_phase_npy")

from typing import TYPE_CHECKING, ClassVar, override

import numpy as np
from kaparoo.filesystem import ensure_file_exists, ensure_file_extension

from iivs.common.data import read_npy_shape, write_npy
from iivs.dhm.data.common import validate_float32_array
from iivs.dhm.data.phase.base import PhaseFileFolder
from iivs.dhm.data.phase.bin import PhaseBinHeader
from iivs.dhm.data.phase.unit import resolve_height_scale

if TYPE_CHECKING:
    from kaparoo.filesystem.types import StrPath
    from numpy.typing import NDArray

    from iivs.dhm.data.common import OnNonFinite, ValidationLevel
    from iivs.dhm.data.phase.unit import PhaseUnit


def load_phase_npy(
    path: StrPath,
    *,
    on_nonfinite: OnNonFinite = "ignore",
) -> NDArray[np.float32]:
    """Load a header-less `.npy` float32 phase image.

    The `.npy` twin of `load_phase_bin` / `load_phase_txt`, but **image only**:
    a `.npy` carries no Koala header, so there is no `return_header` form and no
    `read_phase_npy_header` -- the `pixel_size` / `unit` / `height_scale`
    metadata must be supplied separately (e.g. via `PhaseNpyFolder`). Loaded
    with `numpy.load(allow_pickle=False)`, so a pickled object array is
    rejected.

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


def save_phase_npy(
    path: StrPath,
    data: NDArray[np.float32],
    *,
    overwrite: bool = False,
    on_nonfinite: OnNonFinite = "warn",
) -> None:
    """Save a 2D float32 phase image as an uncompressed `.npy` file.

    Header-less: `.npy` stores only the array, so the `pixel_size` / `unit` /
    `height_scale` metadata the `.bin` / `.txt` formats carry is dropped (supply
    it when reading via `PhaseNpyFolder`). Written atomically.

    Args:
        path: The `.npy` file to write.
        data: The phase image to save, of shape (H, W).
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


class PhaseNpyFolder(PhaseFileFolder):
    """An ordered sequence of header-less `{index:05d}_phase.npy` float32 phase images.

    A `.npy` array carries no Koala header, so the acquisition metadata the
    `.bin` / `.txt` formats embed -- `pixel_size`, `unit`, and the phase-to-height
    `height_scale` -- is supplied to the constructor instead and shared by every
    frame. Files are loaded with `numpy.load(allow_pickle=False)`, so a pickled
    object array is rejected; create them with `numpy.save` (uncompressed `.npy`,
    one 2-D float32 frame per file).

    Args:
        root: The folder to scan.
        pixel_size: Physical size of one (square) pixel, in m.
        unit: Physical unit of the stored phase values.
        height_scale: Height represented by one rad of phase, in m. Mutually
            exclusive with `wavelength` / `refractive_delta`.
        wavelength: Illumination wavelength, in m (with `refractive_delta`).
        refractive_delta: Refractive-index difference (with `wavelength`).
        target_unit: Unit to return loaded images in (None keeps `unit`).
        validate: Run `validate` to this level at construction, or None to skip.
            Defaults to "headers".

    Raises:
        ValueError: If neither or both height-scale forms are given, if
            `target_unit` cannot be reached from `unit`, or if `validate` is set
            and the sequence fails validation.
    """

    FILE_EXT: ClassVar[str] = "npy"

    def __init__(
        self,
        root: StrPath,
        *,
        pixel_size: float,
        unit: PhaseUnit,
        height_scale: float | None = None,
        wavelength: float | None = None,
        refractive_delta: float | None = None,
        target_unit: PhaseUnit | None = None,
        validate: ValidationLevel | None = "headers",
    ) -> None:
        # Set the synthesized-header metadata before super().__init__, which
        # reads the first file's header via _read_header (uses these).
        self._pixel_size = pixel_size
        self._unit = unit
        self._height_scale = resolve_height_scale(
            height_scale, wavelength, refractive_delta
        )
        super().__init__(root, target_unit=target_unit, validate=validate)

    def _header_for(self, height: int, width: int) -> PhaseBinHeader:
        """Synthesize the shared header from the constructor metadata + a shape."""
        return PhaseBinHeader(
            width=width,
            height=height,
            pixel_size=self._pixel_size,
            height_scale=self._height_scale,
            unit=self._unit,
        )

    @override
    def _read_header(self, path: StrPath) -> PhaseBinHeader:
        """Build the header from the `.npy` shape and the shared metadata."""
        height, width = read_npy_shape(path)
        return self._header_for(height, width)

    @override
    def _decode(
        self,
        path: StrPath,
        *,
        on_nonfinite: OnNonFinite = "ignore",
    ) -> tuple[NDArray[np.float32], PhaseBinHeader]:
        """Load the `.npy` float32 image and synthesize its header from the shape."""
        data = load_phase_npy(path, on_nonfinite=on_nonfinite)
        return data, self._header_for(data.shape[0], data.shape[1])
