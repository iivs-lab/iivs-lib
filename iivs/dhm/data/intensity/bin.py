from __future__ import annotations

__all__ = (
    "IntensityBinFolder",
    "IntensityBinHeader",
    "IntensityBinList",
    "load_intensity_bin",
    "read_intensity_bin_header",
    "save_intensity_bin",
)

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, overload, override

from kaparoo.filesystem import ensure_file_exists

from iivs.dhm.data.common import (
    KoalaBinHeader,
    read_bin_pixels,
    validate_float32_image,
    write_bin,
)
from iivs.dhm.data.intensity.base import IntensityFileFolder, IntensityFileList

if TYPE_CHECKING:
    from typing import Literal, Self

    import numpy as np
    from kaparoo.filesystem.types import StrPath
    from numpy.typing import NDArray


# ========================== #
#           Header           #
# ========================== #


@dataclass(frozen=True, slots=True)
class IntensityBinHeader(KoalaBinHeader):
    """The fixed-size header of a Lyncée Tec Koala float32 .bin intensity image.

    Intensity reconstructions share the 23-byte Koala header with phase, but
    carry no height scale or phase unit: Koala writes ``hconv = -1`` and
    ``unit = 0`` as a no-op sentinel. Only the geometry is meaningful, so this
    header adds no fields to the shared `KoalaBinHeader` -- it just pins those
    trailing bytes to the sentinel on write and ignores them on read.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        pixel_size: Physical size of one (square) pixel, in m.
        version: Format version. Fixed at 1.
        endian: Byte-order flag. Fixed at 0 (little-endian).
    """

    # Koala's no-op sentinel for the phase-only bytes intensity does not use.
    SENTINEL_HEIGHT_SCALE: ClassVar[float] = -1.0
    SENTINEL_UNIT: ClassVar[int] = 0

    def to_dtype(self) -> NDArray[np.void]:
        """Serialize to a 1-element `IntensityBinHeader.DTYPE` record array."""
        record = self.base_record()
        record["height_scale"] = self.SENTINEL_HEIGHT_SCALE
        record["unit"] = self.SENTINEL_UNIT
        return record

    @classmethod
    def from_dtype(cls, record: np.void) -> Self:
        """Build a header from a `DTYPE` scalar (the hconv/unit bytes are ignored)."""
        return cls(
            width=int(record["width"]),
            height=int(record["height"]),
            pixel_size=float(record["pixel_size"]),
        )


# ========================== #
#          Reading           #
# ========================== #


def read_intensity_bin_header(path: StrPath) -> IntensityBinHeader:
    """Read only the header of a Koala intensity `.bin` file, without the pixels.

    A thin wrapper over `IntensityBinHeader.from_file`; reads just the
    fixed-size header, so it stays cheap when curating many files by metadata
    (shape, field of view) without decoding the images.

    Raises:
        FileNotFoundError: If `path` does not exist.
        NotAFileError: If `path` exists but is not a regular file.
        ValueError: If the file is too small, declares an unsupported header
            size, or has invalid header fields.
    """
    return IntensityBinHeader.from_file(path)


@overload
def load_intensity_bin(
    path: StrPath,
    *,
    return_header: Literal[False] = False,
    on_nonfinite: Literal["ignore", "warn", "raise"] = ...,
) -> NDArray[np.float32]: ...


@overload
def load_intensity_bin(
    path: StrPath,
    *,
    return_header: Literal[True],
    on_nonfinite: Literal["ignore", "warn", "raise"] = ...,
) -> tuple[NDArray[np.float32], IntensityBinHeader]: ...


@overload
def load_intensity_bin(
    path: StrPath,
    *,
    return_header: bool,
    on_nonfinite: Literal["ignore", "warn", "raise"] = ...,
) -> NDArray[np.float32] | tuple[NDArray[np.float32], IntensityBinHeader]: ...


def load_intensity_bin(
    path: StrPath,
    *,
    return_header: bool = False,
    on_nonfinite: Literal["ignore", "warn", "raise"] = "ignore",
) -> NDArray[np.float32] | tuple[NDArray[np.float32], IntensityBinHeader]:
    """Load a Koala float32 .bin intensity image, and optionally its header.

    Args:
        path: The .bin file to read.
        return_header: Whether to also return the parsed `IntensityBinHeader`.
            Defaults to False.
        on_nonfinite: How to handle non-finite values (NaN, +inf, -inf),
            forwarded to `validate_float32_image`: "ignore" (default) accepts
            silently, "warn" emits a RuntimeWarning, "raise" raises a
            ValueError (useful to reject corrupted files).

    Returns:
        The intensity image as a 2D float32 array, or an (image, header)
        tuple when `return_header` is True.

    Raises:
        FileNotFoundError: If `path` does not exist.
        NotAFileError: If `path` exists but is not a regular file.
        ValueError: If the file is too small, declares an unsupported header
            size, has invalid header fields, holds the wrong number of
            pixels, or holds non-finite values while `on_nonfinite` is
            "raise".
    """
    path = ensure_file_exists(path)
    with path.open("rb") as fb:
        header = IntensityBinHeader.from_stream(fb)
        data = read_bin_pixels(fb, header)

    data = validate_float32_image(data, on_nonfinite=on_nonfinite)
    return (data, header) if return_header else data


# ========================== #
#          Writing           #
# ========================== #


def save_intensity_bin(
    path: StrPath,
    data: NDArray[np.float32],
    *,
    pixel_size: float,
    overwrite: bool = False,
    on_nonfinite: Literal["ignore", "warn", "raise"] = "warn",
) -> None:
    """Save a 2D float32 intensity image as a Koala .bin file.

    The phase-only ``hconv`` / ``unit`` bytes are written as Koala's no-op
    sentinel (``-1`` / ``0``); intensity has no height scale or unit.

    The file is written atomically: content is staged to a temp file in the
    destination's directory and moved into place on success, so a failed
    write never leaves a partial or clobbered file.

    Args:
        path: The .bin file to write.
        data: The intensity image to save, of shape (H, W).
        pixel_size: Physical size of one (square) pixel, in m.
        overwrite: Whether to replace `path` if it already exists. Defaults
            to False.
        on_nonfinite: How to handle non-finite values (NaN, +inf, -inf),
            forwarded to `validate_float32_image`: "ignore" accepts silently,
            "warn" (default) emits a RuntimeWarning, "raise" rejects with a
            ValueError.

    Raises:
        ValueError: If `data` is not a single 2D float32 image, or holds
            non-finite values while `on_nonfinite` is "raise".
        FileExistsError: If `path` exists and `overwrite` is False.
        FileNotFoundError: If the parent directory of `path` does not exist.
    """
    # save stores a single image (allow_stack=False), unlike the loader.
    data = validate_float32_image(data, on_nonfinite=on_nonfinite, allow_stack=False)

    header = IntensityBinHeader(
        width=int(data.shape[1]),
        height=int(data.shape[0]),
        pixel_size=pixel_size,
    )
    write_bin(path, header, data, overwrite=overwrite)


# ========================== #
#          Sequence          #
# ========================== #


class IntensityBinList(IntensityFileList):
    """An intensity sequence over an explicit, arbitrary list of `.bin` files.

    The general case (the `.bin` codec over `IntensityFileList`): no naming,
    contiguity, single-folder, or shared-header constraint; each file is read
    independently. `IntensityBinFolder` is the auto-discovered, same-shape
    special case of this.

    Args:
        files: The `.bin` files to expose, in the given order.

    Raises:
        ValueError: If any path does not have a `.bin` extension.
    """

    FILE_EXT: ClassVar[str] = "bin"

    @override
    def _read_header(self, path: StrPath) -> IntensityBinHeader:
        """Read the `.bin` header."""
        return read_intensity_bin_header(path)

    @override
    def _decode(
        self,
        path: StrPath,
        *,
        on_nonfinite: Literal["ignore", "warn", "raise"] = "ignore",
    ) -> NDArray[np.float32]:
        """Decode the `.bin` image."""
        return load_intensity_bin(path, on_nonfinite=on_nonfinite)


class IntensityBinFolder(IntensityFileFolder, IntensityBinList):
    """An ordered sequence of Koala `.bin` intensity images in a folder.

    The auto-discovered, same-shape special case of `IntensityBinList`: lists
    the direct children matching `{index:05d}_intensity.bin` (exactly five
    digits, case-sensitive), sharing one acquisition `header`. Construction and
    validation are inherited; this supplies only the `.bin` extension.

    Args:
        root: The folder to scan. Must exist, be a directory, and contain at
            least one matching file.
        validate: Run `validate` to this level at construction, or None to
            skip. Defaults to "headers".

    Raises:
        DirectoryNotFoundError: If `root` does not exist.
        NotADirectoryError: If `root` exists but is not a directory.
        FileNotFoundError: If no `NNNNN_intensity.bin` files are found in
            `root`.
        ValueError: If `validate` is set and the sequence fails validation.
    """
