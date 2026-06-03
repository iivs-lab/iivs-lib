"""Building blocks shared across the data modalities (phase, intensity, ...).

Holds the cross-modality primitives that the per-modality packages compose:
the Koala `.bin` header/IO (`KoalaBinHeader`, `read_bin_pixels`, `write_bin`),
the numbered-folder sequence base (`SequentialFileFolder`), the same-shape
mixin (`FrameShapedMixin`), and the float32 image validator
(`validate_float32_image`).
"""

from __future__ import annotations

__all__ = (
    "FrameShapedMixin",
    "KoalaBinHeader",
    "SequentialFileFolder",
    "parse_txt_grid",
    "read_bin_pixels",
    "validate_float32_image",
    "validate_uint8_image",
    "write_bin",
)

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast, override

import numpy as np
from kaparoo.data.sequences import FileFolderSequence
from kaparoo.filesystem import StagedFile, ensure_file_exists
from kaparoo.filesystem.search import search_files
from kaparoo.filesystem.search.filters import Regex
from natsort import natsorted

if TYPE_CHECKING:
    from typing import IO, Literal, Self

    from kaparoo.filesystem.types import StrPath
    from numpy.typing import NDArray


# ========================== #
#         Bin format         #
# ========================== #


_PIXEL_DTYPE = np.dtype("<f4")  # on-disk pixels: little-endian float32


@dataclass(frozen=True, slots=True)
class KoalaBinHeader:
    """Base for the fixed-size 23-byte Lyncée Tec Koala .bin header.

    Holds the geometry (width, height, pixel_size) and the on-disk format
    machinery shared by every `.bin` modality (phase, intensity): the packed
    `DTYPE`, the structural read (size / version / byte-order checks), and the
    geometry conveniences. The trailing ``hconv`` / ``unit`` bytes carry
    modality-specific meaning, so subclasses own them via `from_dtype` /
    `to_dtype` -- phase reads them as a height scale plus `PhaseUnit`, while
    intensity treats them as Koala's no-op sentinel.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        pixel_size: Physical size of one (square) pixel, in m.
        version: Format version. Fixed at 1.
        endian: Byte-order flag. Fixed at 0 (little-endian).
    """

    # Packed (no alignment padding) -> exactly 23 bytes, matching the on-disk
    # Lyncée Tec Koala header layout (cf. Lyncée Tec's pyKoalaUtils
    # `binkoala.py`). Field names are clarified from that reference:
    # header_size=head_size, pixel_size=px_size, height_scale=hconv, unit=unit_code.
    DTYPE: ClassVar[np.dtype[np.void]] = cast(
        "np.dtype[np.void]",
        np.dtype(
            [
                ("version", "u1"),
                ("endian", "u1"),
                ("header_size", "<i4"),
                ("width", "<i4"),
                ("height", "<i4"),
                ("pixel_size", "<f4"),
                ("height_scale", "<f4"),
                ("unit", "u1"),
            ],
        ),
    )

    HEADER_SIZE: ClassVar[int] = DTYPE.itemsize
    SUPPORTED_VERSION: ClassVar[int] = 1

    width: int
    height: int
    pixel_size: float
    version: int = field(default=SUPPORTED_VERSION, init=False)
    endian: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Validate the geometry shared by every `.bin` modality."""
        if self.height <= 0 or self.width <= 0:
            msg = f"height and width must be positive (got {self.height}x{self.width})"
            raise ValueError(msg)

        if self.pixel_size <= 0:
            msg = f"pixel_size must be positive (got {self.pixel_size})"
            raise ValueError(msg)

    @property
    def shape(self) -> tuple[int, int]:
        """Pixel dimensions as (height, width)."""
        return (self.height, self.width)

    @property
    def pixel_count(self) -> int:
        """Total number of pixels (height * width)."""
        return self.height * self.width

    @property
    def field_of_view(self) -> tuple[float, float]:
        """Field of view in m as (height, width)."""
        return (self.height * self.pixel_size, self.width * self.pixel_size)

    @property
    def pixel_size_um(self) -> float:
        """Pixel size in um."""
        return self.pixel_size * 1e6

    @property
    def field_of_view_um(self) -> tuple[float, float]:
        """Field of view in um as (height, width)."""
        height_m, width_m = self.field_of_view
        return (height_m * 1e6, width_m * 1e6)

    def base_record(self) -> NDArray[np.void]:
        """Allocate a `DTYPE` record with the shared fields filled.

        Subclasses fill the remaining ``height_scale`` / ``unit`` bytes in
        their own `to_dtype`.
        """
        record = np.zeros(1, dtype=self.DTYPE)
        record["version"] = self.version
        record["endian"] = self.endian
        record["header_size"] = self.HEADER_SIZE
        record["width"] = self.width
        record["height"] = self.height
        record["pixel_size"] = self.pixel_size
        return record

    def to_dtype(self) -> NDArray[np.void]:
        """Serialize to a 1-element `DTYPE` record array.

        Subclasses implement this, filling the modality-specific
        ``height_scale`` / ``unit`` bytes on top of `base_record`.
        """
        raise NotImplementedError

    @classmethod
    def from_dtype(cls, record: np.void) -> Self:
        """Build a header from a `DTYPE` structured scalar.

        Subclasses implement this, interpreting the ``height_scale`` /
        ``unit`` bytes for their modality.
        """
        raise NotImplementedError

    @classmethod
    def read_record(cls, fb: IO[bytes]) -> np.void:
        """Read and structurally validate the fixed-size header record.

        Checks the declared header size, version, and byte order, leaving the
        stream positioned at the pixel data. Modality-specific field
        validation happens later in the subclass `from_dtype`.

        Raises:
            ValueError: If the stream is too small for a header, or declares
                an unsupported header size, version, or byte order.
        """
        raw = fb.read(cls.HEADER_SIZE)
        if len(raw) < cls.HEADER_SIZE:
            msg = f"file must be at least {cls.HEADER_SIZE} bytes for a header (got {len(raw)})"
            raise ValueError(msg)

        record = np.frombuffer(raw, dtype=cls.DTYPE, count=1)[0]
        record_size = int(record["header_size"])
        if record_size != cls.HEADER_SIZE:
            msg = f"header size must be {cls.HEADER_SIZE} (got {record_size})"
            raise ValueError(msg)

        version = int(record["version"])
        if version != cls.SUPPORTED_VERSION:
            msg = f"unsupported header version {version} (expected {cls.SUPPORTED_VERSION})"
            raise ValueError(msg)

        endian = int(record["endian"])
        if endian != 0:  # 0 == little-endian; DTYPE stores fields little-endian
            msg = f"unsupported byte order (endian flag {endian}; only little-endian)"
            raise ValueError(msg)

        return record

    @classmethod
    def from_stream(cls, fb: IO[bytes]) -> Self:
        """Read and validate a header from an open binary stream.

        Reads exactly the fixed-size header (works on any `IO[bytes]`,
        including `io.BytesIO`), leaving the stream positioned at the pixel
        data so callers can keep reading.

        Raises:
            ValueError: As `read_record`, plus any field validation raised by
                the subclass `from_dtype`.
        """
        return cls.from_dtype(cls.read_record(fb))

    @classmethod
    def from_file(cls, path: StrPath) -> Self:
        """Open `path` and read its header; a thin wrapper over `from_stream`.

        Raises:
            FileNotFoundError: If `path` does not exist.
            NotAFileError: If `path` exists but is not a regular file.
            ValueError: As `from_stream`.
        """
        path = ensure_file_exists(path)
        with path.open("rb") as fb:
            return cls.from_stream(fb)


def read_bin_pixels(fb: IO[bytes], header: KoalaBinHeader) -> NDArray[np.float32]:
    """Read the float32 pixel block after the header as an (H, W) array.

    Validates that the remaining bytes match the pixel count declared by
    `header` before decoding.

    Raises:
        ValueError: If the byte count does not match `header.pixel_count`.
    """
    raw = fb.read()
    expected = header.pixel_count * _PIXEL_DTYPE.itemsize
    if len(raw) != expected:
        msg = f"pixel count must be {header.pixel_count} ({expected} bytes), got {len(raw)}"
        raise ValueError(msg)

    pixels = np.frombuffer(raw, dtype=_PIXEL_DTYPE)
    return pixels.reshape(header.shape).astype(np.float32, copy=True)


def write_bin(
    path: StrPath,
    header: KoalaBinHeader,
    pixels: NDArray[np.float32],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically write `header` followed by `pixels` as a Koala `.bin` file.

    Content is staged to a temp file in the destination's directory and moved
    into place on success, so a failed write never leaves a partial or
    clobbered file.

    Raises:
        FileExistsError: If `path` exists and `overwrite` is False.
        FileNotFoundError: If the parent directory of `path` does not exist.
    """
    record = header.to_dtype()
    block = np.ascontiguousarray(pixels, dtype=_PIXEL_DTYPE)
    with StagedFile(path, binary=True, overwrite=overwrite) as staged:
        record.tofile(staged.file)
        block.tofile(staged.file)


# ========================== #
#       Frame-shape mixin    #
# ========================== #


class FrameShapedMixin(ABC):
    """Mixin marking a sequence whose items all share one `frame_shape`.

    Mix into a modality sequence on a same-shape source (e.g. a single
    acquisition) to force `frame_shape` to be implemented. There is no
    per-modality `Uniform*Sequence`: "a uniform phase sequence" is just
    ``isinstance(x, PhaseSequence) and isinstance(x, FrameShapedMixin)``
    (and likewise for the other modalities). `SequentialFileFolder` mixes this
    in for every numbered folder; a single-file source like `HologramRawFile`
    mixes it in directly.
    """

    @property
    @abstractmethod
    def frame_shape(self) -> tuple[int, int]:
        """The pixel dimensions (height, width) shared by every item."""
        raise NotImplementedError


# ========================== #
#       Folder sequence      #
# ========================== #


class SequentialFileFolder[T](FileFolderSequence[T, Path], FrameShapedMixin):
    """A folder of contiguously numbered `{index:05d}_<stem>.<ext>` files.

    Each such folder is one acquisition's frames, hence same-shape: it mixes in
    `FrameShapedMixin`, and subclasses implement `frame_shape` from their header
    or first file. Factors the discovery and validation shared by every
    modality folder (phase, intensity, hologram, ...): `list_files` (numbered
    discovery), `get_meta` (= source path), the `validate` loop, and the
    name-contiguity check. A subclass declares the filename parts and validation
    depth as class attributes, implements `load_file`, and supplies its
    per-format consistency check by overriding `_validate_content`.

    Class attributes:
        FILE_STEM: The ``<stem>`` in ``{index:05d}_<stem>.<ext>`` (e.g. "phase").
        FILE_EXT: The file extension without the dot (e.g. "bin").
        LEVELS: The validation levels this folder accepts (a subset of
            "names" / "headers" / "data").
        DEFAULT_LEVEL: The level `validate` / `validate_file` use when given none.
    """

    FILE_STEM: ClassVar[str]
    FILE_EXT: ClassVar[str]
    LEVELS: ClassVar[tuple[str, ...]] = ("names",)
    DEFAULT_LEVEL: ClassVar[str] = "names"

    @override
    def list_files(self, root: Path) -> list[Path]:
        """List the `NNNNN_<stem>.<ext>` files under `root`, in index order."""
        pattern = rf"\d{{5}}_{self.FILE_STEM}\.{self.FILE_EXT}"
        files = search_files(root, name_filter=Regex(pattern), max_depth=1)
        if not files:
            msg = f"no NNNNN_{self.FILE_STEM}.{self.FILE_EXT} files found in {root}"
            raise FileNotFoundError(msg)
        return natsorted(files)

    @override
    def get_meta(self, index: int) -> Path:
        """Return the source path of the file at `index`."""
        return self.get_file(index)

    def expected_name(self, index: int) -> str:
        """The contiguous filename expected at `index`."""
        return f"{index:05d}_{self.FILE_STEM}.{self.FILE_EXT}"

    def validate(
        self, *, level: Literal["names", "headers", "data"] | None = None
    ) -> None:
        """Validate every file to `level` (defaults to `DEFAULT_LEVEL`)."""
        for index in range(len(self)):
            self.validate_file(index, level=level)

    def validate_file(
        self,
        index: int,
        *,
        level: Literal["names", "headers", "data"] | None = None,
    ) -> None:
        """Validate the file at `index` to `level` (defaults to `DEFAULT_LEVEL`).

        Always checks the contiguous name; any deeper level defers to the
        per-format `_validate_content`.

        Raises:
            ValueError: If `level` is unsupported, the numbering is
                non-contiguous, or `_validate_content` rejects the file.
        """
        resolved = self.DEFAULT_LEVEL if level is None else level
        if resolved not in self.LEVELS:
            msg = f"level must be one of {self.LEVELS} (got {resolved!r})"
            raise ValueError(msg)

        path = self.get_file(index)
        expected = self.expected_name(index)
        if path.name != expected:
            msg = f"non-contiguous numbering: expected {expected} at index {index}, got {path.name}"
            raise ValueError(msg)

        if resolved != "names":
            self._validate_content(path, level=resolved)

    def _validate_content(self, path: Path, *, level: str) -> None:
        """Per-format consistency check for levels beyond "names" (subclass hook)."""
        raise NotImplementedError


# ========================== #
#         Validation         #
# ========================== #


def _validate_image_dims(data: NDArray[np.generic], *, allow_stack: bool) -> None:
    """Check the leading dimensions of an image array.

    With `allow_stack`, any number of leading axes is allowed (at least two
    dims); without it, exactly a single 2-D image is required.
    """
    if allow_stack:
        if data.ndim < 2:
            msg = f"data must be at least 2-dimensional (got {data.ndim})"
            raise ValueError(msg)
    elif data.ndim != 2:
        msg = f"data must be a single 2D image (got shape {data.shape})"
        raise ValueError(msg)


def validate_float32_image(
    data: NDArray[np.float32],
    *,
    on_nonfinite: Literal["ignore", "warn", "raise"] = "warn",
    allow_stack: bool = True,
) -> NDArray[np.float32]:
    """Validate a float32 image (or stack) and return it.

    The last two axes are the image height and width. By default any number of
    leading axes is allowed (a stack); pass `allow_stack=False` to require a
    single 2-D image. `data` is never modified.

    Modality-agnostic: phase and intensity validate their float32 arrays
    through this.

    Args:
        data: The image or stack to validate, of shape (..., H, W).
        on_nonfinite: How to handle non-finite values (NaN, +inf, -inf):
            "ignore" accepts them silently, "warn" (default) accepts them but
            emits a RuntimeWarning, "raise" raises a ValueError.
        allow_stack: Whether to accept more than two dimensions. Defaults to
            True; set False to require a single 2-D image.

    Raises:
        ValueError: If `data` is not float32, has the wrong dimensionality, or
            holds non-finite values while `on_nonfinite` is "raise".
    """
    _validate_image_dims(data, allow_stack=allow_stack)

    if data.dtype != np.float32:
        msg = f"data must be float32 (got {data.dtype})"
        raise ValueError(msg)

    match on_nonfinite:
        case "ignore":
            return data
        case "warn" | "raise":
            pass
        case _:
            msg = f"on_nonfinite must be 'ignore', 'warn', or 'raise' (got {on_nonfinite!r})"
            raise ValueError(msg)

    if not np.all(np.isfinite(data)):
        nan = int(np.isnan(data).sum())
        posinf = int(np.isposinf(data).sum())
        neginf = int(np.isneginf(data).sum())
        counts = f"{nan} NaN, {posinf} +inf, {neginf} -inf"
        if on_nonfinite == "raise":
            msg = f"data must be finite (got {counts})"
            raise ValueError(msg)
        msg = f"data is not finite ({counts})"
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    return data


def parse_txt_grid(lines: list[str], *, shape: tuple[int, int]) -> NDArray[np.float32]:
    """Parse whitespace-separated float rows into a float32 (H, W) array.

    Used by the Koala `Float/Txt` readers: a modality's text export is a small
    key=value header followed by `height` rows of `width` floats. Blank lines
    are ignored.

    Raises:
        ValueError: If the parsed grid does not match `shape`, or a row is
            malformed.
    """
    rows = [line for line in lines if line.strip()]
    try:
        grid = np.loadtxt(rows, dtype=np.float32, ndmin=2)
    except ValueError as exc:
        msg = f"malformed txt grid: {exc}"
        raise ValueError(msg) from exc
    if grid.shape != shape:
        msg = f"txt grid must be {shape} (got {grid.shape})"
        raise ValueError(msg)
    return grid


def validate_uint8_image(
    data: NDArray[np.uint8], *, allow_stack: bool = True
) -> NDArray[np.uint8]:
    """Validate a uint8 image (or stack) and return it.

    The last two axes are the image height and width. By default any number of
    leading axes is allowed (a stack); pass `allow_stack=False` to require a
    single 2-D image. uint8 is inherently finite, so there is no non-finite
    policy. `data` is never modified.

    Modality-agnostic: holograms (and any future 8-bit `.tif` phase/intensity
    sequences) validate their uint8 arrays through this.

    Args:
        data: The image or stack to validate, of shape (..., H, W).
        allow_stack: Whether to accept more than two dimensions. Defaults to
            True; set False to require a single 2-D image.

    Raises:
        ValueError: If `data` is not uint8 or has the wrong dimensionality.
    """
    _validate_image_dims(data, allow_stack=allow_stack)

    if data.dtype != np.uint8:
        msg = f"data must be uint8 (got {data.dtype})"
        raise ValueError(msg)

    return data
