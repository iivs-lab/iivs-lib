from __future__ import annotations

__all__ = ("KoalaBinHeader", "load_bin", "write_bin")

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np
from kaparoo.filesystem import StagedFile, ensure_file_exists

from iivs.common.data import validate_float32_array

if TYPE_CHECKING:
    from typing import IO, Self

    from kaparoo.filesystem.types import StrPath
    from numpy.typing import NDArray

    from iivs.common.data import OnNonFinite


_PIXEL_DTYPE = np.dtype("<f4")  # on-disk pixels: little-endian float32


@dataclass(frozen=True, slots=True)
class KoalaBinHeader(ABC):
    """Base for the fixed-size 23-byte Lyncée Tec Koala `.bin` header.

    Holds the geometry (width, height, pixel_size) shared by every `.bin` modality
    (phase, intensity). The trailing ``hconv`` / ``unit`` bytes carry modality-specific
    meaning, so subclasses own them via `from_dtype` / `to_dtype`; phase reads them as a
    height scale plus `PhaseUnit`, while intensity treats them as Koala's no-op
    sentinel.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        pixel_size: Physical size of one (square) pixel, in m.
        version: Format version. Fixed at 1.
        endian: Byte-order flag. Fixed at 0 (little-endian).
    """

    # Packed (no alignment padding) -> exactly 23 bytes, matching the on-disk
    # Lyncée Tec Koala header layout (cf. Lyncée Tec's pyKoalaUtils `binkoala.py`).
    # Field names are clarified from that reference: header_size=head_size,
    # pixel_size=px_size, height_scale=hconv, unit=unit_code.
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

        Subclasses fill the remaining ``height_scale`` / ``unit`` bytes in their own
        `to_dtype`.
        """
        record = np.zeros(1, dtype=self.DTYPE)
        record["version"] = self.version
        record["endian"] = self.endian
        record["header_size"] = self.HEADER_SIZE
        record["width"] = self.width
        record["height"] = self.height
        record["pixel_size"] = self.pixel_size
        return record

    @abstractmethod
    def to_dtype(self) -> NDArray[np.void]:
        """Serialize to a 1-element `DTYPE` record array.

        Subclasses implement this, filling the modality-specific ``height_scale`` /
        ``unit`` bytes on top of `base_record`.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_dtype(cls, record: np.void) -> Self:
        """Build a header from a `DTYPE` structured scalar.

        Subclasses implement this, interpreting the ``height_scale`` / ``unit`` bytes
        for their modality.
        """
        raise NotImplementedError

    @classmethod
    def read_record(cls, f: IO[bytes]) -> np.void:
        """Read and structurally validate the fixed-size header record.

        Checks the declared header size, version, and byte order, leaving the stream
        positioned at the pixel data. Modality-specific field validation happens later
        in the subclass `from_dtype`.

        Raises:
            ValueError: If the stream is too small for a header, or declares an
                unsupported header size, version, or byte order.
        """
        raw = f.read(cls.HEADER_SIZE)
        n = len(raw)
        if n < cls.HEADER_SIZE:
            msg = f"file needs at least {cls.HEADER_SIZE} bytes for a header (got {n})"
            raise ValueError(msg)

        record = np.frombuffer(raw, dtype=cls.DTYPE, count=1)[0]
        record_size = int(record["header_size"])
        if record_size != cls.HEADER_SIZE:
            msg = f"header size must be {cls.HEADER_SIZE} (got {record_size})"
            raise ValueError(msg)

        version = int(record["version"])
        if version != cls.SUPPORTED_VERSION:
            msg = f"unsupported version {version} (expected {cls.SUPPORTED_VERSION})"
            raise ValueError(msg)

        endian = int(record["endian"])
        if endian != 0:  # 0 == little-endian; DTYPE stores fields little-endian
            msg = f"unsupported byte order (endian flag {endian}; only little-endian)"
            raise ValueError(msg)

        return record

    @classmethod
    def from_stream(cls, f: IO[bytes]) -> Self:
        """Read and validate a header from an open binary stream.

        Reads exactly the fixed-size header (works on any `IO[bytes]`, including
        `io.BytesIO`), leaving the stream positioned at the pixel data so callers can
        keep reading.

        Raises:
            ValueError: As `read_record`, plus any field validation raised by the
                subclass `from_dtype`.
        """
        return cls.from_dtype(cls.read_record(f))

    @classmethod
    def from_file(cls, path: StrPath) -> Self:
        """Open `path` and read its header; a thin wrapper over `from_stream`.

        Raises:
            FileNotFoundError: If `path` does not exist.
            NotAFileError: If `path` exists but is not a regular file.
            ValueError: As `from_stream`.
        """
        path = ensure_file_exists(path)
        with path.open("rb") as f:
            return cls.from_stream(f)


def load_bin[H: KoalaBinHeader](
    path: StrPath,
    header_cls: type[H],
    *,
    on_nonfinite: OnNonFinite = "ignore",
) -> tuple[NDArray[np.float32], H]:
    """Read a Koala `.bin` file's float32 image and header (the shared engine).

    Opens `path`, parses the fixed-size header as `header_cls`, decodes the pixel block
    (checking its byte count), and validates it. The per-modality `load_*_bin` wrappers
    bind their header type and add the `return_header` ergonomics.

    Args:
        path: The `.bin` file to read.
        header_cls: The `KoalaBinHeader` subclass to parse the header as.
        on_nonfinite: How to handle non-finite values (NaN, +inf, -inf) in the decoded
            data: "ignore" (default) accepts them silently, "warn" emits a
            RuntimeWarning, "raise" raises a ValueError.

    Returns:
        An ``(image, header)`` tuple: the float32 image of shape
        `header.shape` and the parsed `header_cls` instance.

    Raises:
        FileNotFoundError: If `path` does not exist.
        NotAFileError: If `path` exists but is not a regular file.
        ValueError: If the header is invalid, the pixel count is wrong, or the data
            holds non-finite values while `on_nonfinite` is "raise".
    """
    path = ensure_file_exists(path)
    with path.open("rb") as f:
        header = header_cls.from_stream(f)
        raw = f.read()

    n = len(raw)
    expected = header.pixel_count * _PIXEL_DTYPE.itemsize
    if n != expected:
        msg = f"pixel count must be {header.pixel_count} ({expected} bytes), got {n}"
        raise ValueError(msg)

    pixels = np.frombuffer(raw, dtype=_PIXEL_DTYPE)
    data = pixels.reshape(header.shape).astype(np.float32, copy=True)
    return validate_float32_array(data, on_nonfinite=on_nonfinite), header


def write_bin(
    path: StrPath,
    header: KoalaBinHeader,
    pixels: NDArray[np.float32],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically write `header` followed by `pixels` as a Koala `.bin` file.

    A failed write never leaves a partial or clobbered file.

    Args:
        path: The destination file to write (written as-is; the caller ensures the
            `.bin` extension).
        header: The header to write; its `to_dtype` fills the 23-byte record.
        pixels: The float32 image to write, of shape `header.shape`.
        overwrite: Whether to replace `path` if it already exists. Defaults to False.

    Raises:
        ValueError: If `pixels`' shape does not match `header.shape`.
        FileExistsError: If `path` exists and `overwrite` is False.
        FileNotFoundError: If the parent directory of `path` does not exist.
    """
    if pixels.shape != header.shape:
        msg = f"pixels shape must match header {header.shape} (got {pixels.shape})"
        raise ValueError(msg)

    record = header.to_dtype()
    block = np.ascontiguousarray(pixels, dtype=_PIXEL_DTYPE)
    with StagedFile(path, binary=True, overwrite=overwrite) as staged:
        record.tofile(staged.file)
        block.tofile(staged.file)
