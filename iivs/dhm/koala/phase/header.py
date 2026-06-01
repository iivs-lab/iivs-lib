from __future__ import annotations

__all__ = ("PhaseBinHeader", "PhaseUnit")

import warnings
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np
from kaparoo.filesystem import ensure_file_exists

if TYPE_CHECKING:
    from typing import IO, Self

    from kaparoo.filesystem.types import StrPath
    from numpy.typing import NDArray


class PhaseUnit(IntEnum):
    """Physical unit of the phase values stored in a .bin file."""

    UNKNOWN = 0
    RADIANS = 1
    METERS = 2


@dataclass(frozen=True, slots=True)
class PhaseBinHeader:
    """The fixed-size header of a Lyncée Tec Koala float32 .bin phase image.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        pixel_size: Physical size of one (square) pixel, in meters.
        height_scale: Height represented by one radian of phase, in
            meters; the phase-to-height conversion factor.
        unit: Physical unit of the stored phase values.
        version: Format version. Fixed at 1.
        endian: Byte-order flag. Fixed at 0 (little-endian).
    """

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

    width: int
    height: int
    pixel_size: float
    height_scale: float
    unit: PhaseUnit
    version: int = field(default=1, init=False)
    endian: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            msg = f"height and width must be positive (got {self.height}x{self.width})"
            raise ValueError(msg)

        if self.pixel_size <= 0:
            msg = f"pixel_size must be positive (got {self.pixel_size})"
            raise ValueError(msg)

        if self.height_scale <= 0:
            msg = f"height_scale must be positive (got {self.height_scale})"
            raise ValueError(msg)

        if self.unit not in PhaseUnit:
            msg = f"unit must be one of {list(PhaseUnit)} (got {self.unit!r})"
            raise ValueError(msg)

        if self.unit == PhaseUnit.UNKNOWN:
            msg = "unit is UNKNOWN; physical interpretation is undefined"
            warnings.warn(msg, stacklevel=2)

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
        """Field of view in meters as (height, width)."""
        return (self.height * self.pixel_size, self.width * self.pixel_size)

    @property
    def pixel_size_um(self) -> float:
        """Pixel size in micrometers."""
        return self.pixel_size * 1e6

    @property
    def field_of_view_um(self) -> tuple[float, float]:
        """Field of view in micrometers as (height, width)."""
        height_m, width_m = self.field_of_view
        return (height_m * 1e6, width_m * 1e6)

    @property
    def height_scale_nm(self) -> float:
        """Height scale (height per radian) in nanometers."""
        return self.height_scale * 1e9

    def to_dtype(self) -> NDArray[np.void]:
        record = np.zeros(1, dtype=PhaseBinHeader.DTYPE)
        record["version"] = self.version
        record["endian"] = self.endian
        record["header_size"] = PhaseBinHeader.HEADER_SIZE
        record["width"] = self.width
        record["height"] = self.height
        record["pixel_size"] = self.pixel_size
        record["height_scale"] = self.height_scale
        record["unit"] = int(self.unit)
        return record

    @classmethod
    def from_dtype(cls, record: np.void) -> Self:
        return cls(
            width=int(record["width"]),
            height=int(record["height"]),
            pixel_size=float(record["pixel_size"]),
            height_scale=float(record["height_scale"]),
            unit=PhaseUnit(int(record["unit"])),
        )

    @classmethod
    def from_stream(cls, fb: IO[bytes]) -> Self:
        """Read and validate a header from an open binary stream.

        Reads exactly the fixed-size header (works on any `IO[bytes]`,
        including `io.BytesIO`), leaving the stream positioned at the pixel
        data so callers can keep reading.

        Raises:
            ValueError: If the stream is too small for a header, or declares
                an unsupported header size.
        """

        raw = fb.read(cls.HEADER_SIZE)
        if len(raw) < cls.HEADER_SIZE:
            msg = f"file must be at least {cls.HEADER_SIZE} bytes for a header (got {len(raw)})"
            raise ValueError(msg)

        record = np.frombuffer(raw, dtype=cls.DTYPE, count=1)
        record_size = int(record[0]["header_size"])
        if record_size != cls.HEADER_SIZE:
            msg = f"header size must be {cls.HEADER_SIZE} (got {record_size})"
            raise ValueError(msg)

        return cls.from_dtype(record[0])

    @classmethod
    def from_file(cls, path: StrPath) -> Self:
        """Open `path` and read its header; a thin wrapper over `from_stream`.

        Raises:
            FileNotFoundError: If `path` does not exist.
            NotAFileError: If `path` exists but is not a regular file.
            ValueError: If the file is too small for a header, or declares an
                unsupported header size.
        """

        path = ensure_file_exists(path)
        with path.open("rb") as fb:
            return cls.from_stream(fb)
