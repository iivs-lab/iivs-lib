from __future__ import annotations

__all__ = ("PhaseBinHeader", "PhaseUnit", "load_bin", "save_bin")

import warnings
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast, overload

import numpy as np
from kaparoo.filesystem import file_exists

if TYPE_CHECKING:
    from typing import Literal, Self

    from kaparoo.filesystem.types import StrPath
    from numpy.typing import NDArray


class PhaseUnit(IntEnum):
    UNKNOWN = 0
    RADIANS = 1
    METERS = 2


@dataclass(frozen=True, slots=True)
class PhaseBinHeader:
    DTYPE: ClassVar[np.dtype[np.void]] = cast(
        "np.dtype[np.void]",
        np.dtype(
            [
                ("version", "u1"),
                ("endian", "u1"),
                ("head_size", "<i4"),
                ("width", "<i4"),
                ("height", "<i4"),
                ("px_size", "<f4"),
                ("hconv", "<f4"),
                ("unit_code", "u1"),
            ],
        ),
    )

    width: int
    height: int
    pixel_size: float
    height_per_radian: float
    unit: PhaseUnit
    version: int = field(default=1, init=False)
    endian: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Validate field invariants."""
        if self.height <= 0 or self.width <= 0:
            msg = f"height and width must be positive, got {self.height}x{self.width}"
            raise ValueError(msg)

        if self.pixel_size <= 0:
            msg = f"pixel_size must be positive, got {self.pixel_size}"
            raise ValueError(msg)

        if self.height_per_radian <= 0:
            msg = f"height_per_radian must be positive, got {self.height_per_radian}"
            raise ValueError(msg)

        if self.unit not in PhaseUnit:
            msg = f"unit must be one of {list(PhaseUnit)}, got {self.unit}"
            raise ValueError(msg)

        if self.unit == PhaseUnit.UNKNOWN:
            msg = "unit is UNKNOWN; physical interpretation is undefined"
            warnings.warn(msg, stacklevel=2)

    @property
    def shape(self) -> tuple[int, int]:
        """Pixel dimensions as ``(height, width)``."""
        return (self.height, self.width)

    @property
    def field_of_view(self) -> tuple[float, float]:
        """Field of view in meters as ``(height, width)``."""
        return (self.height * self.pixel_size, self.width * self.pixel_size)

    @property
    def pixel_size_um(self) -> float:
        """Pixel size in micrometers."""
        return self.pixel_size * 1e6

    @property
    def field_of_view_um(self) -> tuple[float, float]:
        """Field of view in micrometers as ``(height, width)``."""
        height_m, width_m = self.field_of_view
        return (height_m * 1e6, width_m * 1e6)

    @property
    def height_per_radian_nm(self) -> float:
        """Height-per-radian in nanometers."""
        return self.height_per_radian * 1e9

    @classmethod
    def from_dtype(cls, record: np.void) -> Self:
        """Build a header from a `PhaseBinHeader.DTYPE` structured scalar."""
        return cls(
            width=int(record["width"]),
            height=int(record["height"]),
            pixel_size=float(record["px_size"]),
            height_per_radian=float(record["hconv"]),
            unit=PhaseUnit(int(record["unit_code"])),
        )

    def to_dtype(self) -> NDArray[np.void]:
        """Serialize to a 1-element `PhaseBinHeader.DTYPE` array."""
        record = np.zeros(1, dtype=PhaseBinHeader.DTYPE)
        record["version"] = self.version
        record["endian"] = self.endian
        record["head_size"] = PhaseBinHeader.DTYPE.itemsize
        record["width"] = self.width
        record["height"] = self.height
        record["px_size"] = self.pixel_size
        record["hconv"] = self.height_per_radian
        record["unit_code"] = int(self.unit)
        return record


@overload
def load_bin(
    path: StrPath, *, return_header: Literal[False] = ...
) -> NDArray[np.float32]: ...


@overload
def load_bin(
    path: StrPath, *, return_header: Literal[True]
) -> tuple[NDArray[np.float32], PhaseBinHeader]: ...


def load_bin(
    path: StrPath, *, return_header: bool = False
) -> NDArray[np.float32] | tuple[NDArray[np.float32], PhaseBinHeader]:

    raw = Path(path).read_bytes()
    head_size = PhaseBinHeader.DTYPE.itemsize
    if len(raw) < head_size:
        msg = f"{path}: file is too short to contain a Koala .bin header"
        raise ValueError(msg)

    record = np.frombuffer(raw, dtype=PhaseBinHeader.DTYPE, count=1)[0]
    stored_size = int(record["head_size"])
    if stored_size != head_size:
        msg = (
            f"{path}: unexpected header size {stored_size} "
            f"(this reader supports the {head_size}-byte layout)"
        )
        raise ValueError(msg)

    header = PhaseBinHeader.from_dtype(record)
    expected = header.width * header.height
    pixels = np.frombuffer(raw, dtype="<f4", offset=head_size)
    if pixels.size != expected:
        msg = (
            f"{path}: expected {expected} float32 values for "
            f"{header.width}x{header.height}, found {pixels.size}"
        )
        raise ValueError(msg)

    image = pixels.reshape(header.shape).astype(np.float32, copy=True)
    if return_header:
        return image, header
    return image


def save_bin(
    path: StrPath,
    data: NDArray[np.float32],
    *,
    pixel_size: float,
    height_per_radian: float,
    unit: PhaseUnit = PhaseUnit.RADIANS,
    overwrite: bool = False,
) -> None:
    if data.ndim != 2:
        msg = f"data must be 2D, got shape {data.shape}"
        raise ValueError(msg)

    if data.dtype != np.float32:
        msg = f"data must be float32, got dtype {data.dtype}"
        raise ValueError(msg)

    header = PhaseBinHeader(
        width=int(data.shape[1]),
        height=int(data.shape[0]),
        pixel_size=pixel_size,
        height_per_radian=height_per_radian,
        unit=unit,
    ).to_dtype()

    if file_exists(file := Path(path)) and not overwrite:
        msg = f"{path} already exists; pass overwrite=True to replace it"
        raise FileExistsError(msg)

    pixels = np.ascontiguousarray(data, dtype="<f4")
    with file.open("wb") as fd:
        header.tofile(fd)
        pixels.tofile(fd)
