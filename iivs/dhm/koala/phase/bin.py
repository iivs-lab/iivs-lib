from __future__ import annotations

__all__ = (
    "PhaseBinHeader",
    "PhaseBinSequence",
    "load_phase_bin",
    "read_phase_bin_header",
    "save_phase_bin",
)

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast, overload

import numpy as np
from kaparoo.data.sequences import FileFolderSequence
from kaparoo.filesystem import StagedFile, ensure_file_exists
from kaparoo.filesystem.search import search_files
from kaparoo.filesystem.search.filters import Regex
from kaparoo.utils import replace_if_none
from natsort import natsorted
from numpy.typing import NDArray

from iivs.dhm.koala.phase.base import PhaseSequence
from iivs.dhm.koala.phase.core import PhaseUnit, convert_phase_unit, validate_phase

if TYPE_CHECKING:
    from typing import IO, Literal, Self

    from kaparoo.filesystem.types import StrPath


_PIXEL_DTYPE = np.dtype("<f4")  # on-disk pixels: little-endian float32


# ========================== #
#           Header           #
# ========================== #


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

    # Packed (no alignment padding) -> exactly 23 bytes, matching the
    # on-disk Lyncée Tec Koala header layout.
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
    height_scale: float
    unit: PhaseUnit
    version: int = field(default=SUPPORTED_VERSION, init=False)
    endian: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Validate the fields."""
        if self.height <= 0 or self.width <= 0:
            msg = f"height and width must be positive (got {self.height}x{self.width})"
            raise ValueError(msg)

        if self.pixel_size <= 0:
            msg = f"pixel_size must be positive (got {self.pixel_size})"
            raise ValueError(msg)

        if self.height_scale <= 0:
            msg = f"height_scale must be positive (got {self.height_scale})"
            raise ValueError(msg)

        if self.unit not in (PhaseUnit.UNKNOWN, PhaseUnit.RADIANS, PhaseUnit.METERS):
            msg = f"unit must be one of UNKNOWN, RADIANS, METERS (got {self.unit!r})"
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
        """Serialize to a 1-element `PhaseBinHeader.DTYPE` record array."""
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
        """Build a header from a `PhaseBinHeader.DTYPE` structured scalar."""
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

        return cls.from_dtype(record)

    @classmethod
    def from_file(cls, path: StrPath) -> Self:
        """Open `path` and read its header; a thin wrapper over `from_stream`.

        Raises:
            FileNotFoundError: If `path` does not exist.
            NotAFileError: If `path` exists but is not a regular file.
            ValueError: If the file is too small for a header, or declares an
                unsupported header size, version, or byte order.
        """
        path = ensure_file_exists(path)
        with path.open("rb") as fb:
            return cls.from_stream(fb)


# ========================== #
#          Reading           #
# ========================== #


def read_phase_bin_header(path: StrPath) -> PhaseBinHeader:
    """Read only the header of a Lyncée Tec Koala .bin file, without the pixels.

    A thin wrapper over `PhaseBinHeader.from_file`; reads just the
    fixed-size header, so it stays cheap when curating many files by
    metadata (shape, field of view) without decoding the images.

    Raises:
        FileNotFoundError: If `path` does not exist.
        NotAFileError: If `path` exists but is not a regular file.
        ValueError: If the file is too small, declares an unsupported header
            size, or has invalid header fields.
    """
    return PhaseBinHeader.from_file(path)


def _read_pixels(fb: IO[bytes], header: PhaseBinHeader) -> NDArray[np.float32]:
    """Read the float32 pixel block after the header as an (H, W) array.

    Validates that the remaining bytes match the pixel count declared by
    `header` before decoding.
    """
    raw = fb.read()
    expected = header.pixel_count * _PIXEL_DTYPE.itemsize
    if len(raw) != expected:
        msg = f"pixel count must be {header.pixel_count} ({expected} bytes), got {len(raw)}"
        raise ValueError(msg)

    pixels = np.frombuffer(raw, dtype=_PIXEL_DTYPE)
    return pixels.reshape(header.shape).astype(np.float32, copy=True)


@overload
def load_phase_bin(
    path: StrPath,
    *,
    return_header: Literal[False] = False,
    on_nonfinite: Literal["ignore", "warn", "raise"] = ...,
) -> NDArray[np.float32]: ...


@overload
def load_phase_bin(
    path: StrPath,
    *,
    return_header: Literal[True],
    on_nonfinite: Literal["ignore", "warn", "raise"] = ...,
) -> tuple[NDArray[np.float32], PhaseBinHeader]: ...


@overload
def load_phase_bin(
    path: StrPath,
    *,
    return_header: bool,
    on_nonfinite: Literal["ignore", "warn", "raise"] = ...,
) -> NDArray[np.float32] | tuple[NDArray[np.float32], PhaseBinHeader]: ...


def load_phase_bin(
    path: StrPath,
    *,
    return_header: bool = False,
    on_nonfinite: Literal["ignore", "warn", "raise"] = "ignore",
) -> NDArray[np.float32] | tuple[NDArray[np.float32], PhaseBinHeader]:
    """Load a Lyncée Tec Koala float32 .bin phase image, and optionally its header.

    Args:
        path: The .bin file to read.
        return_header: Whether to also return the parsed `PhaseBinHeader`.
            Defaults to False.
        on_nonfinite: How to handle non-finite values (NaN, +inf, -inf),
            forwarded to `validate_phase`: "ignore" (default) accepts
            silently, "warn" emits a RuntimeWarning, "raise" raises a
            ValueError (useful to reject corrupted files). Defaults to
            "ignore", since a structurally valid file's contents are
            accepted by default.

    Returns:
        The phase image as a 2D float32 array, or an (image, header) tuple
        when `return_header` is True.

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
        header = PhaseBinHeader.from_stream(fb)
        data = _read_pixels(fb, header)

    data = validate_phase(data, on_nonfinite=on_nonfinite)
    return (data, header) if return_header else data


# ========================== #
#          Writing           #
# ========================== #


def _resolve_height_scale(
    height_scale: float | None,
    wavelength: float | None,
    refractive_delta: float | None,
) -> float:
    """Return `height_scale`, or derive it from `wavelength` and `refractive_delta`."""
    match height_scale, wavelength, refractive_delta:
        case scale, None, None if scale is not None:
            return scale
        case None, wave, delta if wave is not None and delta is not None:
            # height per radian = wavelength / (2*pi * refractive_delta)
            return wave / (2.0 * np.pi * delta)
        case _:
            msg = "give height_scale, or wavelength and refractive_delta (not both)"
            raise ValueError(msg)


def _to_storable_unit(
    data: NDArray[np.float32], unit: PhaseUnit, height_scale: float
) -> tuple[NDArray[np.float32], PhaseUnit]:
    """Coerce a (data, unit) pair into a unit the file can store.

    The on-disk format holds only UNKNOWN/RADIANS/METERS, so the code-only
    NANOMETERS unit is converted to METERS; every other unit passes through
    unchanged.
    """
    if unit is PhaseUnit.NANOMETERS:
        data = convert_phase_unit(
            data, source=unit, target=PhaseUnit.METERS, height_scale=height_scale
        )
        return data, PhaseUnit.METERS
    return data, unit


@overload
def save_phase_bin(
    path: StrPath,
    data: NDArray[np.float32],
    *,
    pixel_size: float,
    height_scale: float,
    unit: PhaseUnit = ...,
    overwrite: bool = ...,
    on_nonfinite: Literal["ignore", "warn", "raise"] = ...,
) -> None: ...


@overload
def save_phase_bin(
    path: StrPath,
    data: NDArray[np.float32],
    *,
    pixel_size: float,
    wavelength: float,
    refractive_delta: float,
    unit: PhaseUnit = ...,
    overwrite: bool = ...,
    on_nonfinite: Literal["ignore", "warn", "raise"] = ...,
) -> None: ...


def save_phase_bin(
    path: StrPath,
    data: NDArray[np.float32],
    *,
    pixel_size: float,
    height_scale: float | None = None,
    wavelength: float | None = None,
    refractive_delta: float | None = None,
    unit: PhaseUnit = PhaseUnit.RADIANS,
    overwrite: bool = False,
    on_nonfinite: Literal["ignore", "warn", "raise"] = "warn",
) -> None:
    """Save a 2D float32 phase image as a Lyncée Tec Koala .bin file.

    The phase-to-height scale is given either directly as `height_scale`,
    or as a `wavelength` and refractive-index difference `refractive_delta`
    pair (then height per radian = wavelength / (2*pi * refractive_delta)).
    Exactly one of the two forms must be given.

    The file is written atomically: content is staged to a temp file in the
    destination's directory and moved into place on success, so a failed
    write never leaves a partial or clobbered file.

    Args:
        path: The .bin file to write.
        data: The phase image to save, of shape (H, W).
        pixel_size: Physical size of one (square) pixel, in meters.
        height_scale: Height represented by one radian of phase, in meters.
            Mutually exclusive with `wavelength`/`refractive_delta`.
        wavelength: Illumination wavelength, in meters. Requires
            `refractive_delta`.
        refractive_delta: Refractive-index difference n1 - n2 (the plain
            difference, not the normalized contrast). Requires `wavelength`.
        unit: Physical unit of `data`. Defaults to RADIANS. NANOMETERS is
            converted to METERS before storing (the file cannot store it);
            UNKNOWN is stored as-is but emits a warning.
        overwrite: Whether to replace `path` if it already exists. Defaults
            to False.
        on_nonfinite: How to handle non-finite values (NaN, +inf, -inf),
            forwarded to `validate_phase`: "ignore" accepts silently,
            "warn" (default) emits a RuntimeWarning, "raise" rejects with a
            ValueError.

    Raises:
        ValueError: If neither or both scale forms are given, if `data` is
            not a single 2D float32 image, or if `data` holds non-finite
            values while `on_nonfinite` is "raise".
        FileExistsError: If `path` exists and `overwrite` is False.
        FileNotFoundError: If the parent directory of `path` does not exist.
    """
    height_scale = _resolve_height_scale(height_scale, wavelength, refractive_delta)

    # save stores a single image; validate_phase would also accept a stack.
    if data.ndim != 2:
        msg = f"data must be a single 2D image (got shape {data.shape})"
        raise ValueError(msg)
    data = validate_phase(data, on_nonfinite=on_nonfinite)
    data, unit = _to_storable_unit(data, unit, height_scale)

    if unit is PhaseUnit.UNKNOWN:
        msg = "saving with unit=UNKNOWN; physical interpretation is undefined"
        warnings.warn(msg, stacklevel=2)

    header = PhaseBinHeader(
        width=int(data.shape[1]),
        height=int(data.shape[0]),
        pixel_size=pixel_size,
        height_scale=height_scale,
        unit=unit,
    ).to_dtype()
    pixels = np.ascontiguousarray(data, dtype=_PIXEL_DTYPE)

    with StagedFile(path, binary=True, overwrite=overwrite) as staged:
        header.tofile(staged.file)
        pixels.tofile(staged.file)


# ========================== #
#          Sequence          #
# ========================== #


class PhaseBinSequence(
    FileFolderSequence[NDArray[np.float32], Path], PhaseSequence[Path]
):
    """An ordered sequence of Lyncée Tec Koala `.bin` phase images in a folder.

    Lists the direct children matching `{index:05d}_phase.bin` (exactly five
    digits, case-sensitive), in index order. All images are assumed to share
    one acquisition `header`, read once from the first file; that header drives
    the optional unit conversion and `validate`. Each item is the decoded
    float32 image (optionally converted to `target_unit`) and its metadata is
    the source path.

    Args:
        root: The folder to scan. Must exist, be a directory, and contain at
            least one matching file.
        target_unit: Unit to return loaded images in. When it differs from the
            stored unit, images are converted on load via the header's
            `height_scale`; an unreachable unit (e.g. converting to/from
            UNKNOWN) is rejected at construction. Defaults to None, which keeps
            the stored unit.
        validate: Run `validate` to this level ("names", "headers", or
            "data") at construction, or None to skip. Defaults to "headers".

    Raises:
        DirectoryNotFoundError: If `root` does not exist.
        NotADirectoryError: If `root` exists but is not a directory.
        FileNotFoundError: If no `NNNNN_phase.bin` files are found in `root`.
        ValueError: If `target_unit` cannot be converted from the stored unit,
            or if `validate` is set and the sequence fails validation.
    """

    def __init__(
        self,
        root: StrPath,
        *,
        target_unit: PhaseUnit | None = None,
        validate: Literal["names", "headers", "data"] | None = "headers",
    ) -> None:
        super().__init__(root)  # list_files rejects an empty folder

        self._header = read_phase_bin_header(self.get_file(0))
        self._target_unit = replace_if_none(target_unit, self._header.unit)

        # Fail fast: surface an unreachable target unit now, not lazily on every
        # get_item. convert_phase_unit no-ops when the units already match, so
        # this runs unconditionally; the empty array keeps it a pure pair check.
        convert_phase_unit(
            np.empty((0, 0), dtype=np.float32),
            source=self._header.unit,
            target=self._target_unit,
            height_scale=self._header.height_scale,
        )

        if validate is not None:
            self.validate(level=validate)

    @property
    def header(self) -> PhaseBinHeader:
        """The shared acquisition header, read from the first file."""
        return self._header

    @property
    def target_unit(self) -> PhaseUnit:
        """The unit that loaded images are returned in."""
        return self._target_unit

    def get_meta(self, index: int) -> Path:
        return self.get_file(index)

    def list_files(self, root: Path) -> list[Path]:
        files = search_files(root, name_filter=Regex(r"\d{5}_phase\.bin"), max_depth=1)
        if not files:
            msg = f"no NNNNN_phase.bin files found in {root}"
            raise FileNotFoundError(msg)
        return natsorted(files)

    def load_file(self, path: Path) -> NDArray[np.float32]:
        return convert_phase_unit(
            load_phase_bin(path),
            source=self._header.unit,
            target=self._target_unit,
            height_scale=self._header.height_scale,
        )

    def validate(
        self, *, level: Literal["names", "headers", "data"] = "headers"
    ) -> None:
        """Validate the sequence to the given `level`.

        Args:
            level: How deep to validate (cumulative): "names" checks only
                that files are numbered contiguously from 0 (index `i` is
                `{i:05d}_phase.bin`); "headers" (default) also that every
                file shares the first file's header; "data" also that every
                image decodes without error (expensive -- it reads every
                pixel).

        Raises:
            ValueError: If the numbering has gaps, a header differs from the
                first, or (at "data") an image fails to load.
        """
        for index in range(len(self)):
            self.validate_file(index, level=level)

    def validate_file(
        self, index: int, *, level: Literal["names", "headers", "data"] = "headers"
    ) -> None:
        """Validate the file at `index` to `level` (see `validate`)."""
        if level not in ("names", "headers", "data"):
            msg = f"level must be 'names', 'headers', or 'data' (got {level!r})"
            raise ValueError(msg)

        path = self.get_file(index)

        expected = f"{index:05d}_phase.bin"
        if path.name != expected:
            msg = f"non-contiguous numbering: expected {expected} at index {index}, got {path.name}"
            raise ValueError(msg)

        if level == "names":
            return

        # The first file is the reference header, so it is never compared.
        if index != 0 and read_phase_bin_header(path) != self.header:
            msg = f"header of {path.name} differs from the first file"
            raise ValueError(msg)

        if level == "data":
            load_phase_bin(path, on_nonfinite="raise")
