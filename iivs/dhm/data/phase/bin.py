from __future__ import annotations

__all__ = (
    "PhaseBinFolder",
    "PhaseBinHeader",
    "PhaseBinList",
    "load_phase_bin",
    "read_phase_bin_header",
    "save_phase_bin",
)

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, overload, override

import numpy as np
from kaparoo.data.sequences import FileListSequence
from kaparoo.filesystem import ensure_file_exists
from kaparoo.utils import replace_if_none
from numpy.typing import NDArray

from iivs.dhm.data.binfile import KoalaBinHeader, read_bin_pixels, write_bin
from iivs.dhm.data.folder import SequentialFileFolderSequence
from iivs.dhm.data.phase.base import PhaseSequence
from iivs.dhm.data.phase.core import PhaseUnit, convert_phase_unit, validate_phase
from iivs.dhm.data.sequence import FrameShapedMixin

if TYPE_CHECKING:
    from typing import Literal, Self

    from kaparoo.filesystem.types import StrPath, StrPaths


# ========================== #
#           Header           #
# ========================== #


@dataclass(frozen=True, slots=True)
class PhaseBinHeader(KoalaBinHeader):
    """The fixed-size header of a Lyncée Tec Koala float32 .bin phase image.

    Extends the shared `KoalaBinHeader` with the phase reading of the
    trailing bytes: a positive `height_scale` (the phase-to-height factor,
    m per rad) and a `PhaseUnit`.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        pixel_size: Physical size of one (square) pixel, in m.
        height_scale: Height represented by one rad of phase, in m;
            the phase-to-height conversion factor.
        unit: Physical unit of the stored phase values.
        version: Format version. Fixed at 1.
        endian: Byte-order flag. Fixed at 0 (little-endian).
    """

    height_scale: float
    unit: PhaseUnit

    def __post_init__(self) -> None:
        """Validate the fields."""
        super().__post_init__()

        if self.height_scale <= 0:
            msg = f"height_scale must be positive (got {self.height_scale})"
            raise ValueError(msg)

        if self.unit not in (PhaseUnit.UNKNOWN, PhaseUnit.RADIANS, PhaseUnit.METERS):
            msg = f"unit must be one of UNKNOWN, RADIANS, METERS (got {self.unit!r})"
            raise ValueError(msg)

    @property
    def height_scale_nm(self) -> float:
        """Height scale (height per rad) in nm."""
        return self.height_scale * 1e9

    def to_dtype(self) -> NDArray[np.void]:
        """Serialize to a 1-element `PhaseBinHeader.DTYPE` record array."""
        record = self.base_record()
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
        data = read_bin_pixels(fb, header)

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
            # height per rad = wavelength / (2*pi * refractive_delta)
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
    pair (then height per rad = wavelength / (2*pi * refractive_delta)).
    Exactly one of the two forms must be given.

    The file is written atomically: content is staged to a temp file in the
    destination's directory and moved into place on success, so a failed
    write never leaves a partial or clobbered file.

    Args:
        path: The .bin file to write.
        data: The phase image to save, of shape (H, W).
        pixel_size: Physical size of one (square) pixel, in m.
        height_scale: Height represented by one rad of phase, in m.
            Mutually exclusive with `wavelength`/`refractive_delta`.
        wavelength: Illumination wavelength, in m. Requires
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
    )
    write_bin(path, header, data, overwrite=overwrite)


# ========================== #
#          Sequence          #
# ========================== #


class PhaseBinFolder(
    SequentialFileFolderSequence[NDArray[np.float32]],
    PhaseSequence[Path],
    FrameShapedMixin,
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

    FILE_STEM: ClassVar[str] = "phase"
    FILE_EXT: ClassVar[str] = "bin"
    LEVELS: ClassVar[tuple[str, ...]] = ("names", "headers", "data")
    DEFAULT_LEVEL: ClassVar[str] = "headers"

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

        # Fail fast: reject an unreachable target unit at construction, not
        # lazily on each get_item. The empty array makes it a pure pair check.
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

    @property
    @override
    def frame_shape(self) -> tuple[int, int]:
        """The (height, width) of each image, from the shared header."""
        return self._header.shape

    @override
    def load_file(self, path: Path) -> NDArray[np.float32]:
        """Load the image at `path`, converted to `target_unit`."""
        return convert_phase_unit(
            load_phase_bin(path),
            source=self._header.unit,
            target=self._target_unit,
            height_scale=self._header.height_scale,
        )

    @override
    def _validate_content(self, path: Path, *, level: str) -> None:
        """Check `path`'s header matches the reference; at "data", decode too.

        The "headers" and "data" levels both require every file to share the
        first file's `header`; "data" additionally decodes the pixels.
        """
        if read_phase_bin_header(path) != self.header:
            msg = f"header of {path.name} differs from the first file"
            raise ValueError(msg)

        if level == "data":
            load_phase_bin(path, on_nonfinite="raise")


class PhaseBinList(FileListSequence[NDArray[np.float32], Path], PhaseSequence[Path]):
    """A phase sequence over an explicit, arbitrary list of `.bin` files.

    Unlike `PhaseBinFolder`, imposes no naming, contiguity, single-folder,
    or shared-header constraint: the files may live anywhere and each is read
    independently, its own header driving any per-file unit conversion. The
    images may therefore differ in shape, so this is a plain `PhaseSequence`
    (no `frame_shape`). Each item is the decoded float32 image (optionally
    converted to `target_unit`) and its metadata is the source path.

    Args:
        files: The `.bin` files to expose, in the given order.
        target_unit: Unit to return images in, applied per file via that
            file's own `height_scale`. Defaults to None, which keeps each
            file's stored unit. A file whose stored unit cannot reach
            `target_unit` raises `ValueError` when that item is accessed.
    """

    def __init__(
        self, files: StrPaths, *, target_unit: PhaseUnit | None = None
    ) -> None:
        super().__init__(files)
        self._target_unit = target_unit

    @property
    def target_unit(self) -> PhaseUnit | None:
        """The unit images are converted to on load, or None to keep each file's."""
        return self._target_unit

    @override
    def get_meta(self, index: int) -> Path:
        """Return the source path of the file at `index`."""
        return self.get_file(index)

    @override
    def load_file(self, path: Path) -> NDArray[np.float32]:
        """Load the image at `path`, converted to `target_unit` if one is set."""
        image, header = load_phase_bin(path, return_header=True)
        target = self._target_unit if self._target_unit is not None else header.unit
        return convert_phase_unit(
            image, source=header.unit, target=target, height_scale=header.height_scale
        )
