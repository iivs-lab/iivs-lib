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
from typing import TYPE_CHECKING, ClassVar, overload, override

from kaparoo.filesystem import ensure_file_extension

from iivs.common.data import validate_float32_array
from iivs.dhm.data.koala import KoalaBinHeader, load_bin, write_bin
from iivs.dhm.data.phase.base import PhaseFileFolder, PhaseFileList
from iivs.dhm.data.phase.unit import PhaseUnit, convert_phase_unit, resolve_height_scale

if TYPE_CHECKING:
    from typing import Literal, Self

    import numpy as np
    from kaparoo.filesystem.types import StrPath
    from numpy.typing import NDArray

    from iivs.common.data import OnNonFinite


# ========================== #
#           Header           #
# ========================== #


@dataclass(frozen=True, slots=True)
class PhaseBinHeader(KoalaBinHeader):
    """The fixed-size header of a Lyncée Tec Koala float32 .bin phase image.

    Extends the shared `KoalaBinHeader` with the phase reading of the trailing bytes: a
    positive `height_scale` (the phase-to-height factor, m per rad) and a `PhaseUnit`.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        pixel_size: Physical size of one (square) pixel, in m.
        height_scale: Height represented by one rad of phase, in m; the phase-to-height
            conversion factor.
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

    Reads just the fixed-size header, so it stays cheap when curating many files by
    metadata (shape, field of view) without decoding the images.

    Raises:
        FileNotFoundError: If `path` does not exist.
        NotAFileError: If `path` exists but is not a regular file.
        ValueError: If the file is too small, declares an unsupported header size, or
            has invalid header fields.
    """
    return PhaseBinHeader.from_file(path)


@overload
def load_phase_bin(
    path: StrPath,
    *,
    return_header: Literal[False] = False,
    on_nonfinite: OnNonFinite = ...,
) -> NDArray[np.float32]: ...


@overload
def load_phase_bin(
    path: StrPath,
    *,
    return_header: Literal[True],
    on_nonfinite: OnNonFinite = ...,
) -> tuple[NDArray[np.float32], PhaseBinHeader]: ...


@overload
def load_phase_bin(
    path: StrPath,
    *,
    return_header: bool,
    on_nonfinite: OnNonFinite = ...,
) -> NDArray[np.float32] | tuple[NDArray[np.float32], PhaseBinHeader]: ...


def load_phase_bin(
    path: StrPath,
    *,
    return_header: bool = False,
    on_nonfinite: OnNonFinite = "ignore",
) -> NDArray[np.float32] | tuple[NDArray[np.float32], PhaseBinHeader]:
    """Load a Lyncée Tec Koala float32 .bin phase image, and optionally its header.

    Args:
        path: The .bin file to read.
        return_header: Whether to also return the parsed `PhaseBinHeader`. Defaults to
            False.
        on_nonfinite: How to handle non-finite values (NaN, +inf, -inf) in the decoded
            data: "ignore" (default) accepts silently, "warn" emits a RuntimeWarning,
            "raise" raises a ValueError (useful to reject corrupted files). Defaults to
            "ignore", since a structurally valid file's contents are accepted by
            default.

    Returns:
        The phase image as a 2D float32 array, or an (image, header) tuple
        when `return_header` is True.

    Raises:
        FileNotFoundError: If `path` does not exist.
        NotAFileError: If `path` exists but is not a regular file.
        ValueError: If the file is too small, declares an unsupported header size, has
            invalid header fields, holds the wrong number of pixels, or holds non-finite
            values while `on_nonfinite` is "raise".
    """
    data, header = load_bin(path, PhaseBinHeader, on_nonfinite=on_nonfinite)
    return (data, header) if return_header else data


# ========================== #
#          Writing           #
# ========================== #


def _to_storable_unit(
    data: NDArray[np.float32], unit: PhaseUnit, height_scale: float
) -> tuple[NDArray[np.float32], PhaseUnit]:
    """Coerce a (data, unit) pair into a unit the file can store.

    The on-disk format holds only UNKNOWN/RADIANS/METERS, so the code-only NANOMETERS
    unit is converted to METERS; every other unit passes through unchanged.
    """
    if unit is PhaseUnit.NANOMETERS:
        data = convert_phase_unit(
            data, source=unit, target=PhaseUnit.METERS, height_scale=height_scale
        )
        return data, PhaseUnit.METERS
    return data, unit


def _prepare_phase_write(
    data: NDArray[np.float32],
    *,
    pixel_size: float,
    height_scale: float | None,
    wavelength: float | None,
    refractive_delta: float | None,
    unit: PhaseUnit,
    on_nonfinite: OnNonFinite,
) -> tuple[NDArray[np.float32], PhaseBinHeader]:
    """Resolve and validate what a phase file needs, whatever encodes it.

    Everything the `.bin` and `.txt` writers must agree on: the scale form is resolved,
    the array checked as one image (a save takes a single frame, unlike the loaders), the
    code-only NANOMETERS coerced to a storable METERS (so the values and the recorded
    `unit` always match), and UNKNOWN warned about, since a file saved with it cannot be
    interpreted physically later.

    The header is a `PhaseBinHeader` for both: `.txt` serializes the same fields as text.

    Raises:
        ValueError: If neither or both scale forms are given, `data` is not a single 2D
            float32 image, or it holds non-finite values while `on_nonfinite` is
            `"raise"`.
    """
    height_scale = resolve_height_scale(height_scale, wavelength, refractive_delta)
    data = validate_float32_array(data, on_nonfinite=on_nonfinite, allow_stack=False)
    data, unit = _to_storable_unit(data, unit, height_scale)

    if unit is PhaseUnit.UNKNOWN:
        msg = "saving with unit=UNKNOWN; physical interpretation is undefined"
        warnings.warn(msg, stacklevel=3)  # 3: the public saver is the caller's frame

    header = PhaseBinHeader(
        width=int(data.shape[1]),
        height=int(data.shape[0]),
        pixel_size=pixel_size,
        height_scale=height_scale,
        unit=unit,
    )
    return data, header


@overload
def save_phase_bin(
    path: StrPath,
    data: NDArray[np.float32],
    *,
    pixel_size: float,
    height_scale: float,
    unit: PhaseUnit = ...,
    overwrite: bool = ...,
    on_nonfinite: OnNonFinite = ...,
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
    on_nonfinite: OnNonFinite = ...,
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
    on_nonfinite: OnNonFinite = "warn",
) -> None:
    """Save a 2D float32 phase image as a Lyncée Tec Koala .bin file.

    The phase-to-height scale is given either directly as `height_scale`, or as a
    `wavelength` and refractive-index difference `refractive_delta` pair (then height
    per rad = wavelength / (2*pi * refractive_delta)). Exactly one of the two forms must
    be given.

    Written atomically, so a failed write never leaves a partial or clobbered file.

    Args:
        path: The .bin file to write.
        data: The phase image to save, of shape (H, W).
        pixel_size: Physical size of one (square) pixel, in m.
        height_scale: Height represented by one rad of phase, in m. Mutually exclusive
            with `wavelength`/`refractive_delta`.
        wavelength: Illumination wavelength, in m. Requires `refractive_delta`.
        refractive_delta: Refractive-index difference n1 - n2 (the plain difference, not
            the normalized contrast). Requires `wavelength`.
        unit: Physical unit of `data`. Defaults to RADIANS. NANOMETERS is converted to
            METERS before storing (the file cannot store it); UNKNOWN is stored as-is
            but emits a warning.
        overwrite: Whether to replace `path` if it already exists. Defaults to False.
        on_nonfinite: How to handle non-finite values (NaN, +inf, -inf) in `data`:
            "ignore" accepts silently, "warn" (default) emits a RuntimeWarning, "raise"
            rejects with a ValueError.

    Raises:
        ValueError: If `path` has a non-`.bin` extension, neither or both scale forms
            are given, `data` is not a single 2D float32 image, or it holds non-finite
            values while `on_nonfinite` is "raise".
        FileExistsError: If `path` exists and `overwrite` is False.
        FileNotFoundError: If the parent directory of `path` does not exist.
    """
    path = ensure_file_extension(path, "bin", add=True)
    data, header = _prepare_phase_write(
        data,
        pixel_size=pixel_size,
        height_scale=height_scale,
        wavelength=wavelength,
        refractive_delta=refractive_delta,
        unit=unit,
        on_nonfinite=on_nonfinite,
    )
    write_bin(path, header, data, overwrite=overwrite)


# ========================== #
#          Sequence          #
# ========================== #


class PhaseBinList(PhaseFileList):
    """A phase sequence over an explicit, arbitrary list of `.bin` files.

    The general case: no naming, contiguity, single-folder, or shared-header constraint;
    each file is read independently with per-file unit conversion. `PhaseBinFolder` is
    the auto-discovered, same-shape special case of this.

    Args:
        files: The `.bin` files to expose, in the given order.
        target_unit: Unit to return images in, applied per file via that file's own
            `height_scale`. Defaults to None, which keeps each file's stored unit. A
            file whose stored unit cannot reach `target_unit` raises `ValueError` when
            that item is accessed.

    Raises:
        ValueError: If any path does not have a `.bin` extension.
    """

    FILE_EXT: ClassVar[str] = "bin"

    @override
    def _read_header(self, path: StrPath) -> PhaseBinHeader:
        """Read the `.bin` header."""
        return read_phase_bin_header(path)

    @override
    def _decode(
        self,
        path: StrPath,
        *,
        on_nonfinite: OnNonFinite = "ignore",
    ) -> tuple[NDArray[np.float32], PhaseBinHeader]:
        """Decode the `.bin` image and its header."""
        return load_phase_bin(path, return_header=True, on_nonfinite=on_nonfinite)


class PhaseBinFolder(PhaseFileFolder, PhaseBinList):
    """An ordered sequence of Lyncée Tec Koala `.bin` phase images in a folder.

    The auto-discovered, same-shape special case of `PhaseBinList`: lists the direct
    children matching `{index:05d}_phase.bin` (exactly five digits, case-sensitive), in
    index order, sharing one acquisition `header` read from the first file.

    Args:
        root: The folder to scan. Must exist, be a directory, and contain at least one
            matching file.
        target_unit: Unit to return loaded images in (None keeps the stored).
        validate: Run `validate` to this level at construction, or None to skip.
            Defaults to "headers".

    Raises:
        DirectoryNotFoundError: If `root` does not exist.
        NotADirectoryError: If `root` exists but is not a directory.
        FileNotFoundError: If no `NNNNN_phase.bin` files are found in `root`.
        ValueError: If `target_unit` cannot be converted from the stored unit, or if
            `validate` is set and the sequence fails validation.
    """
