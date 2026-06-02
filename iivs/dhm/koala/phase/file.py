from __future__ import annotations

__all__ = ("convert_phase", "load_bin", "read_header", "save_bin", "validate_phase")

import warnings
from typing import TYPE_CHECKING, overload

import numpy as np
from kaparoo.filesystem import StagedFile, ensure_file_exists

from iivs.dhm.koala.phase.header import PhaseBinHeader, PhaseUnit

if TYPE_CHECKING:
    from typing import IO, Literal

    from kaparoo.filesystem.types import StrPath
    from numpy.typing import NDArray


# ========================== #
#         Validation         #
# ========================== #


def validate_phase(
    data: NDArray[np.float32],
    *,
    on_nonfinite: Literal["ignore", "warn", "raise"] = "warn",
) -> NDArray[np.float32]:
    """Validate a float32 phase image or stack and return it.

    The last two axes are the image height and width; any number of
    leading axes are allowed, so both a single 2-D image and a
    higher-dimensional stack are accepted. `data` is never modified.

    Args:
        data: The phase image or stack to validate, of shape (..., H, W).
        on_nonfinite: How to handle non-finite values (NaN, +inf, -inf):
            "ignore" accepts them silently, "warn" (default) accepts them
            but emits a RuntimeWarning, "raise" raises a ValueError.

    Returns:
        The validated `data`, unchanged.

    Raises:
        ValueError: If `data` is not a float32 array with at least two
            dimensions.
        ValueError: If `data` contains non-finite values and `on_nonfinite`
            is "raise".
    """
    if data.ndim < 2:
        msg = f"data must be at least 2-dimensional (got {data.ndim})"
        raise ValueError(msg)

    if data.dtype != np.float32:
        msg = f"data must be float32 (got {data.dtype})"
        raise ValueError(msg)

    if on_nonfinite != "ignore" and not np.all(np.isfinite(data)):
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


# ========================== #
#         Conversion         #
# ========================== #


_NM_PER_M = 1e9  # nanometers per meter


def convert_phase(
    data: NDArray[np.float32],
    *,
    from_unit: PhaseUnit,
    to_unit: PhaseUnit,
    height_scale: float,
) -> NDArray[np.float32]:
    """Convert phase image `data` from `from_unit` to `to_unit`.

    RADIANS <-> METERS uses `height_scale` (meters per radian); METERS <->
    NANOMETERS uses the fixed 1e9 nm/m. Returns `data` unchanged when the
    units already match.

    Args:
        data: The phase or height image to convert.
        from_unit: The unit `data` is currently in.
        to_unit: The unit to convert to.
        height_scale: Height represented by one radian of phase, in meters.

    Returns:
        The converted image, or `data` itself when `from_unit == to_unit`.

    Raises:
        ValueError: If the conversion is undefined (e.g. an UNKNOWN unit).
    """
    if from_unit is to_unit:
        return data
    # `scale` is defined in ascending unit order (RADIANS < METERS <
    # NANOMETERS); converting the other way uses its reciprocal.
    match sorted((from_unit, to_unit)):
        case [PhaseUnit.RADIANS, PhaseUnit.METERS]:
            scale = height_scale
        case [PhaseUnit.METERS, PhaseUnit.NANOMETERS]:
            scale = _NM_PER_M
        case [PhaseUnit.RADIANS, PhaseUnit.NANOMETERS]:
            scale = height_scale * _NM_PER_M
        case _:
            msg = f"cannot convert phase from {from_unit.name} to {to_unit.name}"
            raise ValueError(msg)
    if from_unit > to_unit:
        scale = 1.0 / scale
    return (data * scale).astype(np.float32, copy=False)


# ========================== #
#          Reading           #
# ========================== #


def read_header(path: StrPath) -> PhaseBinHeader:
    """Read only the header of a Lyncée Tec Koala .bin file, without the pixels.

    A thin wrapper over `PhaseBinHeader.from_file`; reads just the
    fixed-size header, so it stays cheap when curating many files by
    metadata (shape, field of view) without decoding the images.

    Returns:
        The parsed header.

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
    expected = header.pixel_count * 4  # float32 is 4 bytes
    if len(raw) != expected:
        msg = f"pixel count must be {header.pixel_count} ({expected} bytes), got {len(raw)}"
        raise ValueError(msg)
    pixels = np.frombuffer(raw, dtype="<f4")
    return pixels.reshape(header.shape).astype(np.float32, copy=True)


@overload
def load_bin(
    path: StrPath,
    *,
    return_header: Literal[False] = ...,
    on_nonfinite: Literal["ignore", "warn", "raise"] = ...,
) -> NDArray[np.float32]: ...


@overload
def load_bin(
    path: StrPath,
    *,
    return_header: Literal[True],
    on_nonfinite: Literal["ignore", "warn", "raise"] = ...,
) -> tuple[NDArray[np.float32], PhaseBinHeader]: ...


def load_bin(
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
            msg = (
                "exactly one of height_scale, or both wavelength and "
                "refractive_delta, must be given"
            )
            raise ValueError(msg)


@overload
def save_bin(
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
def save_bin(
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


def save_bin(
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
            converted to METERS before storing (the file cannot store it).
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

    # Reject any non-2D input before validate_phase scans the data.
    if data.ndim != 2:
        msg = f"data must be a single 2D image (got shape {data.shape})"
        raise ValueError(msg)
    data = validate_phase(data, on_nonfinite=on_nonfinite)

    # NANOMETERS is code-only; store it as METERS.
    if unit is PhaseUnit.NANOMETERS:
        data = convert_phase(
            data, from_unit=unit, to_unit=PhaseUnit.METERS, height_scale=height_scale
        )
        unit = PhaseUnit.METERS

    header = PhaseBinHeader(
        width=int(data.shape[1]),
        height=int(data.shape[0]),
        pixel_size=pixel_size,
        height_scale=height_scale,
        unit=unit,
    ).to_dtype()
    pixels = np.ascontiguousarray(data, dtype="<f4")

    with StagedFile(path, binary=True, overwrite=overwrite) as staged:
        header.tofile(staged.file)
        pixels.tofile(staged.file)
