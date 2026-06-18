from __future__ import annotations

__all__ = (
    "load_phase",
    "phase_folder",
    "phase_list",
    "read_phase_header",
    "save_phase",
)

import warnings
from typing import TYPE_CHECKING, overload

from iivs.dhm.data.common import (
    FLOAT_FORMATS,
    detect_numbered_format,
    file_extension,
    unsupported_extension,
)
from iivs.dhm.data.phase.bin import (
    PhaseBinFolder,
    PhaseBinList,
    load_phase_bin,
    read_phase_bin_header,
    save_phase_bin,
)
from iivs.dhm.data.phase.npy import PhaseNpyFolder, load_phase_npy, save_phase_npy
from iivs.dhm.data.phase.txt import (
    PhaseTxtFolder,
    PhaseTxtList,
    load_phase_txt,
    read_phase_txt_header,
    save_phase_txt,
)
from iivs.dhm.data.phase.unit import PhaseUnit, resolve_height_scale

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from kaparoo.filesystem.types import StrPath, StrPaths
    from numpy.typing import NDArray

    from iivs.dhm.data.common import FloatFormat, OnNonFinite, ValidationLevel
    from iivs.dhm.data.phase.base import PhaseFileFolder, PhaseFileList
    from iivs.dhm.data.phase.bin import PhaseBinHeader


def load_phase(
    path: StrPath,
    *,
    on_nonfinite: OnNonFinite = "ignore",
) -> NDArray[np.float32]:
    """Load a float32 phase image, picking the reader by `path`'s extension.

    Dispatches `.bin` / `.txt` / `.npy` to `load_phase_bin` / `load_phase_txt` /
    `load_phase_npy`. Returns the **image only** (uniform across formats, since
    `.npy` is header-less); for the header too, use the per-format loader's
    `return_header` (`.bin` / `.txt`) or `read_phase_header`.

    Raises:
        ValueError: If `path`'s extension is not bin, txt, or npy (plus the
            per-format errors of the chosen reader).
    """
    ext = file_extension(path)
    if ext == "bin":
        return load_phase_bin(path, on_nonfinite=on_nonfinite)
    if ext == "txt":
        return load_phase_txt(path, on_nonfinite=on_nonfinite)
    if ext == "npy":
        return load_phase_npy(path, on_nonfinite=on_nonfinite)
    raise unsupported_extension(ext, kind="phase", formats=FLOAT_FORMATS)


def read_phase_header(path: StrPath) -> PhaseBinHeader:
    """Read just the header of a `.bin` / `.txt` phase file, picking by extension.

    Dispatches to `read_phase_bin_header` / `read_phase_txt_header`. `.npy` is
    excluded: it carries no header (supply the metadata via `PhaseNpyFolder`).

    Raises:
        ValueError: If `path` is `.npy` (header-less) or its extension is not
            bin or txt.
    """
    ext = file_extension(path)
    if ext == "bin":
        return read_phase_bin_header(path)
    if ext == "txt":
        return read_phase_txt_header(path)
    if ext == "npy":
        msg = "`.npy` is header-less; supply metadata via PhaseNpyFolder"
        raise ValueError(msg)
    raise unsupported_extension(ext, kind="phase", formats=FLOAT_FORMATS)


@overload
def save_phase(
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
def save_phase(
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


@overload
def save_phase(
    path: StrPath,
    data: NDArray[np.float32],
    *,
    overwrite: bool = ...,
    on_nonfinite: OnNonFinite = ...,
) -> None: ...


def save_phase(
    path: StrPath,
    data: NDArray[np.float32],
    *,
    pixel_size: float | None = None,
    height_scale: float | None = None,
    wavelength: float | None = None,
    refractive_delta: float | None = None,
    unit: PhaseUnit = PhaseUnit.RADIANS,
    overwrite: bool = False,
    on_nonfinite: OnNonFinite = "warn",
) -> None:
    """Save a 2D float32 phase image, picking the writer by `path`'s extension.

    Dispatches `.bin` / `.txt` to `save_phase_bin` / `save_phase_txt` (both need
    `pixel_size` and a phase-to-height scale -- `height_scale`, or `wavelength`
    + `refractive_delta`), and `.npy` to the header-less `save_phase_npy`. For
    `.npy` the metadata args do not apply; passing any emits a warning and they
    are dropped.

    Raises:
        ValueError: If `path`'s extension is not bin, txt, or npy; for `.bin` /
            `.txt`, if `pixel_size` is missing or neither/both scale forms are
            given.
    """
    ext = file_extension(path)
    if ext in ("bin", "txt"):
        if pixel_size is None:
            msg = "pixel_size is required for .bin / .txt"
            raise ValueError(msg)
        # Resolve the scale here (rejecting neither/both forms) so the call
        # matches a single delegate overload regardless of which form was given.
        resolved = resolve_height_scale(height_scale, wavelength, refractive_delta)
        save = save_phase_bin if ext == "bin" else save_phase_txt
        save(
            path,
            data,
            pixel_size=pixel_size,
            height_scale=resolved,
            unit=unit,
            overwrite=overwrite,
            on_nonfinite=on_nonfinite,
        )
        return
    if ext == "npy":
        if (
            pixel_size is not None
            or height_scale is not None
            or wavelength is not None
            or refractive_delta is not None
            or unit is not PhaseUnit.RADIANS
        ):
            msg = "`.npy` is header-less; ignoring pixel_size / unit / height scale"
            warnings.warn(msg, stacklevel=2)
        save_phase_npy(path, data, overwrite=overwrite, on_nonfinite=on_nonfinite)
        return
    raise unsupported_extension(ext, kind="phase", formats=FLOAT_FORMATS)


def phase_list(
    files: StrPaths,
    *,
    target_unit: PhaseUnit | None = None,
) -> PhaseFileList:
    """Build a phase file list, picking the class by the files' shared extension.

    Dispatches `.bin` / `.txt` to `PhaseBinList` / `PhaseTxtList`. All `files`
    must share one extension. `.npy` has no list form (each file is header-less,
    with no shared acquisition header) -- use `PhaseNpyFolder`.

    Raises:
        ValueError: If `files` is empty, mixes extensions, is `.npy`, or shares
            an extension that is not bin or txt.
    """
    files = list(files)
    if not files:
        msg = "files must be non-empty"
        raise ValueError(msg)

    exts = {file_extension(f) for f in files}
    if len(exts) != 1:
        msg = f"all files must share one extension (got {sorted(exts)})"
        raise ValueError(msg)

    ext = exts.pop()
    if ext == "bin":
        return PhaseBinList(files, target_unit=target_unit)
    if ext == "txt":
        return PhaseTxtList(files, target_unit=target_unit)
    if ext == "npy":
        msg = "no .npy phase list; use PhaseNpyFolder (npy is header-less)"
        raise ValueError(msg)
    raise unsupported_extension(ext, kind="phase", formats=FLOAT_FORMATS)


@overload
def phase_folder(
    root: StrPath,
    *,
    pixel_size: float,
    unit: PhaseUnit,
    height_scale: float,
    target_unit: PhaseUnit | None = ...,
    validate: ValidationLevel | None = ...,
    prefer: FloatFormat | Sequence[FloatFormat] | None = ...,
) -> PhaseFileFolder: ...


@overload
def phase_folder(
    root: StrPath,
    *,
    pixel_size: float,
    unit: PhaseUnit,
    wavelength: float,
    refractive_delta: float,
    target_unit: PhaseUnit | None = ...,
    validate: ValidationLevel | None = ...,
    prefer: FloatFormat | Sequence[FloatFormat] | None = ...,
) -> PhaseFileFolder: ...


@overload
def phase_folder(
    root: StrPath,
    *,
    target_unit: PhaseUnit | None = ...,
    validate: ValidationLevel | None = ...,
    prefer: FloatFormat | Sequence[FloatFormat] | None = ...,
) -> PhaseFileFolder: ...


def phase_folder(
    root: StrPath,
    *,
    pixel_size: float | None = None,
    unit: PhaseUnit | None = None,
    height_scale: float | None = None,
    wavelength: float | None = None,
    refractive_delta: float | None = None,
    target_unit: PhaseUnit | None = None,
    validate: ValidationLevel | None = "headers",
    prefer: FloatFormat | Sequence[FloatFormat] | None = None,
) -> PhaseFileFolder:
    """Open a numbered phase folder, picking the class by the format it holds.

    Discovers the `{index:05d}_phase.<ext>` files under `root` (via
    `data.common.detect_numbered_format`) and dispatches to `PhaseBinFolder` /
    `PhaseTxtFolder` / `PhaseNpyFolder`. The `.bin` / `.txt` folders read their
    metadata from the files, so the `pixel_size` / `unit` / scale args must be
    omitted for them; the header-less `.npy` folder instead **requires**
    `pixel_size` and `unit` (and a scale form).

    Args:
        root: The folder to scan.
        pixel_size, unit, height_scale, wavelength, refractive_delta: The
            metadata for a `.npy` folder (omit for `.bin` / `.txt`).
        target_unit: Unit to return loaded images in (None keeps the stored).
        validate: Validation level at construction, or None to skip.
        prefer: How to resolve a `root` that holds more than one format -- `None`
            (default) raises, while a format or a priority sequence picks the
            first present one (e.g. `prefer=("bin", "txt")`).

    Raises:
        FileNotFoundError: If `root` holds no `NNNNN_phase.{bin,txt,npy}` files.
        ValueError: If `root` mixes formats and `prefer` does not resolve it, if
            metadata args are given for a `.bin` / `.txt` folder, or if a `.npy`
            folder is missing `pixel_size` / `unit`.
    """
    ext = detect_numbered_format(
        root, stem="phase", formats=FLOAT_FORMATS, prefer=prefer
    )
    if ext in ("bin", "txt"):
        if any(
            arg is not None
            for arg in (pixel_size, unit, height_scale, wavelength, refractive_delta)
        ):
            msg = f".{ext} folders read metadata from the files; drop the metadata args"
            raise ValueError(msg)
        folder = PhaseBinFolder if ext == "bin" else PhaseTxtFolder
        return folder(root, target_unit=target_unit, validate=validate)

    if pixel_size is None or unit is None:
        msg = "`.npy` folders need pixel_size and unit (npy is header-less)"
        raise ValueError(msg)
    return PhaseNpyFolder(
        root,
        pixel_size=pixel_size,
        unit=unit,
        height_scale=height_scale,
        wavelength=wavelength,
        refractive_delta=refractive_delta,
        target_unit=target_unit,
        validate=validate,
    )
