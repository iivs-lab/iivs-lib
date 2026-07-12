from __future__ import annotations

__all__ = (
    "convert_phase_folder",
    "convert_phase_list",
    "load_phase",
    "phase_folder",
    "phase_list",
    "read_phase_header",
    "save_phase",
    "save_phase_folder",
)

import warnings
from functools import partial
from typing import TYPE_CHECKING, overload

from kaparoo.filesystem import (
    StagedDirectory,
    UnsupportedExtensionError,
    file_extension,
)
from kaparoo.utils import ensure_one_of, replace_if_none

from iivs.dhm.data.koala import FLOAT_FORMATS, detect_koala_format, koala_frame_name
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
    from collections.abc import Iterable, Sequence
    from typing import Literal

    import numpy as np
    from kaparoo.filesystem.types import StrPath, StrPaths
    from numpy.typing import NDArray

    from iivs.common.data import OnNonFinite
    from iivs.dhm.data.koala import FloatFormat, ValidationLevel
    from iivs.dhm.data.phase.base import PhaseFileFolder, PhaseFileList
    from iivs.dhm.data.phase.bin import PhaseBinHeader


# ========================== #
#        Single file         #
# ========================== #


@overload
def load_phase(
    path: StrPath,
    *,
    return_header: Literal[False] = False,
    on_nonfinite: OnNonFinite = ...,
) -> NDArray[np.float32]: ...


@overload
def load_phase(
    path: StrPath,
    *,
    return_header: Literal[True],
    on_nonfinite: OnNonFinite = ...,
) -> tuple[NDArray[np.float32], PhaseBinHeader | None]: ...


@overload
def load_phase(
    path: StrPath,
    *,
    return_header: bool,
    on_nonfinite: OnNonFinite = ...,
) -> NDArray[np.float32] | tuple[NDArray[np.float32], PhaseBinHeader | None]: ...


def load_phase(
    path: StrPath,
    *,
    return_header: bool = False,
    on_nonfinite: OnNonFinite = "ignore",
) -> NDArray[np.float32] | tuple[NDArray[np.float32], PhaseBinHeader | None]:
    """Load a float32 phase image, picking the reader by `path`'s extension.

    Dispatches `.bin` / `.txt` / `.npy` to `load_phase_bin` / `load_phase_txt` /
    `load_phase_npy`. With `return_header` it also returns the parsed header (`None` for
    `.npy`, which is header-less; its `pixel_size`, `unit`, and `height_scale` live only
    on `PhaseNpyFolder`, not in the file). Contrast `read_phase_header`, whose sole job
    *is* the header and so raises on `.npy`; here the header is an optional extra, so an
    absent one is `None`, not an error.

    Args:
        path: The `.bin` / `.txt` / `.npy` file to read.
        return_header: Whether to also return the parsed header (`None` for the
            header-less `.npy`). Defaults to False.
        on_nonfinite: How to handle non-finite values (NaN, +inf, -inf) in the decoded
            data: "ignore" (default) accepts silently, "warn" emits a RuntimeWarning,
            "raise" rejects with a ValueError.

    Returns:
        The phase image as a 2D float32 array, or an `(image, header)` tuple
        when `return_header` is True (with `header` `None` for `.npy`).

    Raises:
        ValueError: If `path`'s extension is not bin, txt, or npy (plus the per-format
            errors of the chosen reader).
    """
    ext = file_extension(path)
    if ext == "bin":
        return load_phase_bin(
            path, return_header=return_header, on_nonfinite=on_nonfinite
        )
    if ext == "txt":
        return load_phase_txt(
            path, return_header=return_header, on_nonfinite=on_nonfinite
        )
    if ext == "npy":
        data = load_phase_npy(path, on_nonfinite=on_nonfinite)
        return (data, None) if return_header else data
    raise UnsupportedExtensionError(ext, FLOAT_FORMATS, kind="phase")


def read_phase_header(path: StrPath) -> PhaseBinHeader:
    """Read just the header of a `.bin` / `.txt` phase file, picking by extension.

    Dispatches to `read_phase_bin_header` / `read_phase_txt_header`. `.npy` is excluded:
    it carries no header (supply the metadata via `PhaseNpyFolder`).

    Raises:
        ValueError: If `path` is `.npy` (header-less) or its extension is not bin or
            txt.
    """
    ext = file_extension(path)
    if ext == "bin":
        return read_phase_bin_header(path)
    if ext == "txt":
        return read_phase_txt_header(path)
    if ext == "npy":
        msg = "`.npy` is header-less; supply metadata via PhaseNpyFolder"
        raise ValueError(msg)
    raise UnsupportedExtensionError(ext, FLOAT_FORMATS, kind="phase")


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
    `pixel_size` and a phase-to-height scale, given as `height_scale` or as `wavelength`
    + `refractive_delta`), and `.npy` to the header-less `save_phase_npy`. For `.npy`
    the metadata args do not apply; passing any emits a warning and they are dropped.

    Raises:
        ValueError: If `path`'s extension is not bin, txt, or npy; for `.bin` / `.txt`,
            if `pixel_size` is missing or neither/both scale forms are given.
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
    raise UnsupportedExtensionError(ext, FLOAT_FORMATS, kind="phase")


# ========================== #
#       Open a sequence      #
# ========================== #


def phase_list(
    files: StrPaths,
    *,
    target_unit: PhaseUnit | None = None,
) -> PhaseFileList:
    """Build a phase file list, picking the class by the files' shared extension.

    Dispatches `.bin` / `.txt` to `PhaseBinList` / `PhaseTxtList`. All `files` must
    share one extension. `.npy` has no list form (each file is header-less, with no
    shared acquisition header); use `PhaseNpyFolder`.

    Raises:
        ValueError: If `files` is empty, mixes extensions, is `.npy`, or shares an
            extension that is not bin or txt.
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
    raise UnsupportedExtensionError(ext, FLOAT_FORMATS, kind="phase")


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
    `detect_koala_format`) and dispatches to `PhaseBinFolder` /
    `PhaseTxtFolder` / `PhaseNpyFolder`. The `.bin` / `.txt` folders read their metadata
    from the files, so the `pixel_size` / `unit` / scale args must be omitted for them;
    the header-less `.npy` folder instead **requires** `pixel_size` and `unit` (and a
    scale form).

    Args:
        root: The folder to scan.
        pixel_size, unit, height_scale, wavelength, refractive_delta: The metadata for a
            `.npy` folder (omit for `.bin` / `.txt`).
        target_unit: Unit to return loaded images in (None keeps the stored).
        validate: Validation level at construction, or None to skip.
        prefer: How to resolve a `root` that holds more than one format. `None`
            (default) raises, while a format or a priority sequence picks the first
            present one (e.g. `prefer=("bin", "txt")`).

    Raises:
        FileNotFoundError: If `root` holds no `NNNNN_phase.{bin,txt,npy}` files.
        ValueError: If `root` mixes formats and `prefer` does not resolve it, if
            metadata args are given for a `.bin` / `.txt` folder, or if a `.npy` folder
            is missing `pixel_size` / `unit`.
    """
    ext = detect_koala_format(root, stem="phase", formats=FLOAT_FORMATS, prefer=prefer)
    if ext in ("bin", "txt"):
        args = [pixel_size, unit, height_scale, wavelength, refractive_delta]
        if any(arg is not None for arg in args):
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


# ========================== #
#      Write a sequence      #
# ========================== #


@overload
def save_phase_folder(
    root: StrPath,
    images: Iterable[NDArray[np.float32]],
    *,
    ext: FloatFormat,
    pixel_size: float,
    height_scale: float,
    unit: PhaseUnit = ...,
    stem: str = ...,
    overwrite: bool = ...,
) -> None: ...


@overload
def save_phase_folder(
    root: StrPath,
    images: Iterable[NDArray[np.float32]],
    *,
    ext: FloatFormat,
    pixel_size: float,
    wavelength: float,
    refractive_delta: float,
    unit: PhaseUnit = ...,
    stem: str = ...,
    overwrite: bool = ...,
) -> None: ...


@overload
def save_phase_folder(
    root: StrPath,
    images: Iterable[NDArray[np.float32]],
    *,
    ext: FloatFormat,
    stem: str = ...,
    overwrite: bool = ...,
) -> None: ...


def save_phase_folder(
    root: StrPath,
    images: Iterable[NDArray[np.float32]],
    *,
    ext: FloatFormat,
    pixel_size: float | None = None,
    height_scale: float | None = None,
    wavelength: float | None = None,
    refractive_delta: float | None = None,
    unit: PhaseUnit = PhaseUnit.RADIANS,
    stem: str = "phase",
    overwrite: bool = False,
) -> None:
    """Write any phase image sequence to `root` as numbered `ext` files.

    The composer-friendly export: `images` is any iterable of float32 phase frames (a
    file sequence, a `kaparoo` composer such as `ConcatSequence` or a sliced or windowed
    view, a `to_float` reconstruction, or a plain list), so it accepts sources that
    carry no Koala header. Because that header cannot be recovered from a composed
    sequence, the `bin` / `txt` metadata (`pixel_size`, the phase-to-height scale, and
    the `unit` the frames are already in) is given here explicitly; the header-less
    `npy` ignores it. `convert_phase_folder` is the convenience that reads this metadata
    off a file folder's header for you.

    Each frame becomes `{index:05d}_<stem>.<ext>`. The folder is built atomically, so a
    failed run leaves any existing `root` untouched.

    Args:
        root: Destination folder to create and fill.
        images: The phase frames to write, in order (each a 2D float32 image).
        ext: Target format ("bin", "txt", or "npy").
        pixel_size: Physical size of one (square) pixel, in m. Required for "bin" /
            "txt"; ignored (with a warning) for "npy".
        height_scale: Height per rad, in m. Mutually exclusive with `wavelength` /
            `refractive_delta`; one form is required for "bin" / "txt", ignored for
            "npy".
        wavelength: Illumination wavelength, in m (with `refractive_delta`).
        refractive_delta: Refractive-index difference (with `wavelength`).
        unit: The unit `images` are already in, recorded in the "bin" / "txt" header.
            Defaults to RADIANS; ignored for "npy".
        stem: The ``<stem>`` in ``{index:05d}_<stem>.<ext>``. Defaults to "phase".
        overwrite: Whether to replace `root` if it already exists. Defaults to False.

    Raises:
        ValueError: If `ext` is not "bin" / "txt" / "npy", or (for "bin" / "txt")
            `pixel_size` is missing or neither/both scale forms are given.
        FileExistsError: If `root` exists and `overwrite` is False.
    """
    ensure_one_of(ext, FLOAT_FORMATS, name="ext")

    if ext == "npy":
        args = [pixel_size, height_scale, wavelength, refractive_delta]
        if any(arg is not None for arg in args) or unit is not PhaseUnit.RADIANS:
            msg = "`.npy` is header-less; ignoring pixel_size / unit / height scale"
            warnings.warn(msg, stacklevel=2)
        save = partial(save_phase_npy, overwrite=overwrite)
    else:
        if pixel_size is None:
            msg = "pixel_size is required for .bin / .txt"
            raise ValueError(msg)
        resolved = resolve_height_scale(height_scale, wavelength, refractive_delta)
        writer = save_phase_bin if ext == "bin" else save_phase_txt
        save = partial(
            writer,
            pixel_size=pixel_size,
            height_scale=resolved,
            unit=unit,
            overwrite=overwrite,
        )

    with StagedDirectory(root, overwrite=overwrite) as staged:
        for index, image in enumerate(images):
            save(staged.workdir / koala_frame_name(index, stem=stem, ext=ext), image)


def convert_phase_folder(
    root: StrPath,
    folder: PhaseFileFolder,
    *,
    ext: FloatFormat,
    overwrite: bool = False,
) -> None:
    """Re-encode a phase `folder` into `root` in the `ext` format.

    The file-folder convenience over `save_phase_folder`: each frame becomes one
    numbered file sharing the folder's single header. `bin` / `txt` preserve
    `pixel_size`, `height_scale`, and the effective `unit` (read from the folder); the
    header-less `npy` drops them. For a composed or transformed sequence (e.g. a
    `kaparoo` `ConcatSequence`, or a `to_float` view), which has no folder header, use
    `save_phase_folder` directly with explicit metadata. The new folder is built
    atomically, so a failed run leaves any existing `root` untouched.

    Args:
        root: Destination folder to create and fill with the re-encoded frames.
        folder: Source phase folder to read.
        ext: Target format ("bin", "txt", or "npy").
        overwrite: Whether to replace `root` if it already exists. Defaults to False.

    Raises:
        ValueError: If `ext` is not "bin", "txt", or "npy".
        FileExistsError: If `root` exists and `overwrite` is False.
    """
    kwargs = {}

    if ext in ("bin", "txt"):
        header = folder.header
        kwargs = {
            "pixel_size": header.pixel_size,
            "height_scale": header.height_scale,
            "unit": replace_if_none(folder.target_unit, header.unit),
        }

    save_phase_folder(
        root,
        folder,
        ext=ext,
        stem=folder.FILE_STEM,
        overwrite=overwrite,
        **kwargs,
    )


def convert_phase_list(
    sequence: PhaseFileList,
    *,
    ext: FloatFormat,
    overwrite: bool = False,
) -> None:
    """Re-encode each file of a phase `sequence` in place, changing only the suffix.

    A list's files may live anywhere, so each is rewritten beside the original with the
    new ``.{ext}`` suffix, keeping its own `pixel_size`, `height_scale`, and effective
    `unit`; the header-less `npy` drops them. Each file is written atomically, but the
    set as a whole is not.

    Args:
        sequence: Source phase file list to re-encode in place.
        ext: Target format ("bin", "txt", or "npy").
        overwrite: Whether to replace an existing target sibling. Defaults to False.

    Raises:
        ValueError: If `ext` is not "bin", "txt", or "npy".
        FileExistsError: If a target sibling exists and `overwrite` is False.
    """
    ensure_one_of(ext, FLOAT_FORMATS, name="ext")

    if ext == "npy":
        save = partial(save_phase_npy, overwrite=overwrite)
        for index, image in enumerate(sequence):
            save(sequence.get_file(index).with_suffix(f".{ext}"), image)
        return

    writer = save_phase_bin if ext == "bin" else save_phase_txt

    for index in range(len(sequence)):
        # decode + header in one read, rather than get_item + a separate get_header.
        image, header = sequence.load_with_header(index)
        writer(
            sequence.get_file(index).with_suffix(f".{ext}"),
            image,
            pixel_size=header.pixel_size,
            height_scale=header.height_scale,
            unit=replace_if_none(sequence.target_unit, header.unit),
            overwrite=overwrite,
        )
