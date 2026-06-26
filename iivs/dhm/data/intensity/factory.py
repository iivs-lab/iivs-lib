from __future__ import annotations

__all__ = (
    "intensity_folder",
    "intensity_list",
    "load_intensity",
    "read_intensity_header",
    "save_intensity",
)

import warnings
from typing import TYPE_CHECKING, overload

from kaparoo.filesystem import UnsupportedExtensionError, file_extension

from iivs.dhm.data.common import FLOAT_FORMATS, detect_numbered_format
from iivs.dhm.data.intensity.bin import (
    IntensityBinFolder,
    IntensityBinList,
    load_intensity_bin,
    read_intensity_bin_header,
    save_intensity_bin,
)
from iivs.dhm.data.intensity.npy import (
    IntensityNpyFolder,
    load_intensity_npy,
    save_intensity_npy,
)
from iivs.dhm.data.intensity.txt import (
    IntensityTxtFolder,
    IntensityTxtList,
    load_intensity_txt,
    read_intensity_txt_header,
    save_intensity_txt,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal

    import numpy as np
    from kaparoo.filesystem.types import StrPath, StrPaths
    from numpy.typing import NDArray

    from iivs.dhm.data.common import FloatFormat, OnNonFinite, ValidationLevel
    from iivs.dhm.data.intensity.base import IntensityFileFolder, IntensityFileList
    from iivs.dhm.data.intensity.bin import IntensityBinHeader


@overload
def load_intensity(
    path: StrPath,
    *,
    return_header: Literal[False] = False,
    on_nonfinite: OnNonFinite = ...,
) -> NDArray[np.float32]: ...


@overload
def load_intensity(
    path: StrPath,
    *,
    return_header: Literal[True],
    on_nonfinite: OnNonFinite = ...,
) -> tuple[NDArray[np.float32], IntensityBinHeader | None]: ...


@overload
def load_intensity(
    path: StrPath,
    *,
    return_header: bool,
    on_nonfinite: OnNonFinite = ...,
) -> NDArray[np.float32] | tuple[NDArray[np.float32], IntensityBinHeader | None]: ...


def load_intensity(
    path: StrPath,
    *,
    return_header: bool = False,
    on_nonfinite: OnNonFinite = "ignore",
) -> NDArray[np.float32] | tuple[NDArray[np.float32], IntensityBinHeader | None]:
    """Load a float32 intensity image, picking the reader by `path`'s extension.

    Dispatches `.bin` / `.txt` / `.npy` to `load_intensity_bin` /
    `load_intensity_txt` / `load_intensity_npy`. With `return_header` it also
    returns the parsed header -- but `None` for `.npy`, which is header-less
    (its `pixel_size` lives only on `IntensityNpyFolder`, not in the file).
    Contrast `read_intensity_header`, whose sole job *is* the header and so
    raises on `.npy`: here the header is an optional extra, so an absent one is
    `None`, not an error.

    Args:
        path: The `.bin` / `.txt` / `.npy` file to read.
        return_header: Whether to also return the parsed header (`None` for the
            header-less `.npy`). Defaults to False.
        on_nonfinite: How to handle non-finite values (NaN, +inf, -inf) in the
            decoded data: "ignore" (default) accepts silently, "warn" emits a
            RuntimeWarning, "raise" rejects with a ValueError.

    Returns:
        The intensity image as a 2D float32 array, or an `(image, header)` tuple
        when `return_header` is True -- with `header` `None` for `.npy`.

    Raises:
        ValueError: If `path`'s extension is not bin, txt, or npy.
    """
    ext = file_extension(path)
    if ext == "bin":
        return load_intensity_bin(
            path, return_header=return_header, on_nonfinite=on_nonfinite
        )
    if ext == "txt":
        return load_intensity_txt(
            path, return_header=return_header, on_nonfinite=on_nonfinite
        )
    if ext == "npy":
        data = load_intensity_npy(path, on_nonfinite=on_nonfinite)
        return (data, None) if return_header else data
    raise UnsupportedExtensionError(ext, FLOAT_FORMATS, kind="intensity")


def read_intensity_header(path: StrPath) -> IntensityBinHeader:
    """Read just the header of a `.bin` / `.txt` intensity file, by extension.

    Dispatches to `read_intensity_bin_header` / `read_intensity_txt_header`.
    `.npy` is excluded: it carries no header (supply `pixel_size` via
    `IntensityNpyFolder`).

    Raises:
        ValueError: If `path` is `.npy` (header-less) or its extension is not
            bin or txt.
    """
    ext = file_extension(path)
    if ext == "bin":
        return read_intensity_bin_header(path)
    if ext == "txt":
        return read_intensity_txt_header(path)
    if ext == "npy":
        msg = "`.npy` is header-less; supply metadata via IntensityNpyFolder"
        raise ValueError(msg)
    raise UnsupportedExtensionError(ext, FLOAT_FORMATS, kind="intensity")


def save_intensity(
    path: StrPath,
    data: NDArray[np.float32],
    *,
    pixel_size: float | None = None,
    overwrite: bool = False,
    on_nonfinite: OnNonFinite = "warn",
) -> None:
    """Save a 2D float32 intensity image, picking the writer by `path`'s extension.

    Dispatches `.bin` / `.txt` to `save_intensity_bin` / `save_intensity_txt`
    (both need `pixel_size`), and `.npy` to the header-less `save_intensity_npy`.
    For `.npy` the `pixel_size` does not apply; passing it emits a warning and it
    is dropped.

    Raises:
        ValueError: If `path`'s extension is not bin, txt, or npy; or, for
            `.bin` / `.txt`, if `pixel_size` is missing.
    """
    ext = file_extension(path)
    if ext in ("bin", "txt"):
        if pixel_size is None:
            msg = "pixel_size is required for .bin / .txt"
            raise ValueError(msg)
        save = save_intensity_bin if ext == "bin" else save_intensity_txt
        save(
            path,
            data,
            pixel_size=pixel_size,
            overwrite=overwrite,
            on_nonfinite=on_nonfinite,
        )
        return
    if ext == "npy":
        if pixel_size is not None:
            msg = "`.npy` is header-less; ignoring pixel_size"
            warnings.warn(msg, stacklevel=2)
        save_intensity_npy(path, data, overwrite=overwrite, on_nonfinite=on_nonfinite)
        return
    raise UnsupportedExtensionError(ext, FLOAT_FORMATS, kind="intensity")


def intensity_list(files: StrPaths) -> IntensityFileList:
    """Build an intensity file list, picking the class by the files' shared extension.

    Dispatches `.bin` / `.txt` to `IntensityBinList` / `IntensityTxtList`. All
    `files` must share one extension. `.npy` has no list form (each file is
    header-less, with no shared acquisition header) -- use `IntensityNpyFolder`.

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
        return IntensityBinList(files)
    if ext == "txt":
        return IntensityTxtList(files)
    if ext == "npy":
        msg = "no .npy intensity list; use IntensityNpyFolder (npy is header-less)"
        raise ValueError(msg)
    raise UnsupportedExtensionError(ext, FLOAT_FORMATS, kind="intensity")


def intensity_folder(
    root: StrPath,
    *,
    pixel_size: float | None = None,
    validate: ValidationLevel | None = "headers",
    prefer: FloatFormat | Sequence[FloatFormat] | None = None,
) -> IntensityFileFolder:
    """Open a numbered intensity folder, picking the class by the format it holds.

    Discovers the `{index:05d}_intensity.<ext>` files under `root` (via
    `data.common.detect_numbered_format`) and dispatches to `IntensityBinFolder`
    / `IntensityTxtFolder` / `IntensityNpyFolder`. The `.bin` / `.txt` folders
    read `pixel_size` from the files, so it must be omitted for them; the
    header-less `.npy` folder instead **requires** `pixel_size`.

    Args:
        root: The folder to scan.
        pixel_size: The pixel size for a `.npy` folder (omit for `.bin` / `.txt`).
        validate: Validation level at construction, or None to skip.
        prefer: How to resolve a `root` that holds more than one format -- `None`
            (default) raises, while a format or a priority sequence picks the
            first present one (e.g. `prefer=("bin", "txt")`).

    Raises:
        FileNotFoundError: If `root` holds no `NNNNN_intensity.{bin,txt,npy}`
            files.
        ValueError: If `root` mixes formats and `prefer` does not resolve it, if
            `pixel_size` is given for a `.bin` / `.txt` folder, or if a `.npy`
            folder is missing it.
    """
    ext = detect_numbered_format(
        root, stem="intensity", formats=FLOAT_FORMATS, prefer=prefer
    )
    if ext in ("bin", "txt"):
        if pixel_size is not None:
            msg = f".{ext} folders read pixel_size from the files; drop the argument"
            raise ValueError(msg)
        folder = IntensityBinFolder if ext == "bin" else IntensityTxtFolder
        return folder(root, validate=validate)

    if pixel_size is None:
        msg = "`.npy` folders need pixel_size (npy is header-less)"
        raise ValueError(msg)
    return IntensityNpyFolder(root, pixel_size=pixel_size, validate=validate)
