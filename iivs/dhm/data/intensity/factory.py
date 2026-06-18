from __future__ import annotations

__all__ = (
    "intensity_folder",
    "intensity_list",
    "load_intensity",
    "read_intensity_header",
    "save_intensity",
)

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

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
    from typing import Literal

    import numpy as np
    from kaparoo.filesystem.types import StrPath, StrPaths
    from numpy.typing import NDArray

    from iivs.dhm.data.intensity.base import IntensityFileFolder, IntensityFileList
    from iivs.dhm.data.intensity.bin import IntensityBinHeader

# The float intensity formats this package dispatches over, by file extension.
_FORMATS = ("bin", "txt", "npy")


def _ext(path: StrPath) -> str:
    """The lower-case extension of `path`, without the leading dot."""
    return Path(path).suffix.casefold().removeprefix(".")


def _unsupported(ext: str) -> ValueError:
    """A `ValueError` for an extension that is none of the intensity formats."""
    return ValueError(
        f"unsupported intensity extension {ext!r} (expected bin, txt, or npy)"
    )


def load_intensity(
    path: StrPath,
    *,
    on_nonfinite: Literal["ignore", "warn", "raise"] = "ignore",
) -> NDArray[np.float32]:
    """Load a float32 intensity image, picking the reader by `path`'s extension.

    Dispatches `.bin` / `.txt` / `.npy` to `load_intensity_bin` /
    `load_intensity_txt` / `load_intensity_npy`. Returns the **image only**
    (uniform across formats, since `.npy` is header-less); for the header too,
    use the per-format loader's `return_header` (`.bin` / `.txt`) or
    `read_intensity_header`.

    Raises:
        ValueError: If `path`'s extension is not bin, txt, or npy.
    """
    ext = _ext(path)
    if ext == "bin":
        return load_intensity_bin(path, on_nonfinite=on_nonfinite)
    if ext == "txt":
        return load_intensity_txt(path, on_nonfinite=on_nonfinite)
    if ext == "npy":
        return load_intensity_npy(path, on_nonfinite=on_nonfinite)
    raise _unsupported(ext)


def read_intensity_header(path: StrPath) -> IntensityBinHeader:
    """Read just the header of a `.bin` / `.txt` intensity file, by extension.

    Dispatches to `read_intensity_bin_header` / `read_intensity_txt_header`.
    `.npy` is excluded: it carries no header (supply `pixel_size` via
    `IntensityNpyFolder`).

    Raises:
        ValueError: If `path` is `.npy` (header-less) or its extension is not
            bin or txt.
    """
    ext = _ext(path)
    if ext == "bin":
        return read_intensity_bin_header(path)
    if ext == "txt":
        return read_intensity_txt_header(path)
    if ext == "npy":
        msg = "`.npy` is header-less; supply metadata via IntensityNpyFolder"
        raise ValueError(msg)
    raise _unsupported(ext)


def save_intensity(
    path: StrPath,
    data: NDArray[np.float32],
    *,
    pixel_size: float | None = None,
    overwrite: bool = False,
    on_nonfinite: Literal["ignore", "warn", "raise"] = "warn",
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
    ext = _ext(path)
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
    raise _unsupported(ext)


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

    exts = {_ext(f) for f in files}
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
    raise _unsupported(ext)


def intensity_folder(
    root: StrPath,
    *,
    pixel_size: float | None = None,
    validate: Literal["names", "headers", "data"] | None = "headers",
) -> IntensityFileFolder:
    """Open a numbered intensity folder, picking the class by the format it holds.

    Scans `root` for `{index:05d}_intensity.<ext>` files and dispatches to
    `IntensityBinFolder` / `IntensityTxtFolder` / `IntensityNpyFolder`. The
    `.bin` / `.txt` folders read `pixel_size` from the files, so it must be
    omitted for them; the header-less `.npy` folder instead **requires**
    `pixel_size`.

    Raises:
        FileNotFoundError: If `root` holds no `NNNNN_intensity.{bin,txt,npy}`
            files.
        ValueError: If `root` mixes formats, if `pixel_size` is given for a
            `.bin` / `.txt` folder, or if a `.npy` folder is missing it.
    """
    root_path = Path(root)
    present = [
        ext
        for ext in _FORMATS
        if next(root_path.glob(f"*_intensity.{ext}"), None) is not None
    ]
    if not present:
        msg = f"no NNNNN_intensity.(bin|txt|npy) files found in {root}"
        raise FileNotFoundError(msg)
    if len(present) > 1:
        msg = f"ambiguous: {root} holds multiple intensity formats ({present})"
        raise ValueError(msg)

    ext = present[0]
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
