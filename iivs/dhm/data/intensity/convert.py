from __future__ import annotations

__all__ = ("convert_intensity_folder",)

from functools import partial
from typing import TYPE_CHECKING

from kaparoo.filesystem import StagedDirectory

from iivs.dhm.data.intensity.bin import save_intensity_bin
from iivs.dhm.data.intensity.npy import save_intensity_npy
from iivs.dhm.data.intensity.txt import save_intensity_txt

if TYPE_CHECKING:
    from typing import Literal

    from kaparoo.filesystem.types import StrPath

    from iivs.dhm.data.intensity.base import IntensityFileFolder


def convert_intensity_folder(
    root: StrPath,
    folder: IntensityFileFolder,
    *,
    ext: Literal["bin", "txt", "npy"],
    overwrite: bool = False,
) -> None:
    """Re-encode every frame of an intensity `folder` into `root` in `ext` format.

    `folder` is an `IntensityFileFolder` (`IntensityBinFolder` /
    `IntensityTxtFolder` / `IntensityNpyFolder`) -- a numbered, same-shape
    acquisition sharing one header. Each frame is written as
    ``{index:05d}_{folder.FILE_STEM}.{ext}``. `bin` and `txt` carry the shared
    `pixel_size`; `npy` is header-less, so that metadata is dropped (the float
    pixels stay exact -- resupply it via `IntensityNpyFolder` on read).

    The output folder is built atomically: frames are staged in a temp directory
    and moved into place only on success, so a reader never sees a half-built
    folder and a failed conversion leaves any existing `root` untouched.

    Raises:
        ValueError: If `ext` is not one of "bin", "txt", "npy".
        FileExistsError: If `root` exists and `overwrite` is False.
    """
    if ext not in ("bin", "txt", "npy"):
        msg = f"ext must be 'bin', 'txt', or 'npy' (got {ext!r})"
        raise ValueError(msg)

    if ext == "npy":
        save = partial(save_intensity_npy, overwrite=overwrite)
    else:
        writer = save_intensity_bin if ext == "bin" else save_intensity_txt
        save = partial(writer, pixel_size=folder.header.pixel_size, overwrite=overwrite)

    template = f"{{index:05d}}_{folder.FILE_STEM}.{ext}"
    with StagedDirectory(root, overwrite=overwrite) as staged:
        for index, image in enumerate(folder):
            save(staged.workdir / template.format(index=index), image)
