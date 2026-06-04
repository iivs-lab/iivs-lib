from __future__ import annotations

__all__ = (
    "convert_intensity_folder",
    "convert_intensity_list",
)

from functools import partial
from typing import TYPE_CHECKING

from kaparoo.filesystem import StagedDirectory

from iivs.dhm.data.intensity.bin import save_intensity_bin
from iivs.dhm.data.intensity.npy import save_intensity_npy
from iivs.dhm.data.intensity.txt import save_intensity_txt

if TYPE_CHECKING:
    from typing import Literal

    from kaparoo.filesystem.types import StrPath

    from iivs.dhm.data.intensity.base import IntensityFileFolder, IntensityFileList


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


def convert_intensity_list(
    sequence: IntensityFileList,
    *,
    ext: Literal["bin", "txt", "npy"],
    overwrite: bool = False,
) -> None:
    """Re-encode each file of an arbitrary intensity `sequence` in place.

    An `IntensityFileList` (`IntensityBinList` / `IntensityTxtList`) is a flat
    set of files that may live anywhere, so there is no shared `root` or
    numbering: each source file is rewritten as a sibling with the new ``.{ext}``
    suffix (same directory and stem). Every frame keeps *its own* `pixel_size`
    (read per file); `npy` is header-less, so that metadata is dropped.

    Each file is written atomically, but -- unlike `convert_intensity_folder` --
    the set is not built as one atomic folder.

    Raises:
        ValueError: If `ext` is not one of "bin", "txt", "npy".
        FileExistsError: If a target sibling exists and `overwrite` is False.
    """
    if ext not in ("bin", "txt", "npy"):
        msg = f"ext must be 'bin', 'txt', or 'npy' (got {ext!r})"
        raise ValueError(msg)

    if ext == "npy":
        save = partial(save_intensity_npy, overwrite=overwrite)
        for index, image in enumerate(sequence):
            save(sequence.get_file(index).with_suffix(f".{ext}"), image)
        return

    writer = save_intensity_bin if ext == "bin" else save_intensity_txt
    for index, image in enumerate(sequence):
        path = sequence.get_file(index)
        header = sequence._read_header(path)  # noqa: SLF001
        writer(
            path.with_suffix(f".{ext}"),
            image,
            pixel_size=header.pixel_size,
            overwrite=overwrite,
        )
