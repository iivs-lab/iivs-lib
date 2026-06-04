from __future__ import annotations

__all__ = (
    "convert_intensity_folder",
    "convert_intensity_list",
)

from functools import partial
from typing import TYPE_CHECKING

from kaparoo.filesystem import StagedDirectory

from iivs.dhm.data.common import numbered_name
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
    """Re-encode an intensity `folder` into `root` in the `ext` format.

    Each frame becomes one numbered file sharing the folder's single header.
    `bin` / `txt` preserve `pixel_size`; the header-less `npy` drops it. The new
    folder is built atomically, so a failed run leaves any existing `root`
    untouched.

    Args:
        root: Destination folder to create and fill with the re-encoded frames.
        folder: Source intensity folder to read.
        ext: Target format -- "bin", "txt", or "npy".
        overwrite: Whether to replace `root` if it already exists. Defaults to
            False.

    Raises:
        ValueError: If `ext` is not "bin", "txt", or "npy".
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

    with StagedDirectory(root, overwrite=overwrite) as staged:
        for index, image in enumerate(folder):
            name = numbered_name(index, stem=folder.FILE_STEM, ext=ext)
            save(staged.workdir / name, image)


def convert_intensity_list(
    sequence: IntensityFileList,
    *,
    ext: Literal["bin", "txt", "npy"],
    overwrite: bool = False,
) -> None:
    """Re-encode each file of an intensity `sequence` in place, changing the suffix.

    A list's files may live anywhere, so each is rewritten beside the original
    with the new ``.{ext}`` suffix, keeping its own `pixel_size`; the
    header-less `npy` drops it. Each file is written atomically, but the set as
    a whole is not.

    Args:
        sequence: Source intensity file list to re-encode in place.
        ext: Target format -- "bin", "txt", or "npy".
        overwrite: Whether to replace an existing target sibling. Defaults to
            False.

    Raises:
        ValueError: If `ext` is not "bin", "txt", or "npy".
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

    for index in range(len(sequence)):
        # decode + header in one read, rather than get_item + a separate get_header.
        image, header = sequence.load_with_header(index)
        writer(
            sequence.get_file(index).with_suffix(f".{ext}"),
            image,
            pixel_size=header.pixel_size,
            overwrite=overwrite,
        )
