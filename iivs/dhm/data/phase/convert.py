from __future__ import annotations

__all__ = (
    "convert_phase_folder",
    "convert_phase_list",
)

from functools import partial
from typing import TYPE_CHECKING

from kaparoo.filesystem import StagedDirectory
from kaparoo.utils import replace_if_none

from iivs.dhm.data.common import numbered_name
from iivs.dhm.data.phase.bin import save_phase_bin
from iivs.dhm.data.phase.npy import save_phase_npy
from iivs.dhm.data.phase.txt import save_phase_txt

if TYPE_CHECKING:
    from typing import Literal

    from kaparoo.filesystem.types import StrPath

    from iivs.dhm.data.phase.base import PhaseFileFolder, PhaseFileList


def convert_phase_folder(
    root: StrPath,
    folder: PhaseFileFolder,
    *,
    ext: Literal["bin", "txt", "npy"],
    overwrite: bool = False,
) -> None:
    """Re-encode a phase `folder` into `root` in the `ext` format.

    Each frame becomes one numbered file sharing the folder's single header.
    `bin` / `txt` preserve `pixel_size`, `height_scale`, and the effective
    `unit`; the header-less `npy` drops them. The new folder is built
    atomically, so a failed run leaves any existing `root` untouched.

    Args:
        root: Destination folder to create and fill with the re-encoded frames.
        folder: Source phase folder to read.
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
        save = partial(save_phase_npy, overwrite=overwrite)
    else:
        writer = save_phase_bin if ext == "bin" else save_phase_txt
        save = partial(
            writer,
            pixel_size=folder.header.pixel_size,
            height_scale=folder.header.height_scale,
            unit=replace_if_none(folder.target_unit, folder.header.unit),
            overwrite=overwrite,
        )

    with StagedDirectory(root, overwrite=overwrite) as staged:
        for index, image in enumerate(folder):
            name = numbered_name(index, stem=folder.FILE_STEM, ext=ext)
            save(staged.workdir / name, image)


def convert_phase_list(
    sequence: PhaseFileList,
    *,
    ext: Literal["bin", "txt", "npy"],
    overwrite: bool = False,
) -> None:
    """Re-encode each file of a phase `sequence` in place, changing only the suffix.

    A list's files may live anywhere, so each is rewritten beside the original
    with the new ``.{ext}`` suffix, keeping its own `pixel_size`,
    `height_scale`, and effective `unit`; the header-less `npy` drops them.
    Each file is written atomically, but the set as a whole is not.

    Args:
        sequence: Source phase file list to re-encode in place.
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
        save = partial(save_phase_npy, overwrite=overwrite)
        for index, image in enumerate(sequence):
            save(sequence.get_file(index).with_suffix(f".{ext}"), image)
        return

    writer = save_phase_bin if ext == "bin" else save_phase_txt

    for index, image in enumerate(sequence):
        path = sequence.get_file(index)
        header = sequence.header_at(index)
        writer(
            path.with_suffix(f".{ext}"),
            image,
            pixel_size=header.pixel_size,
            height_scale=header.height_scale,
            unit=replace_if_none(sequence.target_unit, header.unit),
            overwrite=overwrite,
        )
