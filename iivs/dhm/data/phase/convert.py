from __future__ import annotations

__all__ = (
    "convert_phase_folder",
    "convert_phase_list",
)

from functools import partial
from typing import TYPE_CHECKING

from kaparoo.filesystem import StagedDirectory
from kaparoo.utils import replace_if_none

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

    Writes one numbered file per frame, ``{index:05d}_{folder.FILE_STEM}.{ext}``,
    sharing the folder's header. `bin` / `txt` keep `pixel_size`, `height_scale`,
    and the effective `unit`; the header-less `npy` drops them. The output is
    built atomically -- staged, then moved into place on success -- so a failed
    run leaves any existing `root` untouched.

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

    template = f"{{index:05d}}_{folder.FILE_STEM}.{ext}"

    with StagedDirectory(root, overwrite=overwrite) as staged:
        for index, image in enumerate(folder):
            save(staged.workdir / template.format(index=index), image)


def convert_phase_list(
    sequence: PhaseFileList,
    *,
    ext: Literal["bin", "txt", "npy"],
    overwrite: bool = False,
) -> None:
    """Re-encode each file of a phase `sequence` in place, changing only the suffix.

    A list's files may live anywhere, so each is rewritten as a sibling with the
    new ``.{ext}`` suffix (same directory and stem), keeping its own
    `pixel_size`, `height_scale`, and effective `unit`; the header-less `npy`
    drops them. Each file is written atomically, but the set is not one atomic
    folder.

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
