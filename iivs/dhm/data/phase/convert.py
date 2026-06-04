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
    """Re-encode every frame of a phase `folder` into `root` in the `ext` format.

    `folder` is a `PhaseFileFolder` (`PhaseBinFolder` / `PhaseTxtFolder` /
    `PhaseNpyFolder`) -- a numbered, same-shape acquisition sharing one header.
    Each frame is written as ``{index:05d}_{folder.FILE_STEM}.{ext}``. `bin` and
    `txt` carry the shared `pixel_size`, `height_scale`, and effective `unit`;
    `npy` is header-less, so that metadata is dropped (the float pixels stay
    exact -- resupply it via `PhaseNpyFolder` on read).

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
    """Re-encode each file of an arbitrary phase `sequence` in place.

    A `PhaseFileList` (`PhaseBinList` / `PhaseTxtList`) is a flat set of files
    that may live anywhere, so there is no shared `root` or numbering: each
    source file is rewritten as a sibling with the new ``.{ext}`` suffix (same
    directory and stem). Every frame keeps *its own* `pixel_size`,
    `height_scale`, and effective `unit` (read per file), so a mixed-metadata
    list converts faithfully; `npy` is header-less, so that metadata is dropped.

    Each file is written atomically, but -- unlike `convert_phase_folder` -- the
    set is not built as one atomic folder.

    Raises:
        ValueError: If `ext` is not one of "bin", "txt", "npy".
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
        header = sequence._read_header(path)  # noqa: SLF001
        writer(
            path.with_suffix(f".{ext}"),
            image,
            pixel_size=header.pixel_size,
            height_scale=header.height_scale,
            unit=replace_if_none(sequence.target_unit, header.unit),
            overwrite=overwrite,
        )
