from __future__ import annotations

__all__ = ("convert_phase_folder", "convert_phase_list", "save_phase_folder")

import warnings
from functools import partial
from typing import TYPE_CHECKING

from kaparoo.filesystem import StagedDirectory
from kaparoo.utils import ensure_one_of, replace_if_none

from iivs.dhm.data.common import numbered_name
from iivs.dhm.data.phase.bin import save_phase_bin
from iivs.dhm.data.phase.npy import save_phase_npy
from iivs.dhm.data.phase.txt import save_phase_txt
from iivs.dhm.data.phase.unit import PhaseUnit, resolve_height_scale

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Literal

    import numpy as np
    from kaparoo.filesystem.types import StrPath
    from numpy.typing import NDArray

    from iivs.dhm.data.phase.base import PhaseFileFolder, PhaseFileList


def save_phase_folder(
    root: StrPath,
    images: Iterable[NDArray[np.float32]],
    *,
    ext: Literal["bin", "txt", "npy"],
    pixel_size: float | None = None,
    height_scale: float | None = None,
    wavelength: float | None = None,
    refractive_delta: float | None = None,
    unit: PhaseUnit = PhaseUnit.RADIANS,
    stem: str = "phase",
    overwrite: bool = False,
) -> None:
    """Write any phase image sequence to `root` as numbered `ext` files.

    The composer-friendly export: `images` is any iterable of float32 phase
    frames -- a file sequence, a `kaparoo` composer (`ConcatSequence`, a sliced
    or windowed view), a `to_float` reconstruction, or a plain list -- so it
    accepts sources that carry no Koala header. Because that header cannot be
    recovered from a composed sequence, the `bin` / `txt` metadata
    (`pixel_size`, the phase-to-height scale, and the `unit` the frames are
    already in) is given here explicitly; the header-less `npy` ignores it.
    `convert_phase_folder` is the convenience that reads this metadata off a
    file folder's header for you.

    Each frame becomes `{index:05d}_<stem>.<ext>`. The folder is built
    atomically, so a failed run leaves any existing `root` untouched.

    Args:
        root: Destination folder to create and fill.
        images: The phase frames to write, in order (each a 2D float32 image).
        ext: Target format -- "bin", "txt", or "npy".
        pixel_size: Physical size of one (square) pixel, in m. Required for
            "bin" / "txt"; ignored (with a warning) for "npy".
        height_scale: Height per rad, in m. Mutually exclusive with
            `wavelength` / `refractive_delta`; one form is required for "bin" /
            "txt", ignored for "npy".
        wavelength: Illumination wavelength, in m (with `refractive_delta`).
        refractive_delta: Refractive-index difference (with `wavelength`).
        unit: The unit `images` are already in, recorded in the "bin" / "txt"
            header. Defaults to RADIANS; ignored for "npy".
        stem: The ``<stem>`` in ``{index:05d}_<stem>.<ext>``. Defaults to "phase".
        overwrite: Whether to replace `root` if it already exists. Defaults to
            False.

    Raises:
        ValueError: If `ext` is not "bin" / "txt" / "npy", or (for "bin" /
            "txt") `pixel_size` is missing or neither/both scale forms are given.
        FileExistsError: If `root` exists and `overwrite` is False.
    """
    ensure_one_of(ext, ("bin", "txt", "npy"), name="ext")

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
            save(staged.workdir / numbered_name(index, stem=stem, ext=ext), image)


def convert_phase_folder(
    root: StrPath,
    folder: PhaseFileFolder,
    *,
    ext: Literal["bin", "txt", "npy"],
    overwrite: bool = False,
) -> None:
    """Re-encode a phase `folder` into `root` in the `ext` format.

    The file-folder convenience over `save_phase_folder`: each frame becomes one
    numbered file sharing the folder's single header. `bin` / `txt` preserve
    `pixel_size`, `height_scale`, and the effective `unit` (read from the
    folder); the header-less `npy` drops them. For a composed or transformed
    sequence (e.g. a `kaparoo` `ConcatSequence`, or a `to_float` view) -- which
    has no folder header -- use `save_phase_folder` directly with explicit
    metadata. The new folder is built atomically, so a failed run leaves any
    existing `root` untouched.

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
    if ext in ("bin", "txt"):
        save_phase_folder(
            root,
            folder,
            ext=ext,
            pixel_size=folder.header.pixel_size,
            height_scale=folder.header.height_scale,
            unit=replace_if_none(folder.target_unit, folder.header.unit),
            stem=folder.FILE_STEM,
            overwrite=overwrite,
        )
    else:
        save_phase_folder(
            root, folder, ext=ext, stem=folder.FILE_STEM, overwrite=overwrite
        )


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
    ensure_one_of(ext, ("bin", "txt", "npy"), name="ext")

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
