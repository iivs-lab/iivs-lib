from __future__ import annotations

__all__ = (
    "convert_intensity_folder",
    "convert_intensity_list",
    "save_intensity_folder",
)

import warnings
from functools import partial
from typing import TYPE_CHECKING

from kaparoo.filesystem import StagedDirectory
from kaparoo.utils import ensure_one_of

from iivs.dhm.data.common import numbered_name
from iivs.dhm.data.intensity.bin import save_intensity_bin
from iivs.dhm.data.intensity.npy import save_intensity_npy
from iivs.dhm.data.intensity.txt import save_intensity_txt

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Literal

    import numpy as np
    from kaparoo.filesystem.types import StrPath
    from numpy.typing import NDArray

    from iivs.dhm.data.intensity.base import IntensityFileFolder, IntensityFileList


def save_intensity_folder(
    root: StrPath,
    images: Iterable[NDArray[np.float32]],
    *,
    ext: Literal["bin", "txt", "npy"],
    pixel_size: float | None = None,
    stem: str = "intensity",
    overwrite: bool = False,
) -> None:
    """Write any intensity image sequence to `root` as numbered `ext` files.

    The composer-friendly export: `images` is any iterable of float32 intensity
    frames -- a file sequence, a `kaparoo` composer (`ConcatSequence`, a sliced
    or windowed view), or a plain list -- so it accepts sources that carry no
    Koala header. Because that header cannot be recovered from a composed
    sequence, the `bin` / `txt` `pixel_size` is given here explicitly; the
    header-less `npy` ignores it. `convert_intensity_folder` is the convenience
    that reads `pixel_size` off a file folder's header for you.

    Each frame becomes `{index:05d}_<stem>.<ext>`. The folder is built
    atomically, so a failed run leaves any existing `root` untouched.

    Args:
        root: Destination folder to create and fill.
        images: The intensity frames to write, in order (each a 2D float32 image).
        ext: Target format -- "bin", "txt", or "npy".
        pixel_size: Physical size of one (square) pixel, in m. Required for
            "bin" / "txt"; ignored (with a warning) for "npy".
        stem: The ``<stem>`` in ``{index:05d}_<stem>.<ext>``. Defaults to
            "intensity".
        overwrite: Whether to replace `root` if it already exists. Defaults to
            False.

    Raises:
        ValueError: If `ext` is not "bin" / "txt" / "npy", or (for "bin" /
            "txt") `pixel_size` is missing.
        FileExistsError: If `root` exists and `overwrite` is False.
    """
    ensure_one_of(ext, ("bin", "txt", "npy"), name="ext")

    if ext == "npy":
        if pixel_size is not None:
            msg = "`.npy` is header-less; ignoring pixel_size"
            warnings.warn(msg, stacklevel=2)
        save = partial(save_intensity_npy, overwrite=overwrite)
    else:
        if pixel_size is None:
            msg = "pixel_size is required for .bin / .txt"
            raise ValueError(msg)
        writer = save_intensity_bin if ext == "bin" else save_intensity_txt
        save = partial(writer, pixel_size=pixel_size, overwrite=overwrite)

    with StagedDirectory(root, overwrite=overwrite) as staged:
        for index, image in enumerate(images):
            save(staged.workdir / numbered_name(index, stem=stem, ext=ext), image)


def convert_intensity_folder(
    root: StrPath,
    folder: IntensityFileFolder,
    *,
    ext: Literal["bin", "txt", "npy"],
    overwrite: bool = False,
) -> None:
    """Re-encode an intensity `folder` into `root` in the `ext` format.

    The file-folder convenience over `save_intensity_folder`: each frame becomes
    one numbered file sharing the folder's single header. `bin` / `txt` preserve
    `pixel_size` (read from the folder); the header-less `npy` drops it. For a
    composed or transformed sequence (e.g. a `kaparoo` `ConcatSequence`) -- which
    has no folder header -- use `save_intensity_folder` directly with explicit
    `pixel_size`. The new folder is built atomically, so a failed run leaves any
    existing `root` untouched.

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
    if ext in ("bin", "txt"):
        save_intensity_folder(
            root,
            folder,
            ext=ext,
            pixel_size=folder.header.pixel_size,
            stem=folder.FILE_STEM,
            overwrite=overwrite,
        )
    else:
        save_intensity_folder(
            root, folder, ext=ext, stem=folder.FILE_STEM, overwrite=overwrite
        )


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
    ensure_one_of(ext, ("bin", "txt", "npy"), name="ext")

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
