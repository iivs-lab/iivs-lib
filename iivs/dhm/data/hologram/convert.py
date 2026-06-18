from __future__ import annotations

__all__ = ("convert_hologram_sequence",)

from functools import partial
from typing import TYPE_CHECKING

from kaparoo.filesystem import StagedDirectory
from kaparoo.utils import ensure_one_of

from iivs.dhm.data.common import numbered_name
from iivs.dhm.data.hologram.npy import save_hologram_npy
from iivs.dhm.data.hologram.raw import save_hologram_raw
from iivs.dhm.data.hologram.tif import save_hologram_tif

if TYPE_CHECKING:
    from typing import Literal

    from kaparoo.filesystem.types import StrPath

    from iivs.dhm.data.hologram.base import HologramSequence


def convert_hologram_sequence(
    dest: StrPath,
    sequence: HologramSequence[object],
    *,
    ext: Literal["raw", "tif", "npy"],
    overwrite: bool = False,
) -> None:
    """Re-encode a hologram `sequence` to `dest` in the `ext` format.

    Every format is lossless uint8. ``ext="raw"`` writes a single multi-frame
    `.raw` stack at `dest`, streamed frame by frame so a large source is never
    held whole; ``"tif"`` / ``"npy"`` write one numbered file per frame into the
    `dest` folder (named from the source's `FILE_STEM`, else ``holo``). Both
    paths are written atomically.

    Args:
        dest: Destination -- the `.raw` file for "raw", else the folder to
            create and fill.
        sequence: Source hologram sequence to read (a `HologramRawFile`,
            `HologramTifFolder`, `HologramNpyFolder`, or `HologramTifList`).
        ext: Target format -- "raw", "tif", or "npy".
        overwrite: Whether to replace an existing destination. Defaults to
            False.

    Raises:
        ValueError: If `ext` is not "raw", "tif", or "npy".
        FileExistsError: If a destination exists and `overwrite` is False.
    """
    ensure_one_of(ext, ("raw", "tif", "npy"), name="ext")

    if ext == "raw":
        save_hologram_raw(dest, sequence, overwrite=overwrite)
        return

    writer = save_hologram_tif if ext == "tif" else save_hologram_npy
    save = partial(writer, overwrite=overwrite)
    stem = getattr(sequence, "FILE_STEM", "holo")

    with StagedDirectory(dest, overwrite=overwrite) as staged:
        for index, image in enumerate(sequence):
            save(staged.workdir / numbered_name(index, stem=stem, ext=ext), image)
