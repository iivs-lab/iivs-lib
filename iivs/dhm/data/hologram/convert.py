from __future__ import annotations

__all__ = ("convert_hologram_sequence",)

from functools import partial
from typing import TYPE_CHECKING

from kaparoo.filesystem import StagedDirectory

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

    `sequence` is any `HologramSequence` (a single-file `HologramRawFile` or a
    `HologramTifFolder` / `HologramNpyFolder` / `HologramTifList`). ``ext="raw"``
    writes one multi-frame `.raw` stack at `dest`, streamed frame by frame;
    ``"tif"`` / ``"npy"`` write ``{index:05d}_{stem}.{ext}`` (the source's
    `FILE_STEM`, else ``holo``) into the `dest` folder. Both are written
    atomically; every format is lossless uint8.

    Raises:
        ValueError: If `ext` is not "raw", "tif", or "npy".
        FileExistsError: If a destination exists and `overwrite` is False.
    """
    if ext not in ("raw", "tif", "npy"):
        msg = f"ext must be 'raw', 'tif', or 'npy' (got {ext!r})"
        raise ValueError(msg)

    if ext == "raw":
        save_hologram_raw(dest, sequence, overwrite=overwrite)
        return

    writer = save_hologram_tif if ext == "tif" else save_hologram_npy
    save = partial(writer, overwrite=overwrite)
    stem = getattr(sequence, "FILE_STEM", "holo")

    with StagedDirectory(dest, overwrite=overwrite) as staged:
        for index, image in enumerate(sequence):
            save(staged.workdir / numbered_name(index, stem=stem, ext=ext), image)
