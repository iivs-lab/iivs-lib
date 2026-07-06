from __future__ import annotations

__all__ = ("HOLOGRAM_FORMATS", "HologramFormat", "convert_hologram_sequence")

from functools import partial
from typing import TYPE_CHECKING

from kaparoo.filesystem import StagedDirectory
from kaparoo.utils import ensure_one_of

from iivs.dhm.data.hologram.npy import save_hologram_npy
from iivs.dhm.data.hologram.raw import save_hologram_raw
from iivs.dhm.data.hologram.tif import save_hologram_tif
from iivs.dhm.data.koala import numbered_name

if TYPE_CHECKING:
    from typing import Literal

    import numpy as np
    from kaparoo.data.sequences import DataSequence
    from kaparoo.filesystem.types import StrPath
    from numpy.typing import NDArray


type HologramFormat = Literal["raw", "tif", "npy"]
"""A hologram's on-disk format: the multi-frame `.raw` stack, or `.tif` / `.npy`."""

HOLOGRAM_FORMATS: tuple[HologramFormat, ...] = ("raw", "tif", "npy")
"""The hologram formats, for runtime membership checks (the `HologramFormat` values)."""


def convert_hologram_sequence(
    dest: StrPath,
    sequence: DataSequence[NDArray[np.uint8], object],
    *,
    ext: HologramFormat,
    overwrite: bool = False,
) -> None:
    """Re-encode a hologram `sequence` to `dest` in the `ext` format.

    Every format is lossless uint8. ``ext="raw"`` writes a single multi-frame
    `.raw` stack at `dest`, streamed frame by frame so a large source is never
    held whole; ``"tif"`` / ``"npy"`` write one numbered file per frame into the
    `dest` folder (named from the source's `FILE_STEM`, else ``holo``). Both
    paths are written atomically.

    Accepts any uint8 image sequence, not just a file-backed `HologramSequence`:
    a `kaparoo` composer (`ConcatSequence`, a sliced or windowed view) works too,
    falling back to the ``holo`` stem when it has no `FILE_STEM`.

    Args:
        dest: Destination; the `.raw` file for "raw", else the folder to
            create and fill.
        sequence: Source hologram sequence to read (a `HologramRawFile`,
            `HologramTifFolder`, `HologramNpyFolder`, `HologramTifList`, or any
            uint8 `DataSequence` such as a composed sequence).
        ext: Target format; "raw", "tif", or "npy".
        overwrite: Whether to replace an existing destination. Defaults to
            False.

    Raises:
        ValueError: If `ext` is not "raw", "tif", or "npy".
        FileExistsError: If a destination exists and `overwrite` is False.
    """
    ensure_one_of(ext, HOLOGRAM_FORMATS, name="ext")

    if ext == "raw":
        save_hologram_raw(dest, sequence, overwrite=overwrite)
        return

    writer = save_hologram_tif if ext == "tif" else save_hologram_npy
    save = partial(writer, overwrite=overwrite)
    stem = getattr(sequence, "FILE_STEM", "holo")

    with StagedDirectory(dest, overwrite=overwrite) as staged:
        for index, image in enumerate(sequence):
            save(staged.workdir / numbered_name(index, stem=stem, ext=ext), image)
