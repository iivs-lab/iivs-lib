from __future__ import annotations

__all__ = ("HOLOGRAM_TREE", "open_holograms", "search_holograms")

from pathlib import Path
from typing import TYPE_CHECKING

from kaparoo.filesystem import hierarchy
from kaparoo.filters import Glob, Regex

from iivs.dhm.data.hologram.raw import HologramRawFile
from iivs.dhm.data.hologram.tif import HologramTifFolder
from iivs.dhm.data.koala import HOLOGRAMS, open_folder, search_modality_dirs

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from kaparoo.filesystem.exclude import ExcludeRule
    from kaparoo.filesystem.types import StrPath
    from kaparoo.filters import Filter
    from kaparoo.filters.types import FilterDict

    from iivs.dhm.data.hologram.base import HologramSequence

_HOLOGRAM_TIF = r"\d{5}_holo\.tif"

HOLOGRAM_TREE = hierarchy.Directory(
    HOLOGRAMS,
    [
        # A single multi-frame stack XOR numbered per-frame previews; a real
        # acquisition yields only one, so both present is a violation.
        hierarchy.Exclusive(
            hierarchy.File(Glob("*.raw")),
            hierarchy.File(Regex(_HOLOGRAM_TIF)),
        ),
    ],
)
"""The `Holograms/` subtree of a Koala acquisition, as a `hierarchy` spec.

Holds either a `*.raw` stack or numbered `*.tif` previews, never both.
"""


def open_holograms(root: StrPath) -> HologramSequence | None:
    """Open a `Holograms/` folder as a single hologram sequence, or None when absent.

    Returns the `*.raw` stack if present, else the numbered tif folder, tolerating an
    absent or empty folder as None.

    Raises:
        ValueError: If the folder holds both a `.raw` stack and numbered `.tif` previews
            (a real acquisition produces only one).
    """
    holo_dir = Path(root)
    if not holo_dir.is_dir():
        return None

    raws = sorted(holo_dir.glob("*.raw"))
    tif_folder = open_folder(holo_dir, HologramTifFolder)
    if raws and tif_folder is not None:
        msg = "holograms hold both a .raw stack and .tif previews (expected one)"
        raise ValueError(msg)

    return HologramRawFile(raws[0]) if raws else tif_folder


def search_holograms(
    root: StrPath,
    *,
    name_filter: Filter | FilterDict | None = None,
    part_filter: Filter | FilterDict | None = None,
    predicate: Callable[[HologramSequence], bool] | None = None,
    exclude: ExcludeRule | Iterable[ExcludeRule] | None = None,
    min_depth: int = 1,
    max_depth: int | None = None,
    ordered: bool = True,
) -> list[HologramSequence]:
    """Return the holograms of each time-lapse under `root` that holds a `Holograms/` folder.

    Finds the time-lapse folders via `search_modality_dirs` (delegating to `search_dirs`, no
    manual recursion) and opens each `Holograms/` with `open_holograms`; an empty folder
    is skipped, and `predicate` is a final check on the built *`HologramSequence`*.

    Args:
        root: The directory to scan.
        name_filter: Filter on each candidate time-lapse folder's own name.
        part_filter: Filter on each visited parent directory's relative path.
        predicate: A final check on the opened `HologramSequence`; None (default) keeps
            all.
        exclude: Path(s) to prune, as in `search_dirs`.
        min_depth: Shallowest depth to include (>= 1).
        max_depth: Deepest depth to include, or None (default) for unlimited.
        ordered: Sort the results by path. Defaults to True.

    Returns:
        The opened hologram sequences, in `search_dirs` order.

    Raises:
        ValueError: If any matched `Holograms/` holds both a `.raw` stack and `.tif`
            previews (from `open_holograms`).
    """
    opened = (
        open_holograms(directory)
        for directory in search_modality_dirs(
            root,
            HOLOGRAMS,
            name_filter=name_filter,
            part_filter=part_filter,
            exclude=exclude,
            min_depth=min_depth,
            max_depth=max_depth,
            ordered=ordered,
        )
    )
    present = [sequence for sequence in opened if sequence is not None]
    if predicate is None:
        return present
    return [sequence for sequence in present if predicate(sequence)]
