from __future__ import annotations

__all__ = (
    "HOLOGRAM_TREE",
    "MultiFormatHologramsError",
    "open_holograms",
    "search_holograms",
    "search_multi_format_holograms",
)

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from kaparoo.filesystem.hierarchy import Directory, Exclusive, File
from kaparoo.filters import Regex

from iivs.dhm.data.hologram.raw import HologramRawFile
from iivs.dhm.data.hologram.tif import HologramTifFolder
from iivs.dhm.data.koala import HOLOGRAMS, open_folder, search_timelapse_subdirs

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from typing import Literal

    from kaparoo.filesystem.exclude import ExcludeRule
    from kaparoo.filesystem.types import StrPath
    from kaparoo.filters import Filter
    from kaparoo.filters.types import FilterDict

    from iivs.dhm.data.hologram.base import HologramSequence

_HOLOGRAM_RAW = "holo.raw"
_HOLOGRAM_TIF = r"\d{5}_holo\.tif"

HOLOGRAM_TREE = Directory(
    HOLOGRAMS,
    [
        # A single multi-frame stack XOR numbered per-frame previews; a real
        # acquisition yields only one, so both present is a violation.
        Exclusive(
            File(_HOLOGRAM_RAW),
            File(Regex(_HOLOGRAM_TIF)),
        ),
    ],
)
"""The `Holograms/` subtree of a Koala acquisition, as a `hierarchy` spec.

Holds either the `holo.raw` stack or numbered `*.tif` previews, never both.
"""


class MultiFormatHologramsError(ValueError):
    """A `Holograms/` folder holds both a `.raw` stack and numbered `.tif` previews.

    A real Koala acquisition writes its holograms in one format, so the two together
    leave no way to say which is the acquisition: unlike the reconstructions, whose
    `Float/{Bin,Txt}` are two serializations of one source that `quantitative` picks
    between, neither hologram format defers to the other. A layout fault, then, distinct
    from a corrupt file. Subclasses `ValueError` (a broad `except ValueError` still
    catches it); catch it specifically to tolerate or skip only this fault while letting
    genuine content errors surface.
    """


def open_holograms(root: StrPath) -> HologramSequence | None:
    """Open a `Holograms/` folder as a single hologram sequence, or None when absent.

    Returns the `holo.raw` stack if present, else the numbered tif folder, tolerating an
    absent or empty folder as None.

    Raises:
        MultiFormatHologramsError: If the folder holds both the `holo.raw` stack and
            numbered `.tif` previews (a real acquisition produces only one).
        ValueError: If the `holo.raw` stack is present but corrupt (a bad header or a
            size that does not match it).
    """
    holo_dir = Path(root)
    if not holo_dir.is_dir():
        return None

    raw = holo_dir / _HOLOGRAM_RAW
    has_raw = raw.is_file()
    tif_folder = open_folder(holo_dir, HologramTifFolder)
    if has_raw and tif_folder is not None:
        msg = "holograms hold both a .raw stack and .tif previews (expected one)"
        raise MultiFormatHologramsError(msg)

    return HologramRawFile(raw) if has_raw else tif_folder


def search_holograms(
    root: StrPath,
    *,
    on_conflict: Literal["skip", "raise"] = "skip",
    name_filter: Filter | FilterDict | None = None,
    part_filter: Filter | FilterDict | None = None,
    predicate: Callable[[HologramSequence], bool] | None = None,
    exclude: ExcludeRule | Iterable[ExcludeRule] | None = None,
    min_depth: int = 1,
    max_depth: int | None = None,
    ordered: bool = True,
) -> list[HologramSequence]:
    """Return the holograms of every time-lapse under `root` that has a `Holograms/`.

    Finds each time-lapse folder holding a `Holograms/` and opens it; an empty folder is
    skipped, and `predicate` is a final check on the built *`HologramSequence`*. A
    `Holograms/` holding both a `.raw` stack and `.tif` previews has no single format to
    open; `on_conflict` decides whether such a time-lapse is dropped (so one malformed
    acquisition does not abort the whole scan) or aborts the search.

    Args:
        root: The directory to scan.
        on_conflict: What to do when a matched `Holograms/` holds both a `.raw` stack
            and `.tif` previews. `"skip"` (default) drops that time-lapse and warns;
            `"raise"` aborts the search.
        name_filter: Filter on each candidate time-lapse folder's own name.
        part_filter: Filter on each visited parent directory's relative path.
        predicate: A final check on the opened `HologramSequence`; None (default) keeps
            all.
        exclude: Path(s) to prune from the walk.
        min_depth: Shallowest depth to include (>= 1).
        max_depth: Deepest depth to include, or None (default) for unlimited.
        ordered: Sort the results by path. Defaults to True.

    Returns:
        The opened hologram sequences (excluding any skipped on a conflict).

    Raises:
        MultiFormatHologramsError: If a matched `Holograms/` holds both a `.raw` stack and
            `.tif` previews and `on_conflict` is `"raise"`.
        ValueError: If a matched `holo.raw` is corrupt (surfaced regardless of
            `on_conflict`: that is a content error, not the layout ambiguity).
    """
    sequences: list[HologramSequence] = []
    for directory in search_timelapse_subdirs(
        root,
        HOLOGRAMS,
        name_filter=name_filter,
        part_filter=part_filter,
        exclude=exclude,
        min_depth=min_depth,
        max_depth=max_depth,
        ordered=ordered,
    ):
        try:
            sequence = open_holograms(directory)
        except MultiFormatHologramsError:
            if on_conflict == "raise":
                raise
            warnings.warn(
                f"skipping {directory}: holograms hold both a .raw stack and .tif "
                "previews (expected one)",
                stacklevel=2,
            )
            continue
        if sequence is not None:
            sequences.append(sequence)

    if predicate is None:
        return sequences
    return [sequence for sequence in sequences if predicate(sequence)]


def search_multi_format_holograms(
    root: StrPath,
    *,
    name_filter: Filter | FilterDict | None = None,
    part_filter: Filter | FilterDict | None = None,
    exclude: ExcludeRule | Iterable[ExcludeRule] | None = None,
    min_depth: int = 1,
    max_depth: int | None = None,
    ordered: bool = True,
) -> list[Path]:
    """Return each `Holograms/` under `root` that holds both a `.raw` stack and `.tif`.

    These folders cannot be opened as a single `HologramSequence` until one of the two
    is removed, so `search_holograms(on_conflict="skip")` drops them and `open_holograms`
    raises `MultiFormatHologramsError`. The auditing counterpart to those, it returns the
    offending `Holograms/` folders themselves (their parent is the time-lapse).

    Args:
        root: The directory to scan.
        name_filter: Filter on each candidate time-lapse folder's own name.
        part_filter: Filter on each visited parent directory's relative path.
        exclude: Path(s) to prune from the walk.
        min_depth: Shallowest depth to include (>= 1).
        max_depth: Deepest depth to include, or None (default) for unlimited.
        ordered: Sort the results by path. Defaults to True.
    """
    return [
        directory
        for directory in search_timelapse_subdirs(
            root,
            HOLOGRAMS,
            name_filter=name_filter,
            part_filter=part_filter,
            exclude=exclude,
            min_depth=min_depth,
            max_depth=max_depth,
            ordered=ordered,
        )
        if (directory / _HOLOGRAM_RAW).is_file()
        and open_folder(directory, HologramTifFolder) is not None
    ]
