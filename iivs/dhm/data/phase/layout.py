from __future__ import annotations

__all__ = (
    "PHASE_TREE",
    "PhaseGroup",
    "search_phase_bin_folders",
    "search_phase_tif_folders",
    "search_phase_txt_folders",
)

from typing import TYPE_CHECKING

from iivs.dhm.data.koala import (
    PHASE,
    PHASE_FLOAT_BIN,
    PHASE_FLOAT_TXT,
    PHASE_IMAGE,
    ReconstructionGroup,
    open_timelapse_subfolders,
    reconstruction_tree,
)
from iivs.dhm.data.phase.bin import PhaseBinFolder
from iivs.dhm.data.phase.tif import PhaseTifFolder
from iivs.dhm.data.phase.txt import PhaseTxtFolder

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from kaparoo.filesystem.exclude import ExcludeRule
    from kaparoo.filesystem.types import StrPath
    from kaparoo.filters import Filter
    from kaparoo.filters.types import FilterDict

PHASE_TREE = reconstruction_tree(PHASE)
"""The `Phase/` subtree of a Koala time-lapse (`Float/{Bin,Txt}` + `Image`)."""


class PhaseGroup(ReconstructionGroup[PhaseBinFolder, PhaseTxtFolder, PhaseTifFolder]):
    """The phase modality within a Koala time-lapse, opened from its `Phase/` folder.

    Exposes each format present: `bin_folder` / `txt_folder` (the `Float/{Bin,Txt}`
    sources, which may coexist) and `tif_folder` (the uint8 `Image` preview), plus the
    `.bin`-preferred `quantitative`, the shared `num_frames` / `frame_shape`, the
    tolerant `is_consistent` and the non-vacuous `is_usable` checks, and `validate` for
    per-file content. Each accessor is None when its source is absent.

    There is no `npy` accessor: `.npy` is a re-encoding target, not part of the Koala
    layout. To open one numbered folder by its serialization (including `.npy`), use
    `phase_folder`.

    Args:
        root: The `Phase/` folder. Not required to exist: a missing one makes every
            accessor None.
    """

    def __init__(self, root: StrPath) -> None:
        super().__init__(root, PhaseBinFolder, PhaseTxtFolder, PhaseTifFolder)


def search_phase_bin_folders(
    root: StrPath,
    *,
    name_filter: Filter | FilterDict | None = None,
    part_filter: Filter | FilterDict | None = None,
    predicate: Callable[[PhaseBinFolder], bool] | None = None,
    exclude: ExcludeRule | Iterable[ExcludeRule] | None = None,
    min_depth: int = 1,
    max_depth: int | None = None,
    ordered: bool = True,
) -> list[PhaseBinFolder]:
    """Return the `Phase/Float/Bin` folder of each time-lapse under `root` that has one.

    A time-lapse without a non-empty `Phase/Float/Bin` is skipped, and `predicate`
    checks the opened `PhaseBinFolder`. `name_filter` matches the time-lapse folder's
    own name.

    The walk itself (`part_filter`, `exclude`, `min_depth`, `max_depth`,
    `ordered`) is `open_timelapse_subfolders`'s, passed through unchanged.
    """
    return open_timelapse_subfolders(
        root,
        PHASE_FLOAT_BIN,
        PhaseBinFolder,
        name_filter=name_filter,
        part_filter=part_filter,
        predicate=predicate,
        exclude=exclude,
        min_depth=min_depth,
        max_depth=max_depth,
        ordered=ordered,
    )


def search_phase_txt_folders(
    root: StrPath,
    *,
    name_filter: Filter | FilterDict | None = None,
    part_filter: Filter | FilterDict | None = None,
    predicate: Callable[[PhaseTxtFolder], bool] | None = None,
    exclude: ExcludeRule | Iterable[ExcludeRule] | None = None,
    min_depth: int = 1,
    max_depth: int | None = None,
    ordered: bool = True,
) -> list[PhaseTxtFolder]:
    """Return the `Phase/Float/Txt` folder of each time-lapse under `root` that has one.

    The `.txt` twin of `search_phase_bin_folders`; `predicate` checks the opened
    `PhaseTxtFolder`.

    The walk itself (`part_filter`, `exclude`, `min_depth`, `max_depth`,
    `ordered`) is `open_timelapse_subfolders`'s, passed through unchanged.
    """
    return open_timelapse_subfolders(
        root,
        PHASE_FLOAT_TXT,
        PhaseTxtFolder,
        name_filter=name_filter,
        part_filter=part_filter,
        predicate=predicate,
        exclude=exclude,
        min_depth=min_depth,
        max_depth=max_depth,
        ordered=ordered,
    )


def search_phase_tif_folders(
    root: StrPath,
    *,
    name_filter: Filter | FilterDict | None = None,
    part_filter: Filter | FilterDict | None = None,
    predicate: Callable[[PhaseTifFolder], bool] | None = None,
    exclude: ExcludeRule | Iterable[ExcludeRule] | None = None,
    min_depth: int = 1,
    max_depth: int | None = None,
    ordered: bool = True,
) -> list[PhaseTifFolder]:
    """Return the uint8 `Phase/Image` preview folder of every time-lapse that has one.

    The preview twin of `search_phase_bin_folders`; `predicate` checks the opened
    `PhaseTifFolder`.

    The walk itself (`part_filter`, `exclude`, `min_depth`, `max_depth`,
    `ordered`) is `open_timelapse_subfolders`'s, passed through unchanged.
    """
    return open_timelapse_subfolders(
        root,
        PHASE_IMAGE,
        PhaseTifFolder,
        name_filter=name_filter,
        part_filter=part_filter,
        predicate=predicate,
        exclude=exclude,
        min_depth=min_depth,
        max_depth=max_depth,
        ordered=ordered,
    )
