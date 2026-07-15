from __future__ import annotations

__all__ = (
    "PHASE_TREE",
    "PhaseGroup",
    "search_phase_bin_folders",
    "search_phase_preview_folders",
    "search_phase_txt_folders",
)

from typing import TYPE_CHECKING

from iivs.dhm.data.koala import (
    ModalityGroup,
    float_modality_tree,
    search_modality_folders,
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

_PHASE = "Phase"
_BIN_SUBPATH = f"{_PHASE}/Float/Bin"
_TXT_SUBPATH = f"{_PHASE}/Float/Txt"
_PREVIEW_SUBPATH = f"{_PHASE}/Image"

PHASE_TREE = float_modality_tree(_PHASE)
"""The `Phase/` subtree of a Koala time-lapse (`Float/{Bin,Txt}` + `Image`), a spec."""


class PhaseGroup(ModalityGroup[PhaseBinFolder, PhaseTxtFolder, PhaseTifFolder]):
    """The phase modality within a Koala time-lapse, opened from its `Phase/` folder.

    Exposes each format present: `float_bin` / `float_txt` (the `Float/{Bin,Txt}` sources,
    which may coexist) and `previews` (the uint8 `Image` folder), plus the `.bin`-preferred
    `quantitative` and `frame_counts`. Each accessor is None when its source is absent.

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
    """Return the `Phase/Float/Bin` folder of every time-lapse under `root` that has one.

    Delegates to `search_modality_folders` (no manual recursion); a time-lapse without a
    non-empty `Phase/Float/Bin` is skipped, and `predicate` checks the opened
    `PhaseBinFolder`. `name_filter` matches the time-lapse folder's name; the other
    `search_dirs` controls carry through.
    """
    return search_modality_folders(
        root,
        _BIN_SUBPATH,
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
    """Return the `Phase/Float/Txt` folder of every time-lapse under `root` that has one.

    The `.txt` twin of `search_phase_bin_folders`; `predicate` checks the opened
    `PhaseTxtFolder`.
    """
    return search_modality_folders(
        root,
        _TXT_SUBPATH,
        PhaseTxtFolder,
        name_filter=name_filter,
        part_filter=part_filter,
        predicate=predicate,
        exclude=exclude,
        min_depth=min_depth,
        max_depth=max_depth,
        ordered=ordered,
    )


def search_phase_preview_folders(
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
    """
    return search_modality_folders(
        root,
        _PREVIEW_SUBPATH,
        PhaseTifFolder,
        name_filter=name_filter,
        part_filter=part_filter,
        predicate=predicate,
        exclude=exclude,
        min_depth=min_depth,
        max_depth=max_depth,
        ordered=ordered,
    )
