from __future__ import annotations

__all__ = (
    "PHASE_TREE",
    "PhaseGroup",
    "search_phase_bin_folders",
    "search_phase_preview_folders",
    "search_phase_txt_folders",
)

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from kaparoo.filesystem import hierarchy

from iivs.dhm.data.koala import open_folder, search_modality_folders
from iivs.dhm.data.phase.bin import PhaseBinFolder
from iivs.dhm.data.phase.tif import PhaseTifFolder
from iivs.dhm.data.phase.txt import PhaseTxtFolder

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from kaparoo.filesystem.exclude import ExcludeRule
    from kaparoo.filesystem.types import StrPath
    from kaparoo.filters import Filter
    from kaparoo.filters.types import FilterDict

    from iivs.dhm.data.phase.base import PhaseFloatSequence

_PHASE = "Phase"
_FLOAT = "Float"
_BIN = "Bin"
_TXT = "Txt"
_IMAGE = "Image"

PHASE_TREE = hierarchy.Directory(
    _PHASE,
    [
        hierarchy.Directory(
            _FLOAT, [hierarchy.Directory(_BIN), hierarchy.Directory(_TXT)]
        ),
        hierarchy.Directory(_IMAGE),
    ],
)
"""The `Phase/` subtree of a Koala acquisition, as a `hierarchy` spec.

`Float/Bin` and `Float/Txt` are independent siblings (the same phase in two
serializations may coexist); `Image` is the uint8 preview folder.
"""


class PhaseGroup:
    """The phase modality within a Koala acquisition, opened from its `Phase/` folder.

    Opens each format present: the quantitative `Float/Bin` / `Float/Txt` sources (which
    may coexist) and the uint8 `Image` preview. Each accessor is None when its source is
    absent, and `quantitative` is the `.bin`-preferred convenience over the two floats.

    Args:
        root: The `Phase/` folder. Not required to exist: a missing one makes every
            accessor None.
    """

    def __init__(self, root: StrPath) -> None:
        self._root = Path(root)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self._root)!r})"

    @property
    def root(self) -> Path:
        """The `Phase/` folder."""
        return self._root

    @cached_property
    def float_bin(self) -> PhaseBinFolder | None:
        """The quantitative `Float/Bin` source, or None when it is absent."""
        return open_folder(self._root / _FLOAT / _BIN, PhaseBinFolder)

    @cached_property
    def float_txt(self) -> PhaseTxtFolder | None:
        """The quantitative `Float/Txt` source, or None when it is absent."""
        return open_folder(self._root / _FLOAT / _TXT, PhaseTxtFolder)

    @cached_property
    def previews(self) -> PhaseTifFolder | None:
        """The uint8 `Image` preview folder, or None when it is absent."""
        return open_folder(self._root / _IMAGE, PhaseTifFolder)

    @property
    def quantitative(self) -> PhaseFloatSequence | None:
        """The quantitative source, `Float/Bin` preferred over `Float/Txt`, or None."""
        return self.float_bin or self.float_txt

    @cached_property
    def frame_counts(self) -> dict[str, int]:
        """The frame count of each present source, keyed by accessor name."""
        sources = {
            "float_bin": self.float_bin,
            "float_txt": self.float_txt,
            "previews": self.previews,
        }
        return {name: len(seq) for name, seq in sources.items() if seq is not None}


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
        f"{_PHASE}/{_FLOAT}/{_BIN}",
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
        f"{_PHASE}/{_FLOAT}/{_TXT}",
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
        f"{_PHASE}/{_IMAGE}",
        PhaseTifFolder,
        name_filter=name_filter,
        part_filter=part_filter,
        predicate=predicate,
        exclude=exclude,
        min_depth=min_depth,
        max_depth=max_depth,
        ordered=ordered,
    )
