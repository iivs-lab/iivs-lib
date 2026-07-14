from __future__ import annotations

__all__ = (
    "INTENSITY_TREE",
    "IntensityGroup",
    "search_intensity_bin_folders",
    "search_intensity_preview_folders",
    "search_intensity_txt_folders",
)

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from kaparoo.filesystem import hierarchy

from iivs.dhm.data.intensity.bin import IntensityBinFolder
from iivs.dhm.data.intensity.tif import IntensityTifFolder
from iivs.dhm.data.intensity.txt import IntensityTxtFolder
from iivs.dhm.data.koala import open_folder, search_modality_folders

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from kaparoo.filesystem.exclude import ExcludeRule
    from kaparoo.filesystem.types import StrPath
    from kaparoo.filters import Filter
    from kaparoo.filters.types import FilterDict

    from iivs.dhm.data.intensity.base import IntensityFloatSequence

_INTENSITY = "Intensity"
_FLOAT = "Float"
_BIN = "Bin"
_TXT = "Txt"
_IMAGE = "Image"

INTENSITY_TREE = hierarchy.Directory(
    _INTENSITY,
    [
        hierarchy.Directory(
            _FLOAT, [hierarchy.Directory(_BIN), hierarchy.Directory(_TXT)]
        ),
        hierarchy.Directory(_IMAGE),
    ],
)
"""The `Intensity/` subtree of a Koala acquisition, as a `hierarchy` spec.

`Float/Bin` and `Float/Txt` are independent siblings (the same intensity in two
serializations may coexist); `Image` is the uint8 preview folder.
"""


class IntensityGroup:
    """The intensity modality within a Koala acquisition, from its `Intensity/` folder.

    Opens each format present: the quantitative `Float/Bin` / `Float/Txt` sources (which
    may coexist) and the uint8 `Image` preview. Each accessor is None when its source is
    absent, and `quantitative` is the `.bin`-preferred convenience over the two floats.

    Args:
        root: The `Intensity/` folder. Not required to exist: a missing one makes every
            accessor None.
    """

    def __init__(self, root: StrPath) -> None:
        self._root = Path(root)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self._root)!r})"

    @property
    def root(self) -> Path:
        """The `Intensity/` folder."""
        return self._root

    @cached_property
    def float_bin(self) -> IntensityBinFolder | None:
        """The quantitative `Float/Bin` source, or None when it is absent."""
        return open_folder(self._root / _FLOAT / _BIN, IntensityBinFolder)

    @cached_property
    def float_txt(self) -> IntensityTxtFolder | None:
        """The quantitative `Float/Txt` source, or None when it is absent."""
        return open_folder(self._root / _FLOAT / _TXT, IntensityTxtFolder)

    @cached_property
    def previews(self) -> IntensityTifFolder | None:
        """The uint8 `Image` preview folder, or None when it is absent."""
        return open_folder(self._root / _IMAGE, IntensityTifFolder)

    @property
    def quantitative(self) -> IntensityFloatSequence | None:
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


def search_intensity_bin_folders(
    root: StrPath,
    *,
    name_filter: Filter | FilterDict | None = None,
    part_filter: Filter | FilterDict | None = None,
    predicate: Callable[[IntensityBinFolder], bool] | None = None,
    exclude: ExcludeRule | Iterable[ExcludeRule] | None = None,
    min_depth: int = 1,
    max_depth: int | None = None,
    ordered: bool = True,
) -> list[IntensityBinFolder]:
    """Return the `Intensity/Float/Bin` folder of every time-lapse under `root` with one.

    Delegates to `search_modality_folders` (no manual recursion); a time-lapse without a
    non-empty `Intensity/Float/Bin` is skipped, and `predicate` checks the opened
    `IntensityBinFolder`. `name_filter` matches the time-lapse folder's name; the other
    `search_dirs` controls carry through.
    """
    return search_modality_folders(
        root,
        f"{_INTENSITY}/{_FLOAT}/{_BIN}",
        IntensityBinFolder,
        name_filter=name_filter,
        part_filter=part_filter,
        predicate=predicate,
        exclude=exclude,
        min_depth=min_depth,
        max_depth=max_depth,
        ordered=ordered,
    )


def search_intensity_txt_folders(
    root: StrPath,
    *,
    name_filter: Filter | FilterDict | None = None,
    part_filter: Filter | FilterDict | None = None,
    predicate: Callable[[IntensityTxtFolder], bool] | None = None,
    exclude: ExcludeRule | Iterable[ExcludeRule] | None = None,
    min_depth: int = 1,
    max_depth: int | None = None,
    ordered: bool = True,
) -> list[IntensityTxtFolder]:
    """Return the `Intensity/Float/Txt` folder of every time-lapse under `root` with one.

    The `.txt` twin of `search_intensity_bin_folders`; `predicate` checks the opened
    `IntensityTxtFolder`.
    """
    return search_modality_folders(
        root,
        f"{_INTENSITY}/{_FLOAT}/{_TXT}",
        IntensityTxtFolder,
        name_filter=name_filter,
        part_filter=part_filter,
        predicate=predicate,
        exclude=exclude,
        min_depth=min_depth,
        max_depth=max_depth,
        ordered=ordered,
    )


def search_intensity_preview_folders(
    root: StrPath,
    *,
    name_filter: Filter | FilterDict | None = None,
    part_filter: Filter | FilterDict | None = None,
    predicate: Callable[[IntensityTifFolder], bool] | None = None,
    exclude: ExcludeRule | Iterable[ExcludeRule] | None = None,
    min_depth: int = 1,
    max_depth: int | None = None,
    ordered: bool = True,
) -> list[IntensityTifFolder]:
    """Return the uint8 `Intensity/Image` preview folder of every time-lapse with one.

    The preview twin of `search_intensity_bin_folders`; `predicate` checks the opened
    `IntensityTifFolder`.
    """
    return search_modality_folders(
        root,
        f"{_INTENSITY}/{_IMAGE}",
        IntensityTifFolder,
        name_filter=name_filter,
        part_filter=part_filter,
        predicate=predicate,
        exclude=exclude,
        min_depth=min_depth,
        max_depth=max_depth,
        ordered=ordered,
    )
