from __future__ import annotations

__all__ = (
    "INTENSITY_FLOAT_BIN",
    "INTENSITY_FLOAT_TXT",
    "INTENSITY_IMAGE",
    "INTENSITY_TREE",
    "IntensityGroup",
    "search_intensity_bin_folders",
    "search_intensity_preview_folders",
    "search_intensity_txt_folders",
)

from typing import TYPE_CHECKING

from iivs.dhm.data.intensity.bin import IntensityBinFolder
from iivs.dhm.data.intensity.tif import IntensityTifFolder
from iivs.dhm.data.intensity.txt import IntensityTxtFolder
from iivs.dhm.data.koala import (
    BIN,
    FLOAT,
    IMAGE,
    INTENSITY,
    TXT,
    ModalityGroup,
    float_modality_tree,
    search_modality_folders,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from kaparoo.filesystem.exclude import ExcludeRule
    from kaparoo.filesystem.types import StrPath
    from kaparoo.filters import Filter
    from kaparoo.filters.types import FilterDict

INTENSITY_FLOAT_BIN = f"{INTENSITY}/{FLOAT}/{BIN}"
"""The `Intensity/Float/Bin` folder's time-lapse-relative path."""
INTENSITY_FLOAT_TXT = f"{INTENSITY}/{FLOAT}/{TXT}"
"""The `Intensity/Float/Txt` folder's time-lapse-relative path."""
INTENSITY_IMAGE = f"{INTENSITY}/{IMAGE}"
"""The `Intensity/Image` preview folder's time-lapse-relative path."""

INTENSITY_TREE = float_modality_tree(INTENSITY)
"""The `Intensity/` subtree of a Koala time-lapse (`Float/{Bin,Txt}` + `Image`), a spec."""


class IntensityGroup(
    ModalityGroup[IntensityBinFolder, IntensityTxtFolder, IntensityTifFolder]
):
    """The intensity modality within a Koala time-lapse, from its `Intensity/` folder.

    Exposes each format present: `float_bin` / `float_txt` (the `Float/{Bin,Txt}` sources,
    which may coexist) and `previews` (the uint8 `Image` folder), plus the `.bin`-preferred
    `quantitative` and `frame_counts`. Each accessor is None when its source is absent.

    Args:
        root: The `Intensity/` folder. Not required to exist: a missing one makes every
            accessor None.
    """

    def __init__(self, root: StrPath) -> None:
        super().__init__(
            root, IntensityBinFolder, IntensityTxtFolder, IntensityTifFolder
        )


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
        INTENSITY_FLOAT_BIN,
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
        INTENSITY_FLOAT_TXT,
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
        INTENSITY_IMAGE,
        IntensityTifFolder,
        name_filter=name_filter,
        part_filter=part_filter,
        predicate=predicate,
        exclude=exclude,
        min_depth=min_depth,
        max_depth=max_depth,
        ordered=ordered,
    )
