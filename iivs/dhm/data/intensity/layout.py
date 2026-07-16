from __future__ import annotations

__all__ = (
    "INTENSITY_TREE",
    "IntensityGroup",
    "search_intensity_bin_folders",
    "search_intensity_tif_folders",
    "search_intensity_txt_folders",
)

from typing import TYPE_CHECKING

from iivs.dhm.data.intensity.bin import IntensityBinFolder
from iivs.dhm.data.intensity.tif import IntensityTifFolder
from iivs.dhm.data.intensity.txt import IntensityTxtFolder
from iivs.dhm.data.koala import (
    INTENSITY,
    INTENSITY_FLOAT_BIN,
    INTENSITY_FLOAT_TXT,
    INTENSITY_IMAGE,
    ReconstructionGroup,
    open_timelapse_subfolders,
    reconstruction_tree,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from kaparoo.filesystem.exclude import ExcludeRule
    from kaparoo.filesystem.types import StrPath
    from kaparoo.filters import Filter
    from kaparoo.filters.types import FilterDict

INTENSITY_TREE = reconstruction_tree(INTENSITY)
"""The `Intensity/` subtree of a Koala time-lapse (`Float/{Bin,Txt}` + `Image`)."""


class IntensityGroup(
    ReconstructionGroup[IntensityBinFolder, IntensityTxtFolder, IntensityTifFolder]
):
    """The intensity modality within a Koala time-lapse, from its `Intensity/` folder.

    Exposes each format present: `bin_folder` / `txt_folder` (the `Float/{Bin,Txt}`
    sources, which may coexist) and `tif_folder` (the uint8 `Image` preview), plus the
    `.bin`-preferred `quantitative`, the shared `num_frames` / `frame_shape`, the
    tolerant `is_consistent` and the non-vacuous `is_usable` checks, and `validate` for
    per-file content. Each accessor is None when its source is absent.

    There is no `npy` accessor: `.npy` is a re-encoding target, not part of the Koala
    layout. To open one numbered folder by its serialization (including `.npy`), use
    `intensity_folder`.

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
    """Return the `Intensity/Float/Bin` folder of each time-lapse under `root` with one.

    A time-lapse without a non-empty `Intensity/Float/Bin` is skipped, and `predicate`
    checks the opened `IntensityBinFolder`. `name_filter` matches the time-lapse
    folder's own name.

    The walk itself (`part_filter`, `exclude`, `min_depth`, `max_depth`,
    `ordered`) is `open_timelapse_subfolders`'s, passed through unchanged.
    """
    return open_timelapse_subfolders(
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
    """Return the `Intensity/Float/Txt` folder of each time-lapse under `root` with one.

    The `.txt` twin of `search_intensity_bin_folders`; `predicate` checks the opened
    `IntensityTxtFolder`.

    The walk itself (`part_filter`, `exclude`, `min_depth`, `max_depth`,
    `ordered`) is `open_timelapse_subfolders`'s, passed through unchanged.
    """
    return open_timelapse_subfolders(
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


def search_intensity_tif_folders(
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

    The walk itself (`part_filter`, `exclude`, `min_depth`, `max_depth`,
    `ordered`) is `open_timelapse_subfolders`'s, passed through unchanged.
    """
    return open_timelapse_subfolders(
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
