from __future__ import annotations

__all__ = (
    "PHASE_TREE",
    "PhaseGroup",
    "search_phase_bin_folders",
    "search_phase_folders",
    "search_phase_tif_folders",
    "search_phase_txt_folders",
)

from functools import partial
from typing import TYPE_CHECKING, Unpack

from kaparoo.utils import ensure_one_of

from iivs.dhm.data.koala import (
    PHASE,
    PHASE_FLOAT_BIN,
    PHASE_FLOAT_TXT,
    PHASE_IMAGE,
    ReconstructionGroup,
    open_timelapse_subfolders,
    reconstruction_tree,
    search_timelapse_subdirs,
)
from iivs.dhm.data.phase.bin import PhaseBinFolder
from iivs.dhm.data.phase.tif import PhaseTifFolder
from iivs.dhm.data.phase.txt import PhaseTxtFolder

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Literal

    from kaparoo.filesystem import WalkKwargs
    from kaparoo.filesystem.types import StrPath

    from iivs.dhm.data.phase.base import PhaseFileFolder
    from iivs.dhm.data.phase.unit import PhaseUnit

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
    subpath: str = PHASE_FLOAT_BIN,
    target_unit: PhaseUnit | None = None,
    predicate: Callable[[PhaseBinFolder], bool] | None = None,
    **walk: Unpack[WalkKwargs],
) -> list[PhaseBinFolder]:
    """Return the `Phase/Float/Bin` folder of each time-lapse under `root` that has one.

    A time-lapse without a non-empty `Phase/Float/Bin` is skipped, and `predicate`
    checks the opened `PhaseBinFolder`. `name_filter` matches the time-lapse folder's
    own name. `subpath` is the folder read within each time-lapse; override its
    `Phase/Float/Bin` default to read a re-encoded `.bin` source at another location
    (e.g. `FilteredPhase/Float/Bin`) with the same reader.

    `target_unit` is bound as each folder is opened, so a bulk read in one unit costs
    one construction rather than a `with_unit` rebuild per folder; None (default)
    keeps each folder's stored unit, as the constructor does.

    `**walk` is `open_timelapse_subfolders`'s `WalkKwargs`, passed through
    unchanged: `predicate` over the opened `PhaseBinFolder`s, plus the walk.
    """
    return open_timelapse_subfolders(
        root,
        subpath,
        partial(PhaseBinFolder, target_unit=target_unit),
        predicate=predicate,
        **walk,
    )


def search_phase_txt_folders(
    root: StrPath,
    *,
    subpath: str = PHASE_FLOAT_TXT,
    target_unit: PhaseUnit | None = None,
    predicate: Callable[[PhaseTxtFolder], bool] | None = None,
    **walk: Unpack[WalkKwargs],
) -> list[PhaseTxtFolder]:
    """Return the `Phase/Float/Txt` folder of each time-lapse under `root` that has one.

    The `.txt` twin of `search_phase_bin_folders`; `predicate` checks the opened
    `PhaseTxtFolder`, and `target_unit` is bound as each folder is opened. `subpath`
    (default `Phase/Float/Txt`) overrides the folder read within each time-lapse.

    `**walk` is `open_timelapse_subfolders`'s `WalkKwargs`, passed through
    unchanged.
    """
    return open_timelapse_subfolders(
        root,
        subpath,
        partial(PhaseTxtFolder, target_unit=target_unit),
        predicate=predicate,
        **walk,
    )


def search_phase_tif_folders(
    root: StrPath,
    *,
    subpath: str = PHASE_IMAGE,
    predicate: Callable[[PhaseTifFolder], bool] | None = None,
    **walk: Unpack[WalkKwargs],
) -> list[PhaseTifFolder]:
    """Return the uint8 `Phase/Image` preview folder of every time-lapse that has one.

    The preview twin of `search_phase_bin_folders`; `predicate` checks the opened
    `PhaseTifFolder`. `subpath` (default `Phase/Image`) overrides the folder read
    within each time-lapse.

    `**walk` is `open_timelapse_subfolders`'s `WalkKwargs`, passed through
    unchanged.
    """
    return open_timelapse_subfolders(
        root, subpath, PhaseTifFolder, predicate=predicate, **walk
    )


def search_phase_folders(
    root: StrPath,
    *,
    prefer: Literal["bin", "txt"] | Sequence[Literal["bin", "txt"]] = ("bin", "txt"),
    subpath: str = PHASE,
    predicate: Callable[[PhaseFileFolder], bool] | None = None,
    **walk: Unpack[WalkKwargs],
) -> list[PhaseFileFolder]:
    """Return each time-lapse's quantitative phase folder, whichever format it holds.

    The format-agnostic member of the `search_phase_*_folders` family: each time-lapse
    under `root` holding a `Phase` modality contributes its `Float` source in the first
    `prefer` format present (default: `Float/Bin` over `Float/Txt`, the
    `PhaseGroup.quantitative` preference); one holding none of them drops out. Unlike
    `phase_folder`, coexisting formats are not an error (a bulk scan must not abort on
    the common case): the preference order simply decides. The uint8 `Image` previews
    never participate (`search_phase_tif_folders` covers them), nor does `.npy`, a
    re-encoding target rather than part of the Koala layout.

    `subpath` is the modality subtree searched under each time-lapse (default
    `Phase`); override it to scan a re-exported tree (e.g. `FilteredPhase`), still
    resolving `Float/{Bin,Txt}` beneath it. `**walk` is the `WalkKwargs` set:
    `predicate` over the opened folders, plus the walk `search_timelapse_subdirs`
    passes through.

    Raises:
        ValueError: If `prefer` is empty or names a format other than bin or txt.
    """
    order = (prefer,) if isinstance(prefer, str) else tuple(prefer)
    if not order:
        msg = "prefer must name at least one format"
        raise ValueError(msg)
    for fmt in order:
        ensure_one_of(fmt, ("bin", "txt"), name="prefer")

    folders: list[PhaseFileFolder] = []
    for phase_dir in search_timelapse_subdirs(root, subpath, **walk):
        group = PhaseGroup(phase_dir)
        for fmt in order:
            folder = group.bin_folder if fmt == "bin" else group.txt_folder
            if folder is not None:
                folders.append(folder)
                break

    if predicate is None:
        return folders
    return [folder for folder in folders if predicate(folder)]
