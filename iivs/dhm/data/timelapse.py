from __future__ import annotations

__all__ = ("KOALA_TIMELAPSE_TREE", "KoalaTimelapse", "search_timelapses")

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from kaparoo.filesystem.hierarchy import Directory, File, ValidationReport, validate
from kaparoo.filesystem.search import search_dirs
from kaparoo.filters import Glob

from iivs.common.data.timestamp import TimestampsFixedFPS
from iivs.dhm.data.hologram.layout import HOLOGRAM_TREE, open_holograms
from iivs.dhm.data.intensity.layout import INTENSITY_TREE, IntensityGroup
from iivs.dhm.data.koala import HOLOGRAMS, INTENSITY, PHASE, PHBOUNDS, TIMESTAMPS
from iivs.dhm.data.phase.bounds import read_phbounds
from iivs.dhm.data.phase.layout import PHASE_TREE, PhaseGroup
from iivs.dhm.data.timestamp import TimestampsTxtFile

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from kaparoo.filesystem.exclude import ExcludeRule
    from kaparoo.filesystem.types import StrPath
    from kaparoo.filters import Filter
    from kaparoo.filters.types import FilterDict

    from iivs.common.data.timestamp import TimestampSequence
    from iivs.dhm.data.hologram.base import HologramSequence
    from iivs.dhm.data.phase.bounds import PhaseBounds


# ============================================================ #
#                          layout                              #
# ============================================================ #

# The top-level folders that mark a directory as a Koala time-lapse (its holograms, or a
# reconstruction). `search_timelapses` treats a directory holding any of them as one.
_MARKERS = (PHASE, INTENSITY, HOLOGRAMS)

KOALA_TIMELAPSE_TREE = Directory(
    Glob("*"),  # any time-lapse-root name; matched via `root_as_top`
    [
        PHASE_TREE,
        INTENSITY_TREE,
        HOLOGRAM_TREE,
        File(TIMESTAMPS),
        File(PHBOUNDS),
    ],
)
"""The standard Lyncée Tec Koala time-lapse layout, composed from the per-modality
subtrees (`PHASE_TREE`, `INTENSITY_TREE`, `HOLOGRAM_TREE`) plus the root
`timestamps.txt` / `phbounds.txt`. `KoalaTimelapse.validate` checks a root against it.
"""


# ============================================================ #
#                          opener                              #
# ============================================================ #


class KoalaTimelapse:
    """A whole Lyncée Tec Koala time-lapse acquisition, opened from its root folder.

    Composes the per-modality groups (`phase` / `intensity`, each a `PhaseGroup` /
    `IntensityGroup`) with the holograms, timestamps, and `phbounds.txt` display bounds,
    over the standard Koala layout. Holograms, phase, and intensity are independent (any
    subset may be present); when several are, their frame counts must agree
    (`counts_agree`). Every source opens lazily on first use.

    Args:
        root: The time-lapse root (the folder holding `Phase/`, `Holograms/`,
            `timestamps.txt`, ...). Not required to exist: a missing root simply makes
            every accessor empty / None.
        frame_rate: A fallback frame rate (in fps). When `timestamps.txt` is absent,
            timing is synthesized at this rate over the acquisition's frame count; None
            (default) leaves `timestamps` None instead.
    """

    def __init__(self, root: StrPath, *, frame_rate: float | None = None) -> None:
        self._root = Path(root)
        self._frame_rate = frame_rate

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self._root)!r})"

    @property
    def root(self) -> Path:
        """The time-lapse root folder."""
        return self._root

    # -- modalities --

    @cached_property
    def phase(self) -> PhaseGroup:
        """The phase modality group (its `Float/{Bin,Txt}` sources and preview)."""
        return PhaseGroup(self._root / PHASE)

    @cached_property
    def intensity(self) -> IntensityGroup:
        """The intensity modality group (its `Float/{Bin,Txt}` sources and preview)."""
        return IntensityGroup(self._root / INTENSITY)

    @cached_property
    def holograms(self) -> HologramSequence | None:
        """The holograms: a `*.raw` stack or the numbered tif folder, or None.

        Raises:
            ValueError: If the `Holograms` folder holds both a `.raw` stack and numbered
                `.tif` previews (a real acquisition produces only one).
        """
        return open_holograms(self._root / HOLOGRAMS)

    @cached_property
    def timestamps(self) -> TimestampSequence | None:
        """The per-frame timing: `timestamps.txt`, else fixed-`frame_rate`, else None.

        Reads `timestamps.txt` when present; otherwise, with a `frame_rate` set and a
        known frame count, synthesizes evenly spaced `TimestampsFixedFPS`; else None.
        """
        path = self._root / TIMESTAMPS
        if path.is_file():
            return TimestampsTxtFile(path)

        count = self._frame_count()
        if self._frame_rate is not None and count is not None:
            return TimestampsFixedFPS(frame_rate=self._frame_rate, num_frames=count)
        return None

    @cached_property
    def phase_bounds(self) -> PhaseBounds | None:
        """The `phbounds.txt` display bounds, or None when it is absent."""
        path = self._root / PHBOUNDS
        return read_phbounds(path) if path.is_file() else None

    # -- consistency --

    @cached_property
    def frame_counts(self) -> dict[str, int]:
        """The frame count of each present source, keyed by `<modality>_<format>`.

        Merges the phase / intensity groups' own `frame_counts` (prefixed) with the
        holograms and timing. Absent sources are omitted; `counts_agree` reduces this to
        a single yes/no.
        """
        counts = {
            f"{prefix}_{name}": count
            for prefix, group in (("phase", self.phase), ("intensity", self.intensity))
            for name, count in group.frame_counts.items()
        }
        if self.holograms is not None:
            counts["holograms"] = len(self.holograms)
        if self.timestamps is not None:
            counts["timestamps"] = len(self.timestamps)
        return counts

    @property
    def counts_agree(self) -> bool:
        """Whether every present source has the same frame count.

        Vacuously True when nothing (or one source) is present; one time-lapse
        acquisition's phase, intensity, holograms, and timing must all agree.
        """
        return len(set(self.frame_counts.values())) <= 1

    @property
    def has_reconstruction(self) -> bool:
        """Whether a quantitative reconstruction (phase or intensity) is present.

        False for a holograms-only acquisition not yet reconstructed by Koala.
        """
        return (
            self.phase.quantitative is not None
            or self.intensity.quantitative is not None
        )

    @property
    def has_holograms(self) -> bool:
        """Whether the raw holograms are present."""
        return self.holograms is not None

    def validate(self) -> ValidationReport:
        """Check the root's structure against `KOALA_TIMELAPSE_TREE`.

        Lenient about extras (OME metadata, logs, ...) and optional modalities, so the
        report flags only a `Holograms` folder holding both raw and tif and, via
        `matched`, records which parts of the layout are present. The report's
        truthiness is its `ok`. This checks the layout only; use `counts_agree` and each
        modality's own `validate` for data consistency.
        """
        return validate(
            KOALA_TIMELAPSE_TREE, self._root, root_as_top=True, allow_extra=True
        )

    def _frame_count(self) -> int | None:
        """The frame count from the first present data source, or None if there is none.

        Excludes `timestamps` so it can back the synthesized `TimestampsFixedFPS`.
        """
        sources = (
            self.holograms,
            self.phase.quantitative,
            self.phase.tif_folder,
            self.intensity.quantitative,
            self.intensity.tif_folder,
        )
        for source in sources:
            if source is not None:
                return len(source)
        return None


def _looks_like_timelapse(path: Path) -> bool:
    """Whether `path` holds any Koala modality folder (`Phase` / `Intensity` / ...)."""
    return any((path / marker).is_dir() for marker in _MARKERS)


def _requirer(require: Iterable[str] | None) -> Callable[[Path], bool]:
    """Build the predicate that identifies a time-lapse directory.

    With `require=None` a directory qualifies when it holds any modality folder;
    otherwise it must hold every listed name (a modality folder like `"Phase"`, or a
    file like `"timestamps.txt"`), each checked for existence relative to the directory.
    """
    if require is None:
        return _looks_like_timelapse

    required = tuple(require)

    def matches(path: Path) -> bool:
        return all((path / name).exists() for name in required)

    return matches


def search_timelapses(
    root: StrPath,
    *,
    require: Iterable[str] | None = None,
    name_filter: Filter | FilterDict | None = None,
    part_filter: Filter | FilterDict | None = None,
    predicate: Callable[[KoalaTimelapse], bool] | None = None,
    exclude: ExcludeRule | Iterable[ExcludeRule] | None = None,
    min_depth: int = 1,
    max_depth: int | None = None,
    ordered: bool = True,
    frame_rate: float | None = None,
) -> list[KoalaTimelapse]:
    """Return a `KoalaTimelapse` for each Koala acquisition folder under `root`.

    A directory qualifies when it holds every name in `require` (a modality folder like
    `"Phase"`, or a file like `"timestamps.txt"`); with `require=None` it qualifies on
    holding any modality folder. A candidate must also pass `part_filter` (on its
    parent's relative path), `name_filter` (on the time-lapse folder's own name), and
    lie within `[min_depth, max_depth]`; `exclude` prunes subtrees. Each surviving
    directory is wrapped in a `KoalaTimelapse`, then `predicate` (a check on the
    *`KoalaTimelapse`*, not its path) filters the wrapped objects.

    Args:
        root: The directory to scan.
        require: Names that must all be present for a directory to qualify (modality
            folders and / or files). None (default) requires only any one modality.
        name_filter: Filter on each candidate time-lapse folder's own name.
        part_filter: Filter on each visited parent directory's relative path.
        predicate: A final check on the built `KoalaTimelapse`; None (default) keeps
            all.
        exclude: Path(s) to prune from the walk.
        min_depth: Shallowest depth to include (>= 1, direct children are depth 1).
        max_depth: Deepest depth to include, or None (default) for unlimited.
        ordered: Sort the results by path. Defaults to True.
        frame_rate: Passed to each `KoalaTimelapse` as its fallback frame rate.

    Returns:
        The matching time-lapses.

    Raises:
        DirectoryNotFoundError: If `root` does not exist.
    """
    directories = search_dirs(
        root,
        part_filter=part_filter,
        name_filter=name_filter,
        predicate=_requirer(require),
        exclude=exclude,
        min_depth=min_depth,
        max_depth=max_depth,
        ordered=ordered,
    )
    timelapses = (
        KoalaTimelapse(directory, frame_rate=frame_rate) for directory in directories
    )
    if predicate is None:
        return list(timelapses)
    return [timelapse for timelapse in timelapses if predicate(timelapse)]
