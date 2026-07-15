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

# The names `search_timelapses`'s `require` accepts: the root-relative markers and files.
# A subfolder like `Bin` is never a root child, so it is not requirable here.
_REQUIRABLE = frozenset((*_MARKERS, TIMESTAMPS, PHBOUNDS))

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
    (`is_consistent`). Sources open lazily on first use (a modality's format folders
    together, when the modality is first accessed).

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
        """The holograms: the `holo.raw` stack or the numbered tif folder, or None.

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

    def _hologram_count(self) -> int | None:
        """The hologram frame count, or None when absent or ambiguous (raw+tif).

        Tolerant view for the count / consistency queries: `holograms` raises on the
        raw+tif conflict (`validate`'s domain), which this treats as uncountable.
        """
        try:
            holo = self.holograms
        except ValueError:
            return None
        return len(holo) if holo is not None else None

    @property
    def num_frames(self) -> int | None:
        """The acquisition's frame count, from the first present source, or None.

        Phase, intensity, holograms, and timing share this count in a coherent
        acquisition (`is_consistent`).
        """
        count = self._frame_count()
        if count is not None:
            return count
        return len(self.timestamps) if self.timestamps is not None else None

    @property
    def is_consistent(self) -> bool:
        """Whether the acquisition is coherent: each modality consistent, one length.

        Each phase / intensity group is internally consistent, and every present source
        (phase, intensity, holograms, timing) shares one frame count. Shape is not
        compared across modalities (holograms are raw interferograms); a raw+tif
        `Holograms/` conflict is `validate`'s concern, so it is left uncounted here.
        """
        if not (self.phase.is_consistent and self.intensity.is_consistent):
            return False
        counts = {
            c
            for c in (
                self.phase.num_frames,
                self.intensity.num_frames,
                self._hologram_count(),
                len(self.timestamps) if self.timestamps is not None else None,
            )
            if c is not None
        }
        return len(counts) <= 1

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
        """Whether a `Holograms/` source is present (True even if raw and tif conflict)."""
        try:
            return self.holograms is not None
        except ValueError:
            return True

    def validate(self) -> ValidationReport:
        """Check the root's structure against `KOALA_TIMELAPSE_TREE`.

        Lenient about extras (OME metadata, logs, ...) and optional modalities, so the
        report flags only a `Holograms` folder holding both raw and tif and, via
        `matched`, records which parts of the layout are present. The report's
        truthiness is its `ok`. This checks the layout only; use `is_consistent` for
        frame-count / shape agreement.
        """
        return validate(
            KOALA_TIMELAPSE_TREE, self._root, root_as_top=True, allow_extra=True
        )

    def _frame_count(self) -> int | None:
        """The frame count for synthesizing timing, from phase / intensity / holograms.

        Excludes `timestamps` so it can back the synthesized `TimestampsFixedFPS` without
        a cycle; a raw+tif `Holograms/` is uncountable (`_hologram_count`), not fatal.
        """
        for count in (
            self.phase.num_frames,
            self.intensity.num_frames,
            self._hologram_count(),
        ):
            if count is not None:
                return count
        return None


def _looks_like_timelapse(path: Path) -> bool:
    """Whether `path` holds any Koala modality folder (`Phase` / `Intensity` / ...)."""
    return any((path / marker).is_dir() for marker in _MARKERS)


def _requirer(require: Iterable[str] | None) -> Callable[[Path], bool]:
    """Build the predicate that identifies a time-lapse directory.

    With `require` None or empty a directory qualifies when it holds any modality folder;
    otherwise it must hold every listed name (a modality folder like `"Phase"`, or a
    file like `"timestamps.txt"`), each checked for existence relative to the directory.
    """
    if not require:
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
    `"Phase"`, or a file like `"timestamps.txt"`); with `require` None or empty it
    qualifies on holding any modality folder. A candidate must also pass `part_filter`
    (on its parent's relative path), `name_filter` (on the time-lapse folder's own name),
    and lie within `[min_depth, max_depth]`; `exclude` prunes subtrees. Each surviving
    directory is wrapped in a `KoalaTimelapse`, then `predicate` (a check on the
    *`KoalaTimelapse`*, not its path) filters the wrapped objects.

    Args:
        root: The directory to scan.
        require: Names that must all be present for a directory to qualify, each a
            root-level marker or file (`Phase` / `Intensity` / `Holograms` /
            `timestamps.txt` / `phbounds.txt`); an unknown name raises. None (default)
            or empty requires only any one modality.
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
        ValueError: If `require` holds a name outside the root-level markers / files.
        DirectoryNotFoundError: If `root` does not exist.
    """
    if require is not None:
        require = tuple(require)
        unknown = sorted(set(require) - _REQUIRABLE)
        if unknown:
            msg = f"unknown require name(s) {unknown}: expected {sorted(_REQUIRABLE)}"
            raise ValueError(msg)

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
