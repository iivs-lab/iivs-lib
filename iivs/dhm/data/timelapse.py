from __future__ import annotations

__all__ = ("KOALA_TIMELAPSE_TREE", "KoalaTimelapse", "search_timelapses")

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from kaparoo.filesystem.hierarchy import Directory, File
from kaparoo.filesystem.search import search_dirs
from kaparoo.filters import Any
from kaparoo.utils import fold_optional

from iivs.dhm.data.hologram.layout import (
    HOLOGRAM_TREE,
    MultiFormatHologramsError,
    open_holograms,
)
from iivs.dhm.data.intensity.layout import INTENSITY_TREE, IntensityGroup
from iivs.dhm.data.koala import HOLOGRAMS, INTENSITY, PHASE, PHBOUNDS, TIMESTAMPS
from iivs.dhm.data.koala.frame import KoalaFrameFolder
from iivs.dhm.data.phase.bounds import read_phbounds
from iivs.dhm.data.phase.layout import PHASE_TREE, PhaseGroup
from iivs.dhm.data.timestamp import TimestampsTxtFile

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from kaparoo.filesystem.exclude import ExcludeRule
    from kaparoo.filesystem.types import StrPath
    from kaparoo.filters import Filter
    from kaparoo.filters.types import FilterDict

    from iivs.common.data.timestamp import TimestampSequence
    from iivs.dhm.data.hologram.base import HologramSequence
    from iivs.dhm.data.koala.frame import ValidationLevel
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
    Any(),  # any time-lapse-root name; matched via `root_as_top`
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
`timestamps.txt` / `phbounds.txt`. Pass it to `hierarchy.validate` to check a root's
structure against it.
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
    """

    def __init__(self, root: StrPath) -> None:
        self._root = Path(root)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self._root)!r})"

    @property
    def root(self) -> Path:
        """The time-lapse root folder."""
        return self._root

    # -- sources --

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

        The per-item metadata type is left unspecified here: narrow to the concrete
        source if you need it, since a `HologramRawFile` reports a frame-index `int`
        from `get_meta` and a tif folder a source `Path`. Committing to one is a
        consumer's concern, not this composing layer's; the frames are uint8 either way.

        Raises:
            MultiFormatHologramsError: If the `Holograms` folder holds both a `.raw` stack
                and numbered `.tif` previews (a real acquisition produces only one).
        """
        return open_holograms(self._root / HOLOGRAMS)

    @cached_property
    def timestamps(self) -> TimestampSequence | None:
        """The per-frame timing read from `timestamps.txt`, or None when it is absent.

        Reflects only what is on disk. To fall back to a constant rate when the file is
        absent, synthesize it yourself: `TimestampsFixedFPS(frame_rate=r,
        num_frames=tl.num_frames)`.
        """
        path = self._root / TIMESTAMPS
        return TimestampsTxtFile(path) if path.is_file() else None

    @cached_property
    def phase_bounds(self) -> PhaseBounds | None:
        """The `phbounds.txt` display bounds, or None when it is absent."""
        path = self._root / PHBOUNDS
        return read_phbounds(path) if path.is_file() else None

    # -- counts --

    @property
    def num_frames(self) -> int | None:
        """The acquisition's frame count, from its most authoritative source, or None.

        Taken from the holograms, else the timing, else phase, else intensity: what
        Koala captured outranks what it later reconstructed, so a partial reconstruction
        does not restate the acquisition's length. All four share this count in a
        coherent acquisition (`is_consistent`); None when no source is present.
        """
        for count in self._source_counts:
            if count is not None:
                return count
        return None

    @property
    def num_holograms(self) -> int | None:
        """The hologram frame count, or None when absent or split across formats.

        Tolerant of the raw+tif layout fault only: `holograms` raises
        `MultiFormatHologramsError` on it, which this reports as None (no one format to
        count). A genuinely corrupt `holo.raw` is not swallowed; its error propagates.
        """
        try:
            holo = self.holograms
        except MultiFormatHologramsError:
            return None
        return fold_optional(holo, len, None)

    @property
    def num_timestamps(self) -> int | None:
        """The number of `timestamps.txt` timing rows, or None when it is absent."""
        return fold_optional(self.timestamps, len, None)

    @property
    def _source_counts(self) -> Iterator[int | None]:
        """Each source's frame count in `(holograms, timing, phase, intensity)` order.

        Ordered by authority over the acquisition's length: what Koala captured
        (`Holograms/`, then the `timestamps.txt` written beside it) outranks what it
        later reconstructed. Yielded lazily, so `num_frames` stops at the first present
        source instead of reading the rest; `is_consistent` draws the whole run and
        checks they agree. Each is None when its source is absent (or, for the
        holograms, split across formats).
        """
        yield self.num_holograms
        yield self.num_timestamps
        yield self.phase.num_frames
        yield self.intensity.num_frames

    # -- status --

    @property
    def has_holograms(self) -> bool:
        """Whether a `Holograms/` source is present (True even if raw and tif conflict)."""
        try:
            return self.holograms is not None
        except MultiFormatHologramsError:
            return True

    @property
    def has_quantitative_phase(self) -> bool:
        """Whether a quantitative phase reconstruction (`Float/{Bin,Txt}`) is present.

        Distinct from the uint8 `Image` preview, which Koala also produces.
        """
        return self.phase.quantitative is not None

    @property
    def has_quantitative_intensity(self) -> bool:
        """Whether a quantitative intensity reconstruction (`Float/{Bin,Txt}`) is present.

        Distinct from the uint8 `Image` preview, which Koala also produces.
        """
        return self.intensity.quantitative is not None

    @property
    def is_reconstructable(self) -> bool:
        """Whether Koala could reconstruct this acquisition.

        True when the holograms and `timestamps.txt` are both present and share a frame
        count (Koala's precondition for a reconstruction). A raw+tif `Holograms/` counts
        as absent (uncountable). Independent of whether a reconstruction output already
        exists (`has_quantitative_phase` / `has_quantitative_intensity`).
        """
        holo = self.num_holograms
        if holo is None:
            return False
        return holo == self.num_timestamps

    @property
    def is_consistent(self) -> bool:
        """Whether the acquisition is coherent: each modality consistent, one length.

        Each phase / intensity group is internally consistent, and every present source
        (phase, intensity, holograms, timing) shares one frame count. Shape is not
        compared across modalities (holograms are raw interferograms); a raw+tif
        `Holograms/` leaves no one format to count (the `holograms` accessor raises), so
        the holograms sit out this check.
        """
        if not (self.phase.is_consistent and self.intensity.is_consistent):
            return False
        counts = {c for c in self._source_counts if c is not None}
        return len(counts) <= 1

    # -- validation --

    def validate(self, *, level: ValidationLevel | None = None) -> None:
        """Content-validate every present source to `level`; raise on the first bad file.

        Checks each present modality (phase / intensity, and a tif hologram folder),
        skipping any format that lacks `level` (so `"headers"` covers only the
        `Float/{Bin,Txt}` sources). A raw+tif `Holograms/` conflict and a corrupt
        `holo.raw` surface here too. At `level="data"` the aux `timestamps.txt` /
        `phbounds.txt` are parsed as well; at shallower levels they are checked only when
        otherwise read.

        `level=None` (default) checks each source to its own depth; `"names"` /
        `"headers"` / `"data"` apply to every source that supports them. This is file
        *content*: use `is_consistent` for frame-count / shape agreement, and
        `hierarchy.validate(KOALA_TIMELAPSE_TREE, root)` for a structural report.

        Raises:
            MultiFormatHologramsError: If the `Holograms/` folder holds both a `.raw`
                stack and `.tif` previews.
            ValueError: If a file fails validation.
        """
        self.phase.validate(level=level)
        self.intensity.validate(level=level)

        holo = self.holograms  # opening validates a raw stack; raw+tif conflict raises
        if isinstance(holo, KoalaFrameFolder):
            holo.validate_if_supported(level=level)  # a tif folder is lazy

        if level == "data":
            _ = self.timestamps  # opening parses / validates timestamps.txt
            _ = self.phase_bounds  # opening parses / validates phbounds.txt


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
    timelapses = (KoalaTimelapse(directory) for directory in directories)
    if predicate is None:
        return list(timelapses)
    return [timelapse for timelapse in timelapses if predicate(timelapse)]
