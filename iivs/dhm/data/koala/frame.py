from __future__ import annotations

__all__ = (
    "KoalaFrameFolder",
    "ValidationLevel",
    "detect_koala_format",
    "koala_frame_name",
    "open_folder",
    "search_modality_dirs",
)

from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, override

from kaparoo.data.sequences import FileFolderSequence
from kaparoo.filesystem import file_extension
from kaparoo.filesystem.search import search_dirs, search_files
from kaparoo.filters import Regex
from kaparoo.utils import ensure_one_of, replace_if_none

from iivs.common.data.mixin import FrameShapedMixin

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from typing import Literal

    from kaparoo.filesystem.exclude import ExcludeRule
    from kaparoo.filesystem.types import StrPath
    from kaparoo.filters import Filter
    from kaparoo.filters.types import FilterDict


type ValidationLevel = Literal["names", "headers", "data"]
"""How deeply a numbered folder checks each file: name, header, or full data."""


@cache
def _koala_frame_filter(stem: str, exts: tuple[str, ...]) -> Regex:
    """A cached name filter for ``{index:05d}_{stem}.<ext>`` files.

    Built once per ``(stem, exts)``, since both are fixed by the folder type; shared by
    `KoalaFrameFolder.list_files` (one `ext`) and `detect_koala_format` (several).
    """
    return Regex(rf"\d{{5}}_{stem}\.({'|'.join(exts)})")


def koala_frame_name(index: int, *, stem: str, ext: str) -> str:
    """The contiguous Koala filename ``{index:05d}_{stem}.{ext}``.

    The single source of truth for Koala's frame-naming convention, used both to
    discover/validate a `KoalaFrameFolder` and to write a converted folder. The
    5-digit zero-padded field caps a folder at 100000 frames; discovery matches
    exactly `\\d{5}`, so a 6-digit name would be silently undiscoverable, hence an
    out-of-range index is rejected here rather than written unreadable.

    Raises:
        ValueError: If `index` is negative or exceeds 99999 (the 5-digit field's max).
    """
    if not 0 <= index <= 99999:  # the 5-digit field's inclusive span
        msg = f"frame index must be in [0, 99999] (got {index})"
        raise ValueError(msg)
    return f"{index:05d}_{stem}.{ext}"


def open_folder[T](path: StrPath, folder: Callable[..., T]) -> T | None:
    """Open `folder(path, validate=None)` when `path` is a populated directory, else None.

    The tolerant opener the modality groups build on: an absent directory, or a
    `FileNotFoundError` from an empty numbered folder, becomes None instead of raising.

    Type Parameters:
        T: The opened folder type (e.g. `PhaseBinFolder`).
    """
    path = Path(path)
    if path.is_dir():
        try:
            return folder(path, validate=None)
        except FileNotFoundError:
            return None
    return None


def search_modality_dirs(
    root: StrPath,
    folder: str,
    *,
    name_filter: Filter | FilterDict | None = None,
    part_filter: Filter | FilterDict | None = None,
    exclude: ExcludeRule | Iterable[ExcludeRule] | None = None,
    min_depth: int = 1,
    max_depth: int | None = None,
    ordered: bool = True,
) -> list[Path]:
    """Return each `<time-lapse>/<folder>` path under `root`, via `search_dirs`.

    Finds the time-lapse directories that hold a `folder` subdirectory (e.g. `Phase`) and
    returns that subdirectory for each, so a modality search can wrap it. The walk is
    delegated to `search_dirs` (no manual recursion): `name_filter` matches the *time-lapse*
    folder's name, `part_filter` its parent's relative path, and the depth / `exclude` /
    `ordered` controls carry through.

    Args:
        root: The directory to scan.
        folder: The modality subdirectory each time-lapse must hold (e.g. "Phase").
        name_filter: Filter on each candidate time-lapse folder's own name.
        part_filter: Filter on each visited parent directory's relative path.
        exclude: Path(s) to prune, as in `search_dirs`.
        min_depth: Shallowest depth to include (>= 1).
        max_depth: Deepest depth to include, or None for unlimited.
        ordered: Sort the results by path. Defaults to True.

    Returns:
        The `<time-lapse>/<folder>` directories, in `search_dirs` order.
    """
    dirs = search_dirs(
        root,
        name_filter=name_filter,
        part_filter=part_filter,
        predicate=lambda path: (path / folder).is_dir(),
        exclude=exclude,
        min_depth=min_depth,
        max_depth=max_depth,
        ordered=ordered,
    )
    return [Path(directory) / folder for directory in dirs]


def search_modality_folders[T](
    root: StrPath,
    subpath: str,
    folder: Callable[..., T],
    *,
    name_filter: Filter | FilterDict | None = None,
    part_filter: Filter | FilterDict | None = None,
    predicate: Callable[[T], bool] | None = None,
    exclude: ExcludeRule | Iterable[ExcludeRule] | None = None,
    min_depth: int = 1,
    max_depth: int | None = None,
    ordered: bool = True,
) -> list[T]:
    """Open each `<time-lapse>/<subpath>` folder under `root`, skipping absent / empty ones.

    Finds the time-lapses holding `subpath` (a relative folder like ``"Phase/Float/Bin"``)
    via `search_modality_dirs`, opens each with `open_folder` (so an empty one drops out
    as None), and keeps the survivors passing `predicate` (a check on the opened folder).
    The `search_dirs` filters (`name_filter` on the time-lapse folder's name, ...) carry
    through.

    Type Parameters:
        T: The opened folder type (e.g. `PhaseBinFolder`).
    """
    opened = (
        open_folder(directory, folder)
        for directory in search_modality_dirs(
            root,
            subpath,
            name_filter=name_filter,
            part_filter=part_filter,
            exclude=exclude,
            min_depth=min_depth,
            max_depth=max_depth,
            ordered=ordered,
        )
    )
    present = [item for item in opened if item is not None]
    if predicate is None:
        return present
    return [item for item in present if predicate(item)]


def detect_koala_format(
    root: StrPath,
    *,
    stem: str,
    formats: Sequence[str],
    prefer: str | Sequence[str] | None = None,
) -> str:
    """Return which of `formats` the ``{index:05d}_{stem}.<ext>`` files in `root` use.

    Scans `root` at depth 1 for the numbered ``{stem}`` files with `search_files` and a
    `Regex`, then resolves a single format. When more than one format is present,
    `prefer` decides, mirroring `kaparoo`'s `hierarchy.Exclusive(on_conflict=...)`:

    - `None`: raise (the conflict is an error; the caller must disambiguate).
    - a format, or a priority sequence of formats: pick the first present
      format in that order (the `"priority"` resolution).

    Args:
        root: The folder to scan.
        stem: The ``<stem>`` in ``{index:05d}_<stem>.<ext>`` (e.g. "phase").
        formats: The candidate extensions, in their natural order.
        prefer: The conflict policy; `None` to error on multiple formats, or a format /
            priority sequence to pick the first present one.

    Raises:
        FileNotFoundError: If `root` holds no ``{NNNNN}_{stem}.<format>`` files.
        ValueError: If multiple formats are present and `prefer` is `None`, or `prefer`
            is given but selects none of the present formats.
    """
    koala_filter = _koala_frame_filter(stem, tuple(formats))
    hits = search_files(root, name_filter=koala_filter, max_depth=1)
    found = {file_extension(hit) for hit in hits}
    present = [fmt for fmt in formats if fmt in found]

    if not present:
        alternation = "|".join(formats)
        msg = f"no NNNNN_{stem}.({alternation}) files found in {root}"
        raise FileNotFoundError(msg)
    if len(present) == 1:
        return present[0]

    if prefer is None:
        msg = f"ambiguous: multiple {stem} formats in {root} ({present}); pass prefer"
        raise ValueError(msg)

    order = [prefer] if isinstance(prefer, str) else list(prefer)
    for fmt in order:
        if fmt in present:
            return fmt
    msg = f"prefer={order} selects none of the present {stem} formats ({present})"
    raise ValueError(msg)


class KoalaFrameFolder[T](FileFolderSequence[T, Path], FrameShapedMixin):
    """A folder of contiguously numbered `{index:05d}_<stem>.<ext>` files.

    Each such folder is one acquisition's frames, hence same-shape; subclasses implement
    `frame_shape` from their header or first file. Factors the discovery and validation
    shared by every modality folder (phase, intensity, hologram, ...). A subclass
    declares the filename parts and validation depth as class attributes, implements
    `load_file`, and supplies its per-format consistency check by overriding
    `_validate_content`.

    Class attributes:
        FILE_STEM: The ``<stem>`` in ``{index:05d}_<stem>.<ext>`` (e.g. "phase").
        FILE_EXT: The file extension without the dot (e.g. "bin").
        LEVELS: The validation levels this folder accepts (a subset of "names" /
            "headers" / "data").
        DEFAULT_LEVEL: The level `validate` / `validate_file` use when given none.
    """

    FILE_STEM: ClassVar[str]
    FILE_EXT: ClassVar[str]
    LEVELS: ClassVar[tuple[str, ...]] = ("names",)
    DEFAULT_LEVEL: ClassVar[str] = "names"

    @override
    def list_files(self, root: Path) -> list[Path]:
        """List the `NNNNN_<stem>.<ext>` files under `root`, in index order.

        `search_files` sorts lexicographically by default, which (with the fixed-width
        ``{index:05d}`` zero-padding) is exactly numeric index order, so no extra sort
        is needed.
        """
        name_filter = _koala_frame_filter(self.FILE_STEM, (self.FILE_EXT,))
        files = search_files(root, name_filter=name_filter, max_depth=1)
        if not files:
            msg = f"no NNNNN_{self.FILE_STEM}.{self.FILE_EXT} files found in {root}"
            raise FileNotFoundError(msg)
        return files

    @override
    def get_meta(self, index: int) -> Path:
        """Return the source path of the file at `index`."""
        return self.get_file(index)

    def validate(self, *, level: ValidationLevel | None = None) -> None:
        """Validate every file to `level` (defaults to `DEFAULT_LEVEL`)."""
        for index in range(len(self)):
            self.validate_file(index, level=level)

    def validate_file(
        self,
        index: int,
        *,
        level: ValidationLevel | None = None,
    ) -> None:
        """Validate the file at `index` to `level` (defaults to `DEFAULT_LEVEL`).

        Always checks the contiguous name; any deeper level defers to the per-format
        `_validate_content`.

        Raises:
            ValueError: If `level` is unsupported, the numbering is non-contiguous, or
                `_validate_content` rejects the file.
        """
        resolved = replace_if_none(level, self.DEFAULT_LEVEL)
        resolved = ensure_one_of(resolved, self.LEVELS, name="level")

        path = self.get_file(index)
        expected = koala_frame_name(index, stem=self.FILE_STEM, ext=self.FILE_EXT)
        if (name := path.name) != expected:
            msg = f"non-contiguous: expected {expected} at index {index}, got {name}"
            raise ValueError(msg)

        if resolved != "names":
            self._validate_content(path, level=resolved)

    def _validate_content(self, path: Path, *, level: str) -> None:
        """Per-format consistency check for levels beyond "names" (subclass hook)."""
        raise NotImplementedError
