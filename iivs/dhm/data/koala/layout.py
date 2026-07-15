from __future__ import annotations

__all__ = (
    "ModalityGroup",
    "float_modality_tree",
    "open_folder",
    "search_modality_dirs",
    "search_modality_folders",
)

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from kaparoo.filesystem import hierarchy
from kaparoo.filesystem.search import search_dirs

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from kaparoo.filesystem.exclude import ExcludeRule
    from kaparoo.filesystem.types import StrPath
    from kaparoo.filters import Filter
    from kaparoo.filters.types import FilterDict

    from iivs.dhm.data.koala.frame import KoalaFrameFolder

_FLOAT = "Float"
_BIN = "Bin"
_TXT = "Txt"
_IMAGE = "Image"


# ============================================================ #
#                        open / search                         #
# ============================================================ #


def open_folder[T](path: StrPath, folder: Callable[..., T]) -> T | None:
    """Open `folder(path, validate=None)` when `path` is a populated directory, else None.

    The tolerant opener the modality groups and searches build on: an absent directory,
    or a `FileNotFoundError` from an empty numbered folder, becomes None instead of
    raising.

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
    delegated to `search_dirs` (no manual recursion): `name_filter` matches the
    *time-lapse* folder's name, `part_filter` its parent's relative path, and the depth /
    `exclude` / `ordered` controls carry through.

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


# ============================================================ #
#                       group / tree                           #
# ============================================================ #


def float_modality_tree(name: str) -> hierarchy.Directory:
    """The `<name>/{Float/{Bin,Txt}, Image}` subtree shared by the float32 modalities.

    `Float/Bin` and `Float/Txt` are independent siblings (the same data in two
    serializations may coexist); `Image` is the uint8 preview folder. Phase and intensity
    each build their spec from this with their own `name`.
    """
    return hierarchy.Directory(
        name,
        [
            hierarchy.Directory(
                _FLOAT, [hierarchy.Directory(_BIN), hierarchy.Directory(_TXT)]
            ),
            hierarchy.Directory(_IMAGE),
        ],
    )


class ModalityGroup[B: KoalaFrameFolder, T: KoalaFrameFolder, P: KoalaFrameFolder]:
    """A float32 modality's format folders within one time-lapse, opened from its folder.

    The shared base for the Koala float32 modality groups (phase, intensity): a
    ``<Modality>/`` folder exposing each format present, `float_bin` / `float_txt` (the
    `Float/{Bin,Txt}` sources, which may coexist) and `previews` (the uint8 `Image`
    folder), plus the `.bin`-preferred `quantitative` convenience and `frame_counts`.
    Each accessor is None when its source is absent. A subclass binds the concrete
    `KoalaFrameFolder` subtypes (its ``__init__`` passing them to `super().__init__`).

    Type Parameters:
        B: The `Float/Bin` folder type (e.g. `PhaseBinFolder`).
        T: The `Float/Txt` folder type.
        P: The `Image` preview folder type.
    """

    def __init__(
        self,
        root: StrPath,
        bin_folder: type[B],
        txt_folder: type[T],
        preview_folder: type[P],
    ) -> None:
        self._root = Path(root)
        self._bin_folder = bin_folder
        self._txt_folder = txt_folder
        self._preview_folder = preview_folder

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self._root)!r})"

    @property
    def root(self) -> Path:
        """The modality folder (e.g. `Phase/`)."""
        return self._root

    @cached_property
    def float_bin(self) -> B | None:
        """The quantitative `Float/Bin` source, or None when it is absent."""
        return open_folder(self._root / _FLOAT / _BIN, self._bin_folder)

    @cached_property
    def float_txt(self) -> T | None:
        """The quantitative `Float/Txt` source, or None when it is absent."""
        return open_folder(self._root / _FLOAT / _TXT, self._txt_folder)

    @cached_property
    def previews(self) -> P | None:
        """The uint8 `Image` preview folder, or None when it is absent."""
        return open_folder(self._root / _IMAGE, self._preview_folder)

    @property
    def quantitative(self) -> B | T | None:
        """The quantitative source, `Float/Bin` preferred over `Float/Txt`, or None."""
        return self.float_bin or self.float_txt

    @cached_property
    def frame_counts(self) -> dict[str, int]:
        """The frame count of each present source, keyed by accessor name."""
        sources: dict[str, KoalaFrameFolder | None] = {
            "float_bin": self.float_bin,
            "float_txt": self.float_txt,
            "previews": self.previews,
        }
        return {name: len(seq) for name, seq in sources.items() if seq is not None}
