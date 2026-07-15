from __future__ import annotations

__all__ = (
    "ReconstructionGroup",
    "open_folder",
    "reconstruction_tree",
    "search_timelapse_subdirs",
    "search_timelapse_subfolders",
)

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from kaparoo.filesystem.hierarchy import Directory
from kaparoo.filesystem.search import search_dirs

from iivs.dhm.data.koala.constants import BIN, FLOAT, IMAGE, TXT

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from kaparoo.filesystem.exclude import ExcludeRule
    from kaparoo.filesystem.types import StrPath
    from kaparoo.filters import Filter
    from kaparoo.filters.types import FilterDict

    from iivs.dhm.data.koala.float import KoalaFloatFileFolder
    from iivs.dhm.data.koala.frame import KoalaFrameFolder
    from iivs.dhm.data.koala.image import ImageTifFolder


# ============================================================ #
#                        open / search                         #
# ============================================================ #


def open_folder[T](path: StrPath, folder: Callable[..., T]) -> T | None:
    """Open `path` with `folder` when it is a populated directory, else None.

    Tolerant: an absent directory, or a present-but-empty numbered folder, yields None
    rather than raising.

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


def search_timelapse_subdirs(
    root: StrPath,
    subpath: str,
    *,
    name_filter: Filter | FilterDict | None = None,
    part_filter: Filter | FilterDict | None = None,
    exclude: ExcludeRule | Iterable[ExcludeRule] | None = None,
    min_depth: int = 1,
    max_depth: int | None = None,
    ordered: bool = True,
) -> list[Path]:
    """Return each `<time-lapse>/<subpath>` path under `root`.

    Finds each time-lapse directory holding `subpath` (a relative folder like `Phase` or
    `Phase/Float/Bin`) and returns that subfolder. `name_filter` matches the
    *time-lapse* folder's own name and `part_filter` its parent's relative path.

    Args:
        root: The directory to scan.
        subpath: The relative subfolder each time-lapse must hold (e.g. "Phase" or
            "Phase/Float/Bin").
        name_filter: Filter on each candidate time-lapse folder's own name.
        part_filter: Filter on each visited parent directory's relative path.
        exclude: Path(s) to prune from the walk.
        min_depth: Shallowest depth to include (>= 1).
        max_depth: Deepest depth to include, or None for unlimited.
        ordered: Sort the results by path. Defaults to True.

    Returns:
        The matching `<time-lapse>/<subpath>` directories.
    """
    dirs = search_dirs(
        root,
        name_filter=name_filter,
        part_filter=part_filter,
        predicate=lambda path: (path / subpath).is_dir(),
        exclude=exclude,
        min_depth=min_depth,
        max_depth=max_depth,
        ordered=ordered,
    )
    return [Path(directory) / subpath for directory in dirs]


def search_timelapse_subfolders[T](
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
    """Open each `<time-lapse>/<subpath>` folder under `root`.

    Finds the time-lapses holding `subpath` (a relative path like
    ``"Phase/Float/Bin"``), opens each (an empty one drops out as None), then keeps
    those passing `predicate` (a check on the opened folder). `name_filter` matches the
    time-lapse folder's own name.

    Type Parameters:
        T: The opened folder type (e.g. `PhaseBinFolder`).
    """
    opened = (
        open_folder(directory, folder)
        for directory in search_timelapse_subdirs(
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


def reconstruction_tree(name: str) -> Directory:
    """The `<name>/{Float/{Bin,Txt}, Image}` subtree shared by the reconstructions.

    The layout of a Koala reconstruction modality (phase, intensity), as opposed to the
    raw `Holograms`. `Float/Bin` and `Float/Txt` are independent siblings (the same data
    in two serializations may coexist); `Image` is the uint8 preview folder. Phase and
    intensity each build their spec from this with their own `name`.
    """
    return Directory(
        name,
        [
            Directory(FLOAT, [Directory(BIN), Directory(TXT)]),
            Directory(IMAGE),
        ],
    )


class ReconstructionGroup[
    B: KoalaFloatFileFolder,
    T: KoalaFloatFileFolder,
    P: ImageTifFolder,
]:
    """A reconstruction's format folders within one time-lapse, opened from its folder.

    The shared base for the Koala reconstruction groups (phase, intensity): a
    ``<Modality>/`` folder exposing each format present: `bin_folder` / `txt_folder`
    (the quantitative `Float/{Bin,Txt}` sources, which may coexist) and `tif_folder`
    (the uint8 `Image` preview), plus the `.bin`-preferred `quantitative` convenience
    and `frame_counts`. Each accessor is None when its source is absent. A subclass
    binds the concrete folder subtypes.

    Type Parameters:
        B: The `Float/Bin` folder type (e.g. `PhaseBinFolder`).
        T: The `Float/Txt` folder type.
        P: The `Image` preview (`.tif`) folder type.
    """

    def __init__(
        self,
        root: StrPath,
        bin_cls: type[B],
        txt_cls: type[T],
        tif_cls: type[P],
    ) -> None:
        self._root = Path(root)
        self._bin_cls = bin_cls
        self._txt_cls = txt_cls
        self._tif_cls = tif_cls

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self._root)!r})"

    @property
    def root(self) -> Path:
        """The modality folder (e.g. `Phase/`)."""
        return self._root

    @cached_property
    def bin_folder(self) -> B | None:
        """The quantitative `Float/Bin` source, or None when it is absent."""
        return open_folder(self._root / FLOAT / BIN, self._bin_cls)

    @cached_property
    def txt_folder(self) -> T | None:
        """The quantitative `Float/Txt` source, or None when it is absent."""
        return open_folder(self._root / FLOAT / TXT, self._txt_cls)

    @cached_property
    def tif_folder(self) -> P | None:
        """The uint8 `Image` preview folder, or None when it is absent."""
        return open_folder(self._root / IMAGE, self._tif_cls)

    @property
    def quantitative(self) -> B | T | None:
        """The quantitative source, `Float/Bin` preferred over `Float/Txt`, or None."""
        return self.bin_folder or self.txt_folder

    @cached_property
    def frame_counts(self) -> dict[str, int]:
        """The frame count of each present source, keyed by `bin` / `txt` / `tif`."""
        sources: dict[str, KoalaFrameFolder | None] = {
            "bin": self.bin_folder,
            "txt": self.txt_folder,
            "tif": self.tif_folder,
        }
        return {name: len(seq) for name, seq in sources.items() if seq is not None}
