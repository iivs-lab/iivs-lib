from __future__ import annotations

__all__ = (
    "ReconstructionGroup",
    "open_folder",
    "open_timelapse_subfolders",
    "reconstruction_tree",
    "search_timelapse_subdirs",
)

from pathlib import Path
from typing import TYPE_CHECKING

from kaparoo.filesystem.hierarchy import Directory
from kaparoo.filesystem.search import search_dirs
from kaparoo.utils import fold_optional

from iivs.dhm.data.koala.constants import BIN, FLOAT, IMAGE, TXT

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from typing import Literal

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


def open_folder[T: KoalaFrameFolder](path: StrPath, folder: type[T]) -> T | None:
    """Open `path` with `folder` when it is a populated directory, else None.

    Tolerant: an absent directory, or a present-but-empty numbered folder, yields None
    rather than raising. Opened without content validation (`validate=None`).

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


def open_timelapse_subfolders[T: KoalaFrameFolder](
    root: StrPath,
    subpath: str,
    folder: type[T],
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
    (the uint8 `Image` preview), plus the `.bin`-preferred `quantitative`, the shared
    `num_frames` / `frame_shape`, the tolerant `is_consistent` cross-format check, and
    the non-vacuous `is_usable` (has quantitative data and is consistent). Each
    accessor is None when its source is absent. `validate` checks per-file content
    across the present formats (distinct from the structural checks above). A subclass
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
        # Opened eagerly (not `cached_property`): ty mis-types a generic `cached_property`
        # return `B | None` as `None`, which then reads as unreachable downstream.
        self._bin_folder = open_folder(self._root / FLOAT / BIN, bin_cls)
        self._txt_folder = open_folder(self._root / FLOAT / TXT, txt_cls)
        self._tif_folder = open_folder(self._root / IMAGE, tif_cls)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self._root)!r})"

    @property
    def root(self) -> Path:
        """The modality folder (e.g. `Phase/`)."""
        return self._root

    @property
    def bin_folder(self) -> B | None:
        """The quantitative `Float/Bin` source, or None when it is absent."""
        return self._bin_folder

    @property
    def txt_folder(self) -> T | None:
        """The quantitative `Float/Txt` source, or None when it is absent."""
        return self._txt_folder

    @property
    def tif_folder(self) -> P | None:
        """The uint8 `Image` preview folder, or None when it is absent."""
        return self._tif_folder

    @property
    def quantitative(self) -> B | T | None:
        """The quantitative source, `Float/Bin` preferred over `Float/Txt`, or None."""
        if self.bin_folder is not None:
            return self.bin_folder
        if self.txt_folder is not None:
            return self.txt_folder
        return None

    @property
    def _reference(self) -> KoalaFrameFolder | None:
        """The reference source: quantitative data if present, else the preview."""
        if self.quantitative is not None:
            return self.quantitative
        return self.tif_folder

    @property
    def num_frames(self) -> int | None:
        """The frame count shared by the present sources, or None when empty.

        The `bin` / `txt` / `tif` counts agree when the acquisition is fully
        reconstructed; `is_consistent` reports whether they actually do.
        """
        return fold_optional(self._reference, len, None)

    @property
    def frame_shape(self) -> tuple[int, int] | None:
        """The (height, width) shared by the present sources, or None when empty."""

        def get_shape(ref: KoalaFrameFolder) -> tuple[int, int]:
            return ref.frame_shape

        return fold_optional(self._reference, get_shape, None)

    @property
    def is_consistent(self) -> bool:
        """Whether the present sources agree in frame count and shape.

        Vacuously True when nothing (or one source) is present; a correct acquisition's
        `bin` / `txt` / `tif` share one shape (and one count when fully reconstructed).
        """
        sources = (self.bin_folder, self.txt_folder, self.tif_folder)
        present = [f for f in sources if f is not None]
        counts = {len(f) for f in present}
        shapes = {f.frame_shape for f in present}
        return len(counts) <= 1 and len(shapes) <= 1

    @property
    def is_usable(self) -> bool:
        """Whether the group holds quantitative data and it is self-consistent.

        The non-vacuous counterpart to `is_consistent`: True when a quantitative source
        (`bin` or `txt`) is present and every present format agrees in count and shape.
        Unlike `is_consistent` (vacuously True for an absent group), this is False for an
        empty or preview-only folder, so it marks a real reconstruction. The `tif`
        preview is optional, and `bin` / `txt` are the same data in two serializations,
        so a single-format export still counts.
        """
        return self.quantitative is not None and self.is_consistent

    def validate(self, *, level: Literal["names", "data"] | None = None) -> None:
        """Validate every present format's files to `level`.

        Delegates to each present format folder's own `validate`, skipping the absent
        ones (an empty group validates nothing). `level=None` (default) lets each folder
        use its own default depth: the quantitative `bin` / `txt` check headers, the
        `tif` preview checks names. `"names"` (contiguous naming) and `"data"` (full
        decode) apply uniformly to every present format. `"headers"` is a
        format-specific depth (the `tif` preview has no header), so it is not offered
        here; call `bin_folder.validate(level="headers")` for that.

        This checks file *content*, unlike the structural `is_consistent` / `is_usable`.

        Raises:
            ValueError: If a file fails validation (a non-contiguous name, or a bad
                header / payload at the deeper levels).
        """
        for folder in (self.bin_folder, self.txt_folder, self.tif_folder):
            if folder is not None:
                folder.validate(level=level)
