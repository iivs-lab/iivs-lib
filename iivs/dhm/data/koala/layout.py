from __future__ import annotations

__all__ = (
    "FrameFolderOpener",
    "ReconstructionGroup",
    "open_folder",
    "open_timelapse_subfolders",
    "reconstruction_tree",
    "search_timelapse_subdirs",
)

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Unpack, cast

from kaparoo.filesystem import contains, dir_exists, search_dirs
from kaparoo.filesystem.hierarchy import Directory
from kaparoo.utils import fold_optional

from iivs.dhm.data.koala.constants import BIN, FLOAT, IMAGE, TXT

if TYPE_CHECKING:
    from collections.abc import Callable

    from kaparoo.filesystem import WalkKwargs
    from kaparoo.filesystem.types import StrPath

    from iivs.dhm.data.koala.float import KoalaFloatFileFolder
    from iivs.dhm.data.koala.frame import KoalaFrameFolder, ValidationLevel
    from iivs.dhm.data.koala.image import ImageTifFolder


# ============================================================ #
#                        open / search                         #
# ============================================================ #


class FrameFolderOpener[T: KoalaFrameFolder](Protocol):
    """Opens one numbered folder at a path, with validation controlled by the caller.

    A `KoalaFrameFolder` subclass satisfies this by construction; so does a
    `functools.partial` of one that pre-binds the extra arguments a format takes
    (`partial(PhaseBinFolder, target_unit=...)`, say). Taking the opener rather than
    the class lets a search open each folder *as the caller wants it* in one pass,
    instead of opening it and rebuilding it.

    `validate` is typed `None` because that is all `open_folder` ever passes: a
    folder accepting only some levels (a preview folder has no `"headers"`) still
    satisfies this, where demanding the whole `ValidationLevel` would exclude it.
    """

    def __call__(self, root: StrPath, *, validate: None) -> T: ...


def open_folder[T: KoalaFrameFolder](
    path: StrPath, folder: FrameFolderOpener[T]
) -> T | None:
    """Open `path` with `folder` when it is a populated directory, else None.

    Tolerant of a missing source: an absent directory or a present-but-empty numbered
    folder yields None rather than raising. Opened without per-file content validation
    (`validate=None`), but a folder that reads a shared header at construction still
    surfaces a corrupt one; a real content error is raised, not hidden as None.

    Args:
        path: The folder to open.
        folder: The folder class, or any `FrameFolderOpener` (e.g. a `partial`
            pre-binding a format's extra arguments).

    Type Parameters:
        T: The opened folder type (e.g. `PhaseBinFolder`).
    """
    if dir_exists(path):
        try:
            return folder(path, validate=None)
        except FileNotFoundError:
            return None
    return None


def search_timelapse_subdirs(
    root: StrPath, subpath: str, **walk: Unpack[WalkKwargs]
) -> list[Path]:
    """Return each `<time-lapse>/<subpath>` path under `root`.

    Finds each time-lapse directory holding `subpath` (a relative folder like `Phase` or
    `Phase/Float/Bin`) and returns that subfolder. `name_filter` matches the
    *time-lapse* folder's own name and `part_filter` its parent's relative path.

    Args:
        root: The directory to scan.
        subpath: The relative subfolder each time-lapse must hold (e.g. "Phase" or
            "Phase/Float/Bin").
        **walk: The `WalkKwargs` set `kaparoo`'s `search_dirs` defines. `predicate`
            is not among them: this function supplies its own, the check for
            `subpath`.
    """
    dirs = search_dirs(root, predicate=contains(subpath, kind="dir"), **walk)
    return [directory / subpath for directory in dirs]


def open_timelapse_subfolders[T: KoalaFrameFolder](
    root: StrPath,
    subpath: str,
    folder: FrameFolderOpener[T],
    *,
    predicate: Callable[[T], bool] | None = None,
    **walk: Unpack[WalkKwargs],
) -> list[T]:
    """Open each `<time-lapse>/<subpath>` folder under `root`.

    Finds each time-lapse holding `subpath` and opens it with `folder`, tolerating an
    empty one (it drops out rather than raising), then keeps those passing `predicate`.

    Args:
        root: The directory to scan.
        subpath: The relative subfolder each time-lapse must hold (e.g. "Phase" or
            "Phase/Float/Bin").
        folder: The class to open each `subpath` with (e.g. `PhaseBinFolder`), or any
            `FrameFolderOpener`.
        predicate: A final check on each opened folder; None (default) keeps all.
        **walk: The `WalkKwargs` set `search_timelapse_subdirs` passes through.

    Returns:
        The opened folders, one per matching time-lapse.

    Type Parameters:
        T: The opened folder type (e.g. `PhaseBinFolder`).
    """
    opened = (
        open_folder(directory, folder)
        for directory in search_timelapse_subdirs(root, subpath, **walk)
    )
    present = [item for item in opened if item is not None]
    if predicate is None:
        return present
    return [item for item in present if predicate(item)]


# ============================================================ #
#                       group / tree                           #
# ============================================================ #


def reconstruction_tree(name: str) -> Directory:
    """Build the `<name>/{Float/{Bin,Txt}, Image}` subtree the reconstructions share.

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
    (the uint8 `Image` preview), with `present_folders` to iterate whichever of the
    three exist, plus the `.bin`-preferred `quantitative`, the shared
    `num_frames` / `frame_shape`, the tolerant `is_consistent` cross-format check, and
    the non-vacuous `is_usable` (has quantitative data and is consistent). Each
    accessor is None when its source is absent, and opens its folder on first access:
    constructing a group touches no disk, and a corrupt source surfaces at the accessor
    that reaches it rather than here. `validate` checks per-file content across the
    present formats (distinct from the structural checks above). A subclass binds the
    concrete folder subtypes.

    Args:
        root: The modality folder (e.g. `Phase/`). Not required to exist: a missing one
            makes every accessor None.
        bin_cls: The class to open `Float/Bin` with.
        txt_cls: The class to open `Float/Txt` with.
        tif_cls: The class to open the `Image` preview with.

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
        self._opened: dict[Path, KoalaFrameFolder | None] = {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self._root)!r})"

    @property
    def root(self) -> Path:
        """The modality folder (e.g. `Phase/`)."""
        return self._root

    # -- sources --

    def _open[F: KoalaFrameFolder](self, path: Path, folder: type[F]) -> F | None:
        """Open `path` with `folder` on first call, cached thereafter (absence too).

        Each path is reached by one accessor, hence always with the same `folder`, so a
        cached entry matches the requested type.

        Not a `cached_property` per accessor, which is the obvious shape and does not
        work: ty (0.0.51) resolves a `cached_property` returning the class type
        parameter `B | None` to plain `None`, and a cast inside the body cannot fix
        it, the mis-typing being at the descriptor's access site. A generic *method*
        resolves `F := B` correctly, so the memoization lives here instead.
        """
        if path not in self._opened:
            self._opened[path] = open_folder(path, folder)
        return cast("F | None", self._opened[path])

    @property
    def bin_folder(self) -> B | None:
        """The quantitative `Float/Bin` source, or None when it is absent."""
        return self._open(self._root / FLOAT / BIN, self._bin_cls)

    @property
    def txt_folder(self) -> T | None:
        """The quantitative `Float/Txt` source, or None when it is absent."""
        return self._open(self._root / FLOAT / TXT, self._txt_cls)

    @property
    def tif_folder(self) -> P | None:
        """The uint8 `Image` preview folder, or None when it is absent."""
        return self._open(self._root / IMAGE, self._tif_cls)

    @property
    def _all_folders(self) -> tuple[B | None, T | None, P | None]:
        """The three format sources in `(bin, txt, tif)` order; None where absent."""
        return (self.bin_folder, self.txt_folder, self.tif_folder)

    @property
    def present_folders(self) -> tuple[B | T | P, ...]:
        """The sources that are present, in `(bin, txt, tif)` order."""
        return tuple(f for f in self._all_folders if f is not None)

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

    # -- counts --

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
        ref = self._reference
        return ref.frame_shape if ref is not None else None

    # -- status --

    @property
    def is_consistent(self) -> bool:
        """Whether the present sources agree in frame count and shape.

        Vacuously True when nothing (or one source) is present; a correct acquisition's
        `bin` / `txt` / `tif` share one shape (and one count when fully reconstructed).
        """
        present = self.present_folders
        counts = {len(f) for f in present}
        shapes = {f.frame_shape for f in present}
        return len(counts) <= 1 and len(shapes) <= 1

    @property
    def is_usable(self) -> bool:
        """Whether the group holds quantitative data and it is self-consistent.

        The non-vacuous counterpart to `is_consistent`: True when a quantitative source
        (`bin` or `txt`) is present and every present format agrees in count and shape.
        Unlike `is_consistent` (vacuously True for an absent group), this is False for
        empty or preview-only folders, so it marks a real reconstruction. The `tif`
        preview is optional, and `bin` / `txt` are the same data in two serializations,
        so a single-format export still counts.
        """
        return self.quantitative is not None and self.is_consistent

    # -- validation --

    def validate(self, *, level: ValidationLevel | None = None) -> None:
        """Validate every present format's files to `level`.

        Checks each present format (`bin` / `txt` / `tif`), skipping absent ones (an
        empty group is a no-op) and any that lack `level`. So `"headers"` is a partial
        pass: the quantitative `bin` / `txt` are checked and the header-less `tif`
        preview is skipped. `level=None` (default) checks each format to its own depth
        (`bin` / `txt` headers, `tif` names); `"names"` (contiguous naming) and `"data"`
        (full decode) apply to every present format.

        This is file *content*, unlike the structural `is_consistent` / `is_usable`.

        Raises:
            ValueError: If a file fails validation (a non-contiguous name, or a bad
                header / payload at the deeper levels).
        """
        for folder in self.present_folders:
            folder.validate_if_supported(level=level)
