from __future__ import annotations

__all__ = (
    "KoalaFrameFolder",
    "ValidationLevel",
    "detect_koala_format",
    "koala_frame_name",
)

from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, override

from kaparoo.data.sequences import FileFolderSequence
from kaparoo.filesystem import file_extension
from kaparoo.filesystem.search import search_files
from kaparoo.filters import Regex
from kaparoo.utils import ensure_one_of, replace_if_none

from iivs.common.data.mixin import FrameShapedMixin

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal

    from kaparoo.filesystem.types import StrPath


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
    discover/validate a `KoalaFrameFolder` and to write a converted folder.
    """
    return f"{index:05d}_{stem}.{ext}"


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
    formats = tuple(formats)
    hits = search_files(
        root, name_filter=_koala_frame_filter(stem, formats), max_depth=1
    )
    found = {file_extension(hit) for hit in hits}
    present = [fmt for fmt in formats if fmt in found]

    if not present:
        alternation = "|".join(formats)
        msg = f"no NNNNN_{stem}.({alternation}) files found in {root}"
        raise FileNotFoundError(msg)
    if len(present) == 1:
        return present[0]

    if prefer is None:
        msg = (
            f"ambiguous: {root} holds multiple {stem} formats ({present}); "
            f"pass prefer to pick one"
        )
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

    def expected_name(self, index: int) -> str:
        """The contiguous filename expected at `index`."""
        return koala_frame_name(index, stem=self.FILE_STEM, ext=self.FILE_EXT)

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
        expected = self.expected_name(index)
        if path.name != expected:
            msg = f"non-contiguous numbering: expected {expected} at index {index}, got {path.name}"
            raise ValueError(msg)

        if resolved != "names":
            self._validate_content(path, level=resolved)

    def _validate_content(self, path: Path, *, level: str) -> None:
        """Per-format consistency check for levels beyond "names" (subclass hook)."""
        raise NotImplementedError
