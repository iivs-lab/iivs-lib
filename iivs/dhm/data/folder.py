from __future__ import annotations

__all__ = ("SequentialFileFolderSequence",)

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, override

from kaparoo.data.sequences import FileFolderSequence
from kaparoo.filesystem.search import search_files
from kaparoo.filesystem.search.filters import Regex
from natsort import natsorted

if TYPE_CHECKING:
    from typing import Literal


class SequentialFileFolderSequence[T](FileFolderSequence[T, Path]):
    """A folder of contiguously numbered `{index:05d}_<stem>.<ext>` files.

    Factors the discovery and validation shared by every modality folder
    (phase, intensity, hologram, ...): `list_files` (numbered discovery),
    `get_meta` (= source path), the `validate` loop, and the name-contiguity
    check. A subclass declares the filename parts and validation depth as class
    attributes, implements `load_file` and (for same-shape sources)
    `frame_shape`, and supplies its per-format consistency check by overriding
    `_validate_content`.

    Class attributes:
        FILE_STEM: The ``<stem>`` in ``{index:05d}_<stem>.<ext>`` (e.g. "phase").
        FILE_EXT: The file extension without the dot (e.g. "bin").
        LEVELS: The validation levels this folder accepts (a subset of
            "names" / "headers" / "data").
        DEFAULT_LEVEL: The level `validate` / `validate_file` use when given none.
    """

    FILE_STEM: ClassVar[str]
    FILE_EXT: ClassVar[str]
    LEVELS: ClassVar[tuple[str, ...]] = ("names",)
    DEFAULT_LEVEL: ClassVar[str] = "names"

    @override
    def list_files(self, root: Path) -> list[Path]:
        """List the `NNNNN_<stem>.<ext>` files under `root`, in index order."""
        pattern = rf"\d{{5}}_{self.FILE_STEM}\.{self.FILE_EXT}"
        files = search_files(root, name_filter=Regex(pattern), max_depth=1)
        if not files:
            msg = f"no NNNNN_{self.FILE_STEM}.{self.FILE_EXT} files found in {root}"
            raise FileNotFoundError(msg)
        return natsorted(files)

    @override
    def get_meta(self, index: int) -> Path:
        """Return the source path of the file at `index`."""
        return self.get_file(index)

    def expected_name(self, index: int) -> str:
        """The contiguous filename expected at `index`."""
        return f"{index:05d}_{self.FILE_STEM}.{self.FILE_EXT}"

    def validate(
        self, *, level: Literal["names", "headers", "data"] | None = None
    ) -> None:
        """Validate every file to `level` (defaults to `DEFAULT_LEVEL`)."""
        for index in range(len(self)):
            self.validate_file(index, level=level)

    def validate_file(
        self,
        index: int,
        *,
        level: Literal["names", "headers", "data"] | None = None,
    ) -> None:
        """Validate the file at `index` to `level` (defaults to `DEFAULT_LEVEL`).

        Always checks the contiguous name; any deeper level defers to the
        per-format `_validate_content`.

        Raises:
            ValueError: If `level` is unsupported, the numbering is
                non-contiguous, or `_validate_content` rejects the file.
        """
        resolved = self.DEFAULT_LEVEL if level is None else level
        if resolved not in self.LEVELS:
            msg = f"level must be one of {self.LEVELS} (got {resolved!r})"
            raise ValueError(msg)

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
