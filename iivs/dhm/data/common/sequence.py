from __future__ import annotations

__all__ = ("FrameShapedMixin", "SequentialFileFolder")

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, override

from kaparoo.data.sequences import FileFolderSequence
from kaparoo.filesystem.search import search_files
from kaparoo.filesystem.search.filters import Regex
from kaparoo.utils import replace_if_none
from natsort import natsorted

from iivs.dhm.data.common.extension import numbered_name

if TYPE_CHECKING:
    from typing import Literal


class FrameShapedMixin(ABC):
    """Mixin marking a sequence whose items all share one `frame_shape`.

    Mix into a modality sequence on a same-shape source (e.g. a single
    acquisition) to force `frame_shape` to be implemented. There is no
    per-modality `Uniform*Sequence`: "a uniform float phase sequence" is just
    ``isinstance(x, PhaseFloatSequence) and isinstance(x, FrameShapedMixin)``
    (and likewise for the other modalities). `SequentialFileFolder` mixes this
    in for every numbered folder; a single-file source like `HologramRawFile`
    mixes it in directly.
    """

    @property
    @abstractmethod
    def frame_shape(self) -> tuple[int, int]:
        """The pixel dimensions (height, width) shared by every item."""
        raise NotImplementedError


class SequentialFileFolder[T](FileFolderSequence[T, Path], FrameShapedMixin):
    """A folder of contiguously numbered `{index:05d}_<stem>.<ext>` files.

    Each such folder is one acquisition's frames, hence same-shape: it mixes in
    `FrameShapedMixin`, and subclasses implement `frame_shape` from their header
    or first file. Factors the discovery and validation shared by every
    modality folder (phase, intensity, hologram, ...): `list_files` (numbered
    discovery), `get_meta` (= source path), the `validate` loop, and the
    name-contiguity check. A subclass declares the filename parts and validation
    depth as class attributes, implements `load_file`, and supplies its
    per-format consistency check by overriding `_validate_content`.

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
        return numbered_name(index, stem=self.FILE_STEM, ext=self.FILE_EXT)

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
        resolved = replace_if_none(level, self.DEFAULT_LEVEL)
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
