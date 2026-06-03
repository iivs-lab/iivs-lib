from __future__ import annotations

__all__ = ("PhaseSequence",)

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, override

import numpy as np
from kaparoo.data.sequences import DataSequence, FileListSequence
from kaparoo.utils import replace_if_none
from numpy.typing import NDArray

from iivs.dhm.data.common import SequentialFileFolder
from iivs.dhm.data.phase.core import convert_phase_unit

if TYPE_CHECKING:
    from typing import Literal

    from kaparoo.filesystem.types import StrPath, StrPaths

    from iivs.dhm.data.phase.bin import PhaseBinHeader
    from iivs.dhm.data.phase.core import PhaseUnit


class PhaseSequence[M](DataSequence[NDArray[np.float32], M]):
    """A read-only sequence of float32 phase images, from any source.

    Common base for every phase sequence -- whether the images come from one
    acquisition (`PhaseBinFolder`) or an arbitrary `PhaseBinList` of unrelated
    files; annotate parameters with it to accept any of them. Each item is a
    float32 phase image; `M` is the per-item metadata type chosen by the
    concrete sequence (e.g. the source `Path`).

    Same-shape sources additionally mix in `data.common.FrameShapedMixin` to
    expose `frame_shape`.
    """


class PhaseFileList(FileListSequence[NDArray[np.float32], Path], PhaseSequence[Path]):
    """Format-agnostic phase file list over a ``(read_header, decode)`` codec.

    Holds the list machinery -- per-file unit conversion, `target_unit`,
    `get_meta` -- once; a concrete subclass (`PhaseBinList`, `PhaseTxtList`)
    supplies only `_read_header` / `_decode` for its on-disk format.
    `PhaseFileFolder` is the auto-discovered, same-shape specialization.

    Args:
        files: The files to expose, in the given order.
        target_unit: Unit to return images in, applied per file via that
            file's own `height_scale`. Defaults to None, which keeps each
            file's stored unit.
    """

    def __init__(
        self, files: StrPaths, *, target_unit: PhaseUnit | None = None
    ) -> None:
        super().__init__(files)
        self._target_unit = target_unit

    @property
    def target_unit(self) -> PhaseUnit | None:
        """The unit images are converted to on load, or None to keep each file's."""
        return self._target_unit

    @override
    def get_meta(self, index: int) -> Path:
        """Return the source path of the file at `index`."""
        return self.get_file(index)

    @override
    def load_file(self, path: Path) -> NDArray[np.float32]:
        """Load the image at `path`, converted to `target_unit` if one is set."""
        image, header = self._decode(path)
        target = self._target_unit if self._target_unit is not None else header.unit
        return convert_phase_unit(
            image, source=header.unit, target=target, height_scale=header.height_scale
        )

    def _read_header(self, path: StrPath) -> PhaseBinHeader:
        """Read the format's header (subclass codec)."""
        raise NotImplementedError

    def _decode(
        self,
        path: StrPath,
        *,
        on_nonfinite: Literal["ignore", "warn", "raise"] = "ignore",
    ) -> tuple[NDArray[np.float32], PhaseBinHeader]:
        """Decode the format's image and its header (subclass codec)."""
        raise NotImplementedError


class PhaseFileFolder(SequentialFileFolder[NDArray[np.float32]], PhaseFileList):
    """Format-agnostic phase folder: numbered discovery + one shared header.

    The auto-discovered, same-shape specialization of `PhaseFileList`; it
    reuses that list's `load_file` codec. Concrete folders (`PhaseBinFolder`,
    `PhaseTxtFolder`) set only `FILE_EXT` -- the `(read_header, decode)` codec
    comes from the matching `*List` they also inherit.

    Args:
        root: The folder to scan.
        target_unit: Unit to return loaded images in (None keeps the stored).
        validate: Validation level at construction, or None to skip.
    """

    LEVELS: ClassVar[tuple[str, ...]] = ("names", "headers", "data")
    DEFAULT_LEVEL: ClassVar[str] = "headers"
    FILE_STEM: ClassVar[str] = "phase"

    def __init__(
        self,
        root: StrPath,
        *,
        target_unit: PhaseUnit | None = None,
        validate: Literal["names", "headers", "data"] | None = "headers",
    ) -> None:
        # super().__init__(root) discovers the files and, via the cooperative
        # MRO, runs PhaseFileList.__init__ (target_unit defaults to None here);
        # the resolved unit is set below once the shared header is known.
        super().__init__(root)

        self._header = self._read_header(self.get_file(0))
        self._target_unit = replace_if_none(target_unit, self._header.unit)

        # Fail fast on an unreachable target unit (empty array -> pure pair check).
        convert_phase_unit(
            np.empty((0, 0), dtype=np.float32),
            source=self._header.unit,
            target=self._target_unit,
            height_scale=self._header.height_scale,
        )

        if validate is not None:
            self.validate(level=validate)

    @property
    def header(self) -> PhaseBinHeader:
        """The shared acquisition header, read from the first file."""
        return self._header

    @property
    @override
    def frame_shape(self) -> tuple[int, int]:
        """The (height, width) of each image, from the shared header."""
        return self._header.shape

    @override
    def _validate_content(self, path: Path, *, level: str) -> None:
        """Check `path`'s header matches the reference; at "data", decode too."""
        if self._read_header(path) != self.header:
            msg = f"header of {path.name} differs from the first file"
            raise ValueError(msg)

        if level == "data":
            self._decode(path, on_nonfinite="raise")
