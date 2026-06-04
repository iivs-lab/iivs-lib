from __future__ import annotations

__all__ = ("PhaseFloatSequence", "PhaseImageSequence", "PhaseSequence")

import math
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, override

import numpy as np
from kaparoo.data.sequences import DataSequence, FileListSequence
from kaparoo.utils import replace_if_none
from numpy.typing import NDArray

from iivs.dhm.data.common import SequentialFileFolder, ensure_file_extension
from iivs.dhm.data.phase.bounds import PhaseBounds
from iivs.dhm.data.phase.core import PhaseUnit, convert_phase_unit

if TYPE_CHECKING:
    from typing import Literal

    from kaparoo.filesystem.types import StrPath, StrPaths

    from iivs.dhm.data.phase.bin import PhaseBinHeader


class PhaseSequence[T, M](DataSequence[T, M]):
    """A read-only sequence of phase images, from any source.

    The modality-level base over both representations Koala exports:
    quantitative float32 (`PhaseFloatSequence`, from `Float/{Bin,Txt}`) and the
    uint8 display preview (`PhaseImageSequence`, from `Image/*.tif`). Annotate
    with it to accept any phase sequence regardless of pixel type; annotate with
    the `Float` / `Image` subtype when the dtype matters.

    Type Parameters:
        T: The item (image) array type -- `NDArray[np.float32]` (quantitative)
            or `NDArray[np.uint8]` (preview).
        M: The per-item metadata type chosen by the concrete sequence (e.g. the
            source `Path`).
    """


class PhaseFloatSequence[M](PhaseSequence[NDArray[np.float32], M]):
    """A read-only sequence of quantitative float32 phase images.

    The phase reconstruction Koala exports as `Float/{Bin,Txt}`; annotate
    parameters with it to accept any float32 phase source -- one acquisition
    (`PhaseBinFolder`) or an arbitrary `PhaseBinList` of unrelated files, and
    their `.txt` twins. Same-shape sources additionally mix in
    `data.common.FrameShapedMixin` to expose `frame_shape`.
    """


class PhaseImageSequence[M](PhaseSequence[NDArray[np.uint8], M]):
    """A read-only sequence of uint8 phase preview images.

    The display-only 8-bit preview Koala renders under `Image/*.tif` (the float
    phase mapped through `phbounds.txt` into 0-255) -- distinct from, and not a
    substitute for, the quantitative `PhaseFloatSequence`. Same-shape sources
    mix in `data.common.FrameShapedMixin` to expose `frame_shape`.
    """


class PhaseFileList(
    FileListSequence[NDArray[np.float32], Path], PhaseFloatSequence[Path]
):
    """Format-agnostic phase file list over a ``(read_header, decode)`` codec.

    Holds the list machinery -- per-file unit conversion, `target_unit`,
    `get_meta`, the `.<FILE_EXT>` check -- once; a concrete subclass
    (`PhaseBinList`, `PhaseTxtList`) supplies only `FILE_EXT` and the
    `_read_header` / `_decode` codec for its on-disk format. `PhaseFileFolder`
    is the auto-discovered, same-shape specialization.

    Args:
        files: The files to expose, in the given order.
        target_unit: Unit to return images in, applied per file via that
            file's own `height_scale`. Defaults to None, which keeps each
            file's stored unit.

    Raises:
        ValueError: If any path does not have the subclass `.<FILE_EXT>` suffix.
    """

    FILE_EXT: ClassVar[str]

    def __init__(
        self, files: StrPaths, *, target_unit: PhaseUnit | None = None
    ) -> None:
        super().__init__([ensure_file_extension(f, self.FILE_EXT) for f in files])
        self._target_unit = target_unit

    @property
    def target_unit(self) -> PhaseUnit | None:
        """The unit images are converted to on load, or None to keep each file's."""
        return self._target_unit

    @override
    def get_meta(self, index: int) -> Path:
        """Return the source path of the file at `index`."""
        return self.get_file(index)

    def get_header(self, index: int) -> PhaseBinHeader:
        """Read the header of the file at `index`.

        A header accessor named to sit beside the sequence's `get_item` (the
        image) and `get_meta` (the source path), though not part of kaparoo's
        `get_item` / `get_meta` protocol. The per-file twin of a folder's shared
        `header`, for a list whose files may each carry a different one.
        """
        return self._read_header(self.get_file(index))

    @override
    def load_file(self, path: Path) -> NDArray[np.float32]:
        """Load the image at `path`, converted to `target_unit` if one is set."""
        image, header = self._decode(path)
        target = self._target_unit if self._target_unit is not None else header.unit
        return convert_phase_unit(
            image, source=header.unit, target=target, height_scale=header.height_scale
        )

    def bounds_nm(self) -> PhaseBounds:
        """Global phase display bounds over every frame, in nanometers.

        Recomputes the `phbounds.txt` values straight from the float source:
        each frame is converted to nanometers via its own `height_scale`, then
        reduced to one global ``(min, max)``. Reads every file once, regardless
        of `target_unit`.

        Raises:
            ValueError: If the sequence is empty, or a frame's stored unit
                cannot be converted to nanometers (e.g. an UNKNOWN unit).
        """
        minimum, maximum = math.inf, -math.inf
        for index in range(len(self)):
            image, header = self._decode(self.get_file(index))
            nm = convert_phase_unit(
                image,
                source=header.unit,
                target=PhaseUnit.NANOMETERS,
                height_scale=header.height_scale,
            )
            minimum = min(minimum, float(nm.min()))
            maximum = max(maximum, float(nm.max()))

        if minimum > maximum:
            msg = "phase bounds are undefined for an empty sequence"
            raise ValueError(msg)
        return PhaseBounds(min_nm=minimum, max_nm=maximum)

    @abstractmethod
    def _read_header(self, path: StrPath) -> PhaseBinHeader:
        """Read the format's header (subclass codec)."""
        raise NotImplementedError

    @abstractmethod
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
