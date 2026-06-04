from __future__ import annotations

__all__ = ("PhaseFloatSequence", "PhaseImageSequence", "PhaseSequence")

import math
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, override

import numpy as np
from kaparoo.data.sequences import DataSequence
from kaparoo.utils import replace_if_none
from numpy.typing import NDArray

from iivs.dhm.data.common import KoalaFloatFileFolder, KoalaFloatFileList
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


class PhaseFileList(KoalaFloatFileList["PhaseBinHeader"], PhaseFloatSequence[Path]):
    """Format-agnostic phase file list over a ``(read_header, decode)`` codec.

    Inherits the float-list machinery from `KoalaFloatFileList` (the `.<FILE_EXT>`
    check, `get_meta`, `get_header`, `load_with_header`); a concrete subclass
    (`PhaseBinList`, `PhaseTxtList`) supplies only `FILE_EXT` and the
    `_read_header` / `_decode` codec. This adds the phase layer: `target_unit`
    plus the per-file unit conversion done in `_postprocess`. `PhaseFileFolder`
    is the auto-discovered, same-shape specialization.

    Args:
        files: The files to expose, in the given order.
        target_unit: Unit to return images in, applied per file via that
            file's own `height_scale`. Defaults to None, which keeps each
            file's stored unit.

    Raises:
        ValueError: If any path does not have the subclass `.<FILE_EXT>` suffix.
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
    def _postprocess(
        self, image: NDArray[np.float32], header: PhaseBinHeader
    ) -> NDArray[np.float32]:
        """Convert the decoded image to `target_unit` via the file's `height_scale`."""
        target = replace_if_none(self._target_unit, header.unit)
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


class PhaseFileFolder(KoalaFloatFileFolder["PhaseBinHeader"], PhaseFileList):
    """Format-agnostic phase folder: numbered discovery + one shared header.

    The auto-discovered, same-shape specialization of `PhaseFileList`; it reuses
    that list's `load_file` codec and adds the shared acquisition `header`.
    Concrete folders (`PhaseBinFolder`, `PhaseTxtFolder`) set only `FILE_EXT` --
    the `(read_header, decode)` codec comes from the matching `*List` they also
    inherit. `target_unit` defaults to the shared header's stored unit.

    Args:
        root: The folder to scan.
        target_unit: Unit to return loaded images in (None keeps the stored).
        validate: Validation level at construction, or None to skip.
    """

    FILE_STEM: ClassVar[str] = "phase"

    def __init__(
        self,
        root: StrPath,
        *,
        target_unit: PhaseUnit | None = None,
        validate: Literal["names", "headers", "data"] | None = "headers",
    ) -> None:
        # Stashed for _after_header, which resolves it once the shared header is
        # known (the cooperative PhaseFileList.__init__ sets target_unit to None).
        self._init_target_unit = target_unit
        super().__init__(root, validate=validate)

    @override
    def _after_header(self) -> None:
        """Resolve `target_unit` against the shared header and fail fast if unreachable."""
        self._target_unit = replace_if_none(self._init_target_unit, self._header.unit)

        # Empty array -> a pure source/target pair check, no pixel work.
        convert_phase_unit(
            np.empty((0, 0), dtype=np.float32),
            source=self._header.unit,
            target=self._target_unit,
            height_scale=self._header.height_scale,
        )
