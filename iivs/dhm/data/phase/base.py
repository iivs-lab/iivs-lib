from __future__ import annotations

__all__ = ("PhaseFloatSequence", "PhaseImageSequence", "PhaseSequence")

import math
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, override

import numpy as np
from kaparoo.data.sequences import DataSequence, TransformedSequence
from kaparoo.utils import replace_if_none
from numpy.typing import NDArray

from iivs.dhm.data.common import KoalaFloatFileFolder, KoalaFloatFileList
from iivs.dhm.data.phase.bounds import PhaseBounds
from iivs.dhm.data.phase.core import (
    PhaseUnit,
    convert_phase_unit,
    resolve_height_scale,
)

if TYPE_CHECKING:
    from collections.abc import Callable
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

    def to_float(
        self,
        bounds: PhaseBounds,
        *,
        target_unit: PhaseUnit = PhaseUnit.NANOMETERS,
        height_scale: float | None = None,
        wavelength: float | None = None,
        refractive_delta: float | None = None,
    ) -> PhaseFloatSequence[M]:
        """A lazy float32 phase reconstruction of these previews (Image -> Float).

        Each uint8 frame is mapped back through `bounds`
        (`PhaseBounds.decode_preview`) on access, then converted from nanometers
        to `target_unit`. The result is a **lossy, 8-bit-quantized
        reconstruction**, not the quantitative `Float` source it imitates -- use
        the real `Float` sequence when the exact values matter. `bounds` must be
        supplied (a preview cannot recover them); read it from the acquisition's
        `phbounds.txt` or recompute it from the `Float` twin's `bounds_nm`.

        NANOMETERS (the default) and METERS need no scale; RADIANS needs
        `height_scale` (m per rad), or `wavelength` + `refractive_delta` to
        derive it.

        Args:
            bounds: The display bounds Koala used to render these previews.
            target_unit: Unit of the reconstructed phase. Defaults to NANOMETERS.
            height_scale: m per rad, needed only to reach RADIANS. Mutually
                exclusive with `wavelength` / `refractive_delta`.
            wavelength: Illumination wavelength, in m (with `refractive_delta`).
            refractive_delta: Refractive-index difference (with `wavelength`).

        Raises:
            ValueError: If `target_unit` is RADIANS but neither (or both) of the
                height-scale forms is given, or `target_unit` is not reachable
                from nm.
        """
        scale = (
            resolve_height_scale(height_scale, wavelength, refractive_delta)
            if target_unit is PhaseUnit.RADIANS
            else 1.0  # METERS <-> NANOMETERS is a fixed rescale; `scale` is unused
        )

        def to_unit(preview: NDArray[np.uint8]) -> NDArray[np.float32]:
            # source == target (nm) short-circuits inside convert_phase_unit.
            return convert_phase_unit(
                bounds.decode_preview(preview),
                source=PhaseUnit.NANOMETERS,
                target=target_unit,
                height_scale=scale,
            )

        return PhaseFloatView(self, to_unit, bounds)


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

    def _decode_nm(self, index: int) -> NDArray[np.float32]:
        """Decode frame `index` and convert it to nanometers via its own header.

        The unit-agnostic frame source for `bounds_nm` and `to_image`: reads
        the raw stored values (ignoring `target_unit`) and converts to nm with
        that frame's `height_scale`.
        """
        image, header = self._decode(self.get_file(index))
        return convert_phase_unit(
            image,
            source=header.unit,
            target=PhaseUnit.NANOMETERS,
            height_scale=header.height_scale,
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
            nm = self._decode_nm(index)
            minimum = min(minimum, float(nm.min()))
            maximum = max(maximum, float(nm.max()))

        if minimum > maximum:
            msg = "phase bounds are undefined for an empty sequence"
            raise ValueError(msg)
        return PhaseBounds(min_nm=minimum, max_nm=maximum)

    def to_image(self, bounds: PhaseBounds | None = None) -> PhaseImageSequence[Path]:
        """A lazy uint8 Koala-style preview of this phase, in nm (Float -> Image).

        Each frame is converted to nanometers via its own `height_scale` (so the
        result is correct regardless of `target_unit`), then rendered through
        `bounds` (`PhaseBounds.encode_preview`) on access. With ``bounds=None`` the
        global `bounds_nm` is used, reading every file once up front (as Koala's
        `phbounds.txt` spans the whole acquisition); pass `bounds` explicitly to
        skip that pass or to match an existing `phbounds.txt`.

        Args:
            bounds: The display bounds to render against, or None to derive them
                from this source via `bounds_nm`.
        """
        if bounds is None:
            bounds = self.bounds_nm()
        return PhaseImageView(self, bounds)


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


class PhaseImageView(PhaseImageSequence[Path]):
    """A lazy uint8 preview over a quantitative phase list (the `to_image` result).

    Wraps a `PhaseFileList`; `get_item` decodes the frame, converts it to nm, and
    renders it through the bound `PhaseBounds` on access. Per-frame metadata (the
    source path) passes through unchanged.
    """

    def __init__(self, source: PhaseFileList, bounds: PhaseBounds) -> None:
        self._source = source
        self._bounds = bounds

    @property
    def source(self) -> PhaseFileList:
        """The quantitative phase list being rendered (e.g. for `frame_shape`)."""
        return self._source

    @property
    def bounds(self) -> PhaseBounds:
        """The display bounds the previews are rendered against."""
        return self._bounds

    def __len__(self) -> int:
        return len(self._source)

    @override
    def get_meta(self, index: int) -> Path:
        return self._source.get_meta(index)

    @override
    def get_item(self, index: int) -> NDArray[np.uint8]:
        return self._bounds.encode_preview(self._source._decode_nm(index))  # noqa: SLF001


class PhaseFloatView[M](
    TransformedSequence[NDArray[np.uint8], M, NDArray[np.float32], M],
    PhaseFloatSequence[M],
):
    """A lazy phase reconstruction over a preview sequence (the `to_float` result).

    A `kaparoo` `TransformedSequence` that maps each uint8 preview back to float32
    phase via a `PhaseBounds`-derived transform, retyped as a `PhaseFloatSequence`.
    The values are 8-bit-quantized -- a reconstruction, never the quantitative
    `Float` source. `source` (the wrapped previews) and the per-frame metadata
    pass-through come from `TransformedSequence`.
    """

    def __init__(
        self,
        source: PhaseImageSequence[M],
        transform: Callable[[NDArray[np.uint8]], NDArray[np.float32]],
        bounds: PhaseBounds,
    ) -> None:
        super().__init__(source, transform)
        self._bounds = bounds

    @property
    def bounds(self) -> PhaseBounds:
        """The display bounds used to map previews back to phase."""
        return self._bounds
