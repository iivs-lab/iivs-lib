from __future__ import annotations

__all__ = ("PhaseFloatSequence", "PhaseImageSequence", "PhaseSequence")

import math
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, overload, override

import numpy as np
from kaparoo.data.sequences import DataSequence, TransformedSequence
from kaparoo.utils import replace_if_none
from numpy.typing import NDArray

from iivs.dhm.data.koala import KoalaFloatFileFolder, KoalaFloatFileList
from iivs.dhm.data.phase.bounds import PhaseBounds
from iivs.dhm.data.phase.unit import PhaseUnit, convert_phase_unit, resolve_height_scale

if TYPE_CHECKING:
    from typing import Literal

    from kaparoo.filesystem.types import StrPath, StrPaths

    from iivs.dhm.data.koala import ValidationLevel
    from iivs.dhm.data.phase.bin import PhaseBinHeader


class PhaseSequence[T, M](DataSequence[T, M]):
    """A read-only sequence of phase images, from any source.

    The modality-level base over both representations Koala exports: quantitative
    float32 (`PhaseFloatSequence`, from `Float/{Bin,Txt}`) and the uint8 display preview
    (`PhaseImageSequence`, from `Image/*.tif`). Annotate with it to accept any phase
    sequence regardless of pixel type; annotate with the `Float` / `Image` subtype when
    the dtype matters.

    Type Parameters:
        T: The item (image) array type: `NDArray[np.float32]` (quantitative) or
            `NDArray[np.uint8]` (preview).
        M: The per-item metadata type chosen by the concrete sequence (e.g. the source
            `Path`).
    """


class PhaseImageSequence[M](PhaseSequence[NDArray[np.uint8], M]):
    """A read-only sequence of uint8 phase preview images.

    The display-only 8-bit preview Koala renders under `Image/*.tif` (the float phase
    mapped through `phbounds.txt` into 0-255); distinct from, and not a substitute for,
    the quantitative `PhaseFloatSequence`. Same-shape sources also expose `frame_shape`.
    """

    @overload
    def to_float(
        self,
        bounds: PhaseBounds,
        *,
        target_unit: Literal[PhaseUnit.NANOMETERS, PhaseUnit.METERS] = ...,
    ) -> PhaseFloatSequence[M]: ...

    @overload
    def to_float(
        self,
        bounds: PhaseBounds,
        *,
        target_unit: Literal[PhaseUnit.RADIANS],
        height_scale: float,
    ) -> PhaseFloatSequence[M]: ...

    @overload
    def to_float(
        self,
        bounds: PhaseBounds,
        *,
        target_unit: Literal[PhaseUnit.RADIANS],
        wavelength: float,
        refractive_delta: float,
    ) -> PhaseFloatSequence[M]: ...

    def to_float(
        self,
        bounds: PhaseBounds,
        *,
        target_unit: PhaseUnit = PhaseUnit.NANOMETERS,
        height_scale: float | None = None,
        wavelength: float | None = None,
        refractive_delta: float | None = None,
    ) -> PhaseFloatSequence[M]:
        """Reconstruct these previews as a lazy float32 phase sequence (Image -> Float).

        Each uint8 frame is mapped back through `bounds` (`PhaseBounds.decode_preview`)
        on access, then converted from nanometers to `target_unit`. The result is a
        **lossy, 8-bit-quantized reconstruction**, not the quantitative `Float` source
        it imitates; use the real `Float` sequence when the exact values matter.
        `bounds` must be supplied (a preview cannot recover them); read it from the
        acquisition's `phbounds.txt` or recompute it from the `Float` twin's
        `value_range(unit=NANOMETERS)`.

        NANOMETERS (the default) and METERS need no scale; RADIANS needs `height_scale`
        (m per rad), or `wavelength` + `refractive_delta` to derive it.

        Args:
            bounds: The display bounds Koala used to render these previews.
            target_unit: Unit of the reconstructed phase. Defaults to NANOMETERS.
            height_scale: m per rad, needed only to reach RADIANS. Mutually exclusive
                with `wavelength` / `refractive_delta`.
            wavelength: Illumination wavelength, in m (with `refractive_delta`).
            refractive_delta: Refractive-index difference (with `wavelength`).

        Raises:
            ValueError: If `target_unit` is RADIANS but neither (or both) of the
                height-scale forms is given, or `target_unit` is not reachable from nm.
        """
        return PhaseFloatView(
            self,
            bounds,
            target_unit=target_unit,
            height_scale=height_scale,
            wavelength=wavelength,
            refractive_delta=refractive_delta,
        )


class PhaseImageView(PhaseImageSequence[Path]):
    """A lazy uint8 preview over a quantitative phase list (the `to_image` result).

    Each frame is the source phase converted to nm and rendered through the bound
    `PhaseBounds` on access; the per-frame metadata (the source path) is unchanged.
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
        nm = self._source._decode_in(index, PhaseUnit.NANOMETERS)  # noqa: SLF001
        return self._bounds.encode_preview(nm)


class PhaseFloatSequence[M](PhaseSequence[NDArray[np.float32], M]):
    """A read-only sequence of quantitative float32 phase images.

    The phase reconstruction Koala exports as `Float/{Bin,Txt}`; annotate parameters
    with it to accept any float32 phase source: one acquisition (`PhaseBinFolder`) or an
    arbitrary `PhaseBinList` of unrelated files, and their `.txt` twins. Same-shape
    sources additionally expose `frame_shape`.
    """


class PhaseFloatView[M](
    TransformedSequence[NDArray[np.uint8], M, NDArray[np.float32], M],
    PhaseFloatSequence[M],
):
    """A lazy phase reconstruction over a preview sequence (the `to_float` result).

    Maps each uint8 preview back to float32 phase (`bounds.decode_preview` to nm, then
    `convert_phase_unit` to `target_unit`). The values are 8-bit-quantized (a
    reconstruction, never the quantitative `Float` source).
    """

    def __init__(
        self,
        source: PhaseImageSequence[M],
        bounds: PhaseBounds,
        *,
        target_unit: PhaseUnit = PhaseUnit.NANOMETERS,
        height_scale: float | None = None,
        wavelength: float | None = None,
        refractive_delta: float | None = None,
    ) -> None:
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

        super().__init__(source, to_unit)
        self._bounds = bounds

    @property
    def bounds(self) -> PhaseBounds:
        """The display bounds used to map previews back to phase."""
        return self._bounds


class PhaseFileList(KoalaFloatFileList["PhaseBinHeader"], PhaseFloatSequence[Path]):
    """Format-agnostic phase file list over a ``(read_header, decode)`` codec.

    An arbitrary list of float32 phase files, each read independently and returned in
    `target_unit` (converted per file via its own `height_scale`). Concrete subclasses
    (`PhaseBinList`, `PhaseTxtList`) bind a format; `PhaseFileFolder` is the
    auto-discovered, same-shape case.

    Args:
        files: The files to expose, in the given order.
        target_unit: Unit to return images in, applied per file via that file's own
            `height_scale`. Defaults to None, which keeps each file's stored unit.

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

    def _decode_in(self, index: int, unit: PhaseUnit) -> NDArray[np.float32]:
        """Decode frame `index` and convert it to `unit` via its own header.

        Reads the raw stored values (ignoring `target_unit`) and converts to `unit`
        with that frame's `height_scale`. Backs `value_range(unit=...)` and the
        `to_image` preview render (which uses nanometers).
        """
        image, header = self._decode(self.get_file(index))
        return convert_phase_unit(
            image, source=header.unit, target=unit, height_scale=header.height_scale
        )

    @override
    def value_range(
        self, index: int | None = None, unit: PhaseUnit | None = None
    ) -> tuple[float, float]:
        """The phase `(min, max)`: over every frame or of frame `index`, in `unit`.

        Widens the inherited `value_range` with a `unit`. With `unit=None` (default)
        the range is over the values as loaded (`target_unit`); with a `PhaseUnit`
        each frame is decoded to that unit via its own `height_scale` first (ignoring
        `target_unit`), so the range is in a fixed unit regardless of how the sequence
        loads. `to_image` uses `value_range(unit=NANOMETERS)` for its display bounds.
        The global range is cached (per unit), like the inherited one. Non-finite
        values are ignored, as in the inherited `value_range`.

        Raises:
            ValueError: If the global range is requested on an empty sequence, the
                sequence (or `index`'s frame) has no finite values, or a frame's stored
                unit cannot be converted to `unit`.
        """
        if unit is None:
            return super().value_range(index)
        if index is not None:
            frame = self._decode_in(index, unit)
            finite = frame[np.isfinite(frame)]
            if finite.size == 0:
                msg = f"phase value range of frame {index} is undefined (every value is non-finite)"
                raise ValueError(msg)
            return float(finite.min()), float(finite.max())
        return self._global_value_range_in(unit)

    @cached_property
    def _value_range_by_unit(self) -> dict[PhaseUnit, tuple[float, float]]:
        return {}

    def _global_value_range_in(self, unit: PhaseUnit) -> tuple[float, float]:
        """The cached global `(min, max)` with every frame decoded to `unit`."""
        cache = self._value_range_by_unit
        if unit not in cache:
            if len(self) == 0:
                msg = "phase value range is undefined for an empty sequence"
                raise ValueError(msg)

            minimum, maximum = math.inf, -math.inf
            for i in range(len(self)):
                frame = self._decode_in(i, unit)
                finite = frame[np.isfinite(frame)]
                if finite.size:
                    minimum = min(minimum, float(finite.min()))
                    maximum = max(maximum, float(finite.max()))
            if minimum > maximum:
                msg = "phase value range is undefined (every value is non-finite)"
                raise ValueError(msg)
            cache[unit] = (minimum, maximum)
        return cache[unit]

    def to_image(self, bounds: PhaseBounds | None = None) -> PhaseImageSequence[Path]:
        """Render this phase as a lazy uint8 Koala preview, in nm (Float -> Image).

        Each frame is converted to nanometers via its own `height_scale` (so the result
        is correct regardless of `target_unit`), then rendered through `bounds`
        (`PhaseBounds.encode_preview`) on access. With ``bounds=None`` the global
        `value_range(unit=NANOMETERS)` is used, reading every file once up front (as
        Koala's `phbounds.txt` spans the whole acquisition); pass `bounds` explicitly to
        skip that pass or to match an existing `phbounds.txt`.

        Args:
            bounds: The display bounds to render against, or None to derive them from
                this source via `value_range(unit=NANOMETERS)`.
        """
        if bounds is None:
            bounds = PhaseBounds(*self.value_range(unit=PhaseUnit.NANOMETERS))
        return PhaseImageView(self, bounds)


class PhaseFileFolder(KoalaFloatFileFolder["PhaseBinHeader"], PhaseFileList):
    """Format-agnostic phase folder: numbered discovery + one shared header.

    The auto-discovered, same-shape case of `PhaseFileList`: one acquisition's numbered
    files sharing a single `header`. Concrete folders (`PhaseBinFolder`,
    `PhaseTxtFolder`) bind a format. `target_unit` defaults to the shared header's
    stored unit.

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
        validate: ValidationLevel | None = "headers",
    ) -> None:
        super().__init__(root, validate=validate)

        # The shared header is known only after super().__init__ reads it, and
        # the validation above never uses target_unit (it decodes raw frames),
        # so resolve it here against the header, failing fast if unreachable.
        self._target_unit = replace_if_none(target_unit, self._header.unit)
        convert_phase_unit(  # empty array -> a pure source/target pair check
            np.empty((0, 0), dtype=np.float32),
            source=self._header.unit,
            target=self._target_unit,
            height_scale=self._header.height_scale,
        )
