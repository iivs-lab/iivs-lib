from __future__ import annotations

__all__ = ("FrameShapedMixin", "ValueRangeMixin")

import math
from abc import ABC, abstractmethod
from functools import cached_property
from typing import ClassVar

import numpy as np
from kaparoo.data.sequences import DataSequence
from numpy.typing import NDArray


class FrameShapedMixin(ABC):
    """Mixin marking a sequence whose items all share one `frame_shape`.

    Mix into a sequence whose items are guaranteed the same shape (e.g. one
    acquisition's frames) to force `frame_shape` to be implemented. There is no separate
    `Uniform*Sequence` type: a uniform source is its role base plus this mixin
    (``isinstance(x, SomeSequence) and isinstance(x, FrameShapedMixin)``). A
    numbered-folder template mixes this in for every folder; a single-file multi-frame
    source mixes it in directly.
    """

    @property
    @abstractmethod
    def frame_shape(self) -> tuple[int, int]:
        """The pixel dimensions (height, width) shared by every item."""
        raise NotImplementedError


class ValueRangeMixin[T: np.generic, M](DataSequence[NDArray[T], M]):
    """Mixin adding `value_range` to a `DataSequence` of numeric image frames.

    Mix into a `DataSequence[NDArray[T], M]` with matching `T` / `M` to expose the
    `(min, max)` of its values: `value_range()` over all frames, or `value_range(index)`
    of one frame. Values are taken as `get_item` yields them; a subclass may widen the
    signature to convert first. The global range reads every frame, so it is computed
    once and cached for the sequence's lifetime; per-frame ranges are cheap and
    recomputed each call.
    """

    _EMPTY_RANGE_MSG: ClassVar[str] = "value range is undefined for an empty sequence"

    @staticmethod
    def _undefined_range_msg(index: int | None) -> str:
        """Build the error text for a value range with no finite input.

        `index` is None for the global range, else the offending frame.
        """
        where = f"of frame {index} " if index is not None else ""
        return f"value range {where}is undefined (all non-finite)"

    @cached_property
    def _global_value_range(self) -> tuple[float, float]:
        if len(self) == 0:
            raise ValueError(self._EMPTY_RANGE_MSG)

        minimum, maximum = math.inf, -math.inf
        for i in range(len(self)):
            frame = self.get_item(i)
            finite: NDArray[T] = frame[np.isfinite(frame)]
            if finite.size:
                minimum = min(minimum, float(finite.min()))
                maximum = max(maximum, float(finite.max()))
        if minimum > maximum:
            raise ValueError(self._undefined_range_msg(None))
        return minimum, maximum

    def value_range(self, index: int | None = None) -> tuple[float, float]:
        """Compute the `(min, max)` over every frame (cached), or of frame `index`.

        Non-finite values (NaN, +/-inf) are ignored, so the range reflects only the
        real data (e.g. a masked frame's background NaNs do not distort it).

        Args:
            index: A single frame to range over, or None (default) for the global
                range, which is computed once and then cached.

        Raises:
            ValueError: If the global range is requested on an empty sequence, or the
                sequence (or `index`'s frame) has no finite values at all.
        """
        if index is None:
            return self._global_value_range
        frame = self.get_item(index)
        finite: NDArray[T] = frame[np.isfinite(frame)]
        if finite.size == 0:
            raise ValueError(self._undefined_range_msg(index))
        return float(finite.min()), float(finite.max())
