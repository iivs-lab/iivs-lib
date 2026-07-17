from __future__ import annotations

__all__ = ("FrameShapedMixin", "ValueRangeMixin")

import math
from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from kaparoo.data.sequences import DataSequence
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Callable


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

    @classmethod
    def _range_of(cls, frame: NDArray[T], index: int | None) -> tuple[float, float]:
        """Compute one frame's `(min, max)` over its finite values.

        Args:
            frame: The frame to range over.
            index: Its position, for the error message; None when it stands alone.

        Raises:
            ValueError: If `frame` holds no finite value.
        """
        finite: NDArray[T] = frame[np.isfinite(frame)]
        if finite.size == 0:
            raise ValueError(cls._undefined_range_msg(index))
        return float(finite.min()), float(finite.max())

    def _range_over_all(self, get: Callable[[int], NDArray[T]]) -> tuple[float, float]:
        """Compute the `(min, max)` across every frame `get` yields.

        Non-finite values are ignored, so a frame holding none contributes nothing; when
        no frame contributes at all, the bounds stay as initialized and inverted, which
        is what makes `minimum > maximum` the all-non-finite test rather than a flag.

        Args:
            get: Yields the frame at an index. A subclass whose range is over something
                other than `get_item` (a converted view, say) passes its own reader.

        Raises:
            ValueError: If the sequence is empty, or no frame holds a finite value.
        """
        if len(self) == 0:
            raise ValueError(self._EMPTY_RANGE_MSG)

        minimum, maximum = math.inf, -math.inf
        for i in range(len(self)):
            frame = get(i)
            finite: NDArray[T] = frame[np.isfinite(frame)]
            if finite.size:
                minimum = min(minimum, float(finite.min()))
                maximum = max(maximum, float(finite.max()))
        if minimum > maximum:
            raise ValueError(self._undefined_range_msg(None))
        return minimum, maximum

    @cached_property
    def _global_value_range(self) -> tuple[float, float]:
        return self._range_over_all(self.get_item)

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
        return self._range_of(self.get_item(index), index)
