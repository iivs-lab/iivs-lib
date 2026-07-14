from __future__ import annotations

__all__ = ("FrameShapedMixin", "ValueRangeMixin")

import math
from abc import ABC, abstractmethod
from functools import cached_property

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
    `(min, max)` of its values: `value_range()` over every frame, or `value_range(index)`
    of one frame. Values are taken as `get_item` yields them (e.g. a modality's
    `target_unit`); a subclass may widen the signature to convert first (see phase). The
    whole-sequence range reads every frame, so it is computed once and cached for the
    sequence's lifetime; per-frame ranges are cheap and recomputed each call.
    """

    @cached_property
    def _whole_value_range(self) -> tuple[float, float]:
        minimum, maximum = math.inf, -math.inf
        for i in range(len(self)):
            frame = self.get_item(i)
            minimum = min(minimum, float(frame.min()))
            maximum = max(maximum, float(frame.max()))
        if minimum > maximum:
            msg = "value range is undefined for an empty sequence"
            raise ValueError(msg)
        return minimum, maximum

    def value_range(self, index: int | None = None) -> tuple[float, float]:
        """The `(min, max)` over every frame (cached), or of frame `index`.

        Args:
            index: A single frame to range over, or None (default) for the whole-
                sequence range, which is computed once and then cached.

        Raises:
            ValueError: If the whole-sequence range is requested but the sequence is
                empty.
        """
        if index is None:
            return self._whole_value_range
        frame = self.get_item(index)
        return float(frame.min()), float(frame.max())
