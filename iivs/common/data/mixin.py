from __future__ import annotations

__all__ = ("FrameShapedMixin", "ValueRangeMixin")

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Protocol

    import numpy as np
    from numpy.typing import NDArray

    class _FrameSource(Protocol):
        def __len__(self) -> int: ...
        def get_item(self, index: int) -> NDArray[np.generic]: ...

    _Base = _FrameSource
else:
    _Base = object


def _min_max(array: NDArray[np.generic]) -> tuple[float, float]:
    return float(array.min()), float(array.max())


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


class ValueRangeMixin(_Base):
    """Mixin adding `value_range` to a sequence of numeric image frames.

    `value_range()` reduces to the global `(min, max)` over every frame;
    `value_range(index)` returns one frame's `(min, max)`. Values are taken as the
    sequence yields them from `get_item` (e.g. a modality's `target_unit`); a subclass
    may widen the signature to convert to a fixed unit first (see phase).
    """

    def value_range(self, index: int | None = None) -> tuple[float, float]:
        """The `(min, max)` over every frame, or of frame `index`.

        Args:
            index: A single frame to range over, or None (default) for the global
                range across all frames.

        Raises:
            ValueError: If the whole-sequence range is requested but the sequence is
                empty.
        """
        if index is not None:
            return _min_max(self.get_item(index))

        minimum, maximum = math.inf, -math.inf
        for i in range(len(self)):
            low, high = _min_max(self.get_item(i))
            minimum, maximum = min(minimum, low), max(maximum, high)
        if minimum > maximum:
            msg = "value range is undefined for an empty sequence"
            raise ValueError(msg)
        return minimum, maximum
