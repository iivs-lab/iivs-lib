from __future__ import annotations

__all__ = ("FrameShapedMixin",)

from abc import ABC, abstractmethod


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
