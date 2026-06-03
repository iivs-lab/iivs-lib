from __future__ import annotations

__all__ = ("FrameShapedMixin",)

from abc import ABC, abstractmethod


class FrameShapedMixin(ABC):
    """Mixin marking a sequence whose items all share one `frame_shape`.

    Mix into a modality sequence on a same-shape source (e.g. a single
    acquisition) to force `frame_shape` to be implemented. There is no
    per-modality `Uniform*Sequence`: "a uniform phase sequence" is just
    ``isinstance(x, PhaseSequence) and isinstance(x, FrameShapedMixin)``
    (and likewise for the other modalities).
    """

    @property
    @abstractmethod
    def frame_shape(self) -> tuple[int, int]:
        """The pixel dimensions (height, width) shared by every item."""
        raise NotImplementedError
