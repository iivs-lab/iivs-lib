from __future__ import annotations

__all__ = ("IntensitySequence", "UniformIntensitySequence")

from abc import abstractmethod

import numpy as np
from kaparoo.data.sequences import DataSequence
from numpy.typing import NDArray


class IntensitySequence[M](DataSequence[NDArray[np.float32], M]):
    """A read-only sequence of float32 intensity images, from any source.

    Common base for every intensity sequence -- whether the images come from
    one acquisition (`IntensityBinFolder`) or an arbitrary `IntensityBinList`
    of unrelated files; annotate parameters with it to accept any of them.
    Each item is a float32 intensity image; `M` is the per-item metadata type
    chosen by the concrete sequence (e.g. the source `Path`).
    """


class UniformIntensitySequence[M](IntensitySequence[M]):
    """An `IntensitySequence` whose images all share the same pixel dimensions.

    Adds `frame_shape`, which is well-defined only under that uniformity (e.g.
    a single-acquisition folder). Heterogeneous sequences stay a plain
    `IntensitySequence` instead.
    """

    @property
    @abstractmethod
    def frame_shape(self) -> tuple[int, int]:
        """The pixel dimensions (height, width) shared by every image."""
        raise NotImplementedError
