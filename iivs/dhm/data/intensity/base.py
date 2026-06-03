from __future__ import annotations

__all__ = ("IntensitySequence",)

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

    Same-shape sources additionally mix in `data.sequence.FrameShapedMixin` to
    expose `frame_shape`.
    """
