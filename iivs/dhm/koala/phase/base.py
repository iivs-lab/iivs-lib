from __future__ import annotations

__all__ = ("PhaseSequence", "UniformPhaseSequence")

from abc import abstractmethod

import numpy as np
from kaparoo.data.sequences import DataSequence
from numpy.typing import NDArray


class PhaseSequence[M](DataSequence[NDArray[np.float32], M]):
    """A read-only sequence of float32 phase images, from any source.

    Common base for every phase sequence -- whether the images come from one
    acquisition (`PhaseBinFolder`) or an arbitrary `PhaseBinList` of
    unrelated files; annotate parameters with it to accept any of them. Each
    item is a float32 phase image; `M` is the per-item metadata type chosen by
    the concrete sequence (e.g. the source `Path`).
    """


class UniformPhaseSequence[M](PhaseSequence[M]):
    """A `PhaseSequence` whose images all share the same pixel dimensions.

    Adds `frame_shape`, which is well-defined only under that uniformity (e.g.
    a single-acquisition folder or `.bin` file). Heterogeneous sequences stay
    a plain `PhaseSequence` instead.
    """

    @property
    @abstractmethod
    def frame_shape(self) -> tuple[int, int]:
        """The pixel dimensions (height, width) shared by every image."""
        raise NotImplementedError
