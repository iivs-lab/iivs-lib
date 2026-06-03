from __future__ import annotations

__all__ = ("HologramSequence", "UniformHologramSequence")

from abc import abstractmethod

import numpy as np
from kaparoo.data.sequences import DataSequence
from numpy.typing import NDArray


class HologramSequence[M](DataSequence[NDArray[np.uint8], M]):
    """A read-only sequence of uint8 hologram images, from any source.

    Common base for every hologram sequence -- a single-acquisition
    `HologramTifSequence` / `HologramRawSequence`, or an arbitrary
    `HologramTifList` of unrelated files; annotate parameters with it
    to accept any of them. Each item is a uint8 hologram; `M` is the per-item
    metadata type chosen by the concrete sequence.
    """


class UniformHologramSequence[M](HologramSequence[M]):
    """A `HologramSequence` whose images all share the same pixel dimensions.

    Adds `frame_shape`, which is well-defined only under that uniformity (a
    single acquisition). Heterogeneous sequences stay a plain
    `HologramSequence` instead.
    """

    @property
    @abstractmethod
    def frame_shape(self) -> tuple[int, int]:
        """The pixel dimensions (height, width) shared by every hologram."""
        raise NotImplementedError
