from __future__ import annotations

__all__ = ("HologramSequence",)

from abc import abstractmethod

import numpy as np
from kaparoo.data.sequences import DataSequence
from numpy.typing import NDArray


class HologramSequence[M](DataSequence[NDArray[np.uint8], M]):
    """A read-only sequence of uint8 hologram images, from any source.

    Common base for `HologramTifSequence` (a `.tif` folder) and
    `HologramRawSequence` (a single multi-frame `.raw` file); annotate
    parameters with it to accept either. Each item is a uint8 hologram; `M`
    is the per-item metadata type chosen by the concrete sequence.
    """

    @property
    @abstractmethod
    def frame_shape(self) -> tuple[int, int]:
        """The pixel dimensions (height, width) shared by every hologram."""
        raise NotImplementedError
