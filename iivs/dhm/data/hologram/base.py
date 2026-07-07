from __future__ import annotations

__all__ = ("HologramSequence",)

import numpy as np
from kaparoo.data.sequences import DataSequence
from numpy.typing import NDArray


class HologramSequence[M](DataSequence[NDArray[np.uint8], M]):
    """A read-only sequence of uint8 hologram images, from any source.

    Common base for every hologram sequence (a single-acquisition `HologramTifFolder` /
    `HologramRawFile`, or an arbitrary `HologramTifList` of unrelated files); annotate
    parameters with it to accept any of them. Each item is a uint8 hologram; `M` is the
    per-item metadata type chosen by the concrete sequence.

    Same-shape sources additionally expose `frame_shape`.
    """
