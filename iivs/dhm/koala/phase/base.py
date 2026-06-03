from __future__ import annotations

__all__ = ("PhaseSequence",)

from abc import abstractmethod

import numpy as np
from kaparoo.data.sequences import DataSequence
from numpy.typing import NDArray


class PhaseSequence[M](DataSequence[NDArray[np.float32], M]):
    """A read-only sequence of float32 phase images, from any source.

    Common base for the concrete phase sequences (`PhaseBinSequence`, and
    future `.tif` / `.txt` variants); annotate parameters with it to accept
    any of them. Each item is a float32 phase image; `M` is the per-item
    metadata type chosen by the concrete sequence (e.g. the source `Path`).
    """

    @property
    @abstractmethod
    def frame_shape(self) -> tuple[int, int]:
        """The pixel dimensions (height, width) shared by every image."""
        raise NotImplementedError
