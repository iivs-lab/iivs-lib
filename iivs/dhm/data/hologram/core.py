from __future__ import annotations

__all__ = ("validate_hologram",)

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def validate_hologram(data: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Validate a single uint8 hologram image and return it.

    `data` is never modified.

    Raises:
        ValueError: If `data` is not a 2D uint8 array.
    """
    if data.ndim != 2:
        msg = f"hologram must be a single 2D image (got shape {data.shape})"
        raise ValueError(msg)

    if data.dtype != np.uint8:
        msg = f"hologram must be uint8 (got {data.dtype})"
        raise ValueError(msg)

    return data
