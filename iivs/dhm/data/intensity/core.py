from __future__ import annotations

__all__ = ("validate_intensity",)

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Literal

    from numpy.typing import NDArray


def validate_intensity(
    data: NDArray[np.float32],
    *,
    on_nonfinite: Literal["ignore", "warn", "raise"] = "warn",
) -> NDArray[np.float32]:
    """Validate a float32 intensity image or stack and return it.

    The last two axes are the image height and width; any number of leading
    axes are allowed, so both a single 2-D image and a higher-dimensional
    stack are accepted. `data` is never modified.

    Intensity carries no physical unit (unlike phase), so this checks only
    dtype, dimensionality, and finiteness.

    Args:
        data: The intensity image or stack to validate, of shape (..., H, W).
        on_nonfinite: How to handle non-finite values (NaN, +inf, -inf):
            "ignore" accepts them silently, "warn" (default) accepts them but
            emits a RuntimeWarning, "raise" raises a ValueError.

    Raises:
        ValueError: If `data` is not a float32 array with at least two
            dimensions.
        ValueError: If `data` contains non-finite values and `on_nonfinite`
            is "raise".
    """
    if data.ndim < 2:
        msg = f"data must be at least 2-dimensional (got {data.ndim})"
        raise ValueError(msg)

    if data.dtype != np.float32:
        msg = f"data must be float32 (got {data.dtype})"
        raise ValueError(msg)

    match on_nonfinite:
        case "ignore":
            return data
        case "warn" | "raise":
            pass
        case _:
            msg = f"on_nonfinite must be 'ignore', 'warn', or 'raise' (got {on_nonfinite!r})"
            raise ValueError(msg)

    if not np.all(np.isfinite(data)):
        nan = int(np.isnan(data).sum())
        posinf = int(np.isposinf(data).sum())
        neginf = int(np.isneginf(data).sum())
        counts = f"{nan} NaN, {posinf} +inf, {neginf} -inf"
        if on_nonfinite == "raise":
            msg = f"data must be finite (got {counts})"
            raise ValueError(msg)
        msg = f"data is not finite ({counts})"
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    return data
