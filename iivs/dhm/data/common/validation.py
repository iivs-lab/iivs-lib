from __future__ import annotations

__all__ = ("OnNonFinite", "validate_float32_image", "validate_uint8_image")

import warnings
from typing import TYPE_CHECKING

import numpy as np
from kaparoo.utils import ensure_one_of

if TYPE_CHECKING:
    from typing import Literal

    from numpy.typing import NDArray


type OnNonFinite = Literal["ignore", "warn", "raise"]
"""How a float32 image validator treats non-finite values (NaN, +/-inf)."""


def _validate_image_dims(data: NDArray[np.generic], *, allow_stack: bool) -> None:
    """Check the leading dimensions of an image array.

    With `allow_stack`, any number of leading axes is allowed (at least two
    dims); without it, exactly a single 2-D image is required.
    """
    if allow_stack:
        if data.ndim < 2:
            msg = f"data must be at least 2-dimensional (got {data.ndim})"
            raise ValueError(msg)
    elif data.ndim != 2:
        msg = f"data must be a single 2D image (got shape {data.shape})"
        raise ValueError(msg)


def validate_float32_image(
    data: NDArray[np.float32],
    *,
    on_nonfinite: OnNonFinite = "warn",
    allow_stack: bool = True,
) -> NDArray[np.float32]:
    """Validate a float32 image (or stack) and return it.

    The last two axes are the image height and width. By default any number of
    leading axes is allowed (a stack); pass `allow_stack=False` to require a
    single 2-D image. `data` is never modified.

    Modality-agnostic: phase and intensity validate their float32 arrays
    through this.

    Args:
        data: The image or stack to validate, of shape (..., H, W).
        on_nonfinite: How to handle non-finite values (NaN, +inf, -inf):
            "ignore" accepts them silently, "warn" (default) accepts them but
            emits a RuntimeWarning, "raise" raises a ValueError.
        allow_stack: Whether to accept more than two dimensions. Defaults to
            True; set False to require a single 2-D image.

    Raises:
        ValueError: If `data` is not float32, has the wrong dimensionality, or
            holds non-finite values while `on_nonfinite` is "raise".
    """
    _validate_image_dims(data, allow_stack=allow_stack)

    if data.dtype != np.float32:
        msg = f"data must be float32 (got {data.dtype})"
        raise ValueError(msg)

    ensure_one_of(on_nonfinite, ("ignore", "warn", "raise"), name="on_nonfinite")
    if on_nonfinite == "ignore":
        return data

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


def validate_uint8_image(
    data: NDArray[np.uint8], *, allow_stack: bool = True
) -> NDArray[np.uint8]:
    """Validate a uint8 image (or stack) and return it.

    The last two axes are the image height and width. By default any number of
    leading axes is allowed (a stack); pass `allow_stack=False` to require a
    single 2-D image. uint8 is inherently finite, so there is no non-finite
    policy. `data` is never modified.

    Modality-agnostic: holograms (and any future 8-bit `.tif` phase/intensity
    sequences) validate their uint8 arrays through this.

    Args:
        data: The image or stack to validate, of shape (..., H, W).
        allow_stack: Whether to accept more than two dimensions. Defaults to
            True; set False to require a single 2-D image.

    Raises:
        ValueError: If `data` is not uint8 or has the wrong dimensionality.
    """
    _validate_image_dims(data, allow_stack=allow_stack)

    if data.dtype != np.uint8:
        msg = f"data must be uint8 (got {data.dtype})"
        raise ValueError(msg)

    return data
