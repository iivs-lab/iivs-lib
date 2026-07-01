from __future__ import annotations

__all__ = (
    "OnNonFinite",
    "validate_float32_array",
    "validate_float_array",
    "validate_uint8_array",
    "validate_uint_array",
)

import warnings
from typing import TYPE_CHECKING

import numpy as np
from kaparoo.utils import ensure_one_of

if TYPE_CHECKING:
    from typing import Literal

    from numpy.typing import DTypeLike, NDArray


type OnNonFinite = Literal["ignore", "warn", "raise"]
"""How a floating array validator treats non-finite values (NaN, +/-inf)."""


def validate_dims[T: np.generic](
    array: NDArray[T], *, dims: int = 2, allow_stack: bool = True
) -> NDArray[T]:
    """Validate an array's dimensionality and return it.

    `dims` is the core (trailing) dimensionality -- 2 for an (H, W) image. With
    `allow_stack`, any number of leading axes may precede it (`array.ndim >=
    dims`); without it, exactly `dims` axes are required. `array` is never
    modified.

    Args:
        array: The array to check.
        dims: The core (trailing) dimensionality. Defaults to 2.
        allow_stack: Whether to accept leading axes beyond `dims`. Defaults to
            True; set False to require exactly `dims` axes.

    Raises:
        ValueError: If `array` has fewer than `dims` axes, or -- when
            `allow_stack` is False -- not exactly `dims`.
    """

    if allow_stack:
        if array.ndim < dims:
            msg = f"array must be at least {dims}-dimensional (got {array.ndim})"
            raise ValueError(msg)

    elif array.ndim != dims:
        msg = f"array must be a single {dims}D array (got shape {array.shape})"
        raise ValueError(msg)

    return array


def validate_dtype[T: np.generic](
    array: NDArray[T], *, dtype: DTypeLike, kind: type[np.generic]
) -> NDArray[T]:
    """Validate `array`'s dtype is exactly `dtype`, a subtype of `kind`, and return it.

    `kind` is the abstract numpy scalar family `dtype` must belong to (e.g.
    `np.floating`); its `__name__` names that family in the error message, so it
    cannot drift from the `issubdtype` check. `array` is never modified.

    Raises:
        ValueError: If `dtype` is not a subtype of `kind`, or if `array`'s dtype
            is not exactly `dtype`.
    """

    expected = np.dtype(dtype)

    if not np.issubdtype(expected, kind):
        msg = f"dtype must be a subtype of {kind.__name__} (got {expected})"
        raise ValueError(msg)

    if array.dtype != expected:
        msg = f"array must be {expected} (got {array.dtype})"
        raise ValueError(msg)

    return array


def validate_float_array[F: np.floating](
    array: NDArray[F],
    *,
    dtype: DTypeLike = np.float32,
    dims: int = 2,
    on_nonfinite: OnNonFinite = "warn",
    allow_stack: bool = True,
) -> NDArray[F]:
    """Validate a floating-point array (or stack) of `dtype` and return it.

    The trailing `dims` axes are the array's core dimensions (2 = an H x W image
    by default). Any number of leading axes is allowed (a stack); pass
    `allow_stack=False` to require exactly `dims` axes. `array` is never modified.

    Modality-agnostic over the floating dtype: phase and intensity validate their
    float32 arrays through the `validate_float32_array` binding.

    Args:
        array: The array or stack to validate, of shape (..., H, W).
        dtype: The exact floating dtype `array` must carry. Defaults to float32.
        dims: The core (trailing) dimensionality, excluding stacked leading axes.
            Defaults to 2 (an H x W image).
        on_nonfinite: How to handle non-finite values (NaN, +inf, -inf):
            "ignore" accepts them silently, "warn" (default) accepts them but
            emits a RuntimeWarning, "raise" raises a ValueError.
        allow_stack: Whether to accept leading axes beyond `dims`. Defaults to
            True; set False to require exactly `dims` axes.

    Raises:
        ValueError: If `dtype` is not a floating type; if `array`'s dtype is not
            `dtype`, has the wrong dimensionality, or holds non-finite values
            while `on_nonfinite` is "raise".
    """
    array = validate_dims(array, dims=dims, allow_stack=allow_stack)
    array = validate_dtype(array, dtype=dtype, kind=np.floating)

    ensure_one_of(on_nonfinite, ("ignore", "warn", "raise"), name="on_nonfinite")

    if on_nonfinite == "ignore":
        return array

    if not np.all(np.isfinite(array)):
        nan = int(np.isnan(array).sum())
        posinf = int(np.isposinf(array).sum())
        neginf = int(np.isneginf(array).sum())
        counts = f"{nan} NaN, {posinf} +inf, {neginf} -inf"

        if on_nonfinite == "raise":
            msg = f"array must be finite (got {counts})"
            raise ValueError(msg)

        msg = f"array is not finite ({counts})"
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    return array


def validate_uint_array[U: np.unsignedinteger](
    array: NDArray[U],
    *,
    dtype: DTypeLike = np.uint8,
    dims: int = 2,
    allow_stack: bool = True,
) -> NDArray[U]:
    """Validate an unsigned-integer array (or stack) of `dtype` and return it.

    The trailing `dims` axes are the array's core dimensions (2 = an H x W image
    by default). Any number of leading axes is allowed (a stack); pass
    `allow_stack=False` to require exactly `dims` axes. Unsigned integers are
    inherently finite, so there is no non-finite policy. `array` is never modified.

    Modality-agnostic over the unsigned dtype: holograms validate their uint8
    arrays through the `validate_uint8_array` binding; a future 16-bit source
    binds `np.uint16`.

    Args:
        array: The array or stack to validate, of shape (..., H, W).
        dtype: The exact unsigned-integer dtype `array` must carry. Defaults to
            uint8.
        dims: The core (trailing) dimensionality, excluding stacked leading axes.
            Defaults to 2 (an H x W image).
        allow_stack: Whether to accept leading axes beyond `dims`. Defaults to
            True; set False to require exactly `dims` axes.

    Raises:
        ValueError: If `dtype` is not an unsigned-integer type, or if `array`'s
            dtype is not `dtype` or has the wrong dimensionality.
    """
    array = validate_dims(array, dims=dims, allow_stack=allow_stack)
    return validate_dtype(array, dtype=dtype, kind=np.unsignedinteger)


def validate_float32_array(
    array: NDArray[np.float32],
    *,
    dims: int = 2,
    on_nonfinite: OnNonFinite = "warn",
    allow_stack: bool = True,
) -> NDArray[np.float32]:
    """Validate a float32 array (or stack) and return it.

    The float32 binding of `validate_float_array` -- see it for the shape,
    `dims`, and `on_nonfinite` details. `array` is never modified.
    """
    return validate_float_array(
        array,
        dtype=np.float32,
        dims=dims,
        on_nonfinite=on_nonfinite,
        allow_stack=allow_stack,
    )


def validate_uint8_array(
    array: NDArray[np.uint8], *, dims: int = 2, allow_stack: bool = True
) -> NDArray[np.uint8]:
    """Validate a uint8 array (or stack) and return it.

    The uint8 binding of `validate_uint_array` -- see it for the shape and `dims`
    details. uint8 is inherently finite, so there is no non-finite policy. `array`
    is never modified.
    """
    return validate_uint_array(
        array, dtype=np.uint8, dims=dims, allow_stack=allow_stack
    )
