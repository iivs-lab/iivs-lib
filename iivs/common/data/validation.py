from __future__ import annotations

__all__ = (
    "OnNonFinite",
    "validate_dtype",
    "validate_float32_array",
    "validate_float_array",
    "validate_ndim",
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


def validate_ndim[T: np.generic](
    array: NDArray[T], *, ndim: int = 2, allow_stack: bool = True
) -> NDArray[T]:
    """Validate an array's dimensionality and return it.

    Only the axis *count* is checked. With `allow_stack` the array needs at least `ndim`
    axes; without it, exactly `ndim`. `array` is never modified.

    Args:
        array: The array to check.
        ndim: The required number of core dimensions. Defaults to 2.
        allow_stack: Whether to accept extra axes beyond `ndim` (a stack). Defaults to
            True; set False to require exactly `ndim` axes.

    Raises:
        ValueError: If `array` has fewer than `ndim` axes, or if `allow_stack` is False
            and it does not have exactly `ndim`.
    """

    if allow_stack:
        if array.ndim < ndim:
            msg = f"array must be at least {ndim}-dimensional (got {array.ndim})"
            raise ValueError(msg)

    elif array.ndim != ndim:
        msg = f"array must be a single {ndim}D array (got shape {array.shape})"
        raise ValueError(msg)

    return array


def validate_dtype[T: np.generic](
    array: NDArray[T], *, dtype: DTypeLike, kind: type[np.generic]
) -> NDArray[T]:
    """Validate `array`'s dtype is exactly `dtype`, a subtype of `kind`, and return it.

    `kind` is the abstract numpy scalar family `dtype` must belong to (e.g.
    `np.floating`); it also names that family in the error message, so the message
    cannot drift from the check. `array` is never modified.

    Raises:
        ValueError: If `dtype` is not a subtype of `kind`, or if `array`'s dtype is not
            exactly `dtype`.
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
    ndim: int = 2,
    on_nonfinite: OnNonFinite = "warn",
    allow_stack: bool = True,
) -> NDArray[F]:
    """Validate a floating-point array (or stack) of `dtype` and return it.

    The `ndim` core dimensions default to 2 (an H x W image). With a stack, extra axes
    add to the count; pass `allow_stack=False` to require exactly `ndim` axes. Only the
    axis count is checked. `array` is never modified.

    Args:
        array: The array or stack to validate.
        dtype: The exact floating dtype `array` must carry. Defaults to float32.
        ndim: The number of core dimensions, excluding any stacked axes. Defaults to 2
            (an H x W image).
        on_nonfinite: How to handle non-finite values (NaN, +inf, -inf): "ignore"
            accepts them silently, "warn" (default) accepts them but emits a
            RuntimeWarning, "raise" raises a ValueError.
        allow_stack: Whether to accept extra axes beyond `ndim` (a stack). Defaults to
            True; set False to require exactly `ndim` axes.

    Raises:
        ValueError: If `dtype` is not a floating type; if `array`'s dtype is not
            `dtype`, has the wrong dimensionality, or holds non-finite values while
            `on_nonfinite` is "raise".
    """
    array = validate_ndim(array, ndim=ndim, allow_stack=allow_stack)
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
    ndim: int = 2,
    allow_stack: bool = True,
) -> NDArray[U]:
    """Validate an unsigned-integer array (or stack) of `dtype` and return it.

    The `ndim` core dimensions default to 2 (an H x W image). With a stack, extra axes
    add to the count; pass `allow_stack=False` to require exactly `ndim` axes. Only the
    axis count is checked. Unsigned integers are inherently finite, so there is no
    non-finite policy. `array` is never modified.

    Args:
        array: The array or stack to validate.
        dtype: The exact unsigned-integer dtype `array` must carry. Defaults to uint8.
        ndim: The number of core dimensions, excluding any stacked axes. Defaults to 2
            (an H x W image).
        allow_stack: Whether to accept extra axes beyond `ndim` (a stack). Defaults to
            True; set False to require exactly `ndim` axes.

    Raises:
        ValueError: If `dtype` is not an unsigned-integer type, or if `array`'s dtype is
            not `dtype` or has the wrong dimensionality.
    """
    array = validate_ndim(array, ndim=ndim, allow_stack=allow_stack)
    return validate_dtype(array, dtype=dtype, kind=np.unsignedinteger)


def validate_float32_array(
    array: NDArray[np.float32],
    *,
    ndim: int = 2,
    on_nonfinite: OnNonFinite = "warn",
    allow_stack: bool = True,
) -> NDArray[np.float32]:
    """Validate a float32 array (or stack) and return it.

    The `ndim` core dimensions default to 2 (an H x W image); with a stack, extra axes
    add to the count (only the axis count is checked). Non-finite values (NaN, +/-inf)
    are handled per `on_nonfinite`. `array` is never modified.

    Args:
        array: The array or stack to validate.
        ndim: The number of core dimensions, excluding any stacked axes. Defaults to 2
            (an H x W image).
        on_nonfinite: How to handle non-finite values (NaN, +inf, -inf): "ignore"
            accepts them silently, "warn" (default) accepts them but emits a
            RuntimeWarning, "raise" raises a ValueError.
        allow_stack: Whether to accept extra axes beyond `ndim` (a stack). Defaults to
            True; set False to require exactly `ndim` axes.

    Raises:
        ValueError: If `array` is not float32, has the wrong dimensionality, or holds
            non-finite values while `on_nonfinite` is "raise".
    """
    return validate_float_array(
        array,
        dtype=np.float32,
        ndim=ndim,
        on_nonfinite=on_nonfinite,
        allow_stack=allow_stack,
    )


def validate_uint8_array(
    array: NDArray[np.uint8], *, ndim: int = 2, allow_stack: bool = True
) -> NDArray[np.uint8]:
    """Validate a uint8 array (or stack) and return it.

    The `ndim` core dimensions default to 2 (an H x W image); with a stack, extra axes
    add to the count (only the axis count is checked). uint8 is inherently finite, so
    there is no non-finite policy. `array` is never modified.

    Args:
        array: The array or stack to validate.
        ndim: The number of core dimensions, excluding any stacked axes. Defaults to 2
            (an H x W image).
        allow_stack: Whether to accept extra axes beyond `ndim` (a stack). Defaults to
            True; set False to require exactly `ndim` axes.

    Raises:
        ValueError: If `array` is not uint8 or has the wrong dimensionality.
    """
    return validate_uint_array(
        array, dtype=np.uint8, ndim=ndim, allow_stack=allow_stack
    )
