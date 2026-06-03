from __future__ import annotations

__all__ = ("PhaseUnit", "convert_phase_unit", "validate_phase")

import warnings
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Literal

    from numpy.typing import NDArray


class PhaseUnit(IntEnum):
    """Physical unit of a phase image.

    UNKNOWN, RADIANS, and METERS are the units stored on disk (the
    ``unit_code`` byte). NANOMETERS is a code-only convenience unit, never
    written to a file: saving converts it to METERS.
    """

    UNKNOWN = 0
    RADIANS = 1
    METERS = 2
    NANOMETERS = 3


def validate_phase(
    data: NDArray[np.float32],
    *,
    on_nonfinite: Literal["ignore", "warn", "raise"] = "warn",
) -> NDArray[np.float32]:
    """Validate a float32 phase image or stack and return it.

    The last two axes are the image height and width; any number of
    leading axes are allowed, so both a single 2-D image and a
    higher-dimensional stack are accepted. `data` is never modified.

    Args:
        data: The phase image or stack to validate, of shape (..., H, W).
        on_nonfinite: How to handle non-finite values (NaN, +inf, -inf):
            "ignore" accepts them silently, "warn" (default) accepts them
            but emits a RuntimeWarning, "raise" raises a ValueError.

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


_NM_PER_M = 1e9


def convert_phase_unit(
    data: NDArray[np.float32],
    *,
    source: PhaseUnit,
    target: PhaseUnit,
    height_scale: float,
) -> NDArray[np.float32]:
    """Rescale a phase/height image between `PhaseUnit` representations.

    The units form the chain RADIANS <-> METERS <-> NANOMETERS: phase in
    RADIANS and height in METERS are bridged by `height_scale` (m per
    rad), while METERS and NANOMETERS differ by the fixed 1e9 nm/m.

    Args:
        data: The phase/height image to convert; never modified.
        source: The unit `data` is currently in.
        target: The unit to convert to.
        height_scale: m per rad. Used only when the conversion
            crosses RADIANS <-> METERS; ignored for a pure METERS <->
            NANOMETERS rescale.

    Returns:
        A new float32 array in `target`, or `data` itself (unchanged) when
        `source == target`.

    Raises:
        ValueError: If the conversion is undefined (e.g. it involves the
            UNKNOWN unit).
    """
    if source is target:
        return data

    # `scale` is defined in ascending unit order (RADIANS < METERS <
    # NANOMETERS); converting the other way uses its reciprocal.
    match sorted((source, target)):
        case [PhaseUnit.RADIANS, PhaseUnit.METERS]:
            scale = height_scale
        case [PhaseUnit.METERS, PhaseUnit.NANOMETERS]:
            scale = _NM_PER_M
        case [PhaseUnit.RADIANS, PhaseUnit.NANOMETERS]:
            scale = height_scale * _NM_PER_M
        case _:
            msg = f"cannot convert phase from {source.name} to {target.name}"
            raise ValueError(msg)

    if source > target:
        scale = 1.0 / scale
    return (data * scale).astype(np.float32, copy=False)
