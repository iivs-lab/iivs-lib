from __future__ import annotations

__all__ = ("PhaseUnit", "convert_phase_unit", "resolve_height_scale")

import math
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
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


_NM_PER_M = 1e9


def resolve_height_scale(
    height_scale: float | None,
    wavelength: float | None,
    refractive_delta: float | None,
) -> float:
    """Return `height_scale`, or derive it from `wavelength` and `refractive_delta`.

    The phase-to-height factor (m per rad) is given either directly, or as a
    `wavelength` / `refractive_delta` pair (`height per rad = wavelength /
    (2*pi * refractive_delta)`). Exactly one of the two forms must be given.

    Raises:
        ValueError: Unless exactly one form is fully given (neither, both, or a
            half-filled pair all raise).
    """
    if height_scale is not None and wavelength is None and refractive_delta is None:
        return height_scale
    if height_scale is None and wavelength is not None and refractive_delta is not None:
        return wavelength / (math.tau * refractive_delta)

    msg = "give exactly one of: height_scale, or wavelength and refractive_delta"
    raise ValueError(msg)


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
