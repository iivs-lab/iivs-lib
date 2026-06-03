from __future__ import annotations

__all__ = ("PhaseUnit", "convert_phase_unit", "validate_phase")

from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

from iivs.dhm.data.image import validate_float32_image

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

    A phase-named alias of `validate_float32_image`; see it for the full
    contract (float32 dtype, at least two dimensions, and the `on_nonfinite`
    policy).
    """
    return validate_float32_image(data, on_nonfinite=on_nonfinite)


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
