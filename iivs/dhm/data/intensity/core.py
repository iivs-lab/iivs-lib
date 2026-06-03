from __future__ import annotations

__all__ = ("validate_intensity",)

from typing import TYPE_CHECKING

from iivs.dhm.data.image import validate_float32_image

if TYPE_CHECKING:
    from typing import Literal

    import numpy as np
    from numpy.typing import NDArray


def validate_intensity(
    data: NDArray[np.float32],
    *,
    on_nonfinite: Literal["ignore", "warn", "raise"] = "warn",
) -> NDArray[np.float32]:
    """Validate a float32 intensity image or stack and return it.

    An intensity-named alias of `validate_float32_image` (intensity carries no
    physical unit); see it for the full contract.
    """
    return validate_float32_image(data, on_nonfinite=on_nonfinite)
