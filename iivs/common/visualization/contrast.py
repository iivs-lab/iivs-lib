from __future__ import annotations

__all__ = ("auto_rescale",)

from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray


def auto_rescale[T: np.number](
    array: NDArray[T],
    *,
    saturated: float = 0.35,
    out_range: tuple[float, float] | None = None,
) -> NDArray[T]:
    """Auto-stretch `array`'s contrast so `saturated`% of its pixels clip.

    ImageJ's Enhance Contrast with Normalize: the range is the percentiles leaving
    `saturated`% of pixels clipped in total (`nanpercentile`, so a masked NaN is
    ignored), stretched onto `out_range` and cast back to `array`'s dtype. Every
    element is pooled for one range, whatever the shape.

    Args:
        array: The array to rescale (a frame, a stack, or any shape).
        saturated: Percent of pixels to let clip, split across both tails. Defaults to
            0.35 (ImageJ's default); 0 keeps the full range.
        out_range: The `(min, max)` output span, or None to fill the dtype's range (an
            integer's limits, or `(0.0, 1.0)` for a float). Defaults to None.

    Returns:
        The rescaled array in `array`'s dtype (integers rounded to nearest).

    Raises:
        ValueError: If `array` is empty, or `saturated` is not in [0, 100).
    """
    if not 0 <= saturated < 100:
        msg = f"saturated must be in [0, 100) (got {saturated})"
        raise ValueError(msg)

    if array.size == 0:
        msg = "array must be non-empty"
        raise ValueError(msg)

    if out_range is None:
        if np.issubdtype(array.dtype, np.floating):
            out_min, out_max = 0.0, 1.0
        else:
            dtype = cast("np.dtype[np.integer[Any]]", array.dtype)
            info = np.iinfo(dtype)
            out_min, out_max = float(info.min), float(info.max)
    else:
        out_min, out_max = out_range

    half = saturated / 2.0
    vmin, vmax = np.nanpercentile(array, (half, 100.0 - half))
    vmin, vmax = float(vmin), float(vmax)
    if vmax <= vmin:
        vmin, vmax = float(np.nanmin(array)), float(np.nanmax(array))

    values = np.asarray(array, dtype=np.float64)
    if vmax <= vmin:
        scaled = np.full(values.shape, out_min)
    else:
        normalized = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
        scaled = out_min + normalized * (out_max - out_min)

    if np.issubdtype(array.dtype, np.integer):
        scaled = np.rint(scaled)

    return scaled.astype(array.dtype)
