from __future__ import annotations

__all__ = ("normalize", "render")

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray


def normalize(
    image: NDArray[np.generic],
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> NDArray[np.float64]:
    """Scale an image to the unit range [0, 1] for display.

    Values are clipped to ``[vmin, vmax]`` then linearly mapped onto [0, 1]. A
    degenerate span (``vmin == vmax``) maps to all-zeros rather than dividing by
    a zero range. `image` is never modified.

    Args:
        image: The array to normalize.
        vmin: Lower bound of the display range; defaults to the array minimum.
        vmax: Upper bound of the display range; defaults to the array maximum.

    Returns:
        A float64 array of the same shape, with values in [0, 1].

    Raises:
        ValueError: If `vmin` exceeds `vmax`.
    """
    data = image.astype(np.float64, copy=False)
    lo = float(data.min()) if vmin is None else float(vmin)
    hi = float(data.max()) if vmax is None else float(vmax)
    if lo > hi:
        msg = f"vmin must be <= vmax (got {lo} > {hi})"
        raise ValueError(msg)

    span = hi - lo
    if span == 0:
        return np.zeros_like(data)

    return np.clip((data - lo) / span, 0.0, 1.0)


def render(
    image: NDArray[np.generic],
    *,
    ax: Axes | None = None,
    cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = False,
    title: str | None = None,
) -> Axes:
    """Draw a 2-D image array on a matplotlib `Axes`.

    Creates a new figure and `Axes` when `ax` is None, and returns the `Axes`
    either way, so the caller owns showing or saving (the data layer holds no
    `.show()` methods). The display range ``[vmin, vmax]`` defaults to the data's
    own min/max, matching `Axes.imshow`.

    Args:
        image: The 2-D image to draw, of shape (H, W).
        ax: The `Axes` to draw on; a new figure / `Axes` is created when None.
        cmap: The matplotlib colormap name. Defaults to "gray".
        vmin: Lower display bound; defaults to the data minimum.
        vmax: Upper display bound; defaults to the data maximum.
        colorbar: Whether to attach a colorbar to `ax`. Defaults to False.
        title: An optional title set on `ax`.

    Returns:
        The `Axes` the image was drawn on.

    Raises:
        ValueError: If `image` is not a 2-D array.
    """
    if image.ndim != 2:
        msg = f"image must be a 2D array (got {image.ndim}D)"
        raise ValueError(msg)

    if ax is None:
        _, ax = plt.subplots()

    mappable = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    if colorbar:
        ax.figure.colorbar(mappable, ax=ax)
    if title is not None:
        ax.set_title(title)

    return ax
