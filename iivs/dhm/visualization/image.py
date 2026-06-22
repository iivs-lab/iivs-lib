from __future__ import annotations

__all__ = ("render_hologram", "render_intensity", "render_phase")

from typing import TYPE_CHECKING

from iivs.common.visualization import render

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from iivs.dhm.data.phase.bounds import PhaseBounds


def render_phase(
    image: NDArray[np.floating],
    *,
    bounds: PhaseBounds | None = None,
    ax: Axes | None = None,
    cmap: str = "viridis",
    colorbar: bool = True,
    title: str | None = None,
) -> Axes:
    """Render a phase image with a colormap and colorbar.

    Phase is a continuous quantity, so it renders with a perceptual colormap and
    a colorbar by default -- unlike the grayscale intensity / hologram views.
    Pass a `PhaseBounds` to fix the display range to the acquisition's global
    ``[min_nm, max_nm]``; `image` must then be in nanometers for that range to be
    meaningful.

    Args:
        image: The 2-D phase image to draw.
        bounds: Optional nm display bounds; sets the range from `bounds.min_nm`
            and `bounds.max_nm`.
        ax: The `Axes` to draw on; a new figure / `Axes` is created when None.
        cmap: The colormap name. Defaults to "viridis".
        colorbar: Whether to attach a colorbar. Defaults to True.
        title: An optional `Axes` title.

    Returns:
        The `Axes` the phase was drawn on.
    """
    vmin = None if bounds is None else bounds.min_nm
    vmax = None if bounds is None else bounds.max_nm
    return render(
        image, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, colorbar=colorbar, title=title
    )


def render_intensity(
    image: NDArray[np.floating],
    *,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Render an intensity image in grayscale.

    Args:
        image: The 2-D intensity image to draw.
        ax: The `Axes` to draw on; a new figure / `Axes` is created when None.
        title: An optional `Axes` title.

    Returns:
        The `Axes` the intensity was drawn on.
    """
    return render(image, ax=ax, cmap="gray", title=title)


def render_hologram(
    image: NDArray[np.uint8],
    *,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Render a hologram image in grayscale.

    Args:
        image: The 2-D uint8 hologram image to draw.
        ax: The `Axes` to draw on; a new figure / `Axes` is created when None.
        title: An optional `Axes` title.

    Returns:
        The `Axes` the hologram was drawn on.
    """
    return render(image, ax=ax, cmap="gray", title=title)
