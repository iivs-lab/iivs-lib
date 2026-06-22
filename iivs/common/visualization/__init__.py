"""Technique-agnostic image rendering via matplotlib.

Functions that draw 2-D image arrays on a matplotlib `Axes` (`render`) and scale
them for display (`normalize`). Matplotlib is a core dependency -- rendering
microscope image data is a primary job of this library -- so this package needs
no extra; each technique's `<technique>.visualization` adapter (e.g.
`iivs.dhm.visualization`) adds its display semantics on top. The data layer
itself stays matplotlib-free.
"""

from __future__ import annotations

__all__ = ("normalize", "render")

from iivs.common.visualization.image import normalize, render
