"""Display rendering for DHM data.

Thin adapters over `iivs.common.visualization` that add dhm display semantics:
phase renders with a colormap + colorbar (optionally over a `PhaseBounds` nm
range), while intensity and holograms render in grayscale.
"""

from __future__ import annotations

__all__ = ("render_hologram", "render_intensity", "render_phase")

from iivs.dhm.visualization.image import (
    render_hologram,
    render_intensity,
    render_phase,
)
