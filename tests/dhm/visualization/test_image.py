from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from iivs.dhm.data.phase.bounds import PhaseBounds
from iivs.dhm.visualization import render_hologram, render_intensity, render_phase

# ========================== #
#        render_phase        #
# ========================== #


def test_render_phase_defaults_to_colormap_and_colorbar():
    ax = render_phase(np.zeros((2, 2), dtype=np.float32))
    assert ax.images[0].get_cmap().name == "viridis"
    assert len(ax.figure.axes) == 2  # image + colorbar
    plt.close(ax.figure)


def test_render_phase_bounds_set_clim():
    bounds = PhaseBounds(min_nm=-400.0, max_nm=600.0)
    ax = render_phase(np.zeros((2, 2), dtype=np.float32), bounds=bounds)
    assert ax.images[0].get_clim() == (-400.0, 600.0)
    plt.close(ax.figure)


def test_render_phase_without_colorbar():
    ax = render_phase(np.zeros((2, 2), dtype=np.float32), colorbar=False)
    assert len(ax.figure.axes) == 1
    plt.close(ax.figure)


# ========================== #
#  render_intensity / holo   #
# ========================== #


def test_render_intensity_is_grayscale_without_colorbar():
    ax = render_intensity(np.zeros((2, 2), dtype=np.float32))
    assert ax.images[0].get_cmap().name == "gray"
    assert len(ax.figure.axes) == 1
    plt.close(ax.figure)


def test_render_hologram_is_grayscale_without_colorbar():
    ax = render_hologram(np.zeros((2, 2), dtype=np.uint8))
    assert ax.images[0].get_cmap().name == "gray"
    assert len(ax.figure.axes) == 1
    plt.close(ax.figure)
