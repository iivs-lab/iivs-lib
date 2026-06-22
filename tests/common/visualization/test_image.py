from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from iivs.common.visualization import normalize, render

# ========================== #
#         normalize          #
# ========================== #


def test_normalize_default_range():
    img = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    out = normalize(img)
    np.testing.assert_allclose(out, [[0.0, 1 / 3], [2 / 3, 1.0]])
    assert out.dtype == np.float64


def test_normalize_explicit_range():
    img = np.array([[0.0, 5.0, 10.0]], dtype=np.float32)
    out = normalize(img, vmin=0.0, vmax=10.0)
    np.testing.assert_allclose(out, [[0.0, 0.5, 1.0]])


def test_normalize_clips_out_of_range():
    img = np.array([[-5.0, 5.0, 15.0]], dtype=np.float32)
    out = normalize(img, vmin=0.0, vmax=10.0)
    np.testing.assert_allclose(out, [[0.0, 0.5, 1.0]])


def test_normalize_degenerate_span_is_zeros():
    img = np.full((2, 2), 7.0, dtype=np.float32)
    np.testing.assert_array_equal(normalize(img), np.zeros((2, 2)))


def test_normalize_rejects_inverted_range():
    with pytest.raises(ValueError, match="vmin must be <= vmax"):
        normalize(np.zeros((2, 2), dtype=np.float32), vmin=1.0, vmax=0.0)


def test_normalize_does_not_mutate_input():
    img = np.array([[0.0, 2.0]], dtype=np.float32)
    original = img.copy()
    normalize(img)
    np.testing.assert_array_equal(img, original)


# ========================== #
#           render           #
# ========================== #


def test_render_creates_axes_and_draws_image():
    img = np.arange(6, dtype=np.float32).reshape(2, 3)
    ax = render(img)
    assert len(ax.images) == 1
    np.testing.assert_array_equal(np.asarray(ax.images[0].get_array()), img)
    plt.close(ax.figure)


def test_render_uses_given_axes():
    _, ax = plt.subplots()
    returned = render(np.zeros((2, 2), dtype=np.float32), ax=ax)
    assert returned is ax
    plt.close(ax.figure)


def test_render_applies_cmap_and_clim():
    ax = render(np.zeros((2, 2), dtype=np.float32), cmap="viridis", vmin=0.0, vmax=1.0)
    assert ax.images[0].get_cmap().name == "viridis"
    assert ax.images[0].get_clim() == (0.0, 1.0)
    plt.close(ax.figure)


def test_render_colorbar_adds_axes():
    # The colorbar is drawn on a second Axes added to the figure.
    ax = render(np.zeros((2, 2), dtype=np.float32), colorbar=True)
    assert len(ax.figure.axes) == 2
    plt.close(ax.figure)


def test_render_without_colorbar_keeps_single_axes():
    ax = render(np.zeros((2, 2), dtype=np.float32), colorbar=False)
    assert len(ax.figure.axes) == 1
    plt.close(ax.figure)


def test_render_sets_title():
    ax = render(np.zeros((2, 2), dtype=np.float32), title="frame 0")
    assert ax.get_title() == "frame 0"
    plt.close(ax.figure)


def test_render_rejects_non_2d():
    with pytest.raises(ValueError, match="image must be a 2D array"):
        render(np.zeros((2, 2, 3), dtype=np.float32))
