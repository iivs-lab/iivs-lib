from __future__ import annotations

import numpy as np
import pytest

from iivs.dhm.data.intensity.core import validate_intensity


def test_validate_intensity_rejects_below_2d():
    with pytest.raises(ValueError, match="2-dimensional"):
        validate_intensity(np.zeros(5, dtype=np.float32))


def test_validate_intensity_rejects_non_float32():
    with pytest.raises(ValueError, match="float32"):
        validate_intensity(np.zeros((2, 2), dtype=np.float64))


def test_validate_intensity_rejects_unknown_on_nonfinite():
    with pytest.raises(ValueError, match="on_nonfinite must be"):
        validate_intensity(np.zeros((2, 2), dtype=np.float32), on_nonfinite="bogus")  # ty: ignore[invalid-argument-type]


def test_validate_intensity_ignore_accepts_nonfinite():
    data = np.array([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float32)
    assert validate_intensity(data, on_nonfinite="ignore") is data


def test_validate_intensity_warns_on_nonfinite():
    data = np.array([[np.inf, 1.0], [2.0, 3.0]], dtype=np.float32)
    with pytest.warns(RuntimeWarning, match="finite"):
        validate_intensity(data, on_nonfinite="warn")


def test_validate_intensity_raises_on_nonfinite():
    data = np.array([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="finite"):
        validate_intensity(data, on_nonfinite="raise")


def test_validate_intensity_clean_returns_input():
    data = np.zeros((2, 2), dtype=np.float32)
    assert validate_intensity(data) is data  # default "warn", finite: returned as-is
