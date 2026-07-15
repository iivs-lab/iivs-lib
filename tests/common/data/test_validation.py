from __future__ import annotations

import numpy as np
import pytest

from iivs.common.data import (
    validate_float32_array,
    validate_float_array,
    validate_uint8_array,
    validate_uint_array,
)

# ========================== #
#  validate_float32_array    #
# ========================== #


def test_rejects_below_2d():
    with pytest.raises(ValueError, match="2-dimensional"):
        validate_float32_array(np.zeros(5, dtype=np.float32))


def test_rejects_non_float32():
    with pytest.raises(ValueError, match="float32"):
        validate_float32_array(np.zeros((2, 2), dtype=np.float64))


def test_rejects_unknown_on_nonfinite():
    with pytest.raises(ValueError, match="on_nonfinite must be"):
        validate_float32_array(
            np.zeros((2, 2), dtype=np.float32),
            on_nonfinite="bogus",  # ty: ignore[invalid-argument-type]
        )


def test_ignore_accepts_nonfinite():
    data = np.array([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float32)
    assert validate_float32_array(data, on_nonfinite="ignore") is data


def test_warns_on_nonfinite():
    data = np.array([[np.inf, 1.0], [2.0, 3.0]], dtype=np.float32)
    with pytest.warns(RuntimeWarning, match="finite"):
        validate_float32_array(data, on_nonfinite="warn")


def test_raises_on_nonfinite():
    data = np.array([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="finite"):
        validate_float32_array(data, on_nonfinite="raise")


def test_clean_returns_input():
    data = np.zeros((2, 2), dtype=np.float32)
    assert validate_float32_array(data) is data  # default "warn", finite: as-is


# ========================== #
#  validate_uint8_array      #
# ========================== #


def test_uint8_rejects_non_uint8():
    with pytest.raises(ValueError, match="uint8"):
        validate_uint8_array(np.zeros((2, 2), dtype=np.float32))


def test_uint8_accepts_stack_by_default():
    data = np.zeros((3, 2, 2), dtype=np.uint8)
    assert validate_uint8_array(data) is data


def test_uint8_clean_2d_returns_input():
    data = np.zeros((2, 2), dtype=np.uint8)
    assert validate_uint8_array(data) is data


def test_no_stack_rejects_higher_dims():
    # allow_stack=False requires a single 2D array, for both dtypes.
    with pytest.raises(ValueError, match="single 2D array"):
        validate_uint8_array(np.zeros((2, 2, 3), dtype=np.uint8), allow_stack=False)
    with pytest.raises(ValueError, match="single 2D array"):
        validate_float32_array(np.zeros((2, 2, 3), dtype=np.float32), allow_stack=False)


def test_no_stack_accepts_2d():
    data = np.zeros((2, 2), dtype=np.uint8)
    assert validate_uint8_array(data, allow_stack=False) is data


# ========================== #
#  dtype-parametric core     #
# ========================== #


def test_float_array_accepts_non_default_dtype():
    data = np.zeros((2, 2), dtype=np.float64)
    assert validate_float_array(data, dtype=np.float64) is data


def test_float_array_rejects_dtype_mismatch():
    # data is float32 but float64 was requested.
    with pytest.raises(ValueError, match="float64"):
        validate_float_array(np.zeros((2, 2), dtype=np.float32), dtype=np.float64)


def test_float_array_rejects_non_floating_dtype():
    with pytest.raises(ValueError, match="dtype must be a subtype of floating"):
        validate_float_array(np.zeros((2, 2), dtype=np.float32), dtype=np.uint8)


def test_float_array_keeps_finite_policy():
    data = np.array([[np.nan, 1.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="finite"):
        validate_float_array(data, dtype=np.float64, on_nonfinite="raise")


def test_uint_array_accepts_uint16():
    data = np.zeros((2, 2), dtype=np.uint16)
    assert validate_uint_array(data, dtype=np.uint16) is data


def test_uint_array_rejects_dtype_mismatch():
    with pytest.raises(ValueError, match="uint16"):
        validate_uint_array(np.zeros((2, 2), dtype=np.uint8), dtype=np.uint16)


def test_uint_array_rejects_non_unsigned_dtype():
    with pytest.raises(ValueError, match="dtype must be a subtype of unsignedinteger"):
        validate_uint_array(np.zeros((2, 2), dtype=np.uint8), dtype=np.int16)


def test_float_array_custom_dims_accepts_volume():
    # ndim=3 treats the trailing three axes as core; a 3-D array is one volume.
    data = np.zeros((2, 2, 2), dtype=np.float32)
    assert validate_float_array(data, ndim=3, allow_stack=False) is data


def test_float_array_custom_dims_rejects_too_few():
    with pytest.raises(ValueError, match="at least 3-dimensional"):
        validate_float_array(np.zeros((2, 2), dtype=np.float32), ndim=3)


def test_uint_array_custom_dims_accepts_stacked_volume():
    data = np.zeros((5, 2, 2, 2), dtype=np.uint16)
    assert validate_uint_array(data, dtype=np.uint16, ndim=3) is data
