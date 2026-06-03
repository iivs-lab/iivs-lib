from __future__ import annotations

import numpy as np
import pytest

from iivs.dhm.data.intensity.core import validate_intensity


def test_validate_intensity_delegates_to_float32_validator():
    # The intensity-named alias enforces the shared float32-image contract.
    data = np.zeros((2, 2), dtype=np.float32)
    assert validate_intensity(data) is data
    with pytest.raises(ValueError, match="float32"):
        validate_intensity(np.zeros((2, 2), dtype=np.float64))
