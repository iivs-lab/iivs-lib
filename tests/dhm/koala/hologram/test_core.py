from __future__ import annotations

import numpy as np
import pytest

from iivs.dhm.koala.hologram.core import validate_hologram


def test_validate_hologram_rejects_non_2d():
    with pytest.raises(ValueError, match="single 2D image"):
        validate_hologram(np.zeros((2, 2, 3), dtype=np.uint8))


def test_validate_hologram_rejects_non_uint8():
    with pytest.raises(ValueError, match="uint8"):
        validate_hologram(np.zeros((2, 2), dtype=np.float32))
