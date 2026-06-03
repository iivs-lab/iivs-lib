from __future__ import annotations

from iivs.dhm.data.intensity.base import IntensitySequence, UniformIntensitySequence
from iivs.dhm.data.intensity.bin import IntensityBinFolder, IntensityBinList


def test_intensity_sequence_hierarchy():
    assert issubclass(UniformIntensitySequence, IntensitySequence)
    # A single-acquisition folder is uniform; an arbitrary file list is not.
    assert issubclass(IntensityBinFolder, UniformIntensitySequence)
    assert issubclass(IntensityBinList, IntensitySequence)
    assert not issubclass(IntensityBinList, UniformIntensitySequence)
