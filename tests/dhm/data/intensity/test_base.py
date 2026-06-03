from __future__ import annotations

from iivs.dhm.data.intensity.base import IntensitySequence
from iivs.dhm.data.intensity.bin import IntensityBinFolder, IntensityBinList
from iivs.dhm.data.sequence import FrameShapedMixin


def test_intensity_sequence_hierarchy():
    # A same-shape folder is an IntensitySequence + FrameShapedMixin; an
    # arbitrary file list is an IntensitySequence only.
    assert issubclass(IntensityBinFolder, IntensitySequence)
    assert issubclass(IntensityBinFolder, FrameShapedMixin)
    assert issubclass(IntensityBinList, IntensitySequence)
    assert not issubclass(IntensityBinList, FrameShapedMixin)
