from __future__ import annotations

from iivs.dhm.data.common import FrameShapedMixin
from iivs.dhm.data.intensity.base import (
    IntensityFloatSequence,
    IntensityImageSequence,
    IntensitySequence,
)
from iivs.dhm.data.intensity.bin import IntensityBinFolder, IntensityBinList


def test_intensity_sequence_hierarchy():
    # Float and Image both specialize the modality base IntensitySequence.
    assert issubclass(IntensityFloatSequence, IntensitySequence)
    assert issubclass(IntensityImageSequence, IntensitySequence)

    # A same-shape float folder is an IntensityFloatSequence + FrameShapedMixin;
    # an arbitrary file list is an IntensityFloatSequence only.
    assert issubclass(IntensityBinFolder, IntensityFloatSequence)
    assert issubclass(IntensityBinFolder, FrameShapedMixin)
    assert issubclass(IntensityBinFolder, IntensityBinList)  # folder is a special list
    assert issubclass(IntensityBinList, IntensityFloatSequence)
    assert not issubclass(IntensityBinList, FrameShapedMixin)
