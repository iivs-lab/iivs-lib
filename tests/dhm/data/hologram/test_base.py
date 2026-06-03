from __future__ import annotations

from iivs.dhm.data.hologram.base import HologramSequence
from iivs.dhm.data.hologram.raw import HologramRawFile
from iivs.dhm.data.hologram.tif import HologramTifFolder, HologramTifList
from iivs.dhm.data.sequence import FrameShapedMixin


def test_hologram_sequence_hierarchy():
    # Single-acquisition sources are HologramSequence + FrameShapedMixin; an
    # arbitrary file list is a HologramSequence only.
    assert issubclass(HologramTifFolder, HologramSequence)
    assert issubclass(HologramTifFolder, FrameShapedMixin)
    assert issubclass(HologramRawFile, HologramSequence)
    assert issubclass(HologramRawFile, FrameShapedMixin)
    assert issubclass(HologramTifList, HologramSequence)
    assert not issubclass(HologramTifList, FrameShapedMixin)
