from __future__ import annotations

from iivs.dhm.data.hologram.base import HologramSequence, UniformHologramSequence
from iivs.dhm.data.hologram.raw import HologramRawFile
from iivs.dhm.data.hologram.tif import HologramTifFolder, HologramTifList


def test_hologram_sequence_hierarchy():
    assert issubclass(UniformHologramSequence, HologramSequence)
    # Single-acquisition sources are uniform; an arbitrary file list is not.
    assert issubclass(HologramTifFolder, UniformHologramSequence)
    assert issubclass(HologramRawFile, UniformHologramSequence)
    assert issubclass(HologramTifList, HologramSequence)
    assert not issubclass(HologramTifList, UniformHologramSequence)
