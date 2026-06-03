from __future__ import annotations

from iivs.dhm.koala.hologram.base import HologramSequence, UniformHologramSequence
from iivs.dhm.koala.hologram.raw import HologramRawSequence
from iivs.dhm.koala.hologram.tif import HologramTifList, HologramTifSequence


def test_hologram_sequence_hierarchy():
    assert issubclass(UniformHologramSequence, HologramSequence)
    # Single-acquisition sources are uniform; an arbitrary file list is not.
    assert issubclass(HologramTifSequence, UniformHologramSequence)
    assert issubclass(HologramRawSequence, UniformHologramSequence)
    assert issubclass(HologramTifList, HologramSequence)
    assert not issubclass(HologramTifList, UniformHologramSequence)
