from __future__ import annotations

from iivs.dhm.koala.hologram.base import HologramSequence
from iivs.dhm.koala.hologram.raw import HologramRawSequence
from iivs.dhm.koala.hologram.tif import HologramTifSequence


def test_tif_and_raw_sequences_subclass_hologram_sequence():
    assert issubclass(HologramTifSequence, HologramSequence)
    assert issubclass(HologramRawSequence, HologramSequence)
