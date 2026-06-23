from __future__ import annotations

from iivs.common.data import FrameShapedMixin
from iivs.dhm.data.phase.base import (
    PhaseFloatSequence,
    PhaseImageSequence,
    PhaseSequence,
)
from iivs.dhm.data.phase.bin import PhaseBinFolder, PhaseBinList


def test_phase_sequence_hierarchy():
    # Float and Image both specialize the modality base PhaseSequence.
    assert issubclass(PhaseFloatSequence, PhaseSequence)
    assert issubclass(PhaseImageSequence, PhaseSequence)

    # A same-shape float folder is a PhaseFloatSequence + FrameShapedMixin; an
    # arbitrary file list is a PhaseFloatSequence only.
    assert issubclass(PhaseBinFolder, PhaseFloatSequence)
    assert issubclass(PhaseBinFolder, FrameShapedMixin)
    assert issubclass(PhaseBinFolder, PhaseBinList)  # a folder is a special list
    assert issubclass(PhaseBinList, PhaseFloatSequence)
    assert not issubclass(PhaseBinList, FrameShapedMixin)
