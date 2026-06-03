from __future__ import annotations

from iivs.dhm.data.phase.base import PhaseSequence
from iivs.dhm.data.phase.bin import PhaseBinFolder, PhaseBinList
from iivs.dhm.data.sequence import FrameShapedMixin


def test_phase_sequence_hierarchy():
    # A same-shape folder is a PhaseSequence + FrameShapedMixin; an arbitrary
    # file list is a PhaseSequence only.
    assert issubclass(PhaseBinFolder, PhaseSequence)
    assert issubclass(PhaseBinFolder, FrameShapedMixin)
    assert issubclass(PhaseBinList, PhaseSequence)
    assert not issubclass(PhaseBinList, FrameShapedMixin)
