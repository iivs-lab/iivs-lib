from __future__ import annotations

from iivs.dhm.data.common import FrameShapedMixin
from iivs.dhm.data.phase.base import PhaseSequence
from iivs.dhm.data.phase.bin import PhaseBinFolder, PhaseBinList


def test_phase_sequence_hierarchy():
    # A same-shape folder is a PhaseSequence + FrameShapedMixin; an arbitrary
    # file list is a PhaseSequence only.
    assert issubclass(PhaseBinFolder, PhaseSequence)
    assert issubclass(PhaseBinFolder, FrameShapedMixin)
    assert issubclass(PhaseBinFolder, PhaseBinList)  # a folder is a special list
    assert issubclass(PhaseBinList, PhaseSequence)
    assert not issubclass(PhaseBinList, FrameShapedMixin)
