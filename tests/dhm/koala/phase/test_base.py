from __future__ import annotations

from iivs.dhm.koala.phase.base import PhaseSequence, UniformPhaseSequence
from iivs.dhm.koala.phase.bin import PhaseBinList, PhaseBinSequence


def test_phase_sequence_hierarchy():
    assert issubclass(UniformPhaseSequence, PhaseSequence)
    # A single-acquisition folder is uniform; an arbitrary file list is not.
    assert issubclass(PhaseBinSequence, UniformPhaseSequence)
    assert issubclass(PhaseBinList, PhaseSequence)
    assert not issubclass(PhaseBinList, UniformPhaseSequence)
