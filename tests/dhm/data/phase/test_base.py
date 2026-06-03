from __future__ import annotations

from iivs.dhm.data.phase.base import PhaseSequence, UniformPhaseSequence
from iivs.dhm.data.phase.bin import PhaseBinFolder, PhaseBinList


def test_phase_sequence_hierarchy():
    assert issubclass(UniformPhaseSequence, PhaseSequence)
    # A single-acquisition folder is uniform; an arbitrary file list is not.
    assert issubclass(PhaseBinFolder, UniformPhaseSequence)
    assert issubclass(PhaseBinList, PhaseSequence)
    assert not issubclass(PhaseBinList, UniformPhaseSequence)
