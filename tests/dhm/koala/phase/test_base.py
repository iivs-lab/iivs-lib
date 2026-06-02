from __future__ import annotations

from iivs.dhm.koala.phase.base import PhaseSequence
from iivs.dhm.koala.phase.bin import PhaseBinSequence


def test_phase_bin_sequence_subclasses_phase_sequence():
    assert issubclass(PhaseBinSequence, PhaseSequence)
