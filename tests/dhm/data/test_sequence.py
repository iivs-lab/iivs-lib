from __future__ import annotations

import numpy as np

from iivs.dhm.data.phase.base import PhaseSequence
from iivs.dhm.data.phase.bin import PhaseBinFolder, PhaseBinList, save_phase_bin
from iivs.dhm.data.sequence import FrameShapedMixin


def _write(root, index, shape=(2, 3)):
    save_phase_bin(
        root / f"{index:05d}_phase.bin",
        np.zeros(shape, dtype=np.float32),
        pixel_size=1e-6,
        height_scale=2e-7,
    )


def test_uniform_sequence_is_role_plus_mixin(tmp_path):
    # "A uniform phase sequence" == PhaseSequence + FrameShapedMixin.
    _write(tmp_path, 0)
    folder = PhaseBinFolder(tmp_path)
    assert isinstance(folder, PhaseSequence)
    assert isinstance(folder, FrameShapedMixin)
    assert folder.frame_shape == (2, 3)


def test_heterogeneous_list_lacks_the_mixin(tmp_path):
    # A plain file list is a PhaseSequence but not FrameShapedMixin.
    _write(tmp_path, 0)
    seq = PhaseBinList([tmp_path / "00000_phase.bin"])
    assert isinstance(seq, PhaseSequence)
    assert not isinstance(seq, FrameShapedMixin)
    assert not hasattr(seq, "frame_shape")
