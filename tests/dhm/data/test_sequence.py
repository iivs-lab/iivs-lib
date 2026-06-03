from __future__ import annotations

import numpy as np

from iivs.dhm.data.phase.bin import PhaseBinFolder, PhaseBinList, save_phase_bin
from iivs.dhm.data.sequence import FrameShaped


def _write(root, index, shape=(2, 3)):
    save_phase_bin(
        root / f"{index:05d}_phase.bin",
        np.zeros(shape, dtype=np.float32),
        pixel_size=1e-6,
        height_scale=2e-7,
    )


def test_uniform_sequence_is_frame_shaped(tmp_path):
    # A same-shape folder exposes frame_shape, so it matches structurally.
    _write(tmp_path, 0)
    folder = PhaseBinFolder(tmp_path)
    assert isinstance(folder, FrameShaped)
    assert folder.frame_shape == (2, 3)


def test_heterogeneous_list_is_not_frame_shaped(tmp_path):
    # A plain file list has no frame_shape, so it is correctly excluded.
    _write(tmp_path, 0)
    seq = PhaseBinList([tmp_path / "00000_phase.bin"])
    assert not isinstance(seq, FrameShaped)
