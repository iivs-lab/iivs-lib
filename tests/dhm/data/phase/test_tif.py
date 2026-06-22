from __future__ import annotations

import numpy as np
import pytest
import tifffile

from iivs.dhm.data.common import FrameShapedMixin
from iivs.dhm.data.phase.base import PhaseFloatSequence, PhaseImageSequence
from iivs.dhm.data.phase.tif import PhaseTifFolder, PhaseTifList


def _write(path, data):
    # A plain uncompressed uint8 tif (the codec path is exercised elsewhere).
    tifffile.imwrite(path, np.asarray(data, dtype=np.uint8))


def _write_idx(root, index, value=0, shape=(2, 3)):
    _write(root / f"{index:05d}_phase.tif", np.full(shape, value, dtype=np.uint8))


# --- hierarchy ---


def test_hierarchy():
    # The preview folder/list are PhaseImageSequence (uint8), not the
    # quantitative PhaseFloatSequence.
    assert issubclass(PhaseTifFolder, PhaseImageSequence)
    assert issubclass(PhaseTifFolder, PhaseTifList)  # a folder is a special list
    assert issubclass(PhaseTifFolder, FrameShapedMixin)
    assert not issubclass(PhaseTifList, FrameShapedMixin)
    assert not issubclass(PhaseTifFolder, PhaseFloatSequence)


# --- file list ---


def test_list_sequence(tmp_path):
    a = tmp_path / "a.tif"
    sub = tmp_path / "nested"
    sub.mkdir()
    b = sub / "b.tif"
    _write(a, np.full((2, 3), 1, dtype=np.uint8))
    _write(b, np.full((4, 5), 2, dtype=np.uint8))

    seq = PhaseTifList([b, a])
    assert len(seq) == 2
    assert [seq.get_meta(i) for i in range(2)] == [b, a]
    assert not hasattr(seq, "frame_shape")  # heterogeneous
    np.testing.assert_array_equal(seq[0], np.full((4, 5), 2, dtype=np.uint8))
    assert seq[0].dtype == np.uint8


# --- folder sequence ---


def test_folder_lists_items_in_index_order(tmp_path):
    for i in range(3):
        _write_idx(tmp_path, i, i)
    folder = PhaseTifFolder(tmp_path)
    assert len(folder) == 3
    assert folder.frame_shape == (2, 3)
    assert folder.get_meta(0) == tmp_path / "00000_phase.tif"
    for i in range(3):
        np.testing.assert_array_equal(folder[i], np.full((2, 3), i, dtype=np.uint8))


def test_folder_is_image_sequence(tmp_path):
    _write_idx(tmp_path, 0)
    folder = PhaseTifFolder(tmp_path)
    assert isinstance(folder, PhaseImageSequence)
    assert isinstance(folder, FrameShapedMixin)


def test_folder_skip_validation(tmp_path):
    _write_idx(tmp_path, 0)
    folder = PhaseTifFolder(tmp_path, validate=None)
    assert len(folder) == 1


def test_folder_validate_rejects_gap(tmp_path):
    _write_idx(tmp_path, 0)
    _write_idx(tmp_path, 2)  # gap at index 1
    with pytest.raises(ValueError, match="non-contiguous"):
        PhaseTifFolder(tmp_path)


def test_folder_validate_data_accepts_uniform(tmp_path):
    for i in range(2):
        _write_idx(tmp_path, i)  # all (2, 3)
    folder = PhaseTifFolder(tmp_path, validate="data")
    assert folder.frame_shape == (2, 3)


def test_folder_validate_data_rejects_shape_mismatch(tmp_path):
    _write_idx(tmp_path, 0, shape=(2, 3))
    _write_idx(tmp_path, 1, shape=(4, 5))
    PhaseTifFolder(tmp_path, validate="names")  # names only: ok
    with pytest.raises(ValueError, match="must match"):
        PhaseTifFolder(tmp_path, validate="data")
