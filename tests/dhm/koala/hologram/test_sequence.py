from __future__ import annotations

import numpy as np
import pytest

from iivs.dhm.koala.hologram.file import save_tif
from iivs.dhm.koala.hologram.sequence import HologramTifSequence


def _write(root, index, value, shape=(2, 3)):
    save_tif(root / f"{index:05d}_holo.tif", np.full(shape, value, dtype=np.uint8))


def test_folder_lists_items_in_index_order(tmp_path):
    for i in range(3):
        _write(tmp_path, i, i)

    folder = HologramTifSequence(tmp_path)

    assert len(folder) == 3
    for i in range(3):
        np.testing.assert_array_equal(folder[i], np.full((2, 3), i, dtype=np.uint8))


def test_folder_get_meta_is_source_path(tmp_path):
    _write(tmp_path, 0, 0)
    folder = HologramTifSequence(tmp_path)
    assert folder.get_meta(0) == tmp_path / "00000_holo.tif"


def test_folder_ignores_non_matching_names(tmp_path):
    _write(tmp_path, 0, 0)
    blank = np.zeros((2, 3), dtype=np.uint8)
    save_tif(tmp_path / "0001_holo.tif", blank)  # 4 digits: ignored
    save_tif(tmp_path / "00002_phase.tif", blank)  # wrong stem: ignored
    assert len(HologramTifSequence(tmp_path)) == 1


def test_empty_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no NNNNN_holo"):
        HologramTifSequence(tmp_path)


def test_init_validate_runs_validation(tmp_path):
    _write(tmp_path, 0, 0)
    _write(tmp_path, 2, 2)  # gap at index 1
    HologramTifSequence(tmp_path, validate=None)  # constructs despite the gap
    with pytest.raises(ValueError, match="non-contiguous"):
        HologramTifSequence(tmp_path, validate="names")


def test_validate_data_level_decodes_each(tmp_path):
    for i in range(2):
        _write(tmp_path, i, i)
    HologramTifSequence(tmp_path).validate(level="data")  # all decode: ok


def test_validate_file_checks_single_index(tmp_path):
    _write(tmp_path, 0, 0)
    _write(tmp_path, 2, 2)  # gap at index 1
    seq = HologramTifSequence(tmp_path, validate=None)
    seq.validate_file(0)  # 00000 at index 0: ok
    with pytest.raises(ValueError, match="non-contiguous"):
        seq.validate_file(1)  # 00002 sits at index 1


def test_validate_file_rejects_unknown_level(tmp_path):
    _write(tmp_path, 0, 0)
    seq = HologramTifSequence(tmp_path, validate=None)
    with pytest.raises(ValueError, match="level must be"):
        seq.validate_file(0, level="bogus")  # ty: ignore[invalid-argument-type]
