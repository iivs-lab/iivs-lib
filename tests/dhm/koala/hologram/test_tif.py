from __future__ import annotations

import numpy as np
import pytest
import tifffile

from iivs.dhm.koala.hologram.tif import (
    HologramTifList,
    HologramTifSequence,
    load_hologram_tif,
    save_hologram_tif,
)


def _write(root, index, value, shape=(2, 3)):
    save_hologram_tif(
        root / f"{index:05d}_holo.tif", np.full(shape, value, dtype=np.uint8)
    )


# --- single-file I/O ---


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, size=(4, 5), dtype=np.uint8)
    path = tmp_path / "00000_holo.tif"

    save_hologram_tif(path, data)
    image = load_hologram_tif(path)

    np.testing.assert_array_equal(image, data)
    assert image.dtype == np.uint8


def test_save_rejects_non_uint8(tmp_path):
    with pytest.raises(ValueError, match="uint8"):
        save_hologram_tif(tmp_path / "bad.tif", np.zeros((2, 2), dtype=np.uint16))


def test_load_rejects_non_uint8_file(tmp_path):
    # A uint16 TIFF decodes but is not a valid hologram.
    path = tmp_path / "00000_holo.tif"
    tifffile.imwrite(path, np.zeros((2, 2), dtype=np.uint16))
    with pytest.raises(ValueError, match="uint8"):
        load_hologram_tif(path)


def test_save_overwrite(tmp_path):
    data = np.zeros((2, 2), dtype=np.uint8)
    path = tmp_path / "00000_holo.tif"
    save_hologram_tif(path, data)
    with pytest.raises(FileExistsError, match="already exists"):
        save_hologram_tif(path, data)
    save_hologram_tif(path, data, overwrite=True)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_hologram_tif(tmp_path / "nope.tif")


# --- folder sequence ---


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


def test_folder_frame_shape(tmp_path):
    _write(tmp_path, 0, 7, shape=(4, 5))  # no header; read from the first file
    assert HologramTifSequence(tmp_path).frame_shape == (4, 5)


def test_folder_ignores_non_matching_names(tmp_path):
    _write(tmp_path, 0, 0)
    blank = np.zeros((2, 3), dtype=np.uint8)
    save_hologram_tif(tmp_path / "0001_holo.tif", blank)  # 4 digits: ignored
    save_hologram_tif(tmp_path / "00002_phase.tif", blank)  # wrong stem: ignored
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


def test_validate_data_level_rejects_shape_mismatch(tmp_path):
    _write(tmp_path, 0, 0, shape=(2, 3))
    _write(tmp_path, 1, 1, shape=(4, 5))  # differs from the first image
    seq = HologramTifSequence(tmp_path)
    seq.validate(level="names")  # names-only: shape ignored
    with pytest.raises(ValueError, match="shape of"):
        seq.validate(level="data")


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


# --- file list ---


def test_list_sequence_loads_arbitrary_unrelated_files(tmp_path):
    # Arbitrary names, nested folder, heterogeneous shapes; order preserved.
    a = tmp_path / "first.tif"
    sub = tmp_path / "nested"
    sub.mkdir()
    b = sub / "whatever.tif"
    save_hologram_tif(a, np.full((2, 3), 1, dtype=np.uint8))
    save_hologram_tif(b, np.full((4, 5), 2, dtype=np.uint8))

    seq = HologramTifList([b, a])

    assert len(seq) == 2
    assert [seq.get_meta(i) for i in range(2)] == [b, a]
    np.testing.assert_array_equal(seq[0], np.full((4, 5), 2, dtype=np.uint8))
    np.testing.assert_array_equal(seq[1], np.full((2, 3), 1, dtype=np.uint8))
    assert not hasattr(seq, "frame_shape")  # heterogeneous: no uniform shape
