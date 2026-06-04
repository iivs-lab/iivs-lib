from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from iivs.dhm.data.common import (
    FrameShapedMixin,
    ensure_file_extension,
    load_uint8_tif,
    numbered_name,
    parse_txt_grid,
    read_npy_shape,
    validate_float32_image,
    validate_uint8_image,
    with_file_extension,
)
from iivs.dhm.data.phase.base import PhaseSequence
from iivs.dhm.data.phase.bin import PhaseBinFolder, PhaseBinList, save_phase_bin

# ========================== #
#  validate_float32_image    #
# ========================== #


def test_rejects_below_2d():
    with pytest.raises(ValueError, match="2-dimensional"):
        validate_float32_image(np.zeros(5, dtype=np.float32))


def test_rejects_non_float32():
    with pytest.raises(ValueError, match="float32"):
        validate_float32_image(np.zeros((2, 2), dtype=np.float64))


def test_rejects_unknown_on_nonfinite():
    with pytest.raises(ValueError, match="on_nonfinite must be"):
        validate_float32_image(np.zeros((2, 2), dtype=np.float32), on_nonfinite="bogus")  # ty: ignore[invalid-argument-type]


def test_ignore_accepts_nonfinite():
    data = np.array([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float32)
    assert validate_float32_image(data, on_nonfinite="ignore") is data


def test_warns_on_nonfinite():
    data = np.array([[np.inf, 1.0], [2.0, 3.0]], dtype=np.float32)
    with pytest.warns(RuntimeWarning, match="finite"):
        validate_float32_image(data, on_nonfinite="warn")


def test_raises_on_nonfinite():
    data = np.array([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="finite"):
        validate_float32_image(data, on_nonfinite="raise")


def test_clean_returns_input():
    data = np.zeros((2, 2), dtype=np.float32)
    assert validate_float32_image(data) is data  # default "warn", finite: as-is


# ========================== #
#  validate_uint8_image      #
# ========================== #


def test_uint8_rejects_non_uint8():
    with pytest.raises(ValueError, match="uint8"):
        validate_uint8_image(np.zeros((2, 2), dtype=np.float32))


def test_uint8_accepts_stack_by_default():
    data = np.zeros((3, 2, 2), dtype=np.uint8)
    assert validate_uint8_image(data) is data


def test_uint8_clean_2d_returns_input():
    data = np.zeros((2, 2), dtype=np.uint8)
    assert validate_uint8_image(data) is data


def test_no_stack_rejects_higher_dims():
    # allow_stack=False requires a single 2D image, for both dtypes.
    with pytest.raises(ValueError, match="single 2D image"):
        validate_uint8_image(np.zeros((2, 2, 3), dtype=np.uint8), allow_stack=False)
    with pytest.raises(ValueError, match="single 2D image"):
        validate_float32_image(np.zeros((2, 2, 3), dtype=np.float32), allow_stack=False)


def test_no_stack_accepts_2d():
    data = np.zeros((2, 2), dtype=np.uint8)
    assert validate_uint8_image(data, allow_stack=False) is data


# ========================== #
#       parse_txt_grid       #
# ========================== #


def test_parse_txt_grid_ok():
    grid = parse_txt_grid(["1 2 3", "4 5 6"], shape=(2, 3))
    np.testing.assert_array_equal(grid, [[1, 2, 3], [4, 5, 6]])
    assert grid.dtype == np.float32


def test_parse_txt_grid_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="txt grid must be"):
        parse_txt_grid(["1 2 3", "4 5 6"], shape=(2, 2))


def test_parse_txt_grid_rejects_malformed():
    with pytest.raises(ValueError, match="malformed txt grid"):
        parse_txt_grid(["1 2", "3 nan x"], shape=(2, 2))


# ========================== #
#       load_uint8_tif       #
# ========================== #


def test_load_uint8_tif_roundtrip(tmp_path):
    data = np.arange(6, dtype=np.uint8).reshape(2, 3)
    path = tmp_path / "00000_phase.tif"
    tifffile.imwrite(path, data)  # uncompressed: no imagecodecs needed
    loaded = load_uint8_tif(path)
    np.testing.assert_array_equal(loaded, data)
    assert loaded.dtype == np.uint8


def test_load_uint8_tif_rejects_non_uint8(tmp_path):
    path = tmp_path / "00000_phase.tif"
    tifffile.imwrite(path, np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="uint8"):
        load_uint8_tif(path)


def test_load_uint8_tif_without_imagecodecs_raises_importerror(tmp_path, monkeypatch):
    # Writing LZW also needs imagecodecs, so simulate tifffile's missing-codec error.
    path = tmp_path / "00000_phase.tif"
    path.write_bytes(b"")  # content is irrelevant; imread is mocked

    def _raise(_):
        msg = "<COMPRESSION.LZW: 5> requires the 'imagecodecs' package"
        raise KeyError(msg)

    monkeypatch.setattr(tifffile, "imread", _raise)
    with pytest.raises(ImportError, match=r"iivs-lib\[image\]"):
        load_uint8_tif(path)


def test_load_uint8_tif_reraises_unrelated_keyerror(tmp_path, monkeypatch):
    path = tmp_path / "00000_phase.tif"
    path.write_bytes(b"")

    def _raise(_):
        msg = "unrelated"
        raise KeyError(msg)

    monkeypatch.setattr(tifffile, "imread", _raise)
    with pytest.raises(KeyError, match="unrelated"):
        load_uint8_tif(path)


# ========================== #
#     file extensions        #
# ========================== #


def test_ensure_file_extension_ok():
    # Case-insensitive; returns the path as a Path.
    assert ensure_file_extension("dir/x.BIN", "bin") == Path("dir/x.BIN")


def test_ensure_file_extension_rejects():
    with pytest.raises(ValueError, match=r"must have a \.bin extension"):
        ensure_file_extension("dir/x.txt", "bin")


def test_file_list_rejects_wrong_extension(tmp_path):
    # PhaseBinList (FILE_EXT="bin") rejects a non-.bin path up front, before any
    # decode -- the file need not even exist.
    with pytest.raises(ValueError, match=r"must have a \.bin extension"):
        PhaseBinList([tmp_path / "00000_phase.txt"])


# ========================== #
#       read_npy_shape       #
# ========================== #


def test_read_npy_shape_2d(tmp_path):
    path = tmp_path / "a.npy"
    np.save(path, np.zeros((3, 4), dtype=np.float32))
    assert read_npy_shape(path) == (3, 4)


def test_read_npy_shape_rejects_non_2d(tmp_path):
    path = tmp_path / "a.npy"
    np.save(path, np.zeros((2, 3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="2D array"):
        read_npy_shape(path)


# ========================== #
#       FrameShapedMixin     #
# ========================== #


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


# ========================== #
#     numbered_name / ext    #
# ========================== #


def test_numbered_name():
    assert numbered_name(7, stem="phase", ext="bin") == "00007_phase.bin"


def test_with_file_extension_appends_when_absent(tmp_path):
    assert (
        with_file_extension(tmp_path / "00000_phase", "bin")
        == tmp_path / "00000_phase.bin"
    )


def test_with_file_extension_keeps_matching(tmp_path):
    path = tmp_path / "out.BIN"  # case-insensitive match
    assert with_file_extension(path, "bin") == path


def test_with_file_extension_rejects_mismatch(tmp_path):
    with pytest.raises(ValueError, match=r"must have a \.bin extension"):
        with_file_extension(tmp_path / "out.txt", "bin")


def test_save_appends_extension_when_absent(tmp_path):
    data = np.zeros((2, 2), dtype=np.float32)
    save_phase_bin(tmp_path / "00000_phase", data, pixel_size=1e-6, height_scale=2e-7)
    assert (tmp_path / "00000_phase.bin").exists()


def test_save_rejects_wrong_extension(tmp_path):
    data = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match=r"must have a \.bin extension"):
        save_phase_bin(tmp_path / "x.txt", data, pixel_size=1e-6, height_scale=2e-7)
