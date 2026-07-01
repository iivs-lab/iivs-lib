from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
from kaparoo.filesystem import ensure_file_extension

from iivs.common.data import FrameShapedMixin, read_npy_shape
from iivs.dhm.data.common import (
    FLOAT_FORMATS,
    detect_numbered_format,
    load_txt,
    load_uint8_tif,
    numbered_name,
    validate_float32_array,
    validate_float_array,
    validate_uint8_array,
    validate_uint_array,
    write_bin,
)
from iivs.dhm.data.intensity.bin import IntensityBinHeader
from iivs.dhm.data.intensity.txt import IntensityTxtHeaderCodec
from iivs.dhm.data.phase.base import PhaseSequence
from iivs.dhm.data.phase.bin import PhaseBinFolder, PhaseBinList, save_phase_bin

# ========================== #
#  validate_float32_array    #
# ========================== #


def test_rejects_below_2d():
    with pytest.raises(ValueError, match="2-dimensional"):
        validate_float32_array(np.zeros(5, dtype=np.float32))


def test_rejects_non_float32():
    with pytest.raises(ValueError, match="float32"):
        validate_float32_array(np.zeros((2, 2), dtype=np.float64))


def test_rejects_unknown_on_nonfinite():
    with pytest.raises(ValueError, match="on_nonfinite must be"):
        validate_float32_array(np.zeros((2, 2), dtype=np.float32), on_nonfinite="bogus")  # ty: ignore[invalid-argument-type]


def test_ignore_accepts_nonfinite():
    data = np.array([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float32)
    assert validate_float32_array(data, on_nonfinite="ignore") is data


def test_warns_on_nonfinite():
    data = np.array([[np.inf, 1.0], [2.0, 3.0]], dtype=np.float32)
    with pytest.warns(RuntimeWarning, match="finite"):
        validate_float32_array(data, on_nonfinite="warn")


def test_raises_on_nonfinite():
    data = np.array([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="finite"):
        validate_float32_array(data, on_nonfinite="raise")


def test_clean_returns_input():
    data = np.zeros((2, 2), dtype=np.float32)
    assert validate_float32_array(data) is data  # default "warn", finite: as-is


# ========================== #
#  validate_uint8_array      #
# ========================== #


def test_uint8_rejects_non_uint8():
    with pytest.raises(ValueError, match="uint8"):
        validate_uint8_array(np.zeros((2, 2), dtype=np.float32))


def test_uint8_accepts_stack_by_default():
    data = np.zeros((3, 2, 2), dtype=np.uint8)
    assert validate_uint8_array(data) is data


def test_uint8_clean_2d_returns_input():
    data = np.zeros((2, 2), dtype=np.uint8)
    assert validate_uint8_array(data) is data


def test_no_stack_rejects_higher_dims():
    # allow_stack=False requires a single 2D array, for both dtypes.
    with pytest.raises(ValueError, match="single 2D array"):
        validate_uint8_array(np.zeros((2, 2, 3), dtype=np.uint8), allow_stack=False)
    with pytest.raises(ValueError, match="single 2D array"):
        validate_float32_array(np.zeros((2, 2, 3), dtype=np.float32), allow_stack=False)


def test_no_stack_accepts_2d():
    data = np.zeros((2, 2), dtype=np.uint8)
    assert validate_uint8_array(data, allow_stack=False) is data


# ========================== #
#  dtype-parametric core     #
# ========================== #


def test_float_array_accepts_non_default_dtype():
    data = np.zeros((2, 2), dtype=np.float64)
    assert validate_float_array(data, dtype=np.float64) is data


def test_float_array_rejects_dtype_mismatch():
    # data is float32 but float64 was requested.
    with pytest.raises(ValueError, match="float64"):
        validate_float_array(np.zeros((2, 2), dtype=np.float32), dtype=np.float64)


def test_float_array_rejects_non_floating_dtype():
    with pytest.raises(ValueError, match="dtype must be a subtype of floating"):
        validate_float_array(np.zeros((2, 2), dtype=np.float32), dtype=np.uint8)


def test_float_array_keeps_finite_policy():
    data = np.array([[np.nan, 1.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="finite"):
        validate_float_array(data, dtype=np.float64, on_nonfinite="raise")


def test_uint_array_accepts_uint16():
    data = np.zeros((2, 2), dtype=np.uint16)
    assert validate_uint_array(data, dtype=np.uint16) is data


def test_uint_array_rejects_dtype_mismatch():
    with pytest.raises(ValueError, match="uint16"):
        validate_uint_array(np.zeros((2, 2), dtype=np.uint8), dtype=np.uint16)


def test_uint_array_rejects_non_unsigned_dtype():
    with pytest.raises(ValueError, match="dtype must be a subtype of unsignedinteger"):
        validate_uint_array(np.zeros((2, 2), dtype=np.uint8), dtype=np.int16)


def test_float_array_custom_dims_accepts_volume():
    # dims=3 treats the trailing three axes as core; a 3-D array is one volume.
    data = np.zeros((2, 2, 2), dtype=np.float32)
    assert validate_float_array(data, dims=3, allow_stack=False) is data


def test_float_array_custom_dims_rejects_too_few():
    with pytest.raises(ValueError, match="at least 3-dimensional"):
        validate_float_array(np.zeros((2, 2), dtype=np.float32), dims=3)


def test_uint_array_custom_dims_accepts_stacked_volume():
    data = np.zeros((5, 2, 2, 2), dtype=np.uint16)
    assert validate_uint_array(data, dtype=np.uint16, dims=3) is data


# ========================== #
#          load_txt          #
# ========================== #


def _write_txt(tmp_path, *, height, width, grid):
    path = tmp_path / "x.txt"
    path.write_text(f"h={height} w={width}\npixel size=1e-06 m\n" + grid)
    return path


def test_load_txt_rows(tmp_path):
    path = _write_txt(tmp_path, height=2, width=3, grid="1 2 3\n4 5 6\n")
    data, header = load_txt(path, IntensityTxtHeaderCodec)
    np.testing.assert_array_equal(data, [[1, 2, 3], [4, 5, 6]])
    assert data.dtype == np.float32
    assert header.shape == (2, 3)


def test_load_txt_single_line_grid(tmp_path):
    # Koala may write the whole grid on one line; row-major reshape handles it.
    path = _write_txt(tmp_path, height=2, width=3, grid="1 2 3 4 5 6\n")
    data, _ = load_txt(path, IntensityTxtHeaderCodec)
    np.testing.assert_array_equal(data, [[1, 2, 3], [4, 5, 6]])


def test_load_txt_rejects_grid_count_mismatch(tmp_path):
    # header declares 2x2 (4 values) but the grid holds 6
    path = _write_txt(tmp_path, height=2, width=2, grid="1 2 3\n4 5 6\n")
    with pytest.raises(ValueError, match="txt grid must hold 4 values"):
        load_txt(path, IntensityTxtHeaderCodec)


def test_load_txt_rejects_malformed_grid(tmp_path):
    path = _write_txt(tmp_path, height=2, width=2, grid="1 2\n3 nan x\n")
    with pytest.raises(ValueError, match="malformed txt grid"):
        load_txt(path, IntensityTxtHeaderCodec)


# ========================== #
#       load_uint8_tif       #
# ========================== #


def test_load_uint8_tif_roundtrip(tmp_path):
    data = np.arange(6, dtype=np.uint8).reshape(2, 3)
    path = tmp_path / "00000_phase.tif"
    tifffile.imwrite(path, data)  # a plain uint8 tif roundtrip
    loaded = load_uint8_tif(path)
    np.testing.assert_array_equal(loaded, data)
    assert loaded.dtype == np.uint8


def test_load_uint8_tif_rejects_non_uint8(tmp_path):
    path = tmp_path / "00000_phase.tif"
    tifffile.imwrite(path, np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="uint8"):
        load_uint8_tif(path)


# ========================== #
#     file extensions        #
# ========================== #


def test_ensure_file_extension_ok():
    # Case-insensitive; returns the path as a Path.
    assert ensure_file_extension("dir/x.BIN", "bin") == Path("dir/x.BIN")


def test_ensure_file_extension_rejects():
    with pytest.raises(
        ValueError, match=r"unsupported extension .* \(supported: 'bin'\)"
    ):
        ensure_file_extension("dir/x.txt", "bin")


def test_file_list_rejects_wrong_extension(tmp_path):
    # PhaseBinList (FILE_EXT="bin") rejects a non-.bin path up front, before any
    # decode -- the file need not even exist.
    with pytest.raises(
        ValueError, match=r"unsupported extension .* \(supported: 'bin'\)"
    ):
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
    with pytest.raises(ValueError, match="expected 2D but got 3D"):
        read_npy_shape(path)


def test_read_npy_shape_honors_expected(tmp_path):
    path = tmp_path / "a.npy"
    np.save(path, np.zeros((2, 3, 4), dtype=np.float32))
    assert read_npy_shape(path, expected=3) == (2, 3, 4)


def test_read_npy_shape_rejects_nonpositive_expected(tmp_path):
    path = tmp_path / "a.npy"
    np.save(path, np.zeros((3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="expected must be positive"):
        read_npy_shape(path, expected=0)


# ========================== #
#         write_bin          #
# ========================== #


def test_write_bin_rejects_pixel_shape_mismatch(tmp_path):
    header = IntensityBinHeader(width=4, height=3, pixel_size=1e-6)  # shape (3, 4)
    pixels = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match=r"pixels shape must match header \(3, 4\)"):
        write_bin(tmp_path / "x.bin", header, pixels)


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


def test_ensure_file_extension_add_appends_when_absent(tmp_path):
    # add=True (kaparoo's, the former `with_file_extension`) appends a missing suffix.
    assert (
        ensure_file_extension(tmp_path / "00000_phase", "bin", add=True)
        == tmp_path / "00000_phase.bin"
    )


def test_ensure_file_extension_add_keeps_matching(tmp_path):
    path = tmp_path / "out.BIN"  # case-insensitive match
    assert ensure_file_extension(path, "bin", add=True) == path


def test_ensure_file_extension_add_rejects_mismatch(tmp_path):
    with pytest.raises(
        ValueError, match=r"unsupported extension .* \(supported: 'bin'\)"
    ):
        ensure_file_extension(tmp_path / "out.txt", "bin", add=True)


def test_save_appends_extension_when_absent(tmp_path):
    data = np.zeros((2, 2), dtype=np.float32)
    save_phase_bin(tmp_path / "00000_phase", data, pixel_size=1e-6, height_scale=2e-7)
    assert (tmp_path / "00000_phase.bin").exists()


def test_save_rejects_wrong_extension(tmp_path):
    data = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(
        ValueError, match=r"unsupported extension .* \(supported: 'bin'\)"
    ):
        save_phase_bin(tmp_path / "x.txt", data, pixel_size=1e-6, height_scale=2e-7)


# ========================== #
#   detect_numbered_format   #
# ========================== #


def _numbered(root, index, *, stem, ext):
    # detect_numbered_format only inspects names, so empty files suffice.
    (root / numbered_name(index, stem=stem, ext=ext)).write_bytes(b"")


def test_detect_single_format(tmp_path):
    _numbered(tmp_path, 0, stem="phase", ext="bin")
    _numbered(tmp_path, 1, stem="phase", ext="bin")
    assert (
        detect_numbered_format(tmp_path, stem="phase", formats=FLOAT_FORMATS) == "bin"
    )


def test_detect_ignores_other_stems_and_loose_files(tmp_path):
    # Only NNNNN_<stem>.<ext> at depth 1 counts.
    _numbered(tmp_path, 0, stem="phase", ext="txt")
    _numbered(tmp_path, 0, stem="intensity", ext="bin")  # other stem
    (tmp_path / "phase.bin").write_bytes(b"")  # unnumbered
    (tmp_path / "0_phase.bin").write_bytes(b"")  # too few digits
    assert (
        detect_numbered_format(tmp_path, stem="phase", formats=FLOAT_FORMATS) == "txt"
    )


def test_detect_no_files_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"no NNNNN_phase"):
        detect_numbered_format(tmp_path, stem="phase", formats=FLOAT_FORMATS)


def test_detect_multiple_without_prefer_raises(tmp_path):
    _numbered(tmp_path, 0, stem="phase", ext="bin")
    _numbered(tmp_path, 0, stem="phase", ext="txt")
    with pytest.raises(ValueError, match=r"ambiguous.*multiple phase formats"):
        detect_numbered_format(tmp_path, stem="phase", formats=FLOAT_FORMATS)


def test_detect_prefer_single_format(tmp_path):
    _numbered(tmp_path, 0, stem="phase", ext="bin")
    _numbered(tmp_path, 0, stem="phase", ext="txt")
    got = detect_numbered_format(
        tmp_path, stem="phase", formats=FLOAT_FORMATS, prefer="txt"
    )
    assert got == "txt"


def test_detect_prefer_priority_sequence_picks_first_present(tmp_path):
    # npy is absent, so the first *present* format in prefer order wins.
    _numbered(tmp_path, 0, stem="phase", ext="bin")
    _numbered(tmp_path, 0, stem="phase", ext="txt")
    got = detect_numbered_format(
        tmp_path, stem="phase", formats=FLOAT_FORMATS, prefer=("npy", "txt", "bin")
    )
    assert got == "txt"


def test_detect_prefer_selects_none_present_raises(tmp_path):
    _numbered(tmp_path, 0, stem="phase", ext="bin")
    _numbered(tmp_path, 0, stem="phase", ext="txt")
    with pytest.raises(ValueError, match=r"prefer=\['npy'\] selects none"):
        detect_numbered_format(
            tmp_path, stem="phase", formats=FLOAT_FORMATS, prefer="npy"
        )
