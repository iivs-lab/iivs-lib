from __future__ import annotations

import io

import numpy as np
import pytest

from iivs.dhm.data.intensity.bin import (
    IntensityBinFolder,
    IntensityBinHeader,
    IntensityBinList,
    load_intensity_bin,
    read_intensity_bin_header,
    save_intensity_bin,
)


def _valid_header_bytes() -> bytearray:
    header = IntensityBinHeader(width=2, height=2, pixel_size=1.0)
    return bytearray(header.to_dtype().tobytes())


def _write(root, index, value, shape=(2, 3)):
    save_intensity_bin(
        root / f"{index:05d}_intensity.bin",
        np.full(shape, float(value), dtype=np.float32),
        pixel_size=1e-6,
    )


def test_value_range_over_folder(tmp_path):
    # Intensity is unit-less, so it uses the shared `ValueRangeMixin` directly.
    _write(tmp_path, 0, 1.0)
    _write(tmp_path, 1, 5.0)
    folder = IntensityBinFolder(tmp_path)
    assert folder.value_range() == pytest.approx((1.0, 5.0))  # global
    assert folder.value_range(1) == pytest.approx((5.0, 5.0))  # one frame, by index


def test_value_range_ignores_non_finite(tmp_path):
    # A NaN pixel is dropped from the range; only the finite values count.
    save_intensity_bin(
        tmp_path / "00000_intensity.bin",
        np.array([[1.0, np.nan], [3.0, 5.0]], dtype=np.float32),
        pixel_size=1e-6,
        on_nonfinite="ignore",
    )
    save_intensity_bin(
        tmp_path / "00001_intensity.bin",
        np.array([[np.nan, 4.0], [2.0, 8.0]], dtype=np.float32),
        pixel_size=1e-6,
        on_nonfinite="ignore",
    )
    folder = IntensityBinFolder(tmp_path)
    assert folder.value_range() == pytest.approx((1.0, 8.0))  # global, NaNs ignored
    assert folder.value_range(0) == pytest.approx((1.0, 5.0))  # per-frame, NaN ignored


def test_value_range_all_non_finite_raises(tmp_path):
    save_intensity_bin(
        tmp_path / "00000_intensity.bin",
        np.full((2, 2), np.nan, dtype=np.float32),
        pixel_size=1e-6,
        on_nonfinite="ignore",
    )
    folder = IntensityBinFolder(tmp_path)
    with pytest.raises(ValueError, match="non-finite"):
        folder.value_range()  # global: every value is NaN
    with pytest.raises(ValueError, match="non-finite"):
        folder.value_range(0)  # per-frame: every value is NaN


def test_value_range_empty_list_raises():
    with pytest.raises(ValueError, match="empty"):
        IntensityBinList([]).value_range()  # the shared mixin's empty guard


# ========================== #
#           Header           #
# ========================== #


def test_header_dtype_is_packed_23_bytes():
    assert IntensityBinHeader.DTYPE.itemsize == 23


def test_header_dtype_roundtrip():
    header = IntensityBinHeader(width=3, height=2, pixel_size=0.5)
    assert IntensityBinHeader.from_dtype(header.to_dtype()[0]) == header


def test_header_writes_koala_sentinel():
    # Intensity pins the phase-only bytes to Koala's no-op sentinel.
    record = IntensityBinHeader(width=2, height=2, pixel_size=1.0).to_dtype()
    assert record["height_scale"][0] == pytest.approx(-1.0)
    assert int(record["unit"][0]) == 0


def test_header_ignores_hconv_and_unit_on_read():
    # from_dtype keeps only the geometry; arbitrary hconv/unit bytes are dropped.
    record = np.zeros(1, dtype=IntensityBinHeader.DTYPE)
    record["width"] = 4
    record["height"] = 2
    record["pixel_size"] = 1.0
    record["height_scale"] = 7.0  # would be rejected by PhaseBinHeader
    record["unit"] = 2
    header = IntensityBinHeader.from_dtype(record[0])
    assert header == IntensityBinHeader(width=4, height=2, pixel_size=1.0)


def test_header_shape_and_geometry():
    header = IntensityBinHeader(width=5, height=3, pixel_size=2e-6)
    assert header.shape == (3, 5)
    assert header.pixel_count == 15
    assert header.field_of_view == pytest.approx((3 * 2e-6, 5 * 2e-6))
    assert header.pixel_size_um == pytest.approx(2.0)
    assert header.field_of_view_um == pytest.approx((3 * 2.0, 5 * 2.0))


def test_header_version_endian_fixed():
    header = IntensityBinHeader(width=2, height=2, pixel_size=1.0)
    assert header.version == 1
    assert header.endian == 0


def test_header_rejects_nonpositive_dims():
    with pytest.raises(ValueError, match="must be positive"):
        IntensityBinHeader(width=0, height=2, pixel_size=1.0)


def test_header_rejects_nonpositive_pixel_size():
    with pytest.raises(ValueError, match="pixel_size must be positive"):
        IntensityBinHeader(width=2, height=2, pixel_size=0.0)


def test_from_stream_rejects_unsupported_version():
    raw = _valid_header_bytes()
    raw[0] = 2  # version byte
    with pytest.raises(ValueError, match="version"):
        IntensityBinHeader.from_stream(io.BytesIO(bytes(raw)))


def test_from_stream_rejects_unsupported_endian():
    raw = _valid_header_bytes()
    raw[1] = 1  # endian byte
    with pytest.raises(ValueError, match="byte order"):
        IntensityBinHeader.from_stream(io.BytesIO(bytes(raw)))


def test_header_is_hashable_and_comparable():
    a = IntensityBinHeader(width=3, height=2, pixel_size=0.5)
    b = IntensityBinHeader(width=3, height=2, pixel_size=0.5)
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


# ========================== #
#       Single-file I/O      #
# ========================== #


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 5)).astype(np.float32)
    path = tmp_path / "intensity.bin"

    save_intensity_bin(path, data, pixel_size=1e-6)
    image, header = load_intensity_bin(path, return_header=True)

    np.testing.assert_array_equal(image, data)
    assert image.dtype == np.float32
    assert image.flags.writeable  # frombuffer view is copied to a writable array
    assert header.shape == (4, 5)
    assert header.pixel_size == pytest.approx(1e-6)


def test_save_load_is_row_major(tmp_path):
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    path = tmp_path / "intensity.bin"
    save_intensity_bin(path, data, pixel_size=1.0)
    np.testing.assert_array_equal(load_intensity_bin(path), data)


def test_save_rejects_non_2d(tmp_path):
    with pytest.raises(ValueError, match="single 2D array"):
        save_intensity_bin(
            tmp_path / "bad.bin", np.zeros(5, dtype=np.float32), pixel_size=1.0
        )


def test_save_rejects_3d_stack(tmp_path):
    with pytest.raises(ValueError, match="single 2D array"):
        save_intensity_bin(
            tmp_path / "bad.bin", np.zeros((2, 2, 2), dtype=np.float32), pixel_size=1.0
        )


def test_save_rejects_non_float32(tmp_path):
    with pytest.raises(ValueError, match="float32"):
        save_intensity_bin(
            tmp_path / "bad.bin", np.zeros((2, 2), dtype=np.float64), pixel_size=1.0
        )


def test_save_overwrite(tmp_path):
    data = np.zeros((2, 2), dtype=np.float32)
    path = tmp_path / "intensity.bin"
    save_intensity_bin(path, data, pixel_size=1.0)
    with pytest.raises(FileExistsError, match="already exists"):
        save_intensity_bin(path, data, pixel_size=1.0)
    save_intensity_bin(path, data, pixel_size=1.0, overwrite=True)


def test_read_header_matches_load(tmp_path):
    data = np.zeros((3, 4), dtype=np.float32)
    path = tmp_path / "intensity.bin"
    save_intensity_bin(path, data, pixel_size=2e-6)
    assert (
        read_intensity_bin_header(path)
        == load_intensity_bin(path, return_header=True)[1]
    )


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no such file"):
        load_intensity_bin(tmp_path / "nope.bin")


def test_load_rejects_truncated_header(tmp_path):
    path = tmp_path / "trunc.bin"
    path.write_bytes(b"\x00\x01\x02")
    with pytest.raises(ValueError, match="bytes for a header"):
        load_intensity_bin(path)


def test_load_rejects_unexpected_header_size(tmp_path):
    data = np.zeros((2, 3), dtype=np.float32)
    path = tmp_path / "intensity.bin"
    save_intensity_bin(path, data, pixel_size=1.0)
    with path.open("r+b") as fd:
        fd.seek(2)
        fd.write(np.int32(99).tobytes())
    with pytest.raises(ValueError, match="header size"):
        load_intensity_bin(path)


def test_load_rejects_pixel_count_mismatch(tmp_path):
    data = np.zeros((2, 3), dtype=np.float32)
    path = tmp_path / "intensity.bin"
    save_intensity_bin(path, data, pixel_size=1.0)
    with path.open("ab") as fd:
        fd.write(np.float32(1.0).tobytes())
    with pytest.raises(ValueError, match="pixel count"):
        load_intensity_bin(path)


def test_save_warns_on_nonfinite(tmp_path):
    # RuntimeWarning alone is numpy's catch-all (overflow, invalid cast, ...), so match
    # the message: this must be the non-finite check, not any warning at all.
    data = np.array([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float32)
    path = tmp_path / "intensity.bin"
    with pytest.warns(RuntimeWarning, match="not finite"):
        save_intensity_bin(path, data, pixel_size=1.0)

    # "warn" is not "reject": the file is written, NaN and all.
    np.testing.assert_array_equal(load_intensity_bin(path), data)


def test_load_on_nonfinite_policy(tmp_path):
    data = np.array([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float32)
    path = tmp_path / "intensity.bin"
    save_intensity_bin(path, data, pixel_size=1.0, on_nonfinite="ignore")

    load_intensity_bin(path)  # default on_nonfinite="ignore": accepts silently
    with pytest.raises(ValueError, match="finite"):
        load_intensity_bin(path, on_nonfinite="raise")
    with pytest.warns(RuntimeWarning, match="finite"):
        load_intensity_bin(path, on_nonfinite="warn")


# ========================== #
#          Sequence          #
# ========================== #


def test_folder_lists_items_in_index_order(tmp_path):
    for i in range(3):
        _write(tmp_path, i, i)

    folder = IntensityBinFolder(tmp_path)

    assert len(folder) == 3
    for i in range(3):
        np.testing.assert_array_equal(
            folder[i], np.full((2, 3), float(i), dtype=np.float32)
        )


def test_folder_header_and_frame_shape(tmp_path):
    _write(tmp_path, 0, 0)
    folder = IntensityBinFolder(tmp_path)
    assert isinstance(folder.header, IntensityBinHeader)
    assert folder.header.shape == (2, 3)
    assert folder.frame_shape == (2, 3)


def test_folder_get_meta_is_source_path(tmp_path):
    _write(tmp_path, 0, 0)
    folder = IntensityBinFolder(tmp_path)
    assert folder.get_meta(0) == tmp_path / "00000_intensity.bin"


def test_folder_ignores_non_matching_names(tmp_path):
    _write(tmp_path, 0, 0)
    blank = np.zeros((2, 3), dtype=np.float32)
    save_intensity_bin(tmp_path / "0001_intensity.bin", blank, pixel_size=1e-6)
    save_intensity_bin(tmp_path / "00002_phase.bin", blank, pixel_size=1e-6)
    assert len(IntensityBinFolder(tmp_path)) == 1


def test_empty_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no NNNNN_intensity"):
        IntensityBinFolder(tmp_path)


def test_init_validate_rejects_gap(tmp_path):
    _write(tmp_path, 0, 0)
    _write(tmp_path, 2, 2)  # gap at index 1
    IntensityBinFolder(tmp_path, validate=None)  # constructs despite the gap
    with pytest.raises(ValueError, match="non-contiguous"):
        IntensityBinFolder(tmp_path, validate="headers")


def test_init_validate_data_level_checks_pixels(tmp_path):
    nan = np.array([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float32)
    save_intensity_bin(
        tmp_path / "00000_intensity.bin", nan, pixel_size=1e-6, on_nonfinite="ignore"
    )
    IntensityBinFolder(tmp_path, validate="headers")  # pixels not inspected: ok
    with pytest.raises(ValueError, match="finite"):
        IntensityBinFolder(tmp_path, validate="data")


def test_validate_names_level_skips_header_check(tmp_path):
    blank = np.zeros((2, 3), dtype=np.float32)
    save_intensity_bin(tmp_path / "00000_intensity.bin", blank, pixel_size=1e-6)
    save_intensity_bin(tmp_path / "00001_intensity.bin", blank, pixel_size=9e-6)
    folder = IntensityBinFolder(tmp_path, validate=None)
    folder.validate(level="names")  # header mismatch ignored at "names"
    with pytest.raises(ValueError, match="header"):
        folder.validate()  # default "headers" detects it


def test_validate_passes_on_clean_folder(tmp_path):
    for i in range(3):
        _write(tmp_path, i, i)
    folder = IntensityBinFolder(tmp_path)
    folder.validate()
    folder.validate(level="data")  # finite data: also passes


def test_validate_file_rejects_unknown_level(tmp_path):
    _write(tmp_path, 0, 0)
    seq = IntensityBinFolder(tmp_path, validate=None)
    with pytest.raises(ValueError, match="level must be"):
        seq.validate_file(0, level="bogus")  # ty: ignore[invalid-argument-type]


def test_validate_rejects_header_mismatch(tmp_path):
    blank = np.zeros((2, 3), dtype=np.float32)
    save_intensity_bin(tmp_path / "00000_intensity.bin", blank, pixel_size=1e-6)
    save_intensity_bin(tmp_path / "00001_intensity.bin", blank, pixel_size=9e-6)
    with pytest.raises(ValueError, match="header"):
        IntensityBinFolder(tmp_path, validate=None).validate()


# ========================== #
#        List sequence       #
# ========================== #


def test_list_sequence_loads_arbitrary_unrelated_files(tmp_path):
    # Arbitrary names, nested folder, heterogeneous shapes -- none allowed by
    # IntensityBinFolder; the input order is preserved verbatim.
    a = tmp_path / "alpha.bin"
    sub = tmp_path / "nested"
    sub.mkdir()
    b = sub / "whatever.bin"
    save_intensity_bin(a, np.full((2, 3), 1.0, dtype=np.float32), pixel_size=1e-6)
    save_intensity_bin(b, np.full((4, 5), 2.0, dtype=np.float32), pixel_size=1e-6)

    seq = IntensityBinList([b, a])

    assert len(seq) == 2
    assert [seq.get_meta(i) for i in range(2)] == [b, a]
    np.testing.assert_array_equal(seq[0], np.full((4, 5), 2.0, dtype=np.float32))
    np.testing.assert_array_equal(seq[1], np.full((2, 3), 1.0, dtype=np.float32))
    assert not hasattr(seq, "frame_shape")  # heterogeneous: no uniform shape
