from __future__ import annotations

import pickle

import numpy as np
import pytest

from iivs.dhm.data.hologram.raw import (
    HologramRawFile,
    HologramRawHeader,
    read_hologram_raw_header,
)


def _write_raw(path, *, n=3, h=2, w=3, bit_depth=8):
    """Write a synthetic `.raw` (header + frames) and return the frames."""
    frames = np.arange(n * h * w, dtype=np.uint8).reshape(n, h, w)
    header = np.array([w, h, bit_depth, n], dtype="<i4")
    path.write_bytes(header.tobytes() + frames.tobytes())
    return frames


def test_header_fields_and_layout(tmp_path):
    path = tmp_path / "holo.raw"
    _write_raw(path, n=5, h=2, w=3)
    hdr = read_hologram_raw_header(path)
    assert (hdr.width, hdr.height, hdr.bit_depth, hdr.frame_count) == (3, 2, 8, 5)
    assert hdr.shape == (2, 3)
    assert hdr.pixel_count == 6
    assert hdr.pixel_dtype == np.uint8
    assert hdr.frame_nbytes == 6
    assert hdr.data_nbytes == 30
    assert HologramRawHeader.HEADER_SIZE == 16


def test_rejects_wrong_extension(tmp_path):
    # A non-.raw path is rejected up front, regardless of contents.
    path = tmp_path / "holo.bin"
    _write_raw(path)
    with pytest.raises(ValueError, match=r"must have a \.raw extension"):
        HologramRawFile(path)


def test_sequence_roundtrip(tmp_path):
    path = tmp_path / "holo.raw"
    frames = _write_raw(path, n=4, h=2, w=3)

    seq = HologramRawFile(path)

    assert len(seq) == 4
    assert seq.header.frame_count == 4
    for i in range(4):
        np.testing.assert_array_equal(seq[i], frames[i])
    assert [seq.get_meta(i) for i in range(4)] == [0, 1, 2, 3]


def test_get_item_returns_writable_copy(tmp_path):
    path = tmp_path / "holo.raw"
    frames = _write_raw(path, n=3, h=2, w=3)
    seq = HologramRawFile(path)

    item = seq[0]
    assert item.flags.writeable  # unlike the underlying read-only memmap

    item[0, 0] = 255  # mutating the copy must not touch the file/memmap
    np.testing.assert_array_equal(seq.frames[0], frames[0])


def test_frame_shape(tmp_path):
    path = tmp_path / "holo.raw"
    _write_raw(path, n=3, h=2, w=3)
    assert HologramRawFile(path).frame_shape == (2, 3)


def test_frames_property_exposes_full_memmap(tmp_path):
    path = tmp_path / "holo.raw"
    frames = _write_raw(path, n=3, h=2, w=3)
    seq = HologramRawFile(path)
    assert seq.frames.shape == (3, 2, 3)
    np.testing.assert_array_equal(seq.frames, frames)


def test_sequence_pickles_to_path_without_copying_frames(tmp_path):
    path = tmp_path / "holo.raw"
    frames = _write_raw(path, n=8, h=64, w=64)  # 32 KiB of pixels
    seq = HologramRawFile(path)

    blob = pickle.dumps(seq)
    assert len(blob) < frames.nbytes  # carries the path, not the frame bytes

    # round-trips our own bytes (test-only), not untrusted input
    restored = pickle.loads(blob)  # noqa: S301
    assert len(restored) == 8
    np.testing.assert_array_equal(restored[0], frames[0])
    np.testing.assert_array_equal(restored.frames, frames)


def test_size_mismatch_raises(tmp_path):
    path = tmp_path / "holo.raw"
    # Header claims 3 frames, but only 2 frames of data are written.
    header = np.array([3, 2, 8, 3], dtype="<i4")  # w, h, bit_depth, frame_count
    data = np.zeros((2, 2, 3), dtype=np.uint8)
    path.write_bytes(header.tobytes() + data.tobytes())
    with pytest.raises(ValueError, match="file size must be"):
        HologramRawFile(path)


def test_rejects_unsupported_bit_depth(tmp_path):
    path = tmp_path / "holo.raw"
    path.write_bytes(np.array([3, 2, 16, 1], dtype="<i4").tobytes())
    with pytest.raises(ValueError, match="bit_depth"):
        read_hologram_raw_header(path)


def test_rejects_nonpositive_dims(tmp_path):
    path = tmp_path / "holo.raw"
    path.write_bytes(np.array([0, 2, 8, 1], dtype="<i4").tobytes())
    with pytest.raises(ValueError, match="must be positive"):
        read_hologram_raw_header(path)


def test_rejects_negative_frame_count(tmp_path):
    path = tmp_path / "holo.raw"
    path.write_bytes(np.array([3, 2, 8, -1], dtype="<i4").tobytes())
    with pytest.raises(ValueError, match="frame_count must be non-negative"):
        read_hologram_raw_header(path)


def test_rejects_truncated_header(tmp_path):
    path = tmp_path / "holo.raw"
    path.write_bytes(b"\x00\x04\x00\x00")  # only 4 bytes
    with pytest.raises(ValueError, match="bytes for a header"):
        read_hologram_raw_header(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_hologram_raw_header(tmp_path / "nope.raw")
