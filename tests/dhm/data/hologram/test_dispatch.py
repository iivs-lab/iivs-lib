from __future__ import annotations

import numpy as np
import pytest
from kaparoo.data.sequences import ConcatSequence

from iivs.dhm.data.hologram.dispatch import convert_hologram_sequence
from iivs.dhm.data.hologram.npy import HologramNpyFolder, save_hologram_npy
from iivs.dhm.data.hologram.raw import (
    HologramRawFile,
    read_hologram_raw_header,
    save_hologram_raw,
)
from iivs.dhm.data.hologram.tif import (
    HologramTifFolder,
    HologramTifList,
    save_hologram_tif,
)


def _stack():
    return np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)


def _raw_source(path, stack=None):
    stack = _stack() if stack is None else stack
    save_hologram_raw(path, stack)
    return HologramRawFile(path)


# --- save_hologram_raw / npy ---


def test_save_hologram_raw_roundtrip(tmp_path):
    stack = _stack()
    path = tmp_path / "holo.raw"
    save_hologram_raw(path, stack)
    header = read_hologram_raw_header(path)
    assert (header.frame_count, header.height, header.width) == (2, 2, 3)
    seq = HologramRawFile(path)
    assert len(seq) == 2
    np.testing.assert_array_equal(seq[0], stack[0])
    np.testing.assert_array_equal(seq[1], stack[1])


def test_save_hologram_raw_single_frame_becomes_stack_of_one(tmp_path):
    frame = np.full((2, 3), 7, dtype=np.uint8)
    path = tmp_path / "h.raw"
    save_hologram_raw(path, frame)
    seq = HologramRawFile(path)
    assert len(seq) == 1
    np.testing.assert_array_equal(seq[0], frame)


def test_save_hologram_raw_rejects_4d(tmp_path):
    with pytest.raises(ValueError, match="2D image or"):
        save_hologram_raw(tmp_path / "h.raw", np.zeros((2, 2, 2, 2), dtype=np.uint8))


def test_save_hologram_npy_roundtrip(tmp_path):
    data = np.full((2, 3), 9, dtype=np.uint8)
    path = tmp_path / "00000_holo.npy"
    save_hologram_npy(path, data)
    np.testing.assert_array_equal(np.load(path), data)


# --- convert_hologram_sequence ---


def test_convert_hologram_to_tif(tmp_path):
    stack = _stack()
    src = _raw_source(tmp_path / "holo.raw", stack)
    out = tmp_path / "tif"
    convert_hologram_sequence(out, src, ext="tif")
    dst = HologramTifFolder(out)
    np.testing.assert_array_equal(dst[0], stack[0])
    np.testing.assert_array_equal(dst[1], stack[1])


def test_convert_hologram_to_npy(tmp_path):
    stack = _stack()
    src = _raw_source(tmp_path / "holo.raw", stack)
    out = tmp_path / "npy"
    convert_hologram_sequence(out, src, ext="npy")
    dst = HologramNpyFolder(out)
    np.testing.assert_array_equal(dst[0], stack[0])
    np.testing.assert_array_equal(dst[1], stack[1])


def test_convert_hologram_to_raw(tmp_path):
    stack = _stack()
    src = _raw_source(tmp_path / "src.raw", stack)
    dest = tmp_path / "out.raw"
    convert_hologram_sequence(dest, src, ext="raw")
    dst = HologramRawFile(dest)
    np.testing.assert_array_equal(np.stack([dst[i] for i in range(len(dst))]), stack)


def test_convert_hologram_rejects_unknown_format(tmp_path):
    src = _raw_source(tmp_path / "holo.raw")
    with pytest.raises(ValueError, match="ext must be"):
        convert_hologram_sequence(tmp_path, src, ext="bin")


@pytest.mark.parametrize("ext", ("tif", "npy"))
def test_convert_hologram_folder_rejects_empty(tmp_path, ext):
    # The raw path already rejects empty; the numbered-folder path does too, rather
    # than committing a folder no reader accepts.
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="empty hologram sequence"):
        convert_hologram_sequence(out, HologramTifList([]), ext=ext)
    assert not out.exists()  # atomic: no unreadable folder left behind


def test_save_hologram_raw_streams_a_sequence(tmp_path):
    stack = _stack()
    src = _raw_source(tmp_path / "src.raw", stack)  # a HologramSequence, not an array
    out = tmp_path / "out.raw"
    save_hologram_raw(out, src)  # streamed frame by frame, no full-stack copy
    dst = HologramRawFile(out)
    np.testing.assert_array_equal(np.stack([dst[i] for i in range(len(dst))]), stack)


def test_save_hologram_raw_rejects_empty_sequence(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        save_hologram_raw(tmp_path / "out.raw", HologramTifList([]))


def test_save_hologram_raw_rejects_empty_stack(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        save_hologram_raw(tmp_path / "out.raw", np.zeros((0, 2, 3), dtype=np.uint8))


def test_save_hologram_raw_rejects_mismatched_frame_shapes(tmp_path):
    save_hologram_tif(tmp_path / "a.tif", np.zeros((2, 3), np.uint8))
    save_hologram_tif(tmp_path / "b.tif", np.zeros((2, 4), np.uint8))  # different width
    seq = HologramTifList([tmp_path / "a.tif", tmp_path / "b.tif"])
    with pytest.raises(ValueError, match="all frames must have"):
        save_hologram_raw(tmp_path / "out.raw", seq)


# --- composer compatibility ---


def test_save_hologram_raw_accepts_composed_sequence(tmp_path):
    # A ConcatSequence is not a HologramSequence, but save_hologram_raw takes
    # any uint8 DataSequence.
    a = _raw_source(tmp_path / "a.raw")
    b = _raw_source(tmp_path / "b.raw", _stack() + 100)
    combined = ConcatSequence(a, b)
    save_hologram_raw(tmp_path / "out.raw", combined)
    out = HologramRawFile(tmp_path / "out.raw")
    assert len(out) == 4
    np.testing.assert_array_equal(out[0], a[0])
    np.testing.assert_array_equal(out[3], b[1])


def test_convert_hologram_sequence_accepts_composed_sequence(tmp_path):
    a = _raw_source(tmp_path / "a.raw")
    b = _raw_source(tmp_path / "b.raw", _stack() + 100)
    combined = ConcatSequence(a, b)

    convert_hologram_sequence(tmp_path / "out.raw", combined, ext="raw")
    assert len(HologramRawFile(tmp_path / "out.raw")) == 4

    convert_hologram_sequence(tmp_path / "tif", combined, ext="tif")
    assert sorted(p.name for p in (tmp_path / "tif").iterdir()) == [
        f"{i:05d}_holo.tif" for i in range(4)
    ]
