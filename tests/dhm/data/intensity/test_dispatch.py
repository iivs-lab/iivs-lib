from __future__ import annotations

import warnings

import numpy as np
import pytest
from kaparoo.data.sequences import ConcatSequence

from iivs.dhm.data.intensity.bin import (
    IntensityBinFolder,
    IntensityBinHeader,
    IntensityBinList,
    save_intensity_bin,
)
from iivs.dhm.data.intensity.dispatch import (
    convert_intensity_folder,
    convert_intensity_list,
    intensity_folder,
    intensity_list,
    load_intensity,
    read_intensity_header,
    save_intensity,
    save_intensity_folder,
)
from iivs.dhm.data.intensity.npy import IntensityNpyFolder, save_intensity_npy
from iivs.dhm.data.intensity.txt import (
    IntensityTxtFolder,
    IntensityTxtList,
    load_intensity_txt,
    read_intensity_txt_header,
    save_intensity_txt,
)

IMG = np.arange(6, dtype=np.float32).reshape(2, 3)


def _save(path, ext):
    """Write IMG at `path` via `save_intensity`, supplying pixel_size for bin/txt."""
    if ext == "npy":
        save_intensity(path, IMG)
    else:
        save_intensity(path, IMG, pixel_size=1e-6)


def _make_folder(root, ext, n=2):
    root.mkdir()
    for i in range(n):
        _save(root / f"{i:05d}_intensity.{ext}", ext)


def _bin_list(tmp_path, pixel_sizes):
    paths = []
    for i, pixel_size in enumerate(pixel_sizes):
        path = tmp_path / f"f{i}.bin"
        save_intensity_bin(
            path, np.full((2, 3), float(i + 1), np.float32), pixel_size=pixel_size
        )
        paths.append(path)
    return IntensityBinList(paths)


def _bin_folder(root, values):
    root.mkdir()
    for i, value in enumerate(values):
        save_intensity_bin(
            root / f"{i:05d}_intensity.bin",
            np.full((2, 3), float(value), dtype=np.float32),
            pixel_size=1e-6,
        )
    return IntensityBinFolder(root)


# --- load_intensity ---


@pytest.mark.parametrize("ext", ("bin", "txt", "npy"))
def test_load_intensity_dispatches_by_extension(tmp_path, ext):
    path = tmp_path / f"x.{ext}"
    _save(path, ext)
    np.testing.assert_allclose(load_intensity(path), IMG, rtol=1e-5)


def test_load_intensity_rejects_unknown_extension(tmp_path):
    with pytest.raises(ValueError, match=r"unsupported extension .* for intensity"):
        load_intensity(tmp_path / "x.foo")


@pytest.mark.parametrize("ext", ("bin", "txt"))
def test_load_intensity_return_header_gives_header_for_bin_txt(tmp_path, ext):
    path = tmp_path / f"x.{ext}"
    _save(path, ext)
    img, header = load_intensity(path, return_header=True)
    np.testing.assert_allclose(img, IMG, rtol=1e-5)
    assert isinstance(header, IntensityBinHeader)
    assert header.shape == IMG.shape
    assert header.pixel_size == pytest.approx(1e-6, rel=1e-3)


def test_load_intensity_return_header_is_none_for_npy(tmp_path):
    path = tmp_path / "x.npy"
    _save(path, "npy")
    img, header = load_intensity(path, return_header=True)
    np.testing.assert_array_equal(img, IMG)
    assert header is None


# --- read_intensity_header ---


@pytest.mark.parametrize("ext", ("bin", "txt"))
def test_read_intensity_header_dispatches(tmp_path, ext):
    path = tmp_path / f"x.{ext}"
    _save(path, ext)
    header = read_intensity_header(path)
    assert isinstance(header, IntensityBinHeader)
    assert header.pixel_size == pytest.approx(1e-6)


def test_read_intensity_header_rejects_npy(tmp_path):
    with pytest.raises(ValueError, match="header-less"):
        read_intensity_header(tmp_path / "x.npy")


def test_read_intensity_header_rejects_unknown_extension(tmp_path):
    with pytest.raises(ValueError, match=r"unsupported extension .* for intensity"):
        read_intensity_header(tmp_path / "x.foo")


# --- save_intensity ---


@pytest.mark.parametrize("ext", ("bin", "txt"))
def test_save_intensity_roundtrip_bin_txt(tmp_path, ext):
    path = tmp_path / f"x.{ext}"
    save_intensity(path, IMG, pixel_size=2e-6)
    np.testing.assert_allclose(load_intensity(path), IMG, rtol=1e-5)
    assert read_intensity_header(path).pixel_size == pytest.approx(2e-6)


def test_save_intensity_npy_roundtrip(tmp_path):
    path = tmp_path / "x.npy"
    save_intensity(path, IMG)
    np.testing.assert_array_equal(load_intensity(path), IMG)


def test_save_intensity_npy_warns_when_pixel_size_given(tmp_path):
    path = tmp_path / "x.npy"
    with pytest.warns(UserWarning, match="header-less"):
        save_intensity(path, IMG, pixel_size=1e-6)


def test_save_intensity_npy_no_warning_without_metadata(tmp_path):
    path = tmp_path / "x.npy"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        save_intensity(path, IMG)


def test_save_intensity_bin_requires_pixel_size(tmp_path):
    with pytest.raises(ValueError, match="pixel_size is required"):
        save_intensity(tmp_path / "x.bin", IMG)


def test_save_intensity_rejects_unknown_extension(tmp_path):
    with pytest.raises(ValueError, match=r"unsupported extension .* for intensity"):
        save_intensity(tmp_path / "x.foo", IMG, pixel_size=1e-6)


# --- intensity_list ---


@pytest.mark.parametrize(
    ("ext", "cls"), (("bin", IntensityBinList), ("txt", IntensityTxtList))
)
def test_intensity_list_dispatches(tmp_path, ext, cls):
    paths = []
    for i in range(2):
        p = tmp_path / f"f{i}.{ext}"
        _save(p, ext)
        paths.append(p)
    seq = intensity_list(paths)
    assert isinstance(seq, cls)
    assert len(seq) == 2


def test_intensity_list_rejects_npy(tmp_path):
    with pytest.raises(ValueError, match=r"no \.npy intensity list"):
        intensity_list([tmp_path / "a.npy"])


def test_intensity_list_rejects_mixed_extensions(tmp_path):
    with pytest.raises(ValueError, match="share one extension"):
        intensity_list([tmp_path / "a.bin", tmp_path / "b.txt"])


def test_intensity_list_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        intensity_list([])


def test_intensity_list_rejects_unknown_extension(tmp_path):
    with pytest.raises(ValueError, match=r"unsupported extension .* for intensity"):
        intensity_list([tmp_path / "a.foo"])


# --- intensity_folder ---


@pytest.mark.parametrize(
    ("ext", "cls"),
    (("bin", IntensityBinFolder), ("txt", IntensityTxtFolder)),
)
def test_intensity_folder_dispatches_bin_txt(tmp_path, ext, cls):
    root = tmp_path / "acq"
    _make_folder(root, ext)
    folder = intensity_folder(root)
    assert isinstance(folder, cls)
    assert len(folder) == 2


def test_intensity_folder_dispatches_npy(tmp_path):
    root = tmp_path / "acq"
    _make_folder(root, "npy")
    folder = intensity_folder(root, pixel_size=1e-6)
    assert isinstance(folder, IntensityNpyFolder)
    assert folder.header.pixel_size == pytest.approx(1e-6)


def test_intensity_folder_npy_requires_pixel_size(tmp_path):
    root = tmp_path / "acq"
    _make_folder(root, "npy")
    with pytest.raises(ValueError, match="need pixel_size"):
        intensity_folder(root)


def test_intensity_folder_bin_rejects_pixel_size(tmp_path):
    root = tmp_path / "acq"
    _make_folder(root, "bin")
    with pytest.raises(ValueError, match="drop the argument"):
        intensity_folder(root, pixel_size=1e-6)


def test_intensity_folder_rejects_empty(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(FileNotFoundError, match="no NNNNN_intensity"):
        intensity_folder(root)


def test_intensity_folder_rejects_ambiguous_formats(tmp_path):
    root = tmp_path / "acq"
    root.mkdir()
    _save(root / "00000_intensity.bin", "bin")
    _save(root / "00000_intensity.txt", "txt")
    with pytest.raises(ValueError, match="multiple intensity formats"):
        intensity_folder(root)


def test_intensity_folder_prefer_resolves_conflict(tmp_path):
    root = tmp_path / "acq"
    root.mkdir()
    _save(root / "00000_intensity.bin", "bin")
    _save(root / "00000_intensity.txt", "txt")
    # txt, against the natural bin-first order: ignoring `prefer` would give bin, so
    # only this direction shows it is passed through at all.
    assert isinstance(intensity_folder(root, prefer="txt"), IntensityTxtFolder)
    assert isinstance(intensity_folder(root, prefer="bin"), IntensityBinFolder)


# --- single-frame writers ---


def test_save_intensity_txt_roundtrip(tmp_path):
    data = np.array([[0.0, 1.5], [2.0, 3.0]], dtype=np.float32)
    path = tmp_path / "00000_intensity.txt"
    save_intensity_txt(path, data, pixel_size=1e-6)
    loaded, header = load_intensity_txt(path, return_header=True)
    np.testing.assert_allclose(loaded, data, rtol=1e-5)
    assert header.pixel_size == pytest.approx(1e-6)


def test_save_intensity_npy_writer_roundtrip(tmp_path):
    data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    path = tmp_path / "00000_intensity.npy"
    save_intensity_npy(path, data)
    np.testing.assert_array_equal(np.load(path), data)


# --- convert_intensity_folder ---


def test_convert_intensity_to_txt(tmp_path):
    src = _bin_folder(tmp_path / "src", [1.0, 2.0])
    out = tmp_path / "out"
    convert_intensity_folder(out, src, ext="txt")
    dst = IntensityTxtFolder(out)
    assert len(dst) == 2
    np.testing.assert_allclose(dst[0], src[0], rtol=1e-5)
    assert dst.header.pixel_size == pytest.approx(src.header.pixel_size)


def test_convert_intensity_to_npy(tmp_path):
    src = _bin_folder(tmp_path / "src", [1.0, 2.0])
    out = tmp_path / "out"
    with pytest.warns(UserWarning, match="header-less"):  # header cannot go into npy
        convert_intensity_folder(out, src, ext="npy")
    dst = IntensityNpyFolder(out, pixel_size=src.header.pixel_size)
    np.testing.assert_array_equal(dst[0], src[0])


def test_convert_intensity_to_bin(tmp_path):
    src = _bin_folder(tmp_path / "src", [1.0, 2.0])
    out = tmp_path / "out"
    convert_intensity_folder(out, src, ext="bin")
    dst = IntensityBinFolder(out)
    np.testing.assert_array_equal(dst[0], src[0])
    assert dst.header == src.header


def test_convert_intensity_rejects_unknown_format(tmp_path):
    src = _bin_folder(tmp_path / "src", [1.0])
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="ext must be"):
        convert_intensity_folder(out, src, ext="raw")


# --- convert_intensity_list ---


def test_convert_intensity_list_converts_each_file_in_place(tmp_path):
    src = _bin_list(tmp_path, [1e-6, 3e-6])  # f0.bin, f1.bin
    convert_intensity_list(src, ext="txt")  # -> f0.txt, f1.txt (siblings)
    hdr = read_intensity_txt_header
    assert hdr(tmp_path / "f0.txt").pixel_size == pytest.approx(1e-6)
    assert hdr(tmp_path / "f1.txt").pixel_size == pytest.approx(3e-6)
    np.testing.assert_allclose(
        load_intensity_txt(tmp_path / "f0.txt"), src[0], rtol=1e-5
    )


def test_convert_intensity_list_to_npy(tmp_path):
    src = _bin_list(tmp_path, [1e-6])
    convert_intensity_list(src, ext="npy")
    np.testing.assert_array_equal(np.load(tmp_path / "f0.npy"), src[0])


def test_convert_intensity_list_rejects_unknown_format(tmp_path):
    src = _bin_list(tmp_path, [1e-6])
    with pytest.raises(ValueError, match="ext must be"):
        convert_intensity_list(src, ext="raw")


def test_intensity_file_list_get_header(tmp_path):
    src = _bin_list(tmp_path, [1e-6, 3e-6])
    assert src.get_header(0).pixel_size == pytest.approx(1e-6)
    assert src.get_header(1).pixel_size == pytest.approx(3e-6)


def test_intensity_file_list_load_with_header(tmp_path):
    src = _bin_list(tmp_path, [1e-6, 3e-6])
    image, header = src.load_with_header(1)
    np.testing.assert_array_equal(image, src[1])
    assert header.pixel_size == pytest.approx(3e-6)


# --- save_intensity_folder (composer-compatible export) ---


def test_save_intensity_folder_from_concat_sequence(tmp_path):
    a = _bin_folder(tmp_path / "a", [0.0, 1.0])
    b = _bin_folder(tmp_path / "b", [10.0, 11.0])
    combined = ConcatSequence(a, b)
    out = tmp_path / "out"
    save_intensity_folder(out, combined, ext="bin", pixel_size=1e-6)
    reopened = IntensityBinFolder(out)
    assert len(reopened) == 4
    for i, value in enumerate((0.0, 1.0, 10.0, 11.0)):
        np.testing.assert_allclose(reopened[i], np.full((2, 3), value), rtol=1e-5)


@pytest.mark.parametrize("ext", ("bin", "txt", "npy"))
def test_save_intensity_folder_writes_numbered_files(tmp_path, ext):
    images = [np.full((2, 2), float(i), np.float32) for i in range(3)]
    out = tmp_path / "out"
    if ext == "npy":
        save_intensity_folder(out, images, ext=ext)
    else:
        save_intensity_folder(out, images, ext=ext, pixel_size=1e-6)
    assert sorted(p.name for p in out.iterdir()) == [
        f"{i:05d}_intensity.{ext}" for i in range(3)
    ]


def test_save_intensity_folder_requires_pixel_size(tmp_path):
    images = [np.zeros((2, 2), np.float32)]
    with pytest.raises(ValueError, match="pixel_size is required"):
        save_intensity_folder(tmp_path / "out", images, ext="bin")


def test_save_intensity_folder_npy_warns_on_pixel_size(tmp_path):
    images = [np.zeros((2, 2), np.float32)]
    with pytest.warns(UserWarning, match="header-less"):
        save_intensity_folder(tmp_path / "out", images, ext="npy", pixel_size=1e-6)


def test_save_intensity_folder_stem_override(tmp_path):
    images = [np.zeros((2, 2), np.float32)]
    out = tmp_path / "out"
    save_intensity_folder(out, images, ext="npy", stem="custom")
    assert (out / "00000_custom.npy").exists()


def test_save_intensity_folder_rejects_unknown_format(tmp_path):
    images = [np.zeros((2, 2), np.float32)]
    with pytest.raises(ValueError, match="ext must be"):
        save_intensity_folder(tmp_path / "out", images, ext="raw")


def test_save_intensity_folder_rejects_empty(tmp_path):
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="empty intensity sequence"):
        save_intensity_folder(out, [], ext="npy")
    assert not out.exists()  # atomic: no unreadable folder left behind
