from __future__ import annotations

import warnings

import numpy as np
import pytest

from iivs.dhm.data.intensity.bin import (
    IntensityBinFolder,
    IntensityBinHeader,
    IntensityBinList,
)
from iivs.dhm.data.intensity.factory import (
    intensity_folder,
    intensity_list,
    load_intensity,
    read_intensity_header,
    save_intensity,
)
from iivs.dhm.data.intensity.npy import IntensityNpyFolder
from iivs.dhm.data.intensity.txt import IntensityTxtFolder, IntensityTxtList

IMG = np.arange(6, dtype=np.float32).reshape(2, 3)


def _save(path, ext):
    """Write IMG at `path` through `save_intensity`, supplying pixel_size for bin/txt."""
    if ext == "npy":
        save_intensity(path, IMG)
    else:
        save_intensity(path, IMG, pixel_size=1e-6)


def _make_folder(root, ext, n=2):
    root.mkdir()
    for i in range(n):
        _save(root / f"{i:05d}_intensity.{ext}", ext)


# --- load_intensity ---


@pytest.mark.parametrize("ext", ("bin", "txt", "npy"))
def test_load_intensity_dispatches_by_extension(tmp_path, ext):
    path = tmp_path / f"x.{ext}"
    _save(path, ext)
    np.testing.assert_allclose(load_intensity(path), IMG, rtol=1e-5)


def test_load_intensity_rejects_unknown_extension(tmp_path):
    with pytest.raises(ValueError, match="unsupported intensity extension"):
        load_intensity(tmp_path / "x.foo")


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
    with pytest.raises(ValueError, match="unsupported intensity extension"):
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
    with pytest.raises(ValueError, match="unsupported intensity extension"):
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
    with pytest.raises(ValueError, match="unsupported intensity extension"):
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
    assert isinstance(intensity_folder(root, prefer="bin"), IntensityBinFolder)
