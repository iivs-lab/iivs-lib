from __future__ import annotations

import warnings

import numpy as np
import pytest

from iivs.dhm.data.phase.bin import PhaseBinFolder, PhaseBinHeader, PhaseBinList
from iivs.dhm.data.phase.factory import (
    load_phase,
    phase_folder,
    phase_list,
    read_phase_header,
    save_phase,
)
from iivs.dhm.data.phase.npy import PhaseNpyFolder
from iivs.dhm.data.phase.txt import PhaseTxtFolder, PhaseTxtList
from iivs.dhm.data.phase.unit import PhaseUnit

IMG = np.arange(6, dtype=np.float32).reshape(2, 3)


def _save(path, ext):
    """Write IMG at `path` through `save_phase`, supplying metadata for bin/txt."""
    if ext == "npy":
        save_phase(path, IMG)
    else:
        save_phase(path, IMG, pixel_size=1e-6, height_scale=2e-7)


def _make_folder(root, ext, n=2):
    root.mkdir()
    for i in range(n):
        _save(root / f"{i:05d}_phase.{ext}", ext)


# --- load_phase ---


@pytest.mark.parametrize("ext", ("bin", "txt", "npy"))
def test_load_phase_dispatches_by_extension(tmp_path, ext):
    path = tmp_path / f"x.{ext}"
    _save(path, ext)
    np.testing.assert_allclose(load_phase(path), IMG, rtol=1e-5)


def test_load_phase_rejects_unknown_extension(tmp_path):
    with pytest.raises(ValueError, match="unsupported phase extension"):
        load_phase(tmp_path / "x.foo")


# --- read_phase_header ---


@pytest.mark.parametrize("ext", ("bin", "txt"))
def test_read_phase_header_dispatches(tmp_path, ext):
    path = tmp_path / f"x.{ext}"
    _save(path, ext)
    header = read_phase_header(path)
    assert isinstance(header, PhaseBinHeader)
    assert header.pixel_size == pytest.approx(1e-6)
    assert header.height_scale == pytest.approx(2e-7)


def test_read_phase_header_rejects_npy(tmp_path):
    with pytest.raises(ValueError, match="header-less"):
        read_phase_header(tmp_path / "x.npy")


def test_read_phase_header_rejects_unknown_extension(tmp_path):
    with pytest.raises(ValueError, match="unsupported phase extension"):
        read_phase_header(tmp_path / "x.foo")


# --- save_phase ---


@pytest.mark.parametrize("ext", ("bin", "txt"))
def test_save_phase_roundtrip_bin_txt(tmp_path, ext):
    path = tmp_path / f"x.{ext}"
    save_phase(path, IMG, pixel_size=1e-6, height_scale=2e-7, unit=PhaseUnit.METERS)
    np.testing.assert_allclose(load_phase(path), IMG, rtol=1e-5)
    assert read_phase_header(path).unit is PhaseUnit.METERS


def test_save_phase_npy_roundtrip(tmp_path):
    path = tmp_path / "x.npy"
    save_phase(path, IMG)
    np.testing.assert_array_equal(load_phase(path), IMG)


def test_save_phase_npy_warns_when_metadata_given(tmp_path):
    path = tmp_path / "x.npy"
    with pytest.warns(UserWarning, match="header-less"):
        save_phase(path, IMG, pixel_size=1e-6)
    np.testing.assert_array_equal(load_phase(path), IMG)


def test_save_phase_npy_no_warning_without_metadata(tmp_path):
    path = tmp_path / "x.npy"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        save_phase(path, IMG)


def test_save_phase_bin_requires_pixel_size(tmp_path):
    with pytest.raises(ValueError, match="pixel_size is required"):
        save_phase(tmp_path / "x.bin", IMG, height_scale=2e-7)


def test_save_phase_bin_requires_a_scale_form(tmp_path):
    with pytest.raises(ValueError, match="height_scale, or wavelength"):
        save_phase(tmp_path / "x.bin", IMG, pixel_size=1e-6)


def test_save_phase_rejects_unknown_extension(tmp_path):
    with pytest.raises(ValueError, match="unsupported phase extension"):
        save_phase(tmp_path / "x.foo", IMG, pixel_size=1e-6, height_scale=2e-7)


# --- phase_list ---


@pytest.mark.parametrize(("ext", "cls"), (("bin", PhaseBinList), ("txt", PhaseTxtList)))
def test_phase_list_dispatches(tmp_path, ext, cls):
    paths = []
    for i in range(2):
        p = tmp_path / f"f{i}.{ext}"
        _save(p, ext)
        paths.append(p)
    seq = phase_list(paths)
    assert isinstance(seq, cls)
    assert len(seq) == 2


def test_phase_list_rejects_npy(tmp_path):
    with pytest.raises(ValueError, match=r"no \.npy phase list"):
        phase_list([tmp_path / "a.npy"])


def test_phase_list_rejects_mixed_extensions(tmp_path):
    with pytest.raises(ValueError, match="share one extension"):
        phase_list([tmp_path / "a.bin", tmp_path / "b.txt"])


def test_phase_list_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        phase_list([])


def test_phase_list_rejects_unknown_extension(tmp_path):
    with pytest.raises(ValueError, match="unsupported phase extension"):
        phase_list([tmp_path / "a.foo"])


# --- phase_folder ---


@pytest.mark.parametrize(
    ("ext", "cls"),
    (("bin", PhaseBinFolder), ("txt", PhaseTxtFolder)),
)
def test_phase_folder_dispatches_bin_txt(tmp_path, ext, cls):
    root = tmp_path / "acq"
    _make_folder(root, ext)
    folder = phase_folder(root)
    assert isinstance(folder, cls)
    assert len(folder) == 2


def test_phase_folder_dispatches_npy(tmp_path):
    root = tmp_path / "acq"
    _make_folder(root, "npy")
    folder = phase_folder(
        root, pixel_size=1e-6, unit=PhaseUnit.RADIANS, height_scale=2e-7
    )
    assert isinstance(folder, PhaseNpyFolder)
    assert folder.header.pixel_size == pytest.approx(1e-6)


def test_phase_folder_npy_requires_metadata(tmp_path):
    root = tmp_path / "acq"
    _make_folder(root, "npy")
    with pytest.raises(ValueError, match="need pixel_size and unit"):
        phase_folder(root)


def test_phase_folder_bin_rejects_metadata_args(tmp_path):
    root = tmp_path / "acq"
    _make_folder(root, "bin")
    with pytest.raises(ValueError, match="drop the metadata args"):
        phase_folder(root, pixel_size=1e-6)


def test_phase_folder_rejects_empty(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(FileNotFoundError, match="no NNNNN_phase"):
        phase_folder(root)


def test_phase_folder_rejects_ambiguous_formats(tmp_path):
    root = tmp_path / "acq"
    root.mkdir()
    _save(root / "00000_phase.bin", "bin")
    _save(root / "00000_phase.txt", "txt")
    with pytest.raises(ValueError, match="multiple phase formats"):
        phase_folder(root)
