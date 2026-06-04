from __future__ import annotations

import numpy as np
import pytest

from iivs.dhm.data.phase.bin import PhaseBinFolder, PhaseBinList, save_phase_bin
from iivs.dhm.data.phase.convert import convert_phase_folder, convert_phase_list
from iivs.dhm.data.phase.core import PhaseUnit
from iivs.dhm.data.phase.npy import PhaseNpyFolder, save_phase_npy
from iivs.dhm.data.phase.txt import (
    PhaseTxtFolder,
    load_phase_txt,
    read_phase_txt_header,
    save_phase_txt,
)


def _bin_list(tmp_path, specs):
    paths = []
    for i, (value, pixel_size) in enumerate(specs):
        path = tmp_path / f"f{i}.bin"
        save_phase_bin(
            path,
            np.full((2, 3), float(value), np.float32),
            pixel_size=pixel_size,
            height_scale=2e-7,
        )
        paths.append(path)
    return PhaseBinList(paths)


def _bin_folder(root, values, *, height_scale=2e-7, unit=PhaseUnit.RADIANS):
    root.mkdir()
    for i, value in enumerate(values):
        save_phase_bin(
            root / f"{i:05d}_phase.bin",
            np.full((2, 3), float(value), dtype=np.float32),
            pixel_size=1e-6,
            height_scale=height_scale,
            unit=unit,
        )
    return PhaseBinFolder(root)


# --- single-frame writers ---


def test_save_phase_txt_roundtrip(tmp_path):
    data = np.array([[0.0, 1.5], [2.0, -3.0]], dtype=np.float32)
    path = tmp_path / "00000_phase.txt"
    save_phase_txt(
        path, data, pixel_size=1e-6, height_scale=2e-7, unit=PhaseUnit.RADIANS
    )
    loaded, header = load_phase_txt(path, return_header=True)
    np.testing.assert_allclose(loaded, data, rtol=1e-5)
    assert header.pixel_size == pytest.approx(1e-6)
    assert header.height_scale == pytest.approx(2e-7)
    assert header.unit is PhaseUnit.RADIANS


def test_save_phase_txt_warns_on_unknown(tmp_path):
    with pytest.warns(UserWarning, match="UNKNOWN"):
        save_phase_txt(
            tmp_path / "p.txt",
            np.zeros((1, 2), dtype=np.float32),
            pixel_size=1e-6,
            height_scale=2e-7,
            unit=PhaseUnit.UNKNOWN,
        )


def test_save_phase_npy_roundtrip(tmp_path):
    data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    path = tmp_path / "00000_phase.npy"
    save_phase_npy(path, data)
    np.testing.assert_array_equal(np.load(path), data)


# --- convert_phase_folder ---


def test_convert_phase_to_txt(tmp_path):
    src = _bin_folder(tmp_path / "src", [1.0, 2.0])
    out = tmp_path / "out"
    convert_phase_folder(out, src, ext="txt")
    dst = PhaseTxtFolder(out)
    assert len(dst) == 2
    np.testing.assert_allclose(dst[0], src[0], rtol=1e-5)
    np.testing.assert_allclose(dst[1], src[1], rtol=1e-5)
    assert dst.header.unit is src.header.unit
    assert dst.header.height_scale == pytest.approx(src.header.height_scale)


def test_convert_phase_to_npy(tmp_path):
    src = _bin_folder(tmp_path / "src", [1.0, 2.0])
    out = tmp_path / "out"
    convert_phase_folder(out, src, ext="npy")
    dst = PhaseNpyFolder(
        out,
        pixel_size=src.header.pixel_size,
        unit=src.header.unit,
        height_scale=src.header.height_scale,
    )
    np.testing.assert_array_equal(dst[0], src[0])
    np.testing.assert_array_equal(dst[1], src[1])


def test_convert_phase_to_bin(tmp_path):
    src = _bin_folder(tmp_path / "src", [1.0, 2.0])
    out = tmp_path / "out"
    convert_phase_folder(out, src, ext="bin")
    dst = PhaseBinFolder(out)
    np.testing.assert_array_equal(dst[0], src[0])
    assert dst.header == src.header


def test_convert_phase_rejects_unknown_format(tmp_path):
    src = _bin_folder(tmp_path / "src", [1.0])
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="ext must be"):
        convert_phase_folder(out, src, ext="raw")


def test_convert_phase_no_overwrite(tmp_path):
    src = _bin_folder(tmp_path / "src", [1.0])
    out = tmp_path / "out"
    convert_phase_folder(out, src, ext="txt")
    with pytest.raises(FileExistsError):
        convert_phase_folder(out, src, ext="txt")
    convert_phase_folder(out, src, ext="txt", overwrite=True)  # whole folder replaced


def test_convert_phase_is_atomic_on_failure(tmp_path, monkeypatch):
    src = _bin_folder(tmp_path / "src", [1.0, 2.0])
    out = tmp_path / "out"
    real, calls = save_phase_txt, {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            msg = "boom"
            raise RuntimeError(msg)
        return real(*args, **kwargs)

    monkeypatch.setattr("iivs.dhm.data.phase.convert.save_phase_txt", flaky)
    with pytest.raises(RuntimeError, match="boom"):
        convert_phase_folder(out, src, ext="txt")
    assert not out.exists()  # StagedDirectory discarded the partially-built folder


# --- convert_phase_list ---


def test_convert_phase_list_converts_each_file_in_place(tmp_path):
    src = _bin_list(tmp_path, [(1.0, 1e-6), (2.0, 3e-6)])  # f0.bin, f1.bin
    convert_phase_list(src, ext="txt")  # -> f0.txt, f1.txt (siblings, same names)
    assert read_phase_txt_header(tmp_path / "f0.txt").pixel_size == pytest.approx(1e-6)
    assert read_phase_txt_header(tmp_path / "f1.txt").pixel_size == pytest.approx(3e-6)
    np.testing.assert_allclose(load_phase_txt(tmp_path / "f0.txt"), src[0], rtol=1e-5)


def test_convert_phase_list_to_npy(tmp_path):
    src = _bin_list(tmp_path, [(1.0, 1e-6)])
    convert_phase_list(src, ext="npy")
    np.testing.assert_array_equal(np.load(tmp_path / "f0.npy"), src[0])


def test_convert_phase_list_rejects_unknown_format(tmp_path):
    src = _bin_list(tmp_path, [(1.0, 1e-6)])
    with pytest.raises(ValueError, match="ext must be"):
        convert_phase_list(src, ext="raw")


def test_phase_file_list_get_header(tmp_path):
    src = _bin_list(tmp_path, [(1.0, 1e-6), (2.0, 3e-6)])
    assert src.get_header(0).pixel_size == pytest.approx(1e-6)
    assert src.get_header(1).pixel_size == pytest.approx(3e-6)
