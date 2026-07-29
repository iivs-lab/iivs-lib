from __future__ import annotations

import warnings

import numpy as np
import pytest
from kaparoo.data.sequences import ConcatSequence

from iivs.dhm.data.phase.bin import (
    PhaseBinFolder,
    PhaseBinHeader,
    PhaseBinList,
    save_phase_bin,
)
from iivs.dhm.data.phase.dispatch import (
    convert_phase_folder,
    convert_phase_list,
    load_phase,
    phase_folder,
    phase_list,
    read_phase_header,
    save_phase,
    save_phase_folder,
)
from iivs.dhm.data.phase.npy import PhaseNpyFolder, save_phase_npy
from iivs.dhm.data.phase.txt import (
    PhaseTxtFolder,
    PhaseTxtList,
    load_phase_txt,
    read_phase_txt_header,
    save_phase_txt,
)
from iivs.dhm.data.phase.unit import PhaseUnit
from tests.dhm.data.helpers import count_reads

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


# --- load_phase ---


@pytest.mark.parametrize("ext", ("bin", "txt", "npy"))
def test_load_phase_dispatches_by_extension(tmp_path, ext):
    path = tmp_path / f"x.{ext}"
    _save(path, ext)
    np.testing.assert_allclose(load_phase(path), IMG, rtol=1e-5)


def test_load_phase_rejects_unknown_extension(tmp_path):
    with pytest.raises(ValueError, match=r"unsupported extension .* for phase"):
        load_phase(tmp_path / "x.foo")


@pytest.mark.parametrize("ext", ("bin", "txt"))
def test_load_phase_return_header_gives_header_for_bin_txt(tmp_path, ext):
    path = tmp_path / f"x.{ext}"
    _save(path, ext)
    img, header = load_phase(path, return_header=True)
    np.testing.assert_allclose(img, IMG, rtol=1e-5)
    assert isinstance(header, PhaseBinHeader)
    assert header.shape == IMG.shape
    assert header.pixel_size == pytest.approx(1e-6, rel=1e-3)


def test_load_phase_return_header_is_none_for_npy(tmp_path):
    path = tmp_path / "x.npy"
    _save(path, "npy")
    img, header = load_phase(path, return_header=True)
    np.testing.assert_array_equal(img, IMG)
    assert header is None


@pytest.mark.parametrize("ext", ("bin", "txt"))
def test_load_phase_target_unit_passes_through(tmp_path, ext):
    path = tmp_path / f"x.{ext}"
    _save(path, ext)  # stored in RADIANS with height_scale 2e-7 m/rad
    nm = load_phase(path, target_unit=PhaseUnit.NANOMETERS)
    np.testing.assert_allclose(nm, IMG * 200.0, rtol=1e-5)  # rad -> nm is *200


def test_load_phase_target_unit_rejected_for_npy(tmp_path):
    path = tmp_path / "x.npy"
    _save(path, "npy")
    with pytest.raises(ValueError, match="header-less"):
        load_phase(path, target_unit=PhaseUnit.RADIANS)


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
    with pytest.raises(ValueError, match=r"unsupported extension .* for phase"):
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
    with pytest.raises(ValueError, match=r"unsupported extension .* for phase"):
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
    with pytest.raises(ValueError, match=r"unsupported extension .* for phase"):
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


def test_the_hub_exports_what_the_dispatchers_declare(tmp_path):
    # A caller who writes `folder = phase_folder(root)` must be able to name its type,
    # so the types the public signatures declare are importable from the package hub.
    from iivs.dhm.data.phase import PhaseFileFolder, PhaseFileList

    root = tmp_path / "acq"
    _make_folder(root, "bin")
    assert isinstance(phase_folder(root), PhaseFileFolder)  # phase_folder's return type
    files = sorted(root.glob("*.bin"))
    assert isinstance(phase_list(files), PhaseFileList)  # phase_list's return type


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


def _mixed_folder(root):
    root.mkdir()
    _save(root / "00000_phase.bin", "bin")
    _save(root / "00000_phase.txt", "txt")


def test_phase_folder_prefer_single_format(tmp_path):
    root = tmp_path / "acq"
    _mixed_folder(root)
    assert isinstance(phase_folder(root, prefer="txt"), PhaseTxtFolder)


def test_phase_folder_prefer_priority_sequence(tmp_path):
    root = tmp_path / "acq"
    _mixed_folder(root)
    # `prefer` is honoured in its own order, not the natural FLOAT_FORMATS one — which
    # would give bin here, so the sequence has to contradict it to prove anything.
    assert isinstance(phase_folder(root, prefer=("txt", "bin")), PhaseTxtFolder)
    assert isinstance(phase_folder(root, prefer=("bin", "txt")), PhaseBinFolder)


def test_phase_folder_prefer_absent_format_raises(tmp_path):
    root = tmp_path / "acq"
    _mixed_folder(root)
    with pytest.raises(ValueError, match="selects none of the present"):
        phase_folder(root, prefer="npy")


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


def test_save_phase_npy_writer_roundtrip(tmp_path):
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
    with pytest.warns(UserWarning, match="header-less"):  # header cannot go into npy
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
    with pytest.raises(FileExistsError, match="already exists"):
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

    monkeypatch.setattr("iivs.dhm.data.phase.dispatch.save_phase_txt", flaky)
    with pytest.raises(RuntimeError, match="boom"):
        convert_phase_folder(out, src, ext="txt")
    assert not out.exists()  # StagedDirectory discarded the partially-built folder


# --- convert_phase_list ---


def test_convert_phase_list_reads_each_source_once(tmp_path, monkeypatch):
    src = _bin_list(tmp_path, [(1.0, 1e-6), (2.0, 3e-6)])
    reads = count_reads(monkeypatch, PhaseBinList)

    convert_phase_list(src, ext="txt")

    # Each source is passed over once for image + header together, not twice.
    assert (reads["decode"], reads["header"]) == (len(src), 0)


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


def test_phase_file_list_load_with_header(tmp_path):
    # The single-read accessor returns the same image as get_item plus the
    # file's own header.
    src = _bin_list(tmp_path, [(1.0, 1e-6), (2.0, 3e-6)])
    image, header = src.load_with_header(1)
    np.testing.assert_array_equal(image, src[1])
    assert header.pixel_size == pytest.approx(3e-6)


def test_phase_file_list_load_with_header_reads_the_file_once(tmp_path, monkeypatch):
    src = _bin_list(tmp_path, [(1.0, 1e-6), (2.0, 3e-6)])
    reads = count_reads(monkeypatch, PhaseBinList)

    _ = src.load_with_header(1)
    assert (reads["decode"], reads["header"]) == (1, 0)  # one pass, header included

    _ = src[1], src.get_header(1)  # the pair it exists to replace: two passes
    assert (reads["decode"], reads["header"]) == (2, 1)


# --- save_phase_folder (composer-compatible export) ---


def test_save_phase_folder_from_concat_sequence(tmp_path):
    # A composed sequence has no folder header, so convert_phase_folder cannot
    # take it; save_phase_folder writes it with explicit metadata.
    a = _bin_folder(tmp_path / "a", [0.0, 1.0])
    b = _bin_folder(tmp_path / "b", [10.0, 11.0])
    combined = ConcatSequence(a, b)
    out = tmp_path / "out"
    save_phase_folder(out, combined, ext="bin", pixel_size=1e-6, height_scale=2e-7)
    reopened = PhaseBinFolder(out)
    assert len(reopened) == 4
    for i, value in enumerate((0.0, 1.0, 10.0, 11.0)):
        np.testing.assert_allclose(reopened[i], np.full((2, 3), value), rtol=1e-5)


@pytest.mark.parametrize("ext", ("bin", "txt", "npy"))
def test_save_phase_folder_writes_numbered_files(tmp_path, ext):
    images = [np.full((2, 2), float(i), np.float32) for i in range(3)]
    out = tmp_path / "out"
    if ext == "npy":
        save_phase_folder(out, images, ext=ext)
    else:
        save_phase_folder(out, images, ext=ext, pixel_size=1e-6, height_scale=2e-7)
    assert sorted(p.name for p in out.iterdir()) == [
        f"{i:05d}_phase.{ext}" for i in range(3)
    ]


def test_save_phase_folder_accepts_plain_list_via_load_phase(tmp_path):
    images = [np.full((2, 2), float(i), np.float32) for i in range(2)]
    out = tmp_path / "out"
    save_phase_folder(out, images, ext="bin", pixel_size=1e-6, height_scale=2e-7)
    folder = PhaseBinFolder(out)
    np.testing.assert_allclose(folder[1], np.full((2, 2), 1.0), rtol=1e-5)


def test_save_phase_folder_requires_pixel_size(tmp_path):
    images = [np.zeros((2, 2), np.float32)]
    with pytest.raises(ValueError, match="pixel_size is required"):
        save_phase_folder(tmp_path / "out", images, ext="bin")


def test_save_phase_folder_requires_a_scale_form(tmp_path):
    images = [np.zeros((2, 2), np.float32)]
    with pytest.raises(ValueError, match="height_scale, or wavelength"):
        save_phase_folder(tmp_path / "out", images, ext="bin", pixel_size=1e-6)


def test_save_phase_folder_npy_warns_on_metadata(tmp_path):
    images = [np.zeros((2, 2), np.float32)]
    with pytest.warns(UserWarning, match="header-less"):
        save_phase_folder(tmp_path / "out", images, ext="npy", pixel_size=1e-6)


def test_save_phase_folder_rejects_empty(tmp_path):
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="empty phase sequence"):
        save_phase_folder(out, [], ext="npy")
    assert not out.exists()  # atomic: no unreadable folder left behind


def test_save_phase_folder_stem_override(tmp_path):
    images = [np.zeros((2, 2), np.float32)]
    out = tmp_path / "out"
    save_phase_folder(out, images, ext="npy", stem="custom")
    assert (out / "00000_custom.npy").exists()


def test_save_phase_folder_rejects_unknown_format(tmp_path):
    images = [np.zeros((2, 2), np.float32)]
    with pytest.raises(ValueError, match="ext must be"):
        save_phase_folder(tmp_path / "out", images, ext="raw")
