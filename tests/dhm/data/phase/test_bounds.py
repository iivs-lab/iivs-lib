from __future__ import annotations

import numpy as np
import pytest

from iivs.dhm.data.phase.base import PhaseFloatSequence, PhaseImageSequence
from iivs.dhm.data.phase.bin import PhaseBinFolder, PhaseBinList, save_phase_bin
from iivs.dhm.data.phase.bounds import PhaseBounds, read_phbounds, write_phbounds
from iivs.dhm.data.phase.unit import PhaseUnit


def _save(path, data, *, height_scale=2e-7):
    # height_scale 2e-7 m/rad -> 200 nm per radian.
    save_phase_bin(
        path,
        np.asarray(data, dtype=np.float32),
        pixel_size=1e-6,
        height_scale=height_scale,
    )


# ========================== #
#         PhaseBounds        #
# ========================== #


def test_phase_bounds_rejects_unordered():
    with pytest.raises(ValueError, match="must not exceed max_nm"):
        PhaseBounds(min_nm=1.0, max_nm=0.0)


def test_phase_bounds_allows_equal():
    assert PhaseBounds(min_nm=5.0, max_nm=5.0).min_nm == 5.0


# ========================== #
#      read/write phbounds   #
# ========================== #


def test_phbounds_roundtrip(tmp_path):
    path = tmp_path / "phbounds.txt"
    bounds = PhaseBounds(min_nm=-403.4911, max_nm=635.9849)
    write_phbounds(path, bounds)
    assert read_phbounds(path) == bounds


def test_phbounds_from_file_to_file_roundtrip(tmp_path):
    # The free functions wrap these methods; exercise the methods directly.
    path = tmp_path / "phbounds.txt"
    bounds = PhaseBounds(min_nm=-1.0, max_nm=2.0)
    bounds.to_file(path)
    assert PhaseBounds.from_file(path) == bounds
    assert PhaseBounds.UNIT_TAG == "[nm]"


def test_phbounds_file_format(tmp_path):
    path = tmp_path / "phbounds.txt"
    write_phbounds(path, PhaseBounds(min_nm=-403.4911, max_nm=635.9849))
    assert path.read_text() == "[nm]\n-403.4911 635.9849\n"


def test_write_phbounds_no_overwrite(tmp_path):
    path = tmp_path / "phbounds.txt"
    write_phbounds(path, PhaseBounds(0.0, 1.0))
    with pytest.raises(FileExistsError):
        write_phbounds(path, PhaseBounds(0.0, 1.0))
    write_phbounds(path, PhaseBounds(0.0, 2.0), overwrite=True)
    assert read_phbounds(path).max_nm == pytest.approx(2.0)


def test_read_phbounds_rejects_wrong_line_count(tmp_path):
    path = tmp_path / "phbounds.txt"
    path.write_text("[nm]\n")
    with pytest.raises(ValueError, match="2 non-blank lines"):
        read_phbounds(path)


def test_read_phbounds_rejects_wrong_tag(tmp_path):
    path = tmp_path / "phbounds.txt"
    path.write_text("[rad]\n0 1\n")
    with pytest.raises(ValueError, match=r"\[nm\]"):
        read_phbounds(path)


def test_read_phbounds_rejects_non_pair_bounds(tmp_path):
    path = tmp_path / "phbounds.txt"
    path.write_text("[nm]\n1 2 3\n")
    with pytest.raises(ValueError, match="min max"):
        read_phbounds(path)


def test_read_phbounds_rejects_non_numeric(tmp_path):
    path = tmp_path / "phbounds.txt"
    path.write_text("[nm]\nlo hi\n")
    with pytest.raises(ValueError, match="could not convert"):  # float() rejects them
        read_phbounds(path)


def test_read_phbounds_rejects_wrong_extension(tmp_path):
    path = tmp_path / "phbounds.bin"
    path.write_text("[nm]\n0 1\n")
    with pytest.raises(
        ValueError, match=r"unsupported extension .* \(supported: 'txt'\)"
    ):
        read_phbounds(path)


def test_write_phbounds_appends_extension(tmp_path):
    # np.save-style: a suffix-less path gets `.txt` appended.
    write_phbounds(tmp_path / "phbounds", PhaseBounds(0.0, 1.0))
    assert (tmp_path / "phbounds.txt").exists()


def test_write_phbounds_rejects_wrong_extension(tmp_path):
    with pytest.raises(
        ValueError, match=r"unsupported extension .* \(supported: 'txt'\)"
    ):
        write_phbounds(tmp_path / "phbounds.bin", PhaseBounds(0.0, 1.0))


# ========================== #
#   preview <-> nm mapping   #
# ========================== #


def test_decode_preview_endpoints_and_dtype():
    bounds = PhaseBounds(min_nm=-100.0, max_nm=300.0)
    preview = np.array([0, 255], dtype=np.uint8)
    nm = bounds.decode_preview(preview)
    assert nm.dtype == np.float32
    assert nm[0] == pytest.approx(-100.0)
    assert nm[1] == pytest.approx(300.0)


def test_encode_preview_endpoints_and_dtype():
    bounds = PhaseBounds(min_nm=-100.0, max_nm=300.0)
    nm = np.array([-100.0, 300.0], dtype=np.float32)
    preview = bounds.encode_preview(nm)
    assert preview.dtype == np.uint8
    assert preview[0] == 0
    assert preview[1] == 255


def test_encode_preview_clamps_out_of_range():
    bounds = PhaseBounds(min_nm=0.0, max_nm=100.0)
    preview = bounds.encode_preview(np.array([-50.0, 150.0], dtype=np.float32))
    assert preview[0] == 0  # below min -> 0
    assert preview[1] == 255  # above max -> 255


def test_preview_roundtrip_within_quantization():
    bounds = PhaseBounds(min_nm=-403.4911, max_nm=635.9849)
    nm = np.linspace(bounds.min_nm, bounds.max_nm, 256, dtype=np.float32)
    recovered = bounds.decode_preview(bounds.encode_preview(nm))
    step = (bounds.max_nm - bounds.min_nm) / 255.0
    assert np.max(np.abs(recovered - nm)) <= step  # 8-bit quantization only


def test_degenerate_bounds_map_to_a_single_value():
    bounds = PhaseBounds(min_nm=42.0, max_nm=42.0)
    # decode: every pixel collapses to the single value (no division by span)
    nm = bounds.decode_preview(np.array([0, 128, 255], dtype=np.uint8))
    assert np.all(nm == np.float32(42.0))
    # encode: zero span maps everything to 0 rather than dividing by zero
    preview = bounds.encode_preview(np.array([42.0, 42.0], dtype=np.float32))
    assert np.all(preview == 0)


# ============================ #
#  PhaseFileList.value_range   #
# ============================ #


def test_value_range_nm_over_folder(tmp_path):
    _save(tmp_path / "00000_phase.bin", np.full((2, 3), 1.0))  # 200 nm
    _save(tmp_path / "00001_phase.bin", np.full((2, 3), 3.0))  # 600 nm
    rng = PhaseBinFolder(tmp_path).value_range(unit=PhaseUnit.NANOMETERS)
    assert rng == pytest.approx((200.0, 600.0))


def test_value_range_within_a_frame(tmp_path):
    path = tmp_path / "00000_phase.bin"
    _save(path, [[0.0, 3.0]])  # 0 .. 600 nm
    rng = PhaseBinList([path]).value_range(unit=PhaseUnit.NANOMETERS)
    assert rng == pytest.approx((0.0, 600.0))


def test_value_range_converts_each_file_by_its_height_scale(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    _save(a, np.full((1, 2), 1.0), height_scale=2e-7)  # 200 nm
    _save(b, np.full((1, 2), 1.0), height_scale=4e-7)  # 400 nm
    rng = PhaseBinList([a, b]).value_range(unit=PhaseUnit.NANOMETERS)
    assert rng == pytest.approx((200.0, 400.0))


def test_value_range_with_unit_is_independent_of_target_unit(tmp_path):
    path = tmp_path / "00000_phase.bin"
    _save(path, np.full((1, 2), 1.0))
    seq = PhaseBinList([path], target_unit=PhaseUnit.RADIANS)
    _, high = seq.value_range(unit=PhaseUnit.NANOMETERS)
    assert high == pytest.approx(200.0)  # nm, not the 1.0 radian


def test_value_range_default_unit_follows_target_unit(tmp_path):
    # No unit -> the range of the values as loaded (target_unit), like any sequence.
    path = tmp_path / "00000_phase.bin"
    _save(path, np.full((1, 2), 1.0))  # 1.0 rad -> 200 nm
    rad = PhaseBinList([path], target_unit=PhaseUnit.RADIANS)
    nm = PhaseBinList([path], target_unit=PhaseUnit.NANOMETERS)
    assert rad.value_range() == pytest.approx((1.0, 1.0))
    assert nm.value_range() == pytest.approx((200.0, 200.0))


def test_value_range_of_a_single_frame_by_index(tmp_path):
    _save(tmp_path / "00000_phase.bin", np.full((1, 2), 1.0))  # 200 nm
    _save(tmp_path / "00001_phase.bin", np.full((1, 2), 3.0))  # 600 nm
    folder = PhaseBinFolder(tmp_path)
    assert folder.value_range(1, unit=PhaseUnit.NANOMETERS) == pytest.approx(
        (600.0, 600.0)
    )


def test_value_range_empty_sequence_raises():
    with pytest.raises(ValueError, match="empty"):
        _ = PhaseBinList([]).value_range()  # via the shared mixin (target_unit)
    with pytest.raises(ValueError, match="empty"):
        _ = PhaseBinList([]).value_range(unit=PhaseUnit.NANOMETERS)  # phase's unit path


def test_value_range_unit_ignores_non_finite(tmp_path):
    # NaN pixels are dropped from the unit-converted range too.
    save_phase_bin(
        tmp_path / "00000_phase.bin",
        np.array([[1.0, np.nan]], dtype=np.float32),  # 1.0 rad -> 200 nm
        pixel_size=1e-6,
        height_scale=2e-7,
        on_nonfinite="ignore",
    )
    save_phase_bin(
        tmp_path / "00001_phase.bin",
        np.array([[np.nan, 3.0]], dtype=np.float32),  # 3.0 rad -> 600 nm
        pixel_size=1e-6,
        height_scale=2e-7,
        on_nonfinite="ignore",
    )
    folder = PhaseBinFolder(tmp_path)
    assert folder.value_range(unit=PhaseUnit.NANOMETERS) == pytest.approx(
        (200.0, 600.0)
    )
    assert folder.value_range(0, unit=PhaseUnit.NANOMETERS) == pytest.approx(
        (200.0, 200.0)
    )


def test_value_range_unit_all_non_finite_raises(tmp_path):
    save_phase_bin(
        tmp_path / "00000_phase.bin",
        np.full((1, 2), np.nan, dtype=np.float32),
        pixel_size=1e-6,
        height_scale=2e-7,
        on_nonfinite="ignore",
    )
    folder = PhaseBinFolder(tmp_path)
    with pytest.raises(ValueError, match="non-finite"):
        folder.value_range(unit=PhaseUnit.NANOMETERS)  # global-in
    with pytest.raises(ValueError, match="non-finite"):
        folder.value_range(0, unit=PhaseUnit.NANOMETERS)  # per-frame unit path


def test_value_range_rejects_unknown_unit(tmp_path):
    path = tmp_path / "00000_phase.bin"
    with pytest.warns(UserWarning, match="UNKNOWN"):  # save_phase_bin warns
        save_phase_bin(
            path,
            np.zeros((1, 2), dtype=np.float32),
            pixel_size=1e-6,
            height_scale=2e-7,
            unit=PhaseUnit.UNKNOWN,
        )
    with pytest.raises(ValueError, match="cannot convert"):
        _ = PhaseBinList([path]).value_range(unit=PhaseUnit.NANOMETERS)


def test_value_range_caches_the_global_but_not_per_frame(tmp_path):
    _save(tmp_path / "00000_phase.bin", np.full((1, 2), 1.0))
    _save(tmp_path / "00001_phase.bin", np.full((1, 2), 3.0))
    seq = PhaseBinFolder(tmp_path)
    assert seq.value_range() is seq.value_range()  # global range is cached
    unit = PhaseUnit.NANOMETERS
    assert seq.value_range(unit=unit) is seq.value_range(unit=unit)  # cached per unit
    assert seq.value_range(0) is not seq.value_range(0)  # per-frame is recomputed


# ========================== #
#    to_image / to_float   #
# ========================== #


def _two_frame_folder(tmp_path, **kwargs):
    # frame 0 = 1.0 rad (200 nm), frame 1 = 3.0 rad (600 nm) at height_scale 2e-7.
    _save(tmp_path / "00000_phase.bin", np.full((2, 3), 1.0))
    _save(tmp_path / "00001_phase.bin", np.full((2, 3), 3.0))
    return PhaseBinFolder(tmp_path, **kwargs)


def test_to_image_is_image_sequence_with_endpoints(tmp_path):
    folder = _two_frame_folder(tmp_path)
    preview = folder.to_image(PhaseBounds(min_nm=200.0, max_nm=600.0))
    assert isinstance(preview, PhaseImageSequence)
    assert len(preview) == 2
    assert preview[0].dtype == np.uint8
    assert np.all(preview[0] == 0)  # 200 nm -> min -> 0
    assert np.all(preview[1] == 255)  # 600 nm -> max -> 255


def test_to_image_renders_nm_regardless_of_target_unit(tmp_path):
    # target_unit=RADIANS, yet the preview must still map by nm (header-derived).
    folder = _two_frame_folder(tmp_path, target_unit=PhaseUnit.RADIANS)
    preview = folder.to_image(PhaseBounds(min_nm=200.0, max_nm=600.0))
    assert np.all(preview[0] == 0)
    assert np.all(preview[1] == 255)


def test_to_image_default_bounds_use_value_range(tmp_path):
    folder = _two_frame_folder(tmp_path)
    expected = PhaseBounds(*folder.value_range(unit=PhaseUnit.NANOMETERS))
    assert folder.to_image().bounds == expected


def test_to_image_reads_up_front_only_to_derive_bounds(tmp_path, monkeypatch):
    folder = _two_frame_folder(tmp_path)
    decoded = []
    # Count the decode both paths go through: the up-front `value_range` scan and the
    # view's render. Nothing public distinguishes "read every file" from "read none".
    real = type(folder)._decode_in  # noqa: SLF001
    monkeypatch.setattr(
        type(folder),
        "_decode_in",
        lambda self, index, unit: (decoded.append(index), real(self, index, unit))[1],
    )

    view = folder.to_image(PhaseBounds(min_nm=200.0, max_nm=600.0))
    assert decoded == []  # given bounds, the up-front pass is skipped...
    _ = view[1]
    assert decoded == [1]  # ...and the view renders on access, that frame alone

    decoded.clear()
    view = folder.to_image()
    assert decoded == list(range(len(folder)))  # bounds=None reads every file, once


def test_to_image_meta_is_source_path(tmp_path):
    folder = _two_frame_folder(tmp_path)
    preview = folder.to_image()
    assert preview.get_meta(0) == folder.get_meta(0)
    assert preview.source is folder


def test_to_float_roundtrip_within_quantization(tmp_path):
    folder = _two_frame_folder(tmp_path, target_unit=PhaseUnit.NANOMETERS)
    bounds = PhaseBounds(*folder.value_range(unit=PhaseUnit.NANOMETERS))
    # Float -> Image -> Float, all in memory (no .tif I/O needed).
    recon = folder.to_image(bounds).to_float(bounds)
    assert isinstance(recon, PhaseFloatSequence)
    assert recon[0].dtype == np.float32
    step = (bounds.max_nm - bounds.min_nm) / 255.0
    for index in range(len(folder)):
        assert np.max(np.abs(recon[index] - folder[index])) <= step


def test_to_float_to_radians_uses_height_scale(tmp_path):
    # frames 1.0 / 3.0 rad at height_scale 2e-7 -> 200 / 600 nm; bounds (200, 600).
    folder = _two_frame_folder(tmp_path)  # default target_unit -> RADIANS
    bounds = PhaseBounds(*folder.value_range(unit=PhaseUnit.NANOMETERS))
    recon = folder.to_image(bounds).to_float(
        bounds, target_unit=PhaseUnit.RADIANS, height_scale=2e-7
    )
    step_rad = (bounds.max_nm - bounds.min_nm) / 255.0 * 1e-9 / 2e-7
    for index in range(len(folder)):
        assert np.max(np.abs(recon[index] - folder[index])) <= step_rad


def test_to_float_to_meters_needs_no_scale(tmp_path):
    folder = _two_frame_folder(tmp_path)
    bounds = PhaseBounds(min_nm=200.0, max_nm=600.0)
    recon = folder.to_image(bounds).to_float(bounds, target_unit=PhaseUnit.METERS)
    assert recon[0].dtype == np.float32
    assert recon[0][0, 0] == pytest.approx(200e-9)  # 200 nm -> 2e-7 m
    assert recon[1][0, 0] == pytest.approx(600e-9)


def test_to_float_to_radians_requires_scale(tmp_path):
    folder = _two_frame_folder(tmp_path)
    preview = folder.to_image(PhaseBounds(min_nm=0.0, max_nm=600.0))
    with pytest.raises(ValueError, match="exactly one"):
        preview.to_float(
            PhaseBounds(min_nm=0.0, max_nm=600.0), target_unit=PhaseUnit.RADIANS
        )


def test_to_float_meta_and_source_passthrough(tmp_path):
    folder = _two_frame_folder(tmp_path)
    preview = folder.to_image()
    bounds = PhaseBounds(min_nm=0.0, max_nm=1.0)
    recon = preview.to_float(bounds)
    assert len(recon) == 2
    assert recon.get_meta(1) == folder.get_meta(1)
    assert recon.source is preview
    assert recon.bounds is bounds
