from __future__ import annotations

import numpy as np
import pytest
import tifffile
from kaparoo.filesystem import DirectoryNotFoundError, hierarchy
from kaparoo.filters import Literal

from iivs.common.data.timestamp import TimestampsFixedFPS
from iivs.dhm.data.hologram.raw import HologramRawFile, save_hologram_raw
from iivs.dhm.data.hologram.tif import HologramTifFolder
from iivs.dhm.data.intensity.bin import save_intensity_bin
from iivs.dhm.data.intensity.group import IntensityGroup
from iivs.dhm.data.intensity.txt import save_intensity_txt
from iivs.dhm.data.phase.bin import PhaseBinFolder, save_phase_bin
from iivs.dhm.data.phase.bounds import PhaseBounds, write_phbounds
from iivs.dhm.data.phase.group import PhaseGroup
from iivs.dhm.data.phase.txt import save_phase_txt
from iivs.dhm.data.timelapse import (
    KOALA_TIMELAPSE_TREE,
    KoalaTimelapse,
    search_timelapses,
)
from iivs.dhm.data.timestamp import TimestampsTxtFile

# ============================== #
#      time-lapse builders       #
# ============================== #


def _u8_tif(path, index):
    tifffile.imwrite(path, np.full((2, 3), index, dtype=np.uint8))


def _write_timestamps(root, n):
    root.joinpath("timestamps.txt").write_text(
        "".join(f"{i:05d} 15:21:47.674 2026.01.15 {i * 50.0}\n" for i in range(n))
    )


def _write_holograms(root, kind, n):
    holo = root / "Holograms"
    holo.mkdir(parents=True)
    if kind == "raw":
        save_hologram_raw(
            holo / "holo.raw", np.arange(n * 2 * 3, dtype=np.uint8).reshape(n, 2, 3)
        )
    else:
        for i in range(n):
            _u8_tif(holo / f"{i:05d}_holo.tif", i)


def _build(root, *, holograms="raw", n=2, timestamps=True):
    """Write a complete synthetic Koala time-lapse under `root`; return `n`."""
    root.mkdir(parents=True, exist_ok=True)

    pbin = root / "Phase" / "Float" / "Bin"
    ptxt = root / "Phase" / "Float" / "Txt"
    pimg = root / "Phase" / "Image"
    ibin = root / "Intensity" / "Float" / "Bin"
    itxt = root / "Intensity" / "Float" / "Txt"
    iimg = root / "Intensity" / "Image"
    for folder in (pbin, ptxt, pimg, ibin, itxt, iimg):
        folder.mkdir(parents=True)

    for i in range(n):
        data = np.full((2, 3), float(i + 1), np.float32)
        save_phase_bin(
            pbin / f"{i:05d}_phase.bin", data, pixel_size=1e-6, height_scale=2e-7
        )
        save_phase_txt(
            ptxt / f"{i:05d}_phase.txt", data, pixel_size=1e-6, height_scale=2e-7
        )
        _u8_tif(pimg / f"{i:05d}_phase.tif", i)
        save_intensity_bin(ibin / f"{i:05d}_intensity.bin", data, pixel_size=1e-6)
        save_intensity_txt(itxt / f"{i:05d}_intensity.txt", data, pixel_size=1e-6)
        _u8_tif(iimg / f"{i:05d}_intensity.tif", i)

    _write_holograms(root, holograms, n)
    if timestamps:
        _write_timestamps(root, n)
    write_phbounds(root / "phbounds.txt", PhaseBounds(min_nm=-100.0, max_nm=300.0))
    return n


# ============================== #
#          composition           #
# ============================== #


def test_composes_modality_groups(tmp_path):
    n = _build(tmp_path, holograms="raw")
    timelapse = KoalaTimelapse(tmp_path)

    assert isinstance(timelapse.phase, PhaseGroup)
    assert isinstance(timelapse.intensity, IntensityGroup)
    assert timelapse.phase.root == tmp_path / "Phase"
    assert isinstance(timelapse.phase.float_bin, PhaseBinFolder)  # group access

    assert isinstance(timelapse.holograms, HologramRawFile)
    assert len(timelapse.holograms) == n
    assert isinstance(timelapse.timestamps, TimestampsTxtFile)
    assert timelapse.phase_bounds == PhaseBounds(min_nm=-100.0, max_nm=300.0)


def test_holograms_tif_variant(tmp_path):
    _build(tmp_path, holograms="tif")
    assert isinstance(KoalaTimelapse(tmp_path).holograms, HologramTifFolder)


def test_holograms_both_raw_and_tif_raises(tmp_path):
    _build(tmp_path, holograms="raw")
    for i in range(2):
        _u8_tif(tmp_path / "Holograms" / f"{i:05d}_holo.tif", i)  # add tifs beside .raw
    timelapse = KoalaTimelapse(tmp_path)

    with pytest.raises(ValueError, match="expected one"):
        _ = timelapse.holograms
    report = timelapse.validate()  # the composed spec flags the same conflict
    assert not report.ok
    assert report.violations


# ============================== #
#           consistency          #
# ============================== #


def test_frame_counts_merge_the_groups(tmp_path):
    n = _build(tmp_path, holograms="raw")
    timelapse = KoalaTimelapse(tmp_path)
    assert timelapse.frame_counts == {
        "phase_float_bin": n,
        "phase_float_txt": n,
        "phase_previews": n,
        "intensity_float_bin": n,
        "intensity_float_txt": n,
        "intensity_previews": n,
        "holograms": n,
        "timestamps": n,
    }
    assert timelapse.counts_agree
    assert timelapse.has_reconstruction
    assert timelapse.has_holograms


def test_counts_disagree_is_detected(tmp_path):
    _build(tmp_path, n=3)
    _write_timestamps(tmp_path, 2)  # one fewer timing row than the frames
    timelapse = KoalaTimelapse(tmp_path)
    assert timelapse.frame_counts["timestamps"] == 2
    assert timelapse.frame_counts["phase_float_bin"] == 3
    assert not timelapse.counts_agree


def test_holograms_only_has_no_reconstruction(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    _write_holograms(tmp_path, "raw", 4)
    timelapse = KoalaTimelapse(tmp_path)
    assert timelapse.phase.quantitative is None
    assert timelapse.intensity.quantitative is None
    assert not timelapse.has_reconstruction
    assert timelapse.has_holograms
    assert timelapse.frame_counts == {"holograms": 4}
    assert timelapse.counts_agree


# ============================== #
#            timestamps          #
# ============================== #


def test_timestamps_fixed_fps_when_no_file(tmp_path):
    n = _build(tmp_path, holograms="raw", timestamps=False)
    ts = KoalaTimelapse(tmp_path, fps=10.0).timestamps
    assert isinstance(ts, TimestampsFixedFPS)
    assert len(ts) == n
    assert ts.mean_frame_rate == pytest.approx(10.0)


def test_timestamps_file_wins_over_fps(tmp_path):
    _build(tmp_path, timestamps=True)
    assert isinstance(KoalaTimelapse(tmp_path, fps=10.0).timestamps, TimestampsTxtFile)


def test_timestamps_none_without_file_or_fps(tmp_path):
    _build(tmp_path, timestamps=False)
    assert KoalaTimelapse(tmp_path).timestamps is None


def test_timestamps_fps_without_frames_is_none(tmp_path):
    (tmp_path / "Phase").mkdir(parents=True)  # a marker dir, but no frame data
    assert KoalaTimelapse(tmp_path, fps=10.0).timestamps is None


# ============================== #
#           tolerance            #
# ============================== #


def test_absent_modalities_are_none(tmp_path):
    timelapse = KoalaTimelapse(tmp_path)  # an empty root
    assert isinstance(timelapse.phase, PhaseGroup)  # the group is always present
    assert timelapse.phase.quantitative is None  # but empty
    assert timelapse.intensity.quantitative is None
    assert timelapse.holograms is None
    assert timelapse.timestamps is None
    assert timelapse.phase_bounds is None
    assert timelapse.frame_counts == {}
    assert timelapse.counts_agree
    assert not timelapse.has_reconstruction
    assert not timelapse.has_holograms


def test_root_and_repr(tmp_path):
    timelapse = KoalaTimelapse(tmp_path)
    assert timelapse.root == tmp_path
    assert repr(timelapse) == f"KoalaTimelapse({str(tmp_path)!r})"


def test_accessors_are_cached(tmp_path):
    _build(tmp_path)
    timelapse = KoalaTimelapse(tmp_path)
    assert timelapse.phase is timelapse.phase
    assert timelapse.holograms is timelapse.holograms
    assert timelapse.frame_counts is timelapse.frame_counts


# ============================== #
#            validate            #
# ============================== #


def test_spec_is_a_directory():
    assert isinstance(KOALA_TIMELAPSE_TREE, hierarchy.Directory)


def test_validate_ok_on_wellformed_timelapse(tmp_path):
    _build(tmp_path, holograms="raw")
    report = KoalaTimelapse(tmp_path).validate()
    assert report.ok
    assert report.matched
    assert not report.violations


def test_validate_is_lenient_on_empty_root(tmp_path):
    report = KoalaTimelapse(tmp_path).validate()
    assert report.ok
    assert not report.missing


# ============================== #
#        search_timelapses       #
# ============================== #


def test_search_timelapses_returns_sorted_list(tmp_path):
    _build(tmp_path / "tlB")
    _build(tmp_path / "tlA")
    (tmp_path / "notimelapse").mkdir()  # no modality folders
    (tmp_path / "loose.txt").write_text("x")  # a file, not a directory

    found = search_timelapses(tmp_path)
    assert isinstance(found, list)
    assert [t.root.name for t in found] == ["tlA", "tlB"]
    assert all(isinstance(t, KoalaTimelapse) for t in found)


def test_search_timelapses_finds_nested_by_depth(tmp_path):
    _build(tmp_path / "group" / "nested")  # a time-lapse at depth 2
    # default (unbounded depth) finds it; max_depth=1 does not ('group' is no time-lapse)
    assert [t.root.name for t in search_timelapses(tmp_path)] == ["nested"]
    assert search_timelapses(tmp_path, max_depth=1) == []


def test_search_timelapses_name_filter_matches_timelapse_folder(tmp_path):
    _build(tmp_path / "keep")
    _build(tmp_path / "drop")
    found = search_timelapses(tmp_path, name_filter=Literal("keep"))
    assert [t.root.name for t in found] == ["keep"]


def test_search_timelapses_predicate_filters_on_koala_timelapse(tmp_path):
    _build(tmp_path / "full")  # has a reconstruction
    holo_only = tmp_path / "holo_only"
    holo_only.mkdir()
    _write_holograms(holo_only, "raw", 2)  # holograms only, no reconstruction

    found = search_timelapses(tmp_path, predicate=lambda t: t.has_reconstruction)
    assert [t.root.name for t in found] == ["full"]


def test_search_timelapses_require_modality(tmp_path):
    _build(tmp_path / "full")  # has Phase
    _write_holograms(tmp_path / "holo_only", "raw", 2)  # only Holograms, no Phase
    found = search_timelapses(tmp_path, require=["Phase"])
    assert [t.root.name for t in found] == ["full"]


def test_search_timelapses_require_file(tmp_path):
    _build(tmp_path / "with_ts", timestamps=True)
    _build(tmp_path / "without_ts", timestamps=False)
    found = search_timelapses(tmp_path, require=["Phase", "timestamps.txt"])
    assert [t.root.name for t in found] == ["with_ts"]


def test_search_timelapses_forwards_fps(tmp_path):
    n = _build(tmp_path / "t", holograms="raw", timestamps=False)
    (timelapse,) = search_timelapses(tmp_path, fps=20.0)
    assert isinstance(timelapse.timestamps, TimestampsFixedFPS)
    assert len(timelapse.timestamps) == n


def test_search_timelapses_missing_root_raises(tmp_path):
    with pytest.raises(DirectoryNotFoundError):
        search_timelapses(tmp_path / "nope")
