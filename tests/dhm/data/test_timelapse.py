from __future__ import annotations

import shutil

import numpy as np
import pytest
import tifffile
from kaparoo.filesystem import DirectoryNotFoundError
from kaparoo.filesystem.hierarchy import Directory
from kaparoo.filters import Literal

from iivs.dhm.data.hologram.raw import HologramRawFile, save_hologram_raw
from iivs.dhm.data.hologram.tif import HologramTifFolder
from iivs.dhm.data.intensity.bin import save_intensity_bin
from iivs.dhm.data.intensity.layout import IntensityGroup
from iivs.dhm.data.intensity.txt import save_intensity_txt
from iivs.dhm.data.phase.bin import PhaseBinFolder, save_phase_bin
from iivs.dhm.data.phase.bounds import PhaseBounds, write_phbounds
from iivs.dhm.data.phase.layout import PhaseGroup
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


def _drop_frames(root, modality, indices):
    """Delete each frame in `indices` from every format of `modality`."""
    name = modality.lower()
    for i in indices:
        (root / modality / "Float" / "Bin" / f"{i:05d}_{name}.bin").unlink()
        (root / modality / "Float" / "Txt" / f"{i:05d}_{name}.txt").unlink()
        (root / modality / "Image" / f"{i:05d}_{name}.tif").unlink()


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
    assert isinstance(timelapse.phase.bin_folder, PhaseBinFolder)  # group access

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

    # Status / count queries stay tolerant: they never raise on the raw+tif conflict,
    # leaving the holograms uncounted rather than propagating the ambiguity error.
    assert timelapse.has_holograms
    assert timelapse.num_holograms is None  # ambiguous -> uncountable
    assert timelapse.num_frames == 2  # falls through to the timing, holograms uncounted
    assert timelapse.is_consistent


# ============================== #
#           consistency          #
# ============================== #


def test_num_frames_and_consistency(tmp_path):
    n = _build(tmp_path, holograms="raw")
    timelapse = KoalaTimelapse(tmp_path)
    assert timelapse.num_frames == n
    assert timelapse.is_consistent
    assert timelapse.has_quantitative_phase
    assert timelapse.has_quantitative_intensity
    assert timelapse.is_reconstructable  # holograms + timestamps, counts match
    assert timelapse.has_holograms


def test_source_counts(tmp_path):
    n = _build(tmp_path, holograms="raw")
    timelapse = KoalaTimelapse(tmp_path)
    assert timelapse.num_holograms == n
    assert timelapse.num_timestamps == n
    assert timelapse.num_frames == n


def test_num_frames_follows_the_source_priority(tmp_path):
    _build(tmp_path, holograms="raw", n=5)  # give each source a distinct count
    _write_timestamps(tmp_path, 4)
    _drop_frames(tmp_path, "Phase", (3, 4))
    _drop_frames(tmp_path, "Intensity", (2, 3, 4))
    timelapse = KoalaTimelapse(tmp_path)
    assert (timelapse.num_holograms, timelapse.num_timestamps) == (5, 4)
    assert (timelapse.phase.num_frames, timelapse.intensity.num_frames) == (3, 2)
    assert timelapse.num_frames == 5  # the holograms outrank every other source
    del timelapse  # `holo.raw` stays memory-mapped while a time-lapse holds it

    # Peeling off the top source each time walks the rest of the priority order.
    shutil.rmtree(tmp_path / "Holograms")
    assert KoalaTimelapse(tmp_path).num_frames == 4  # then the timing beside them
    (tmp_path / "timestamps.txt").unlink()
    assert KoalaTimelapse(tmp_path).num_frames == 3  # only then the reconstructions
    shutil.rmtree(tmp_path / "Phase")
    assert KoalaTimelapse(tmp_path).num_frames == 2  # intensity is the last resort
    shutil.rmtree(tmp_path / "Intensity")
    assert KoalaTimelapse(tmp_path).num_frames is None


def test_num_holograms_surfaces_a_corrupt_raw(tmp_path):
    _build(tmp_path, holograms="raw")
    raw = tmp_path / "Holograms" / "holo.raw"
    raw.write_bytes(raw.read_bytes()[:-6])  # truncate -> byte count mismatch
    # A corrupt raw is not the raw+tif ambiguity, so it surfaces (not silently None).
    with pytest.raises(ValueError, match="file size"):
        _ = KoalaTimelapse(tmp_path).num_holograms


def test_inconsistent_group_fails_timelapse_consistency(tmp_path):
    n = _build(tmp_path, n=3)
    # Drop one phase `.bin` frame so phase's bin (2) disagrees with its txt / tif (3).
    (tmp_path / "Phase" / "Float" / "Bin" / f"{n - 1:05d}_phase.bin").unlink()
    timelapse = KoalaTimelapse(tmp_path)
    assert not timelapse.phase.is_consistent
    assert not timelapse.is_consistent  # a single inconsistent group fails the whole


def test_inconsistent_is_detected(tmp_path):
    _build(tmp_path, n=3)
    _write_timestamps(tmp_path, 2)  # one fewer timing row than the frames
    timelapse = KoalaTimelapse(tmp_path)
    assert timelapse.num_frames == 3  # from the holograms, not the short timing
    assert len(timelapse.timestamps) == 2
    assert not timelapse.is_consistent


def test_holograms_only_has_no_reconstruction(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    _write_holograms(tmp_path, "raw", 4)
    timelapse = KoalaTimelapse(tmp_path)
    assert timelapse.phase.quantitative is None
    assert timelapse.intensity.quantitative is None
    assert not timelapse.has_quantitative_phase
    assert not timelapse.has_quantitative_intensity
    assert not timelapse.is_reconstructable  # holograms present but no timestamps.txt
    assert timelapse.has_holograms
    assert timelapse.num_frames == 4
    assert timelapse.is_consistent


def test_not_reconstructable_on_count_mismatch(tmp_path):
    _build(
        tmp_path, holograms="raw", n=3
    )  # holograms 3, timestamps 3 -> reconstructable
    assert KoalaTimelapse(tmp_path).is_reconstructable
    _write_timestamps(tmp_path, 2)  # now only 2 timing rows
    assert not KoalaTimelapse(tmp_path).is_reconstructable  # holograms 3 vs timing 2


def test_not_reconstructable_on_hologram_conflict(tmp_path):
    _build(tmp_path, holograms="raw")
    _u8_tif(tmp_path / "Holograms" / "00000_holo.tif", 0)  # raw+tif -> uncountable
    assert not KoalaTimelapse(tmp_path).is_reconstructable


# ============================== #
#            timestamps          #
# ============================== #


def test_timestamps_read_from_file(tmp_path):
    n = _build(tmp_path, timestamps=True)
    ts = KoalaTimelapse(tmp_path).timestamps
    assert isinstance(ts, TimestampsTxtFile)
    assert len(ts) == n


def test_timestamps_none_without_file(tmp_path):
    _build(tmp_path, timestamps=False)
    assert KoalaTimelapse(tmp_path).timestamps is None


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
    assert timelapse.num_holograms is None
    assert timelapse.num_timestamps is None
    assert timelapse.num_frames is None
    assert timelapse.is_consistent
    assert not timelapse.has_quantitative_phase
    assert not timelapse.has_quantitative_intensity
    assert not timelapse.is_reconstructable
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


# ============================== #
#       content validation       #
# ============================== #


def test_validate_passes_on_wellformed(tmp_path):
    _build(tmp_path, holograms="raw")
    timelapse = KoalaTimelapse(tmp_path)
    timelapse.validate()  # each folder to its own default depth
    timelapse.validate(level="names")
    timelapse.validate(level="data")  # also parses timestamps.txt / phbounds.txt


def test_validate_raises_on_a_bad_modality_file(tmp_path):
    _build(tmp_path, holograms="raw")
    (
        tmp_path / "Phase" / "Float" / "Bin" / "00000_phase.bin"
    ).unlink()  # gap: 00001 only
    with pytest.raises(ValueError, match="non-contiguous"):
        KoalaTimelapse(tmp_path).validate()


def test_validate_checks_a_tif_hologram_folder(tmp_path):
    _build(tmp_path, holograms="tif")
    (tmp_path / "Holograms" / "00000_holo.tif").unlink()  # gap: 00001 only
    with pytest.raises(ValueError, match="non-contiguous"):
        KoalaTimelapse(tmp_path).validate(level="names")


def test_validate_parses_aux_files_only_at_data_level(tmp_path):
    _build(tmp_path, holograms="raw")
    (tmp_path / "timestamps.txt").write_text("garbage, not a timestamp row\n")
    timelapse = KoalaTimelapse(tmp_path)
    timelapse.validate()  # timestamps.txt not read at the default level
    timelapse.validate(level="names")  # nor at names
    with pytest.raises(ValueError, match="malformed"):
        timelapse.validate(level="data")  # at data the bad timestamps.txt is parsed


def test_validate_raises_on_raw_and_tif_conflict(tmp_path):
    _build(tmp_path, holograms="raw")
    _u8_tif(tmp_path / "Holograms" / "00000_holo.tif", 0)  # add tif beside the .raw
    with pytest.raises(ValueError, match="expected one"):
        KoalaTimelapse(tmp_path).validate()


# ============================== #
#          layout spec           #
# ============================== #


def test_spec_is_a_directory():
    assert isinstance(KOALA_TIMELAPSE_TREE, Directory)


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
    # default (unbounded depth) finds it; max_depth=1 does not ('group' is not one)
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

    found = search_timelapses(
        tmp_path,
        predicate=lambda t: t.has_quantitative_phase or t.has_quantitative_intensity,
    )
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


def test_search_timelapses_rejects_unknown_require(tmp_path):
    # A bare subfolder name ("Bin") is not a root-level marker, so it is rejected up
    # front rather than silently matching nothing.
    with pytest.raises(ValueError, match=r"unknown require name\(s\) \['Bin'\]"):
        search_timelapses(tmp_path, require=["Phase", "Bin"])


def test_search_timelapses_empty_require_is_any_modality(tmp_path):
    _build(tmp_path / "full")  # has Phase
    (tmp_path / "plain").mkdir()  # no modality folder
    found = search_timelapses(tmp_path, require=[])  # empty => any-modality default
    assert [t.root.name for t in found] == ["full"]


def test_search_timelapses_missing_root_raises(tmp_path):
    with pytest.raises(DirectoryNotFoundError):
        search_timelapses(tmp_path / "nope")
