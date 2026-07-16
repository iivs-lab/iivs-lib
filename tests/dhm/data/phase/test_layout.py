from __future__ import annotations

import numpy as np
import pytest
import tifffile
from kaparoo.filesystem.hierarchy import Directory
from kaparoo.filters import Literal

from iivs.dhm.data.phase.base import PhaseFloatSequence
from iivs.dhm.data.phase.bin import PhaseBinFolder, save_phase_bin
from iivs.dhm.data.phase.layout import (
    PHASE_TREE,
    PhaseGroup,
    search_phase_bin_folders,
    search_phase_tif_folders,
    search_phase_txt_folders,
)
from iivs.dhm.data.phase.tif import PhaseTifFolder
from iivs.dhm.data.phase.txt import PhaseTxtFolder, save_phase_txt
from tests.dhm.data.helpers import spy_on_open


def _bin(folder, n):
    folder.mkdir(parents=True)
    for i in range(n):
        save_phase_bin(
            folder / f"{i:05d}_phase.bin",
            np.full((2, 3), float(i + 1), np.float32),
            pixel_size=1e-6,
            height_scale=2e-7,
        )


def _txt(folder, n):
    folder.mkdir(parents=True)
    for i in range(n):
        save_phase_txt(
            folder / f"{i:05d}_phase.txt",
            np.full((2, 3), float(i + 1), np.float32),
            pixel_size=1e-6,
            height_scale=2e-7,
        )


def _img(folder, n):
    folder.mkdir(parents=True)
    for i in range(n):
        tifffile.imwrite(folder / f"{i:05d}_phase.tif", np.full((2, 3), i, np.uint8))


def test_group_opens_every_format(tmp_path):
    phase = tmp_path / "Phase"
    _bin(phase / "Float" / "Bin", 2)
    _txt(phase / "Float" / "Txt", 2)
    _img(phase / "Image", 2)
    group = PhaseGroup(phase)

    assert isinstance(group.bin_folder, PhaseBinFolder)
    assert isinstance(group.txt_folder, PhaseTxtFolder)
    assert isinstance(group.tif_folder, PhaseTifFolder)
    assert isinstance(group.quantitative, PhaseFloatSequence)
    assert group.quantitative is group.bin_folder  # .bin preferred
    assert group.num_frames == 2
    assert group.frame_shape == (2, 3)
    assert group.is_consistent
    assert group.is_usable  # quantitative data present and consistent
    assert group.root == phase


def test_group_quantitative_falls_back_to_txt(tmp_path):
    phase = tmp_path / "Phase"
    _txt(phase / "Float" / "Txt", 2)  # no Float/Bin
    group = PhaseGroup(phase)
    assert group.bin_folder is None
    assert isinstance(group.quantitative, PhaseTxtFolder)
    assert group.num_frames == 2  # from the lone txt source
    assert group.is_consistent
    assert group.is_usable  # a single quantitative format (no bin/tif) still counts


def test_group_inconsistent_counts(tmp_path):
    phase = tmp_path / "Phase"
    _bin(phase / "Float" / "Bin", 3)
    _txt(phase / "Float" / "Txt", 2)  # one fewer -> counts disagree
    group = PhaseGroup(phase)
    assert group.num_frames == 3  # from the .bin reference
    assert not group.is_consistent


def test_group_inconsistent_shapes(tmp_path):
    phase = tmp_path / "Phase"
    _bin(phase / "Float" / "Bin", 2)  # frames are (2, 3)
    txt = phase / "Float" / "Txt"
    txt.mkdir(parents=True)
    for i in range(2):  # same count, but a different frame shape (3, 2)
        save_phase_txt(
            txt / f"{i:05d}_phase.txt",
            np.full((3, 2), float(i + 1), np.float32),
            pixel_size=1e-6,
            height_scale=2e-7,
        )
    group = PhaseGroup(phase)
    assert group.num_frames == 2  # counts agree
    assert not group.is_consistent  # but the shapes differ
    assert not group.is_usable


def test_group_preview_only_is_not_usable(tmp_path):
    phase = tmp_path / "Phase"
    _img(phase / "Image", 2)  # a uint8 preview, but no quantitative data
    group = PhaseGroup(phase)
    assert group.tif_folder is not None
    assert group.quantitative is None
    assert group.is_consistent  # the lone preview trivially agrees with itself
    assert not group.is_usable  # but there is no quantitative data


def test_group_not_usable_when_present_formats_disagree(tmp_path):
    phase = tmp_path / "Phase"
    _bin(phase / "Float" / "Bin", 2)
    _img(phase / "Image", 3)  # quantitative present, but the preview count differs
    group = PhaseGroup(phase)
    assert group.quantitative is not None
    assert not group.is_consistent
    assert not group.is_usable


def test_group_absent_is_all_none(tmp_path):
    group = PhaseGroup(tmp_path / "Phase")  # the folder does not exist
    assert group.bin_folder is None
    assert group.txt_folder is None
    assert group.tif_folder is None
    assert group.quantitative is None
    assert group.num_frames is None
    assert group.frame_shape is None
    assert group.is_consistent  # vacuously, nothing to disagree
    assert not group.is_usable  # but an absent group has no usable data


def test_group_opens_each_format_lazily_and_once(tmp_path, monkeypatch):
    phase = tmp_path / "Phase"
    _bin(phase / "Float" / "Bin", 2)
    _txt(phase / "Float" / "Txt", 2)
    _img(phase / "Image", 2)
    opened = spy_on_open(monkeypatch)

    group = PhaseGroup(phase)
    assert opened == []  # constructing a group touches no disk

    assert isinstance(group.bin_folder, PhaseBinFolder)
    assert opened == [phase / "Float" / "Bin"]  # only the format actually reached

    assert group.bin_folder is group.bin_folder  # cached, so no reopen
    assert opened == [phase / "Float" / "Bin"]

    assert isinstance(group.tif_folder, PhaseTifFolder)
    assert opened == [phase / "Float" / "Bin", phase / "Image"]  # Txt still untouched


def test_group_caches_an_absent_format_too(tmp_path, monkeypatch):
    phase = tmp_path / "Phase"
    _bin(phase / "Float" / "Bin", 2)  # Float/Txt is left absent
    opened = spy_on_open(monkeypatch)
    group = PhaseGroup(phase)

    assert group.txt_folder is None
    assert group.txt_folder is None
    assert opened == [phase / "Float" / "Txt"]  # probed once, not once per access


def test_group_validate_passes_on_good_data(tmp_path):
    phase = tmp_path / "Phase"
    _bin(phase / "Float" / "Bin", 2)
    _txt(phase / "Float" / "Txt", 2)
    _img(phase / "Image", 2)
    group = PhaseGroup(phase)
    group.validate()  # each present format to its own default depth
    group.validate(level="names")  # uniform contiguous-name check
    group.validate(level="data")  # uniform full-decode check


def test_group_validate_raises_on_bad_quantitative(tmp_path):
    phase = tmp_path / "Phase"
    _bin(phase / "Float" / "Bin", 3)
    (phase / "Float" / "Bin" / "00001_phase.bin").unlink()  # gap: 00000, 00002
    group = PhaseGroup(phase)
    with pytest.raises(ValueError, match="non-contiguous"):
        group.validate(level="names")


def test_group_validate_checks_the_preview_too(tmp_path):
    phase = tmp_path / "Phase"
    _img(phase / "Image", 3)  # a preview-only group
    (phase / "Image" / "00001_phase.tif").unlink()  # gap in the tif preview
    group = PhaseGroup(phase)
    with pytest.raises(ValueError, match="non-contiguous"):
        group.validate()  # the tif preview is validated, not just bin/txt


def test_group_validate_headers_skips_the_preview(tmp_path):
    phase = tmp_path / "Phase"
    _bin(phase / "Float" / "Bin", 3)
    _img(phase / "Image", 2)
    (phase / "Image" / "00000_phase.tif").unlink()  # gap: 00001 only -> non-contiguous
    group = PhaseGroup(phase)
    group.validate(level="headers")  # tif has no header level, so it is skipped
    with pytest.raises(ValueError, match="non-contiguous"):
        group.validate(level="names")  # at names the tif preview IS checked


def test_group_validate_headers_checks_the_quantitative(tmp_path):
    phase = tmp_path / "Phase"
    _bin(phase / "Float" / "Bin", 3)
    (phase / "Float" / "Bin" / "00001_phase.bin").unlink()  # gap: 00000, 00002
    group = PhaseGroup(phase)
    with pytest.raises(ValueError, match="non-contiguous"):
        group.validate(level="headers")  # bin supports headers, so it is checked


def test_group_validate_is_noop_when_absent(tmp_path):
    group = PhaseGroup(tmp_path / "Phase")  # nothing present
    group.validate()  # does not raise
    group.validate(level="data")


def test_group_repr(tmp_path):
    group = PhaseGroup(tmp_path / "Phase")
    assert repr(group) == f"PhaseGroup({str(tmp_path / 'Phase')!r})"


def test_phase_tree_models_the_phase_folder():
    assert isinstance(PHASE_TREE, Directory)
    assert PHASE_TREE.name.matches("Phase")


def _phase_timelapse(root, *, bins=True, txts=True, tifs=True, n=2):
    """Build a `Phase/` folder under `root` with the requested formats."""
    phase = root / "Phase"
    if bins:
        _bin(phase / "Float" / "Bin", n)
    if txts:
        _txt(phase / "Float" / "Txt", n)
    if tifs:
        _img(phase / "Image", n)


def _timelapse_name(folder):
    # folder.root is `<timelapse>/Phase/Float/Bin`, so climb three parents.
    return folder.root.parent.parent.parent.name


def test_search_phase_bin_folders(tmp_path):
    _phase_timelapse(tmp_path / "tlA")
    _phase_timelapse(tmp_path / "tlB")
    (tmp_path / "notatimelapse").mkdir()  # no Phase/ folder

    folders = search_phase_bin_folders(tmp_path)
    assert isinstance(folders, list)
    assert all(isinstance(f, PhaseBinFolder) for f in folders)
    assert [_timelapse_name(f) for f in folders] == ["tlA", "tlB"]


def test_search_phase_txt_and_tif_folders(tmp_path):
    _phase_timelapse(tmp_path / "tl")
    txts = search_phase_txt_folders(tmp_path)
    tifs = search_phase_tif_folders(tmp_path)
    assert [type(f).__name__ for f in txts] == ["PhaseTxtFolder"]
    assert [type(f).__name__ for f in tifs] == ["PhaseTifFolder"]


def test_search_phase_bin_folders_name_filter_and_predicate(tmp_path):
    _phase_timelapse(tmp_path / "keep", n=3)
    _phase_timelapse(tmp_path / "other", n=2)

    by_name = search_phase_bin_folders(tmp_path, name_filter=Literal("keep"))
    assert [_timelapse_name(f) for f in by_name] == ["keep"]

    by_predicate = search_phase_bin_folders(tmp_path, predicate=lambda f: len(f) == 3)
    assert [len(f) for f in by_predicate] == [3]


def test_search_phase_bin_folders_skips_empty_or_missing(tmp_path):
    (tmp_path / "empty" / "Phase" / "Float" / "Bin").mkdir(
        parents=True
    )  # exists, empty
    _phase_timelapse(tmp_path / "txt_only", bins=False)  # Phase, but no Float/Bin
    _phase_timelapse(tmp_path / "full")

    folders = search_phase_bin_folders(tmp_path)
    assert [_timelapse_name(f) for f in folders] == ["full"]
