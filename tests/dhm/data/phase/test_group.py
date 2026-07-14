from __future__ import annotations

import numpy as np
import tifffile
from kaparoo.filesystem import hierarchy
from kaparoo.filters import Literal

from iivs.dhm.data.phase.base import PhaseFloatSequence
from iivs.dhm.data.phase.bin import PhaseBinFolder, save_phase_bin
from iivs.dhm.data.phase.group import (
    PHASE_TREE,
    PhaseGroup,
    search_phase_bin_folders,
    search_phase_preview_folders,
    search_phase_txt_folders,
)
from iivs.dhm.data.phase.tif import PhaseTifFolder
from iivs.dhm.data.phase.txt import PhaseTxtFolder, save_phase_txt


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

    assert isinstance(group.float_bin, PhaseBinFolder)
    assert isinstance(group.float_txt, PhaseTxtFolder)
    assert isinstance(group.previews, PhaseTifFolder)
    assert isinstance(group.quantitative, PhaseFloatSequence)
    assert group.quantitative is group.float_bin  # .bin preferred
    assert group.frame_counts == {"float_bin": 2, "float_txt": 2, "previews": 2}
    assert group.root == phase


def test_group_quantitative_falls_back_to_txt(tmp_path):
    phase = tmp_path / "Phase"
    _txt(phase / "Float" / "Txt", 2)  # no Float/Bin
    group = PhaseGroup(phase)
    assert group.float_bin is None
    assert isinstance(group.quantitative, PhaseTxtFolder)
    assert group.frame_counts == {"float_txt": 2}


def test_group_absent_is_all_none(tmp_path):
    group = PhaseGroup(tmp_path / "Phase")  # the folder does not exist
    assert group.float_bin is None
    assert group.float_txt is None
    assert group.previews is None
    assert group.quantitative is None
    assert group.frame_counts == {}


def test_group_repr(tmp_path):
    group = PhaseGroup(tmp_path / "Phase")
    assert repr(group) == f"PhaseGroup({str(tmp_path / 'Phase')!r})"


def test_phase_tree_models_the_phase_folder():
    assert isinstance(PHASE_TREE, hierarchy.Directory)
    assert PHASE_TREE.name.matches("Phase")


def _phase_timelapse(root, *, bins=True, txts=True, previews=True, n=2):
    """Build a `Phase/` folder under `root` with the requested formats."""
    phase = root / "Phase"
    if bins:
        _bin(phase / "Float" / "Bin", n)
    if txts:
        _txt(phase / "Float" / "Txt", n)
    if previews:
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


def test_search_phase_txt_and_preview_folders(tmp_path):
    _phase_timelapse(tmp_path / "tl")
    txts = search_phase_txt_folders(tmp_path)
    previews = search_phase_preview_folders(tmp_path)
    assert [type(f).__name__ for f in txts] == ["PhaseTxtFolder"]
    assert [type(f).__name__ for f in previews] == ["PhaseTifFolder"]


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
