from __future__ import annotations

import numpy as np
import tifffile
from kaparoo.filesystem.hierarchy import Directory
from kaparoo.filters import Literal

from iivs.dhm.data.intensity.base import IntensityFloatSequence
from iivs.dhm.data.intensity.bin import IntensityBinFolder, save_intensity_bin
from iivs.dhm.data.intensity.layout import (
    INTENSITY_TREE,
    IntensityGroup,
    search_intensity_bin_folders,
    search_intensity_tif_folders,
    search_intensity_txt_folders,
)
from iivs.dhm.data.intensity.tif import IntensityTifFolder
from iivs.dhm.data.intensity.txt import IntensityTxtFolder, save_intensity_txt


def _bin(folder, n):
    folder.mkdir(parents=True)
    for i in range(n):
        save_intensity_bin(
            folder / f"{i:05d}_intensity.bin",
            np.full((2, 3), float(i + 1), np.float32),
            pixel_size=1e-6,
        )


def _txt(folder, n):
    folder.mkdir(parents=True)
    for i in range(n):
        save_intensity_txt(
            folder / f"{i:05d}_intensity.txt",
            np.full((2, 3), float(i + 1), np.float32),
            pixel_size=1e-6,
        )


def _img(folder, n):
    folder.mkdir(parents=True)
    for i in range(n):
        tifffile.imwrite(
            folder / f"{i:05d}_intensity.tif", np.full((2, 3), i, np.uint8)
        )


def test_group_opens_every_format(tmp_path):
    intensity = tmp_path / "Intensity"
    _bin(intensity / "Float" / "Bin", 2)
    _txt(intensity / "Float" / "Txt", 2)
    _img(intensity / "Image", 2)
    group = IntensityGroup(intensity)

    assert isinstance(group.bin_folder, IntensityBinFolder)
    assert isinstance(group.txt_folder, IntensityTxtFolder)
    assert isinstance(group.tif_folder, IntensityTifFolder)
    assert isinstance(group.quantitative, IntensityFloatSequence)
    assert group.quantitative is group.bin_folder  # .bin preferred
    assert group.frame_counts == {"bin": 2, "txt": 2, "tif": 2}
    assert group.root == intensity


def test_group_quantitative_falls_back_to_txt(tmp_path):
    intensity = tmp_path / "Intensity"
    _txt(intensity / "Float" / "Txt", 2)  # no Float/Bin
    group = IntensityGroup(intensity)
    assert group.bin_folder is None
    assert isinstance(group.quantitative, IntensityTxtFolder)
    assert group.frame_counts == {"txt": 2}


def test_group_absent_is_all_none(tmp_path):
    group = IntensityGroup(tmp_path / "Intensity")  # the folder does not exist
    assert group.bin_folder is None
    assert group.txt_folder is None
    assert group.tif_folder is None
    assert group.quantitative is None
    assert group.frame_counts == {}


def test_group_repr(tmp_path):
    group = IntensityGroup(tmp_path / "Intensity")
    assert repr(group) == f"IntensityGroup({str(tmp_path / 'Intensity')!r})"


def test_intensity_tree_models_the_intensity_folder():
    assert isinstance(INTENSITY_TREE, Directory)
    assert INTENSITY_TREE.name.matches("Intensity")


def _intensity_timelapse(root, *, bins=True, txts=True, tifs=True, n=2):
    """Build an `Intensity/` folder under `root` with the requested formats."""
    intensity = root / "Intensity"
    if bins:
        _bin(intensity / "Float" / "Bin", n)
    if txts:
        _txt(intensity / "Float" / "Txt", n)
    if tifs:
        _img(intensity / "Image", n)


def _timelapse_name(folder):
    # folder.root is `<timelapse>/Intensity/Float/Bin`, so climb three parents.
    return folder.root.parent.parent.parent.name


def test_search_intensity_bin_folders(tmp_path):
    _intensity_timelapse(tmp_path / "tlA")
    _intensity_timelapse(tmp_path / "tlB")
    (tmp_path / "notatimelapse").mkdir()  # no Intensity/ folder

    folders = search_intensity_bin_folders(tmp_path)
    assert all(isinstance(f, IntensityBinFolder) for f in folders)
    assert [_timelapse_name(f) for f in folders] == ["tlA", "tlB"]


def test_search_intensity_txt_and_tif_folders(tmp_path):
    _intensity_timelapse(tmp_path / "tl")
    txts = search_intensity_txt_folders(tmp_path)
    tifs = search_intensity_tif_folders(tmp_path)
    assert [type(f).__name__ for f in txts] == ["IntensityTxtFolder"]
    assert [type(f).__name__ for f in tifs] == ["IntensityTifFolder"]


def test_search_intensity_bin_folders_name_filter_and_predicate(tmp_path):
    _intensity_timelapse(tmp_path / "keep", n=3)
    _intensity_timelapse(tmp_path / "other", n=2)

    by_name = search_intensity_bin_folders(tmp_path, name_filter=Literal("keep"))
    assert [_timelapse_name(f) for f in by_name] == ["keep"]

    by_predicate = search_intensity_bin_folders(
        tmp_path, predicate=lambda f: len(f) == 3
    )
    assert [len(f) for f in by_predicate] == [3]
