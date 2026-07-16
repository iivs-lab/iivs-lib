from __future__ import annotations

import numpy as np
import pytest
import tifffile
from kaparoo.filesystem.hierarchy import Directory

from iivs.dhm.data.hologram.layout import (
    HOLOGRAM_TREE,
    open_holograms,
    search_ambiguous_holograms,
    search_holograms,
)
from iivs.dhm.data.hologram.raw import HologramRawFile, save_hologram_raw
from iivs.dhm.data.hologram.tif import HologramTifFolder


def _raw(holo_dir, n):
    holo_dir.mkdir(parents=True)
    save_hologram_raw(
        holo_dir / "holo.raw", np.arange(n * 2 * 3, dtype=np.uint8).reshape(n, 2, 3)
    )


def _tif(holo_dir, n):
    holo_dir.mkdir(parents=True)
    for i in range(n):
        tifffile.imwrite(holo_dir / f"{i:05d}_holo.tif", np.full((2, 3), i, np.uint8))


def test_open_raw_stack(tmp_path):
    holo = tmp_path / "Holograms"
    _raw(holo, 3)
    seq = open_holograms(holo)
    assert isinstance(seq, HologramRawFile)
    assert len(seq) == 3


def test_open_tif_folder(tmp_path):
    holo = tmp_path / "Holograms"
    _tif(holo, 3)
    assert isinstance(open_holograms(holo), HologramTifFolder)


def test_both_raw_and_tif_raises(tmp_path):
    holo = tmp_path / "Holograms"
    _raw(holo, 2)
    for i in range(2):
        tifffile.imwrite(holo / f"{i:05d}_holo.tif", np.full((2, 3), i, np.uint8))
    with pytest.raises(ValueError, match="expected one"):
        open_holograms(holo)


def test_empty_dir_is_none(tmp_path):
    holo = tmp_path / "Holograms"
    holo.mkdir()  # neither .raw nor numbered tifs
    assert open_holograms(holo) is None


def test_absent_dir_is_none(tmp_path):
    assert open_holograms(tmp_path / "nope") is None


def test_hologram_tree_models_the_holograms_folder():
    assert isinstance(HOLOGRAM_TREE, Directory)
    assert HOLOGRAM_TREE.name.matches("Holograms")


def test_search_holograms(tmp_path):
    _raw(tmp_path / "timelapseA" / "Holograms", 2)
    _tif(tmp_path / "timelapseB" / "Holograms", 3)
    (tmp_path / "notatimelapse").mkdir()  # no Holograms/ folder

    found = search_holograms(tmp_path)
    assert isinstance(found, list)
    assert [type(s).__name__ for s in found] == ["HologramRawFile", "HologramTifFolder"]


def test_search_holograms_skips_empty_folder(tmp_path):
    (tmp_path / "empty" / "Holograms").mkdir(
        parents=True
    )  # a Holograms/ with no frames
    _raw(tmp_path / "good" / "Holograms", 2)
    found = search_holograms(tmp_path)
    assert len(found) == 1  # the empty Holograms/ opens to None and is skipped
    assert isinstance(found[0], HologramRawFile)


def test_search_holograms_predicate(tmp_path):
    _raw(tmp_path / "a" / "Holograms", 2)
    _raw(tmp_path / "b" / "Holograms", 3)
    found = search_holograms(tmp_path, predicate=lambda s: len(s) == 3)
    assert [len(s) for s in found] == [3]


def _conflict(holo):
    _raw(holo, 2)  # a .raw stack plus numbered .tif previews -> ambiguous
    for i in range(2):
        tifffile.imwrite(holo / f"{i:05d}_holo.tif", np.full((2, 3), i, np.uint8))


def test_search_holograms_skips_conflict_and_warns(tmp_path):
    _raw(tmp_path / "good" / "Holograms", 2)
    _conflict(tmp_path / "bad" / "Holograms")
    with pytest.warns(UserWarning, match="expected one"):
        found = search_holograms(tmp_path)  # default on_conflict="skip"
    assert [type(s).__name__ for s in found] == ["HologramRawFile"]  # only "good"


def test_search_holograms_on_conflict_raise(tmp_path):
    _raw(tmp_path / "good" / "Holograms", 2)
    _conflict(tmp_path / "bad" / "Holograms")
    with pytest.raises(ValueError, match="expected one"):
        search_holograms(tmp_path, on_conflict="raise")


def test_search_ambiguous_holograms(tmp_path):
    _raw(tmp_path / "raw_only" / "Holograms", 2)  # no conflict
    _tif(tmp_path / "tif_only" / "Holograms", 3)  # no conflict
    _conflict(tmp_path / "bad" / "Holograms")  # both raw and tif
    assert search_ambiguous_holograms(tmp_path) == [tmp_path / "bad" / "Holograms"]


def test_search_holograms_surfaces_a_corrupt_raw(tmp_path):
    holo = tmp_path / "bad" / "Holograms"
    _raw(holo, 2)
    raw = holo / "holo.raw"
    raw.write_bytes(raw.read_bytes()[:-6])  # truncate -> size no longer matches header
    # A corrupt raw is a content error, not the raw+tif ambiguity, so on_conflict="skip"
    # does NOT swallow it.
    with pytest.raises(ValueError, match="file size"):
        search_holograms(tmp_path)
