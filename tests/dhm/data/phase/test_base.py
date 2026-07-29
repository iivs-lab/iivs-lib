from __future__ import annotations

import numpy as np
import pytest

from iivs.common.data import FrameShapedMixin
from iivs.dhm.data.phase.base import (
    PhaseFloatSequence,
    PhaseImageSequence,
    PhaseSequence,
)
from iivs.dhm.data.phase.bin import PhaseBinFolder, PhaseBinList, save_phase_bin
from iivs.dhm.data.phase.unit import PhaseUnit

# rad -> nm through the height scale written by _write: 2e-7 m/rad * 1e9 nm/m
_NM_PER_RAD = 200.0


def _write(root, index, value, shape=(2, 3), unit=PhaseUnit.RADIANS):
    save_phase_bin(
        root / f"{index:05d}_phase.bin",
        np.full(shape, float(value), dtype=np.float32),
        pixel_size=1e-6,
        height_scale=2e-7,
        unit=unit,
    )


def test_phase_sequence_hierarchy():
    # Float and Image both specialize the modality base PhaseSequence.
    assert issubclass(PhaseFloatSequence, PhaseSequence)
    assert issubclass(PhaseImageSequence, PhaseSequence)

    # A same-shape float folder is a PhaseFloatSequence + FrameShapedMixin; an
    # arbitrary file list is a PhaseFloatSequence only.
    assert issubclass(PhaseBinFolder, PhaseFloatSequence)
    assert issubclass(PhaseBinFolder, FrameShapedMixin)
    assert issubclass(PhaseBinFolder, PhaseBinList)  # a folder is a special list
    assert issubclass(PhaseBinList, PhaseFloatSequence)
    assert not issubclass(PhaseBinList, FrameShapedMixin)


# ========================== #
#         with_unit          #
# ========================== #


def test_with_unit_folder_converts_and_leaves_original_alone(tmp_path):
    _write(tmp_path, 0, 1.5)
    _write(tmp_path, 1, -0.5)
    folder = PhaseBinFolder(tmp_path)

    sibling = folder.with_unit(PhaseUnit.NANOMETERS)

    assert type(sibling) is PhaseBinFolder
    assert sibling is not folder
    assert sibling.files == folder.files
    assert sibling.target_unit is PhaseUnit.NANOMETERS
    np.testing.assert_allclose(
        sibling[0], np.full((2, 3), 1.5 * _NM_PER_RAD, dtype=np.float32), rtol=1e-6
    )

    # the original still loads in its own unit
    assert folder.target_unit is PhaseUnit.RADIANS
    np.testing.assert_allclose(folder[0], np.full((2, 3), 1.5, dtype=np.float32))


def test_with_unit_folder_none_resolves_to_stored_unit(tmp_path):
    _write(tmp_path, 0, 1.0)
    folder = PhaseBinFolder(tmp_path, target_unit=PhaseUnit.NANOMETERS)

    sibling = folder.with_unit(None)

    assert sibling.target_unit is PhaseUnit.RADIANS  # the header's stored unit
    np.testing.assert_allclose(sibling[0], np.full((2, 3), 1.0, dtype=np.float32))


def test_with_unit_list_over_the_same_files(tmp_path):
    _write(tmp_path, 0, 2.0)
    _write(tmp_path, 1, 3.0)
    files = sorted(tmp_path.glob("*.bin"))
    seq = PhaseBinList(files)

    sibling = seq.with_unit(PhaseUnit.NANOMETERS)

    assert type(sibling) is PhaseBinList
    assert sibling.files == seq.files
    np.testing.assert_allclose(
        sibling[1], np.full((2, 3), 3.0 * _NM_PER_RAD, dtype=np.float32), rtol=1e-6
    )
    assert seq.target_unit is None  # the original keeps loading stored units


def test_with_unit_value_range_is_fresh_not_stale(tmp_path):
    # The regression the API exists to avoid: a range computed under one unit
    # must never be served for another.
    _write(tmp_path, 0, 1.0)
    _write(tmp_path, 1, 2.0)
    folder = PhaseBinFolder(tmp_path)
    assert folder.value_range() == pytest.approx((1.0, 2.0))

    sibling = folder.with_unit(PhaseUnit.NANOMETERS)

    assert sibling.value_range() == pytest.approx(
        (1.0 * _NM_PER_RAD, 2.0 * _NM_PER_RAD)
    )
    assert folder.value_range() == pytest.approx((1.0, 2.0))


def test_with_unit_folder_skips_validation(tmp_path, monkeypatch):
    _write(tmp_path, 0, 1.0)
    folder = PhaseBinFolder(tmp_path)

    calls = []
    monkeypatch.setattr(
        PhaseBinFolder, "validate", lambda self, **kwargs: calls.append(kwargs)
    )
    folder.with_unit(PhaseUnit.NANOMETERS)

    assert calls == []  # already validated at first construction


def test_with_unit_unreachable_unit_fails_fast(tmp_path):
    with pytest.warns(UserWarning, match="UNKNOWN"):
        _write(tmp_path, 0, 1.0, unit=PhaseUnit.UNKNOWN)
    folder = PhaseBinFolder(tmp_path)  # fine: loads in the stored unit

    with pytest.raises(ValueError, match="cannot convert"):
        folder.with_unit(PhaseUnit.RADIANS)
