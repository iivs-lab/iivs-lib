"""Quantitative analysis derived from DHM phase data (OPD, dry mass, ...).

The NumPy engines and one-shot helpers are re-exported here; the Torch twins live in
`iivs.dhm.analysis.pytorch` (install the ``iivs-lib[torch]`` extra) and are *not*
re-exported, so importing this package never requires PyTorch.
"""

__all__ = (
    "DryMassCalculator",
    "OPDConverter",
    "OpticalHeightConverter",
    "OpticalVolumeCalculator",
    "ProjectedAreaCalculator",
    "calc_drymass",
    "calc_drymass_from_phase",
    "calc_optical_volume",
    "calc_optical_volume_from_height",
    "calc_optical_volume_from_opd",
    "calc_projected_area",
    "height_to_opd",
    "height_to_phase",
    "opd_to_height",
    "opd_to_phase",
    "phase_to_height",
    "phase_to_opd",
)

from iivs.dhm.analysis.area import ProjectedAreaCalculator, calc_projected_area
from iivs.dhm.analysis.drymass import (
    DryMassCalculator,
    calc_drymass,
    calc_drymass_from_phase,
)
from iivs.dhm.analysis.height import (
    OpticalHeightConverter,
    height_to_opd,
    height_to_phase,
    opd_to_height,
    phase_to_height,
)
from iivs.dhm.analysis.opd import OPDConverter, opd_to_phase, phase_to_opd
from iivs.dhm.analysis.volume import (
    OpticalVolumeCalculator,
    calc_optical_volume,
    calc_optical_volume_from_height,
    calc_optical_volume_from_opd,
)
