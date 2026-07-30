"""Torch-native OPD / height / volume / dry-mass helpers (``iivs-lib[torch]``).

Tensor-in / tensor-out twins of the `iivs.dhm.analysis` engines that preserve the
input tensor's device and autograd graph. The physical calibration (the scalar
factors) is reused from the NumPy engines, so only the elementwise ops are
torch-native.
"""

try:
    from iivs.dhm.analysis.pytorch.area import ProjectedArea, calc_projected_area
    from iivs.dhm.analysis.pytorch.drymass import (
        DryMass,
        calc_drymass,
        calc_drymass_from_phase,
    )
    from iivs.dhm.analysis.pytorch.height import (
        OpticalHeight,
        height_to_opd,
        height_to_phase,
        opd_to_height,
        phase_to_height,
    )
    from iivs.dhm.analysis.pytorch.opd import (
        OpticalPathDifference,
        opd_to_phase,
        phase_to_opd,
    )
    from iivs.dhm.analysis.pytorch.volume import (
        OpticalVolume,
        calc_optical_volume,
        calc_optical_volume_from_height,
        calc_optical_volume_from_opd,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    if exc.name != "torch":
        raise
    msg = (
        "iivs.dhm.analysis.pytorch requires PyTorch (install the iivs-lib[torch] extra)"
    )
    raise ImportError(msg) from exc

__all__ = (
    "DryMass",
    "OpticalHeight",
    "OpticalPathDifference",
    "OpticalVolume",
    "ProjectedArea",
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
