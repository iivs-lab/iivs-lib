"""Torch-native OPD / dry-mass helpers (install the ``iivs-lib[torch]`` extra).

Tensor-in / tensor-out twins of `iivs.dhm.analysis.opd` and
`iivs.dhm.analysis.drymass` that preserve the input tensor's device and autograd
graph. The physical calibration (the scalar factors) is reused from the NumPy
engines, so only the elementwise ops are torch-native.
"""

from __future__ import annotations

try:
    from iivs.dhm.analysis.pytorch.drymass import (
        DryMassCalculator,
        calc_drymass,
        calc_drymass_from_phase,
    )
    from iivs.dhm.analysis.pytorch.opd import (
        OPDConverter,
        opd_to_phase,
        phase_to_opd,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    if exc.name != "torch":
        raise
    msg = (
        "iivs.dhm.analysis.pytorch requires PyTorch (install the iivs-lib[torch] extra)"
    )
    raise ImportError(msg) from exc

__all__ = (
    "DryMassCalculator",
    "OPDConverter",
    "calc_drymass",
    "calc_drymass_from_phase",
    "opd_to_phase",
    "phase_to_opd",
)
