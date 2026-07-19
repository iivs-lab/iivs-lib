"""Torch-native masked region reductions (install the ``iivs-lib[torch]`` extra).

Tensor-in / tensor-out twin of `iivs.common.data.reduction` that preserves the input
tensor's device, dtype, and autograd graph. Only the elementwise ops are torch-native;
the reduction semantics (mask normalization, empty-region fill, single-region squeeze)
mirror the NumPy engine.
"""

try:
    from iivs.common.data.pytorch.reduction import (
        MaskedReduction,
        Mean,
        MomentReduction,
        Norm,
        Std,
        Sum,
        Variance,
        apply_mask,
        region_stack,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    if exc.name != "torch":
        raise
    msg = (
        "iivs.common.data.pytorch requires PyTorch (install the iivs-lib[torch] extra)"
    )
    raise ImportError(msg) from exc

__all__ = (
    "MaskedReduction",
    "Mean",
    "MomentReduction",
    "Norm",
    "Std",
    "Sum",
    "Variance",
    "apply_mask",
    "region_stack",
)
