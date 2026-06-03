from __future__ import annotations

__all__ = ("FrameShaped",)

from typing import Protocol, runtime_checkable


@runtime_checkable
class FrameShaped(Protocol):
    """Structural protocol for a sequence exposing a uniform `frame_shape`.

    Lets consumers accept any same-shape source -- a `UniformPhaseSequence`,
    `UniformIntensitySequence`, `UniformHologramSequence`, `HologramRawFile`,
    ... -- structurally, without depending on the modality class hierarchy.
    The abstract ``Uniform*Sequence`` bases declare the same `frame_shape` as
    their *role*; this is the cross-cutting *structural* counterpart, so a
    plain (heterogeneous) sequence without `frame_shape` is correctly excluded.
    """

    @property
    def frame_shape(self) -> tuple[int, int]: ...
