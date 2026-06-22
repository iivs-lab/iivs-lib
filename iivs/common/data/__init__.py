"""Format-agnostic, technique-agnostic data building blocks.

The shared data-layer primitives that carry no Koala / `dhm` specifics, so a
future technique (`epi`, `rcm`) reuses them without importing `dhm`:

- `npy` — the header-less `.npy` shape reader and writer (`read_npy_shape`,
  `write_npy`).
- `sequence` — the same-shape marker mixin (`FrameShapedMixin`).

The Koala `.bin` / `.txt` codecs and the numbered-folder template stay in
`iivs.dhm.data.common`; more moves here once `epi` / `rcm` exercise the boundary
(see the `common.data` hoist note in `TODO.md`).
"""

from __future__ import annotations

__all__ = ("FrameShapedMixin", "read_npy_shape", "write_npy")

from iivs.common.data.npy import read_npy_shape, write_npy
from iivs.common.data.sequence import FrameShapedMixin
