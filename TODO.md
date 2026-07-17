# TODO

Actionable work on the data layer and beyond, in rough priority order. Not
formal milestones.

## `iivs.dhm.analysis` — split `MaskedReduction` out of dry mass

Make `DryMass` / `DryMassCalculator` compute only the per-pixel mass-density
map (`opd * scale`) and move masking + reduction (sum / norm / mean) into a
separate `MaskedReduction`, which also accepts label images (scikit-image
style, 0 = background) alongside the current one-hot `(N, H, W)` masks. Two
payoffs beyond label support:

- **Clean `nn.Module` composition.** A per-pixel `forward(self, opd) ->
  opd * scale` drops the keyword-only `mask` / `reduce` and the Python
  validation / `raise`, so `DryMass` becomes a shape-preserving pointwise
  layer (fits `nn.Sequential`, hooks, `jit` / `compile`) rather than a
  reduction head, matching `OpticalPathDifference`'s already-clean form.
  Gradients still flow; the reduction stays differentiable when called
  explicitly.
- **Drops the NumPy/Torch duplication.** The mask shape-validation and
  branching leave both engines for the one shared reduction. This is measured,
  not estimated: `calc_from_opd`'s validation block is currently **16 lines
  identical down to the whitespace** between `drymass.py` and
  `pytorch/drymass.py` — it touches only `.ndim` / `.shape`, which is why it
  reads the same in both, and why it belongs to neither. (If this split slips,
  a `_validate_opd_mask(opd, mask) -> use_mask` helper in `drymass.py` is the
  stopgap: `pytorch/drymass.py` already imports `DryMassCalculator` for
  `drymass_scale`, so it adds no coupling. The split then deletes it.)

Sequence: ① shrink `DryMass` / `DryMassCalculator` to per-pixel; ② add the
shared `MaskedReduction` (NumPy + Torch, label + one-hot).

Whatever `MaskedReduction`'s sum ends up looking like, it must not repeat the
bug below: it is the reduction, so the float64 accumulation becomes its problem.

## `iivs.dhm.analysis` — sum in float64, do not copy into it

`calc_from_opd` upcasts the whole stack before reducing, where `np.sum`'s
`dtype=` would accumulate in float64 without materializing anything:

```python
        if reduce:
            opd = opd.astype(np.float64, copy=False)   # copies the entire stack
            if use_mask:
                result = np.tensordot(opd, mask, axes=([-2, -1], [-2, -1]))
            else:
                result = np.sum(opd, axis=(-2, -1))    # dtype= was enough
```

The docstring promises the sum is "in float64", and `np.sum(..., dtype=np.float64)`
delivers exactly that. Measured on a 94 MB float32 input: **56.8 ms and a 188 MB
temporary** versus **13.5 ms and none**, agreeing to 1e-9 (float64 either way; only
the accumulation order differs). On a 1000-frame 1MP time-lapse (3.2 GB) the copy is
**6.5 GB** — an OOM, not a slowdown.

Move the upcast inside the `use_mask` branch, where it is real (`np.tensordot`
accumulates in the input dtype), and give the unmasked path `dtype=np.float64`.

`pytorch/drymass.py`'s `opd.double()` is the same shape. There the win is not speed —
CPU timings are identical and bit-for-bit equal — but the avoided 2x allocation, which
is what bites on CUDA for a large batch. `out_dtype` is already captured, so the same
restructure is safe.

## Release

`CHANGELOG.md`'s `[Unreleased]` has accumulated substantial public-API changes
since `0.1.0`: the suffix-dispatch factories, the `load_bin` / `load_txt` engine
consolidation, `bounds_nm` replaced by `value_range`, the `imagecodecs` core
dependency, and the whole `KoalaTimelapse` / `ReconstructionGroup` time-lapse
layout (per-modality groups, the `search_*` finders, content `validate`, and the
status flags). The data layer has settled; cut `0.2.0`.

Only two of those changes break a `0.1.0` caller, since `iivs.common` and the
time-lapse layout are both new this cycle: `iivs.dhm.data.constants` moved to
`iivs.dhm.constants`, and `convert_phase_folder` / `convert_intensity_folder`
renamed their first parameter `root` to `dest` (keyword callers only).
