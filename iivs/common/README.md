# `iivs.common`

Technique-agnostic building blocks — nothing here knows what a hologram, phase,
or intensity image is, so a future technique (or an outside project) can build on
them without importing `dhm`. Reach everything by submodule path
(`iivs.common.data`, `iivs.common.visualization`); the package root deliberately
re-exports nothing.

## `data`

Dtype-generic file sequences, header-less `.npy` I/O, array validation, and
acquisition timing.

- **`ArrayFileList[U]`** — the dtype-generic base every file-backed sequence
  extends. **`FrameShapedMixin`** marks the same-shape sources (adding
  `frame_shape`); **`ValueRangeMixin`** adds a cached `value_range()`, the
  `(min, max)` over all frames or of a single frame.
- **`.npy` I/O**, keyed by dtype (all that varies): `load_float32_npy` /
  `save_float32_npy` and the `uint8` pair, plus `read_npy_shape` (the shape
  without the data) and `write_npy`. Pickle is disabled, so an object array is
  never written and one written elsewhere is refused rather than unpickled.
- **Array validation** — `validate_float32_array` / `validate_uint8_array` (and
  the wider `validate_float_array` / `validate_uint_array`): dtype, shape, and
  non-finite checks, with `OnNonFinite` (`"ignore"` / `"warn"` / `"raise"`) the
  policy every loader and saver threads through. The composable parts
  (`validate_ndim` / `validate_dtype`) stay behind the module path.
- **Timing** — the `Timestamp` record, the abstract `TimestampSequence` interface
  (`mean_interval_ms` / `mean_frame_rate`), and `TimestampsFixedFPS` (synthesized
  from a frame rate). Any time-lapse acquisition has per-frame timing, so a
  concrete reader can implement this interface without `iivs.common` depending
  on it.

## `visualization`

Technique-agnostic display helpers.

- **`auto_rescale`** — ImageJ-style auto-contrast (Enhance Contrast + Normalize):
  clips `saturated`% of pixels via NaN-safe percentile bounds, stretches onto
  `out_range` (or the dtype's full span when `None`), and casts back to the input
  dtype. Works on any numeric image or stack.
