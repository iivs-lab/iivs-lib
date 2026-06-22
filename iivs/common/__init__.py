"""Shared, technique-agnostic infrastructure for the `iivs` namespaces.

`common` is "the shared technique": every `iivs` namespace (`dhm`, and the
planned `epi` / `rcm`) mirrors a `<ns>.data` / `<ns>.visualization` layout, and
the pieces shared across techniques live here. `common.visualization` renders
image arrays via matplotlib (a core dependency); `common.data` holds the
format-agnostic data primitives (currently the `.npy` reader / writer, the
`FrameShapedMixin`, and the technique-agnostic `timestamp` timing types), with
more hoisted from `iivs.dhm.data.common` as `epi` / `rcm` land.
"""
