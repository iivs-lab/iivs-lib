"""Shared, technique-agnostic infrastructure for the `iivs` namespaces.

`common` is "the shared technique": the pieces shared across every `iivs`
namespace (`dhm`, and the planned `epi` / `rcm`) live here. `common.data` holds
the format-agnostic data primitives (currently the `.npy` reader / writer, the
`FrameShapedMixin`, and the technique-agnostic `timestamp` timing types), with
more hoisted from `iivs.dhm.data.common` as `epi` / `rcm` land.
"""
