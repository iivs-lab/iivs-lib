"""Shared, technique-agnostic infrastructure for the `iivs` namespaces.

`common` is "the shared technique": every `iivs` namespace (`dhm`, and the
planned `epi` / `rcm`) mirrors a `<ns>.data` / `<ns>.visualization` layout, and
the pieces shared across techniques live here. `common.visualization` (this
package's first occupant) renders image arrays via matplotlib (a core
dependency); the format-agnostic data layer (`common.data`) is hoisted here
once a second technique lands.
"""
