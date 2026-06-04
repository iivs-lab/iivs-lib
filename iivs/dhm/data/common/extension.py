from __future__ import annotations

__all__ = ("ensure_file_extension", "numbered_name")

# `ensure_file_extension` (incl. its `add=True` suffix-appending mode, the former
# `with_file_extension`) lives in `kaparoo.filesystem`; re-exported here so the
# modality modules keep reaching it through `iivs.dhm.data.common`.
from kaparoo.filesystem import ensure_file_extension


def numbered_name(index: int, *, stem: str, ext: str) -> str:
    """The contiguous Koala filename ``{index:05d}_{stem}.{ext}``.

    The single source of truth for the numbered-folder naming convention, used
    both to discover/validate a `SequentialFileFolder` and to write a converted
    folder.
    """
    return f"{index:05d}_{stem}.{ext}"
