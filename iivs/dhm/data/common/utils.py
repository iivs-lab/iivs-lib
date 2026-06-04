from __future__ import annotations

__all__ = ("numbered_name",)


def numbered_name(index: int, *, stem: str, ext: str) -> str:
    """The contiguous Koala filename ``{index:05d}_{stem}.{ext}``.

    The single source of truth for the numbered-folder naming convention, used
    both to discover/validate a `SequentialFileFolder` and to write a converted
    folder.
    """
    return f"{index:05d}_{stem}.{ext}"
