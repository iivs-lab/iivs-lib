from __future__ import annotations

__all__ = ("ensure_file_extension", "numbered_name", "with_file_extension")

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kaparoo.filesystem.types import StrPath


def ensure_file_extension(path: StrPath, ext: str) -> Path:
    """Return `path` as a `Path`, requiring a case-insensitive `.<ext>` suffix.

    The explicit `*List` / `*File` sequences validate each given path with this
    at construction (the auto-discovering folders already filtered by extension),
    so a wrong-format file is caught up front rather than on decode.

    Raises:
        ValueError: If `path`'s suffix is not `.<ext>`.
    """
    path = Path(path)
    if path.suffix.lower() != f".{ext.lower()}":
        msg = f"{path.name} must have a .{ext} extension"
        raise ValueError(msg)
    return path


def numbered_name(index: int, *, stem: str, ext: str) -> str:
    """The contiguous Koala filename ``{index:05d}_{stem}.{ext}``.

    The single source of truth for the numbered-folder naming convention, used
    both to discover/validate a `SequentialFileFolder` and to write a converted
    folder.
    """
    return f"{index:05d}_{stem}.{ext}"


def with_file_extension(path: StrPath, ext: str) -> Path:
    """Return `path` with a ``.<ext>`` suffix, appending it if absent.

    `np.save`-style: a path with no suffix gets ``.<ext>`` appended, while a
    path that already has one must match ``.<ext>`` (case-insensitive). Used by
    the `save_*` writers, so ``out/00000_phase`` becomes
    ``out/00000_phase.<ext>`` while a wrong extension fails fast.

    Args:
        path: The destination path, with or without an extension.
        ext: The expected extension, without the leading dot (e.g. "bin").

    Returns:
        The path as a `Path`, guaranteed to end in ``.<ext>``.

    Raises:
        ValueError: If `path` has a suffix other than ``.<ext>``.
    """
    path = Path(path)
    if not path.suffix:
        return path.with_suffix(f".{ext}")
    if path.suffix.lower() != f".{ext.lower()}":
        msg = f"{path.name} must have a .{ext} extension (got {path.suffix})"
        raise ValueError(msg)
    return path
