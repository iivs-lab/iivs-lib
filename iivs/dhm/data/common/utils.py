from __future__ import annotations

__all__ = (
    "detect_numbered_format",
    "file_extension",
    "numbered_name",
    "unsupported_extension",
)

from pathlib import Path
from typing import TYPE_CHECKING

from kaparoo.filesystem.search import search_files
from kaparoo.filters import RegexFilter

if TYPE_CHECKING:
    from collections.abc import Sequence

    from kaparoo.filesystem.types import StrPath


def numbered_name(index: int, *, stem: str, ext: str) -> str:
    """The contiguous Koala filename ``{index:05d}_{stem}.{ext}``.

    The single source of truth for the numbered-folder naming convention, used
    both to discover/validate a `SequentialFileFolder` and to write a converted
    folder.
    """
    return f"{index:05d}_{stem}.{ext}"


def file_extension(path: StrPath) -> str:
    """The lower-case extension of `path`, without the leading dot.

    Normalizes for suffix dispatch (`Path.suffix` keeps the dot and case): both
    ``"a/b.BIN"`` and ``"a/b.bin"`` yield ``"bin"``, and an extension-less path
    yields ``""``.
    """
    return Path(path).suffix.casefold().removeprefix(".")


def unsupported_extension(ext: str, *, kind: str, formats: Sequence[str]) -> ValueError:
    """A `ValueError` for `ext` being none of a modality's `formats`.

    Centralizes the dispatch-rejection message the per-modality factories raise,
    e.g. ``unsupported phase extension 'foo' (expected bin, txt, or npy)``.
    """
    *head, last = formats
    expected = f"{', '.join(head)}, or {last}" if head else last
    return ValueError(f"unsupported {kind} extension {ext!r} (expected {expected})")


def detect_numbered_format(
    root: StrPath,
    *,
    stem: str,
    formats: Sequence[str],
    prefer: str | Sequence[str] | None = None,
) -> str:
    """Return which of `formats` the ``{index:05d}_{stem}.<ext>`` files in `root` use.

    Scans `root` at depth 1 for the numbered ``{stem}`` files with `search_files`
    and a `RegexFilter`, then resolves a single format. When more than one format
    is present, `prefer` decides -- mirroring `kaparoo`'s
    `hierarchy.Exclusive(on_conflict=...)`:

    - `None` -- raise (the conflict is an error; the caller must disambiguate).
    - a format, or a priority sequence of formats -- pick the first present
      format in that order (the `"priority"` resolution).

    Args:
        root: The folder to scan.
        stem: The ``<stem>`` in ``{index:05d}_<stem>.<ext>`` (e.g. "phase").
        formats: The candidate extensions, in their natural order.
        prefer: The conflict policy -- `None` to error on multiple formats, or a
            format / priority sequence to pick the first present one.

    Raises:
        FileNotFoundError: If `root` holds no ``{NNNNN}_{stem}.<format>`` files.
        ValueError: If multiple formats are present and `prefer` is `None`, or
            `prefer` is given but selects none of the present formats.
    """
    alternation = "|".join(formats)
    hits = search_files(
        root, name_filter=RegexFilter(rf"\d{{5}}_{stem}\.({alternation})"), max_depth=1
    )
    found = {file_extension(hit) for hit in hits}
    present = [fmt for fmt in formats if fmt in found]

    if not present:
        msg = f"no NNNNN_{stem}.({alternation}) files found in {root}"
        raise FileNotFoundError(msg)
    if len(present) == 1:
        return present[0]

    if prefer is None:
        msg = (
            f"ambiguous: {root} holds multiple {stem} formats ({present}); "
            f"pass prefer to pick one"
        )
        raise ValueError(msg)

    order = [prefer] if isinstance(prefer, str) else list(prefer)
    for fmt in order:
        if fmt in present:
            return fmt
    msg = f"prefer={order} selects none of the present {stem} formats ({present})"
    raise ValueError(msg)
