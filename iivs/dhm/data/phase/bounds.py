from __future__ import annotations

__all__ = ("PhaseBounds", "read_phbounds", "write_phbounds")

from dataclasses import dataclass
from typing import TYPE_CHECKING

from kaparoo.filesystem import StagedFile, ensure_file_exists

if TYPE_CHECKING:
    from kaparoo.filesystem.types import StrPath


_UNIT_TAG = "[nm]"


@dataclass(frozen=True, slots=True)
class PhaseBounds:
    """The phase display bounds Koala records in ``phbounds.txt``, in nanometers.

    Koala renders the quantitative float phase into the uint8 `Image/*.tif`
    previews by linearly mapping ``[min_nm, max_nm]`` onto ``0-255``; these are
    the global min and max of that phase over the acquisition. A
    `PhaseFloatSequence` can recompute them straight from its `Float` source via
    `bounds_nm`, so the previews are never the authoritative source.

    Attributes:
        min_nm: Lower display bound, in nanometers.
        max_nm: Upper display bound, in nanometers.
    """

    min_nm: float
    max_nm: float

    def __post_init__(self) -> None:
        """Validate that the bounds are ordered."""
        if self.min_nm > self.max_nm:
            msg = f"min_nm must not exceed max_nm (got {self.min_nm} > {self.max_nm})"
            raise ValueError(msg)


def read_phbounds(path: StrPath) -> PhaseBounds:
    """Read a Koala ``phbounds.txt`` into a `PhaseBounds`.

    The file is a ``[nm]`` unit-tag line followed by a ``min max`` line (e.g.
    ``-403.4911 635.9849``).

    Raises:
        FileNotFoundError: If `path` does not exist.
        NotAFileError: If `path` exists but is not a regular file.
        ValueError: If the file is not a `[nm]` tag plus a numeric `min max`
            line, or if the bounds are not ordered.
    """
    path = ensure_file_exists(path)
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) != 2:
        msg = f"{path}: expected 2 non-blank lines (got {len(lines)})"
        raise ValueError(msg)
    if lines[0] != _UNIT_TAG:
        msg = f"{path}: first line must be {_UNIT_TAG!r} (got {lines[0]!r})"
        raise ValueError(msg)
    parts = lines[1].split()
    if len(parts) != 2:
        msg = f"{path}: bounds line must be 'min max' (got {lines[1]!r})"
        raise ValueError(msg)
    return PhaseBounds(min_nm=float(parts[0]), max_nm=float(parts[1]))


def write_phbounds(
    path: StrPath, bounds: PhaseBounds, *, overwrite: bool = False
) -> None:
    """Write `bounds` as a Koala ``phbounds.txt`` (a `[nm]` tag then `min max`).

    Written atomically: staged to a temp file and moved into place on success.

    Raises:
        FileExistsError: If `path` exists and `overwrite` is False.
        FileNotFoundError: If the parent directory of `path` does not exist.
    """
    content = f"{_UNIT_TAG}\n{bounds.min_nm} {bounds.max_nm}\n"
    with StagedFile(path, overwrite=overwrite, encoding="utf-8") as staged:
        staged.write(content)
