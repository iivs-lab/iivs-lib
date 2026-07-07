from __future__ import annotations

__all__ = ("PhaseBounds", "read_phbounds", "write_phbounds")

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from kaparoo.filesystem import StagedFile, ensure_file_exists, ensure_file_extension

if TYPE_CHECKING:
    from typing import Self

    from kaparoo.filesystem.types import StrPath
    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PhaseBounds:
    """The phase display bounds in a Lyncée Tec Koala ``phbounds.txt``, in nanometers.

    Lyncée Tec Koala renders the quantitative float phase into the uint8 `Image/*.tif`
    previews by linearly mapping ``[min_nm, max_nm]`` onto ``0-255``; these are
    the global min and max of that phase over the acquisition.
    A `PhaseFloatSequence` can recompute them straight from its `Float` source
    via `bounds_nm`, so the previews are never the authoritative source.

    Attributes:
        min_nm: Lower display bound, in nanometers.
        max_nm: Upper display bound, in nanometers.
    """

    UNIT_TAG: ClassVar[str] = "[nm]"

    min_nm: float
    max_nm: float

    def __post_init__(self) -> None:
        """Validate that the bounds are ordered."""
        if self.min_nm > self.max_nm:
            msg = f"min_nm must not exceed max_nm (got {self.min_nm} > {self.max_nm})"
            raise ValueError(msg)

    @classmethod
    def from_file(cls, path: StrPath) -> Self:
        """Read a Lyncée Tec Koala ``phbounds.txt`` into a `PhaseBounds`.

        The file is a ``[nm]`` unit-tag line followed by a ``min max`` line (e.g.
        ``-403.4911 635.9849``).

        Raises:
            FileNotFoundError: If `path` does not exist.
            NotAFileError: If `path` exists but is not a regular file.
            ValueError: If `path` does not have a `.txt` extension, the file is
                not a `[nm]` tag plus a numeric `min max` line, or the bounds
                are not ordered.
        """
        path = ensure_file_exists(path, ext="txt")
        lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]

        if len(lines) != 2:
            msg = f"{path}: expected 2 non-blank lines (got {len(lines)})"
            raise ValueError(msg)

        if lines[0] != cls.UNIT_TAG:
            msg = f"{path}: first line must be {cls.UNIT_TAG!r} (got {lines[0]!r})"
            raise ValueError(msg)

        parts = lines[1].split()
        if len(parts) != 2:
            msg = f"{path}: bounds line must be 'min max' (got {lines[1]!r})"
            raise ValueError(msg)

        return cls(min_nm=float(parts[0]), max_nm=float(parts[1]))

    def to_file(self, path: StrPath, *, overwrite: bool = False) -> None:
        """Write to a Lyncée Tec Koala ``phbounds.txt`` (a `[nm]` tag then `min max`).

        Written atomically: staged to a temp file and moved into place on
        success. `np.save`-style: a path with no suffix gets ``.txt`` appended.

        Raises:
            ValueError: If `path` has a non-`.txt` extension.
            FileExistsError: If `path` exists and `overwrite` is False.
            FileNotFoundError: If the parent directory of `path` does not exist.
        """
        path = ensure_file_extension(path, "txt", add=True)
        content = f"{self.UNIT_TAG}\n{self.min_nm} {self.max_nm}\n"
        with StagedFile(path, overwrite=overwrite, encoding="utf-8") as staged:
            staged.write(content)

    def decode_preview(self, preview: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Map a uint8 Koala preview back toward phase, in nanometers (lossy).

        The inverse of Koala's display rendering: ``0`` maps to `min_nm`, ``255``
        to `max_nm`, linearly. The result is 8-bit quantized (a coarse
        reconstruction with step ``(max_nm - min_nm) / 255``), never a substitute
        for the quantitative `Float` source. A degenerate ``min_nm == max_nm``
        maps every pixel to that single value.

        Args:
            preview: A uint8 preview image (or stack), values in ``0-255``.
        """
        step = np.float32((self.max_nm - self.min_nm) / 255.0)
        return np.asarray(preview, dtype=np.float32) * step + np.float32(self.min_nm)

    def encode_preview(self, phase_nm: NDArray[np.floating]) -> NDArray[np.uint8]:
        """Render phase (nm) into a uint8 Koala-style preview (the forward map).

        Linearly maps ``[min_nm, max_nm]`` onto ``0-255`` with rounding, clamping
        out-of-range values to the ends, as Koala renders `Image/*.tif`. The
        round trip ``decode_preview(encode_preview(x))`` recovers `x` only up to
        the 8-bit quantization. A degenerate ``min_nm == max_nm`` maps everything
        to ``0`` (division by a zero span is avoided).

        Args:
            phase_nm: Phase image(s) in nanometers (e.g. a `Float` frame put in
                nm via `convert_phase_unit`).
        """
        span = self.max_nm - self.min_nm
        values = np.asarray(phase_nm, dtype=np.float64)
        if span == 0:
            normalized = np.zeros_like(values)
        else:
            normalized = np.clip((values - self.min_nm) / span, 0.0, 1.0)
        return np.round(normalized * 255.0).astype(np.uint8)


def read_phbounds(path: StrPath) -> PhaseBounds:
    """Read a Lyncée Tec Koala ``phbounds.txt`` into a `PhaseBounds`.

    Raises:
        As `PhaseBounds.from_file` (wrong extension, missing file, or a malformed
        or unordered record).
    """
    return PhaseBounds.from_file(path)


def write_phbounds(
    path: StrPath, bounds: PhaseBounds, *, overwrite: bool = False
) -> None:
    """Write `bounds` to a Lyncée Tec Koala ``phbounds.txt``.

    Args:
        path: The `.txt` file to write (``.txt`` is appended if `path` has no suffix).
        bounds: The display bounds to record.
        overwrite: Whether to replace `path` if it already exists. Defaults to False.

    Raises:
        As `PhaseBounds.to_file` (wrong extension, an existing target without
        `overwrite`, or a missing parent directory).
    """
    bounds.to_file(path, overwrite=overwrite)
