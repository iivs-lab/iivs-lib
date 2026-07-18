from __future__ import annotations

__all__ = ("TimestampsTxtFile",)

import re
from functools import cached_property
from typing import TYPE_CHECKING, override

from kaparoo.data.sequences.templates import SingleFileSequence
from kaparoo.filesystem import ensure_file_exists

from iivs.common.data.timestamp import Timestamp, TimestampSequence

if TYPE_CHECKING:
    from typing import ClassVar

    from kaparoo.filesystem.types import StrPath


class TimestampsTxtFile(SingleFileSequence[Timestamp, int], TimestampSequence):
    """A `TimestampSequence` read from a Lyncée Tec Koala ``timestamps.txt``.

    Each line is ``<index> <time> <date> <elapsed_ms>`` (space-separated), and the frame
    indices must run contiguously from 0.

    Raises:
        FileNotFoundError: If `path` does not exist.
        NotAFileError: If `path` exists but is not a regular file.
        ValueError: If `path` does not have a `.txt` extension, the file is malformed,
            or its frame indices are not contiguous from 0 (see `parse`).
    """

    LINE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"""
        (?P<index>\d{5})\s+          # 5-digit frame index
        \d{2}:\d{2}:\d{2}\.\d+\s+    # time, HH:MM:SS.fff
        \d{4}\.\d{2}\.\d{2}\s+       # date, YYYY.MM.DD
        (?P<elapsed>\d+(?:\.\d+)?)   # elapsed ms
        """,
        re.VERBOSE,
    )

    def __init__(self, path: StrPath) -> None:
        super().__init__(path)
        self._timestamps = self.parse(self.path)

    @cached_property
    @override
    def mean_interval_ms(self) -> float:
        """The mean interval between consecutive frames, in ms.

        Raises:
            ValueError: If the sequence has fewer than two frames.
        """
        if len(self) < 2:
            msg = f"mean interval requires at least two frames (got {len(self)})"
            raise ValueError(msg)

        return sum((ts.interval_ms for ts in self), 0.0) / (len(self) - 1)

    @classmethod
    def parse(cls, path: StrPath) -> tuple[Timestamp, ...]:
        """Read, validate, and parse a ``timestamps.txt`` into `Timestamp`s.

        Every non-blank line must match ``<5-digit index> <HH:MM:SS.fff> <YYYY.MM.DD>
        <elapsed_ms>``, the frame indices must run contiguously from 0, and the elapsed
        times must be non-decreasing. Blank lines are ignored.

        Raises:
            FileNotFoundError: If `path` does not exist.
            NotAFileError: If `path` exists but is not a regular file.
            ValueError: If `path` does not have a `.txt` extension, a line does not
                match the expected format, the frame indices are not contiguous from 0,
                or an elapsed time decreases.
        """
        path = ensure_file_exists(path, ext="txt")

        elapsed_times_ms: list[float] = []

        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue

            matched = cls.LINE_PATTERN.fullmatch(stripped)
            if matched is None:
                msg = f"line {lineno} is malformed (got {line!r}): {path}"
                raise ValueError(msg)

            index = int(matched["index"])
            expected = len(elapsed_times_ms)  # next contiguous index, from 0
            if index != expected:
                msg = f"index at line {lineno} must be {expected} (got {index}): {path}"
                raise ValueError(msg)

            elapsed = float(matched["elapsed"])
            if elapsed_times_ms and elapsed < elapsed_times_ms[-1]:
                prev = elapsed_times_ms[-1]
                msg = f"elapsed time at line {lineno} must be >= {prev}: {path}"
                raise ValueError(msg)

            elapsed_times_ms.append(elapsed)

        return Timestamp.series_from_elapsed_times(elapsed_times_ms)
