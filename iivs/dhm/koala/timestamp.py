from __future__ import annotations

__all__ = (
    "Timestamp",
    "TimestampFpsSequence",
    "TimestampSequence",
    "TimestampTxtSequence",
)

import itertools
import re
from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

from kaparoo.data.sequences.base import DataSequence
from kaparoo.data.sequences.templates import SingleFileSequence
from kaparoo.filesystem import ensure_file_exists

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import ClassVar

    from kaparoo.filesystem.types import StrPath


@dataclass(frozen=True, slots=True)
class Timestamp:
    """Acquisition timing of one frame, in milliseconds.

    Attributes:
        elapsed_ms: Time since acquisition start.
        interval_ms: Time since the previous frame (0.0 for the first).
    """

    elapsed_ms: float
    interval_ms: float

    def __post_init__(self) -> None:
        """Validate that the timings are non-negative and consistent."""
        if self.elapsed_ms < 0.0:
            msg = f"elapsed_ms must be non-negative (got {self.elapsed_ms})"
            raise ValueError(msg)

        if self.interval_ms < 0.0:
            msg = f"interval_ms must be non-negative (got {self.interval_ms})"
            raise ValueError(msg)

        if self.interval_ms > self.elapsed_ms:
            msg = f"interval_ms must not exceed elapsed_ms (got {self.interval_ms} > {self.elapsed_ms})"
            raise ValueError(msg)

    @classmethod
    def series_from_elapsed_times(
        cls, elapsed_times_ms: Sequence[float]
    ) -> tuple[Timestamp, ...]:
        """Build a series of `Timestamp`s from cumulative elapsed times.

        The first frame's interval is 0.0; each later frame's interval is its
        gap from the previous frame.
        """
        if not elapsed_times_ms:
            return ()

        # Pad the front so the first frame's interval comes out as t0 - t0 = 0.0.
        padded = (elapsed_times_ms[0], *elapsed_times_ms)
        return tuple(
            cls(elapsed_ms=current, interval_ms=current - previous)
            for previous, current in itertools.pairwise(padded)
        )


class TimestampSequence(DataSequence[Timestamp, int]):
    """A read-only sequence of per-frame `Timestamp`s, from any source.

    Annotate parameters with this interface to accept either implementation:
    `TimestampTxtSequence` (read from a Lyncée Tec Koala ``timestamps.txt``) or
    `TimestampFpsSequence` (synthesized from a frame rate). Each item is a
    `Timestamp` and its metadata is the frame index.

    Subclasses populate `self._timestamps` (the ordered frames) in their
    ``__init__``; the sequence protocol (`__len__`, `get_item`, `get_meta`)
    is served from it here.
    """

    _timestamps: tuple[Timestamp, ...]

    def __len__(self) -> int:
        return len(self._timestamps)

    def get_item(self, index: int) -> Timestamp:
        return self._timestamps[index]

    def get_meta(self, index: int) -> int:
        return index

    @property
    def timestamps(self) -> tuple[Timestamp, ...]:
        """The frames as an immutable tuple, in index order."""
        return self._timestamps

    @property
    @abstractmethod
    def mean_interval_ms(self) -> float:
        """Mean interval between consecutive frames, in milliseconds."""
        raise NotImplementedError

    @property
    def mean_frame_rate(self) -> float:
        """Mean frame rate in frames per second (fps).

        Raises:
            ValueError: If the mean interval is zero, leaving the frame rate
                undefined.
        """
        interval_ms = self.mean_interval_ms
        if interval_ms == 0.0:
            msg = "mean frame rate is undefined when the mean interval is zero"
            raise ValueError(msg)
        return 1000.0 / interval_ms


class TimestampTxtSequence(SingleFileSequence[Timestamp, int], TimestampSequence):
    """A `TimestampSequence` read from a Lyncée Tec Koala ``timestamps.txt``.

    Each line is ``<index> <time> <date> <elapsed_ms>`` (space-separated), and
    the frame indices must run contiguously from 0.

    Raises:
        FileNotFoundError: If `path` does not exist.
        NotAFileError: If `path` exists but is not a regular file.
        ValueError: If the file is malformed or its frame indices are not
            contiguous from 0 (see `parse`).
    """

    LINE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"""
        (?P<index>\d{5})\s+          # 5-digit frame index
        \d{2}:\d{2}:\d{2}\.\d+\s+    # time, HH:MM:SS.fff
        \d{4}\.\d{2}\.\d{2}\s+       # date, YYYY.MM.DD
        (?P<elapsed>\d+(?:\.\d+)?)   # elapsed milliseconds
        """,
        re.VERBOSE,
    )

    def __init__(self, path: StrPath) -> None:
        super().__init__(path)
        self._timestamps = self.parse(self.path)

    @cached_property
    def mean_interval_ms(self) -> float:
        """Mean interval between consecutive frames, in milliseconds.

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

        Every non-blank line must match ``<5-digit index> <HH:MM:SS.fff>
        <YYYY.MM.DD> <elapsed_ms>``, the frame indices must run contiguously
        from 0, and the elapsed times must be non-decreasing. Blank lines are
        ignored.

        Raises:
            FileNotFoundError: If `path` does not exist.
            NotAFileError: If `path` exists but is not a regular file.
            ValueError: If a line does not match the expected format, the
                frame indices are not contiguous from 0, or an elapsed time
                decreases.
        """
        path = ensure_file_exists(path)

        elapsed_times_ms: list[float] = []

        for lineno, line in enumerate(path.read_text().splitlines()):
            stripped = line.strip()
            if not stripped:
                continue

            matched = cls.LINE_PATTERN.fullmatch(stripped)
            if matched is None:
                msg = f"line {lineno} is malformed (got {line!r}): {path}"
                raise ValueError(msg)

            index = int(matched["index"])
            expected_index = len(elapsed_times_ms)  # next contiguous index, from 0
            if index != expected_index:
                msg = f"frame index at line {lineno} must be {expected_index} (got {index}): {path}"
                raise ValueError(msg)

            elapsed = float(matched["elapsed"])
            if elapsed_times_ms and elapsed < elapsed_times_ms[-1]:
                msg = f"elapsed time at line {lineno} must be >= {elapsed_times_ms[-1]} (got {elapsed}): {path}"
                raise ValueError(msg)

            elapsed_times_ms.append(elapsed)

        return Timestamp.series_from_elapsed_times(elapsed_times_ms)


class TimestampFpsSequence(TimestampSequence):
    """A synthetic `TimestampSequence` with a constant frame rate.

    Frames are evenly spaced from `frame_rate`; see `generate` for the timing.

    Raises:
        ValueError: If `frame_rate` is not positive, or `num_frames` is
            negative.
    """

    def __init__(self, *, frame_rate: float, num_frames: int) -> None:
        self._timestamps = self.generate(frame_rate=frame_rate, num_frames=num_frames)
        self._frame_rate = frame_rate
        self._interval_ms = 1000.0 / frame_rate

    @property
    def mean_frame_rate(self) -> float:
        return self._frame_rate

    @property
    def mean_interval_ms(self) -> float:
        return self._interval_ms

    @classmethod
    def generate(cls, *, frame_rate: float, num_frames: int) -> tuple[Timestamp, ...]:
        """Generate `num_frames` timestamps at a constant `frame_rate` (in fps).

        Frame ``i`` has ``elapsed_ms = i * 1000 / frame_rate``.

        Raises:
            ValueError: If `frame_rate` is not positive, or `num_frames` is
                negative.
        """
        if frame_rate <= 0.0:
            msg = f"frame_rate must be positive (got {frame_rate})"
            raise ValueError(msg)

        if num_frames < 0:
            msg = f"num_frames must be non-negative (got {num_frames})"
            raise ValueError(msg)

        interval_ms = 1000.0 / frame_rate
        elapsed_times_ms = [i * interval_ms for i in range(num_frames)]
        return Timestamp.series_from_elapsed_times(elapsed_times_ms)
