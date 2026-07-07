from __future__ import annotations

__all__ = ("Timestamp", "TimestampSequence", "TimestampsFixedFPS")

from abc import abstractmethod
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, override

from kaparoo.data.sequences.base import DataSequence

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class Timestamp:
    """Acquisition timing of one frame, in ms.

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
            got = f"{self.interval_ms} > {self.elapsed_ms}"
            msg = f"interval_ms must not exceed elapsed_ms (got {got})"
            raise ValueError(msg)

    @classmethod
    def series_from_elapsed_times(
        cls, elapsed_times_ms: Sequence[float]
    ) -> tuple[Timestamp, ...]:
        """Build a series of `Timestamp`s from cumulative elapsed times.

        The first frame's interval is 0.0; each later frame's interval is its gap from
        the previous frame.
        """
        if not elapsed_times_ms:
            return ()

        # Pad the front so the first frame's interval comes out as t0 - t0 = 0.0.
        padded: tuple[float, ...] = (elapsed_times_ms[0], *elapsed_times_ms)
        return tuple(
            cls(elapsed_ms=current, interval_ms=current - previous)
            for previous, current in pairwise(padded)
        )


class TimestampSequence(DataSequence[Timestamp, int]):
    """A read-only sequence of per-frame `Timestamp`s, from any source.

    Annotate parameters with this interface to accept any timing source, however the
    timestamps are obtained. Each item is a `Timestamp` and its metadata is the frame
    index.

    Subclasses populate `self._timestamps` (the ordered frames) in their ``__init__``
    and implement `mean_interval_ms`; the sequence protocol (`__len__`, `get_item`,
    `get_meta`) is served from it here.
    """

    _timestamps: tuple[Timestamp, ...]

    def __len__(self) -> int:
        return len(self._timestamps)

    @override
    def get_item(self, index: int) -> Timestamp:
        return self._timestamps[index]

    @override
    def get_meta(self, index: int) -> int:
        return index

    @property
    def timestamps(self) -> tuple[Timestamp, ...]:
        """The per-frame `Timestamp`s, as an immutable tuple in index order."""
        return self._timestamps

    @property
    @abstractmethod
    def mean_interval_ms(self) -> float:
        """The mean interval between consecutive frames, in ms."""
        raise NotImplementedError

    @property
    def mean_frame_rate(self) -> float:
        """The mean frame rate, in fps.

        Raises:
            ValueError: If the mean interval is zero, leaving the frame rate undefined.
        """
        interval_ms = self.mean_interval_ms
        if interval_ms == 0.0:
            msg = "mean frame rate is undefined when the mean interval is zero"
            raise ValueError(msg)
        return 1000.0 / interval_ms


class TimestampsFixedFPS(TimestampSequence):
    """A synthetic `TimestampSequence` with a constant frame rate.

    Frames are evenly spaced at ``1000 / frame_rate`` ms; `mean_frame_rate` /
    `mean_interval_ms` return the bound rate exactly (no averaging drift).

    Args:
        frame_rate: Frames per second; must be positive.
        num_frames: Number of frames to synthesize; must be non-negative.

    Raises:
        ValueError: If `frame_rate` is not positive, or `num_frames` is negative.
    """

    def __init__(self, *, frame_rate: float, num_frames: int) -> None:
        self._timestamps = self.generate(frame_rate=frame_rate, num_frames=num_frames)
        self._frame_rate = frame_rate
        self._interval_ms = 1000.0 / frame_rate

    @property
    @override
    def mean_frame_rate(self) -> float:
        """The mean frame rate, in fps (constant: the bound `frame_rate`)."""
        return self._frame_rate

    @property
    @override
    def mean_interval_ms(self) -> float:
        """The mean interval between consecutive frames, in ms (constant 1000 / fps)."""
        return self._interval_ms

    @classmethod
    def generate(cls, *, frame_rate: float, num_frames: int) -> tuple[Timestamp, ...]:
        """Generate `num_frames` timestamps at a constant `frame_rate` (in fps).

        Frame ``i`` has ``elapsed_ms = i * 1000 / frame_rate``.

        Raises:
            ValueError: If `frame_rate` is not positive, or `num_frames` is negative.
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
