from __future__ import annotations

__all__ = (
    "Timestamp",
    "TimestampSequence",
    "TimestampSeries",
    "TimestampsFixedFPS",
)

from abc import abstractmethod
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, override

from kaparoo.data.sequences.base import DataSequence
from kaparoo.utils import ensure_one_of

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Literal


@dataclass(frozen=True, slots=True)
class Timestamp:
    """Acquisition timing of one frame, in ms.

    `elapsed_ms` stands on its own, but `interval_ms` is a relation to the
    *neighbouring* frame, so it is only meaningful within the series it was built
    for: dropping frames (a slice, a `SlicedSequence`, a filter) leaves it
    describing a gap that is no longer there. Rebuild a subset through
    `TimestampSequence.select` / `subsample`, which recompute it.

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

    Subclasses populate `self._timestamps` (the ordered frames) in their `__init__`
    and implement `mean_interval_ms`; the sequence protocol (`__len__`, `get_item`,
    `get_meta`) is served from it here.

    Take a subset through `select` / `subsample` rather than a slice or a generic
    composer: those hand back the items untouched, and an item's `interval_ms`
    measures the gap to a neighbour the subset no longer has.
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

    def _mean_interval_over_series(self) -> float:
        """Average the per-frame intervals; the mean for a measured series.

        Shared by the sources whose spacing is measured rather than declared, where
        `TimestampsFixedFPS` answers from its bound rate instead.

        Raises:
            ValueError: If the sequence has fewer than two frames.
        """
        if len(self) < 2:
            msg = f"mean interval requires at least two frames (got {len(self)})"
            raise ValueError(msg)

        return sum((ts.interval_ms for ts in self), 0.0) / (len(self) - 1)

    def select(self, indices: Sequence[int]) -> TimestampSeries:
        """A series over the frames at `indices`, with the intervals recomputed.

        `elapsed_ms` carries over untouched, since it is measured from acquisition
        start and dropping frames does not move it. `interval_ms` is rebuilt from
        the gaps between the frames that remain, so the first is 0.0 as a first
        frame always is, and skipping every other frame doubles the rest.

        Args:
            indices: The frames to keep, in increasing order.

        Raises:
            IndexError: If an index is out of range.
            ValueError: If `indices` is not increasing, which would put a frame
                before its predecessor and make an interval negative.
        """
        elapsed_times_ms = [self._timestamps[index].elapsed_ms for index in indices]
        return TimestampSeries(Timestamp.series_from_elapsed_times(elapsed_times_ms))

    def subsample(
        self,
        step: int = 1,
        *,
        start: int = 0,
        count: int | None = None,
        on_short: Literal["allow", "raise"] = "allow",
    ) -> TimestampSeries:
        """Take every `step`-th frame from `start`, at most `count` of them.

        A `select` over a regular stride, so the intervals are recomputed the same
        way: reading one frame in two doubles them while leaving `elapsed_ms` alone.

        Args:
            step: Stride between kept frames; 1 (default) keeps every frame.
            start: Index of the first frame to keep.
            count: How many frames to keep, or None (default) for as many as the
                stride yields.
            on_short: What to do when fewer than `count` frames are available:
                `"allow"` (default) returns the shorter series, `"raise"` rejects
                it. Ignored when `count` is None.

        Raises:
            ValueError: If `step` is not positive, `start` or `count` is negative,
                `on_short` is not one of the two policies, or fewer than `count`
                frames are available while `on_short` is `"raise"`.
        """
        if step < 1:
            msg = f"step must be positive (got {step})"
            raise ValueError(msg)

        if start < 0:
            msg = f"start must be non-negative (got {start})"
            raise ValueError(msg)

        ensure_one_of(on_short, ("allow", "raise"), name="on_short")

        indices = range(start, len(self), step)
        if count is not None:
            if count < 0:
                msg = f"count must be non-negative (got {count})"
                raise ValueError(msg)

            available = len(indices)
            if count > available and on_short == "raise":
                msg = f"count must not exceed the {available} available (got {count})"
                raise ValueError(msg)

            indices = indices[:count]

        return self.select(indices)

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


class TimestampSeries(TimestampSequence):
    """A `TimestampSequence` over an explicit series of `Timestamp`s.

    What `select` / `subsample` return, and the way to hold a series computed
    elsewhere. Deliberately not a `TimestampsFixedFPS` even when the frames are
    evenly spaced: that class answers `mean_interval_ms` from the rate bound at
    construction, which a subsample would no longer have.

    Args:
        timestamps: The frames, in order. Their `elapsed_ms` must not decrease.

    Raises:
        ValueError: If an `elapsed_ms` decreases.
    """

    def __init__(self, timestamps: Iterable[Timestamp]) -> None:
        self._timestamps = tuple(timestamps)

        for index, (previous, current) in enumerate(
            pairwise(self._timestamps), start=1
        ):
            if current.elapsed_ms < previous.elapsed_ms:
                got = f"{current.elapsed_ms} < {previous.elapsed_ms}"
                msg = f"elapsed_ms must not decrease at frame {index} (got {got})"
                raise ValueError(msg)

    @property
    @override
    def mean_interval_ms(self) -> float:
        """The mean interval between consecutive frames, in ms.

        Raises:
            ValueError: If the sequence has fewer than two frames.
        """
        return self._mean_interval_over_series()


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
