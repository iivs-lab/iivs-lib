from __future__ import annotations

import pytest

from iivs.common.data import (
    Timestamp,
    TimestampSequence,
    TimestampSeries,
    TimestampsFixedFPS,
)

# --- Timestamp ---


def test_series_from_elapsed_times():
    assert Timestamp.series_from_elapsed_times([0.0, 50.0, 150.0]) == (
        Timestamp(elapsed_ms=0.0, interval_ms=0.0),
        Timestamp(elapsed_ms=50.0, interval_ms=50.0),
        Timestamp(elapsed_ms=150.0, interval_ms=100.0),
    )
    assert Timestamp.series_from_elapsed_times([]) == ()


def test_timestamp_rejects_negative_elapsed():
    with pytest.raises(ValueError, match="elapsed_ms must be non-negative"):
        Timestamp(elapsed_ms=-1.0, interval_ms=0.0)


def test_timestamp_rejects_negative_interval():
    with pytest.raises(ValueError, match="interval_ms must be non-negative"):
        Timestamp(elapsed_ms=1.0, interval_ms=-1.0)


def test_timestamp_rejects_interval_exceeding_elapsed():
    with pytest.raises(ValueError, match="must not exceed elapsed_ms"):
        Timestamp(elapsed_ms=1.0, interval_ms=2.0)


# --- TimestampsFixedFPS ---


def test_fps_is_a_timestamp_sequence():
    assert isinstance(
        TimestampsFixedFPS(frame_rate=20.0, num_frames=3), TimestampSequence
    )


def test_fps_constant_rate():
    seq = TimestampsFixedFPS(frame_rate=20.0, num_frames=3)  # 50 ms/frame
    assert len(seq) == 3
    assert seq[0] == Timestamp(elapsed_ms=0.0, interval_ms=0.0)
    assert seq[1] == Timestamp(elapsed_ms=50.0, interval_ms=50.0)
    assert seq[2] == Timestamp(elapsed_ms=100.0, interval_ms=50.0)
    assert [seq.get_meta(i) for i in range(3)] == [0, 1, 2]


def test_fps_generate_returns_timestamps():
    assert TimestampsFixedFPS.generate(frame_rate=20.0, num_frames=3) == (
        Timestamp(elapsed_ms=0.0, interval_ms=0.0),
        Timestamp(elapsed_ms=50.0, interval_ms=50.0),
        Timestamp(elapsed_ms=100.0, interval_ms=50.0),
    )


def test_fps_zero_frames_is_empty():
    seq = TimestampsFixedFPS(frame_rate=20.0, num_frames=0)
    assert len(seq) == 0


def test_fps_supports_negative_index_and_slice():
    seq = TimestampsFixedFPS(frame_rate=20.0, num_frames=3)
    assert seq[-1] == seq[2]
    assert list(seq[:2]) == [seq[0], seq[1]]


def test_fps_out_of_range_raises():
    seq = TimestampsFixedFPS(frame_rate=20.0, num_frames=3)
    with pytest.raises(IndexError, match="index out of range"):
        _ = seq[3]


def test_fps_rejects_nonpositive_frame_rate():
    with pytest.raises(ValueError, match="frame_rate must be positive"):
        TimestampsFixedFPS(frame_rate=0.0, num_frames=3)


def test_fps_rejects_negative_num_frames():
    with pytest.raises(ValueError, match="num_frames must be non-negative"):
        TimestampsFixedFPS(frame_rate=20.0, num_frames=-1)


def test_fps_mean_is_exact():
    seq = TimestampsFixedFPS(frame_rate=20.0, num_frames=3)
    assert seq.mean_interval_ms == pytest.approx(50.0)
    assert seq.mean_frame_rate == pytest.approx(20.0)


def test_fps_mean_defined_even_for_single_frame():
    # FPS knows its rate from construction, so a 1-frame sequence still reports it.
    seq = TimestampsFixedFPS(frame_rate=20.0, num_frames=1)
    assert seq.mean_interval_ms == pytest.approx(50.0)


# --- TimestampSequence interface ---


def test_timestamps_property_matches_iteration():
    seq = TimestampsFixedFPS(frame_rate=20.0, num_frames=3)
    assert isinstance(seq.timestamps, tuple)
    assert seq.timestamps == tuple(seq)  # same frames, in index order


# --- select / subsample ---


def _elapsed_and_intervals(seq):
    return [(ts.elapsed_ms, ts.interval_ms) for ts in seq]


def test_subsample_widens_the_intervals_and_keeps_elapsed():
    # 10 fps: frames 100 ms apart. Every other frame is 200 ms apart, but each
    # kept frame is still the same distance from acquisition start.
    seq = TimestampsFixedFPS(frame_rate=10.0, num_frames=6)

    half = seq.subsample(2)

    assert _elapsed_and_intervals(half) == [(0.0, 0.0), (200.0, 200.0), (400.0, 200.0)]
    assert half.mean_interval_ms == pytest.approx(200.0)
    assert half.mean_frame_rate == pytest.approx(5.0)
    # the source is untouched
    assert len(seq) == 6
    assert seq.mean_interval_ms == pytest.approx(100.0)


def test_subsample_result_is_a_timestamp_sequence_not_fixed_fps():
    # a subsample of a fixed-rate source must not keep answering from the bound
    # rate, which no longer describes the spacing
    seq = TimestampsFixedFPS(frame_rate=10.0, num_frames=4)
    half = seq.subsample(2)
    assert isinstance(half, TimestampSequence)
    assert isinstance(half, TimestampSeries)
    assert not isinstance(half, TimestampsFixedFPS)


def test_subsample_start_and_count():
    seq = TimestampsFixedFPS(frame_rate=10.0, num_frames=10)

    taken = seq.subsample(3, start=1, count=2)

    # indices 1, 4 -> elapsed 100, 400; the first kept frame restarts the interval
    assert _elapsed_and_intervals(taken) == [(100.0, 0.0), (400.0, 300.0)]


def test_subsample_contiguous_restarts_the_first_interval():
    # even step=1 needs the rebuild: the first kept frame has no predecessor
    seq = TimestampsFixedFPS(frame_rate=10.0, num_frames=5)
    tail = seq.subsample(start=2)
    assert _elapsed_and_intervals(tail) == [
        (200.0, 0.0),
        (300.0, 100.0),
        (400.0, 100.0),
    ]


def test_subsample_on_short_policies():
    seq = TimestampsFixedFPS(frame_rate=10.0, num_frames=5)

    # "allow" (default) yields what there is: indices 0, 2, 4
    assert len(seq.subsample(2, count=10)) == 3

    with pytest.raises(ValueError, match="count must not exceed the 3 available"):
        seq.subsample(2, count=10, on_short="raise")

    # exactly the available count is not short
    assert len(seq.subsample(2, count=3, on_short="raise")) == 3


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"step": 0}, "step must be positive"),
        ({"step": -1}, "step must be positive"),
        ({"start": -1}, "start must be non-negative"),
        ({"count": -1}, "count must be non-negative"),
        ({"on_short": "warn"}, "on_short must be one of"),
    ),
)
def test_subsample_rejects_bad_arguments(kwargs, match):
    seq = TimestampsFixedFPS(frame_rate=10.0, num_frames=4)
    with pytest.raises(ValueError, match=match):
        seq.subsample(**kwargs)


def test_select_takes_arbitrary_frames():
    seq = TimestampsFixedFPS(frame_rate=10.0, num_frames=6)
    picked = seq.select([0, 3, 5])
    assert _elapsed_and_intervals(picked) == [
        (0.0, 0.0),
        (300.0, 300.0),
        (500.0, 200.0),
    ]


def test_select_rejects_out_of_order_indices():
    # a decreasing pick would make an interval negative
    seq = TimestampsFixedFPS(frame_rate=10.0, num_frames=4)
    with pytest.raises(ValueError, match="interval_ms must be non-negative"):
        seq.select([2, 1])


def test_select_empty_gives_an_empty_series():
    seq = TimestampsFixedFPS(frame_rate=10.0, num_frames=4)
    empty = seq.select([])
    assert len(empty) == 0
    with pytest.raises(ValueError, match="at least two frames"):
        _ = empty.mean_interval_ms


# --- TimestampSeries ---


def test_series_rejects_decreasing_elapsed():
    good = Timestamp(elapsed_ms=10.0, interval_ms=0.0)
    back = Timestamp(elapsed_ms=5.0, interval_ms=0.0)
    with pytest.raises(ValueError, match="elapsed_ms must not decrease at frame 1"):
        TimestampSeries([good, back])
