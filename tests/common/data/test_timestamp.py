from __future__ import annotations

import pytest

from iivs.common.data import Timestamp, TimestampSequence, TimestampsFixedFPS

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
