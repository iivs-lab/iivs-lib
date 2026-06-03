from __future__ import annotations

import pytest

from iivs.dhm.koala.timestamp import (
    Timestamp,
    TimestampSequence,
    TimestampsFixedFPS,
    TimestampsTxtFile,
)

_SAMPLE = (
    "00000 15:21:47.674 2026.01.15 0\n"
    "00001 15:21:47.674 2026.01.15 51.3082\n"
    "00002 15:21:47.675 2026.01.15 103.1792\n"
)


def _write(tmp_path, text=_SAMPLE):
    path = tmp_path / "timestamps.txt"
    path.write_text(text)
    return path


# --- TimestampsTxtFile ---


def test_txt_is_a_timestamp_sequence(tmp_path):
    assert isinstance(TimestampsTxtFile(_write(tmp_path)), TimestampSequence)


def test_txt_len_and_frame_indices(tmp_path):
    seq = TimestampsTxtFile(_write(tmp_path))
    assert len(seq) == 3
    assert [seq.get_meta(i) for i in range(3)] == [0, 1, 2]


def test_txt_elapsed_and_interval(tmp_path):
    seq = TimestampsTxtFile(_write(tmp_path))
    assert seq[0] == Timestamp(elapsed_ms=0.0, interval_ms=0.0)
    assert seq[1].elapsed_ms == pytest.approx(51.3082)
    assert seq[1].interval_ms == pytest.approx(51.3082)  # since previous frame
    assert seq[2].interval_ms == pytest.approx(103.1792 - 51.3082)


def test_txt_tolerates_trailing_whitespace_in_line(tmp_path):
    # trailing spaces within a data line are stripped; a lone final newline is fine
    seq = TimestampsTxtFile(_write(tmp_path, "00000 15:21:47.674 2026.01.15 0 \n"))
    assert len(seq) == 1
    assert seq[0].elapsed_ms == 0.0


@pytest.mark.parametrize(
    ("text", "expected_len"),
    (
        ("\n00000 15:21:47.674 2026.01.15 0\n", 1),  # leading
        ("00000 15:21:47.674 2026.01.15 0\n\n", 1),  # trailing
        (  # in between
            "00000 15:21:47.674 2026.01.15 0\n\n00001 15:21:47.674 2026.01.15 51.3082\n",
            2,
        ),
    ),
)
def test_txt_ignores_blank_lines(tmp_path, text, expected_len):
    assert len(TimestampsTxtFile(_write(tmp_path, text))) == expected_len


def test_txt_rejects_malformed_line(tmp_path):
    with pytest.raises(ValueError, match="malformed"):
        TimestampsTxtFile(_write(tmp_path, "00000 15:21:47.674\n"))


def test_txt_rejects_wrong_field_format(tmp_path):
    with pytest.raises(ValueError, match="malformed"):
        TimestampsTxtFile(_write(tmp_path, "0 15:21:47.674 2026/01/15 0\n"))


def test_txt_rejects_noncontiguous_index(tmp_path):
    text = "00000 15:21:47.674 2026.01.15 0\n00002 15:21:47.675 2026.01.15 103.1792\n"
    with pytest.raises(ValueError, match="index at line"):
        TimestampsTxtFile(_write(tmp_path, text))


def test_txt_rejects_decreasing_elapsed(tmp_path):
    text = "00000 15:21:47.674 2026.01.15 100\n00001 15:21:47.675 2026.01.15 50\n"
    with pytest.raises(ValueError, match="elapsed time at line"):
        TimestampsTxtFile(_write(tmp_path, text))


def test_txt_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        TimestampsTxtFile(tmp_path / "nope.txt")


def test_txt_parse_returns_timestamps(tmp_path):
    assert TimestampsTxtFile.parse(_write(tmp_path)) == (
        Timestamp(elapsed_ms=0.0, interval_ms=0.0),
        Timestamp(elapsed_ms=51.3082, interval_ms=51.3082),
        Timestamp(elapsed_ms=103.1792, interval_ms=103.1792 - 51.3082),
    )


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
    with pytest.raises(IndexError):
        _ = seq[3]


def test_fps_rejects_nonpositive_frame_rate():
    with pytest.raises(ValueError, match="frame_rate must be positive"):
        TimestampsFixedFPS(frame_rate=0.0, num_frames=3)


def test_fps_rejects_negative_num_frames():
    with pytest.raises(ValueError, match="num_frames must be non-negative"):
        TimestampsFixedFPS(frame_rate=20.0, num_frames=-1)


# --- TimestampSequence interface ---


def test_timestamps_property_matches_iteration(tmp_path):
    seq = TimestampsTxtFile(_write(tmp_path))
    assert isinstance(seq.timestamps, tuple)
    assert seq.timestamps == tuple(seq)  # same frames, in index order


# --- means ---


def test_txt_mean_interval_and_frame_rate(tmp_path):
    seq = TimestampsTxtFile(_write(tmp_path))
    # 3 frames -> 2 gaps; mean gap = total elapsed / (N - 1)
    assert seq.mean_interval_ms == pytest.approx(103.1792 / 2)
    assert seq.mean_frame_rate == pytest.approx(1000.0 / (103.1792 / 2))


def test_txt_mean_interval_is_cached(tmp_path):
    seq = TimestampsTxtFile(_write(tmp_path))
    assert "mean_interval_ms" not in seq.__dict__  # not computed until first access
    _ = seq.mean_interval_ms
    assert "mean_interval_ms" in seq.__dict__  # cached afterwards


def test_txt_mean_undefined_for_short_sequence(tmp_path):
    seq = TimestampsTxtFile(_write(tmp_path, "00000 15:21:47.674 2026.01.15 0\n"))
    with pytest.raises(ValueError, match="at least two frames"):
        _ = seq.mean_interval_ms


def test_txt_mean_frame_rate_undefined_for_zero_interval(tmp_path):
    # Two frames at the same elapsed time -> mean interval 0 -> rate undefined.
    text = "00000 15:21:47.674 2026.01.15 0\n00001 15:21:47.674 2026.01.15 0\n"
    seq = TimestampsTxtFile(_write(tmp_path, text))
    assert seq.mean_interval_ms == 0.0
    with pytest.raises(ValueError, match="undefined"):
        _ = seq.mean_frame_rate


def test_fps_mean_is_exact():
    seq = TimestampsFixedFPS(frame_rate=20.0, num_frames=3)
    assert seq.mean_interval_ms == pytest.approx(50.0)
    assert seq.mean_frame_rate == pytest.approx(20.0)


def test_fps_mean_defined_even_for_single_frame():
    # FPS knows its rate from construction, so a 1-frame sequence still reports it.
    seq = TimestampsFixedFPS(frame_rate=20.0, num_frames=1)
    assert seq.mean_interval_ms == pytest.approx(50.0)


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
