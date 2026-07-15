from __future__ import annotations

import pytest

from iivs.common.data.timestamp import Timestamp, TimestampSequence
from iivs.dhm.data.timestamp import TimestampsTxtFile

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


def test_txt_rejects_wrong_extension(tmp_path):
    path = tmp_path / "timestamps.dat"
    path.write_text(_SAMPLE)
    with pytest.raises(
        ValueError, match=r"unsupported extension .* \(supported: 'txt'\)"
    ):
        TimestampsTxtFile(path)


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
            "00000 15:21:47.674 2026.01.15 0\n\n"
            "00001 15:21:47.674 2026.01.15 51.3082\n",
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


def test_txt_timestamps_property_matches_iteration(tmp_path):
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
    # Exercises TimestampSequence.mean_frame_rate's zero-interval guard, which a
    # positive-rate TimestampsFixedFPS can never reach.
    text = "00000 15:21:47.674 2026.01.15 0\n00001 15:21:47.674 2026.01.15 0\n"
    seq = TimestampsTxtFile(_write(tmp_path, text))
    assert seq.mean_interval_ms == 0.0
    with pytest.raises(ValueError, match="undefined"):
        _ = seq.mean_frame_rate
