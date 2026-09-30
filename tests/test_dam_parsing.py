"""Reading a DAM file: the status rule, duplicate slots, and the time axis.

``dam_integrity`` decides which readings are data at all, and nothing tested it
directly — the end-to-end import tests only ever saw example_data, which is
clean (no failed reads, no gaps), so every branch that handles a bad recording
was unexercised. The rows here are REAL rows from example_data's Monitor 17,
edited the way a real recording goes wrong: a failed read, a doubled
timestamp, a drifting seconds field, a clock change.

The rule under test, which every analysis downstream depends on: **only
status 1 is a reading.** Any other status is NaN — never zero, because zero is
a real measurement meaning "the fly was still".
"""

import numpy as np
import pandas as pd
import pytest

from clockwork.core import dam_integrity
from clockwork.core.dam_processor import _read_monitor_file
from conftest import EXAMPLE_DIR, requires_example_data

pytestmark = requires_example_data

N_ROWS = 600  # the first ten hours of Monitor 17


@pytest.fixture(scope="module")
def raw_lines():
    with open(EXAMPLE_DIR / "Monitor17.txt") as fh:
        return [next(fh).rstrip("\n").split("\t") for _ in range(N_ROWS)]


def _write(tmp_path, lines, name="Monitor17.txt"):
    path = tmp_path / name
    path.write_text("\n".join("\t".join(row) for row in lines) + "\n")
    return path


def _read(tmp_path, lines):
    return _read_monitor_file(_write(tmp_path, lines))


def _with(row, **fields):
    """A copy of a raw row with some fields replaced (status is column 3)."""
    row = list(row)
    columns = {"date": 1, "time": 2, "status": 3}
    for key, value in fields.items():
        if key.startswith("channel_"):
            row[9 + int(key.split("_")[1])] = str(value)
        else:
            row[columns[key]] = str(value)
    return row


def _channels(df):
    return [c for c in df.columns if c.startswith("channel_")]


# ---------------------------------------------------------------------------
# The file reader
# ---------------------------------------------------------------------------


def test_real_rows_parse_to_a_one_minute_axis(tmp_path, raw_lines):
    df = _read(tmp_path, raw_lines)
    assert len(df) == N_ROWS
    assert len(_channels(df)) == 32
    assert df["datetime"].iloc[0] == pd.Timestamp("2025-01-08 00:00:00")
    assert (df["datetime"].diff().dropna() == pd.Timedelta(minutes=1)).all()


def test_drifting_seconds_are_snapped_to_the_minute(tmp_path, raw_lines):
    """FlyBox writes HH:MM:SS with SS creeping forward; left alone, no reading
    would land on the :00 grid and the whole file would read as missing."""
    drifting = []
    for i, row in enumerate(raw_lines[:120]):
        hh, mm, _ = row[2].split(":")
        drifting.append(_with(row, time=f"{hh}:{mm}:{(6 + i // 3) % 30:02d}"))
    df = _read(tmp_path, drifting)
    assert (df["datetime"].dt.second == 0).all()
    np.testing.assert_array_equal(
        df["datetime"].values, _read(tmp_path, raw_lines[:120])["datetime"].values
    )


def test_an_unrecognised_date_format_raises_rather_than_guessing(tmp_path, raw_lines):
    bad = [_with(raw_lines[0], date="2025-01-08")] + raw_lines[1:10]
    with pytest.raises(ValueError):
        _read(tmp_path, bad)


# ---------------------------------------------------------------------------
# The status rule and duplicate slots
# ---------------------------------------------------------------------------


def test_a_failed_read_is_nan_never_zero(tmp_path, raw_lines):
    lines = list(raw_lines)
    lines[100] = _with(lines[100], status=51, channel_1=7)
    lines[101] = _with(lines[101], status=24)
    df, info = dam_integrity.resolve_status_and_duplicates(_read(tmp_path, lines))
    assert df[_channels(df)].iloc[100:102].isna().all().all()
    assert df[_channels(df)].drop(index=[100, 101]).notna().all().all()
    assert info["n_status_bad"] == 2
    assert info["bad_status_counts"] == {51: 1, 24: 1}
    assert info["undocumented_status_counts"] == {}


def test_an_all_zero_status1_row_is_kept_as_zero(tmp_path, raw_lines):
    """Zero is a real reading. The discriminator is the status, not zero-ness."""
    lines = list(raw_lines)
    lines[200] = _with(lines[200], **{f"channel_{i}": 0 for i in range(1, 33)})
    df, _ = dam_integrity.resolve_status_and_duplicates(_read(tmp_path, lines))
    assert (df[_channels(df)].iloc[200] == 0).all()


def test_undocumented_codes_are_nan_and_reported_separately(tmp_path, raw_lines):
    lines = list(raw_lines)
    lines[300] = _with(lines[300], status=55)
    df, info = dam_integrity.resolve_status_and_duplicates(_read(tmp_path, lines))
    assert df[_channels(df)].iloc[300].isna().all()
    assert info["undocumented_status_counts"] == {55: 1}


def test_at_a_doubled_slot_the_status1_row_wins_whatever_the_order(tmp_path, raw_lines):
    real = raw_lines[50]
    error = _with(real, status=50, channel_1=999)
    for order in ([error, real], [real, error]):
        lines = raw_lines[:50] + order + raw_lines[51:]
        df, info = dam_integrity.resolve_status_and_duplicates(_read(tmp_path, lines))
        assert df["datetime"].is_unique
        assert len(df) == N_ROWS
        row = df[df["datetime"] == pd.Timestamp("2025-01-08 00:50:00")]
        assert row["channel_1"].iloc[0] == float(real[10])
        assert info["n_rows_dropped"] == 1


def test_a_slot_with_only_failed_reads_keeps_one_nan_row(tmp_path, raw_lines):
    failed = _with(raw_lines[60], status=50)
    lines = raw_lines[:60] + [failed, failed] + raw_lines[61:]
    df, _ = dam_integrity.resolve_status_and_duplicates(_read(tmp_path, lines))
    slot = df[df["datetime"] == pd.Timestamp("2025-01-08 01:00:00")]
    assert len(slot) == 1
    assert slot[_channels(df)].isna().all().all()


# ---------------------------------------------------------------------------
# Gaps and the two-tier irregularity report
# ---------------------------------------------------------------------------


def test_a_skipped_hour_is_one_gap_and_sixty_lost_slots(tmp_path, raw_lines):
    lines = raw_lines[:120] + raw_lines[180:]  # 02:00-02:59 never written
    df, _ = dam_integrity.resolve_status_and_duplicates(_read(tmp_path, lines))
    times = df["datetime"].values

    scan = dam_integrity.scan_time_integrity(times, status1_times=times)
    assert len(scan["gaps"]) == 1
    assert scan["gaps"][0]["minutes"] == 61.0  # 01:59 -> 03:00
    assert scan["n_missing_slots"] == 60

    cls = dam_integrity.classify_irregularities(times, times)
    assert cls["n_dataloss_slots"] == 60
    assert cls["dataloss_absent"] == 60
    assert cls["dataloss_spans"] == [
        {
            "start": pd.Timestamp("2025-01-08 02:00"),
            "end": pd.Timestamp("2025-01-08 02:59"),
            "n_slots": 60,
        }
    ]


def test_cosmetic_and_data_loss_slots_are_told_apart(tmp_path, raw_lines):
    lines = list(raw_lines)
    redundant = _with(lines[10], status=50)
    lines[20] = _with(lines[20], status=51)  # a failed read, no real one: data loss
    lines = lines[:11] + [redundant] + lines[11:]  # doubled slot: cosmetic
    raw = _read(tmp_path, lines)
    df, _ = dam_integrity.resolve_status_and_duplicates(raw)
    valid = raw["monitor_status"] == 1
    cls = dam_integrity.classify_irregularities(
        df["datetime"].values,
        raw.loc[valid, "datetime"].values,
        error_times=raw.loc[~valid, "datetime"].values,
    )
    assert cls["n_cosmetic_slots"] == 1
    assert cls["n_dataloss_slots"] == 1
    assert cls["dataloss_error_only"] == 1


def test_a_file_that_stops_early_loses_the_rest_of_the_window(tmp_path, raw_lines):
    df, _ = dam_integrity.resolve_status_and_duplicates(_read(tmp_path, raw_lines))
    times = df["datetime"].values
    cls = dam_integrity.classify_irregularities(
        times, times, grid_start=times[0], grid_end=pd.Timestamp(times[-1]) + pd.Timedelta("30min")
    )
    assert cls["dataloss_absent"] == 30


# ---------------------------------------------------------------------------
# Clock changes: pinned CURRENT behaviour, not a design
# ---------------------------------------------------------------------------
#
# The loader works in naive local time: nothing in core/ knows about time zones.
# A monitor PC that follows daylight saving writes the repeated autumn hour twice
# and skips the spring hour. These tests record what that does today, so a change
# to it is deliberate: the spring hour reads as a 60-slot data-loss gap, and the
# autumn repeat is de-duplicated to its FIRST readings, silently discarding an
# hour of real data. If time-zone handling is ever added, these are the tests to
# rewrite.


def _hour_of(lines, hour):
    return [row for row in lines if row[2].startswith(f"{hour:02d}:")]


def test_spring_forward_reads_as_an_hour_of_data_loss(tmp_path, raw_lines):
    lines = [row for row in raw_lines if not row[2].startswith("02:")]
    df, _ = dam_integrity.resolve_status_and_duplicates(_read(tmp_path, lines))
    cls = dam_integrity.classify_irregularities(df["datetime"].values, df["datetime"].values)
    assert cls["n_dataloss_slots"] == 60


def test_fall_back_keeps_the_first_pass_of_the_repeated_hour(tmp_path, raw_lines):
    first_pass = _hour_of(raw_lines, 1)
    second_pass = [_with(row, channel_1=int(row[10]) + 1000) for row in first_pass]
    cut = raw_lines.index(first_pass[-1]) + 1
    lines = raw_lines[:cut] + second_pass + raw_lines[cut:]
    df, info = dam_integrity.resolve_status_and_duplicates(_read(tmp_path, lines))
    assert df["datetime"].is_unique
    assert info["n_rows_dropped"] == 60
    hour = df[df["datetime"].dt.hour == 1]
    assert (hour["channel_1"] < 1000).all()  # the second pass is gone
