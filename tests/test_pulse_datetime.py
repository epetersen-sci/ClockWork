"""A light pulse written as a date-time, in the same format as start_datetime.

start_datetime is the fly's ZT0, so the pulse's minute — and from it both its day
and its ZT — follows from the difference, with no assumption about which day of the
protocol it fell on. The ZT-hour shorthand is still read the old way.
"""

import numpy as np
import pandas as pd
import pytest

from clockwork.core import dam_utilities


def _build(pulse_cells, first_dd="2025-01-19 09:00"):
    rows = [
        {
            "id": f"f{i}",
            "Monitor": 1,
            "genotype": "w",
            "pulse_time": cell,
            "pulse_duration_min": 0 if cell == "" else 60,
            "start_datetime": pd.Timestamp("2025-01-15 09:00"),
            "stop_datetime": pd.Timestamp("2025-01-22 09:00"),
            "first_DD_day": pd.Timestamp(first_dd),
        }
        for i, cell in enumerate(pulse_cells)
    ]
    meta = pd.DataFrame(rows)
    n_time = 7 * 1440
    data = pd.DataFrame(
        np.ones((n_time, len(rows))), columns=list(meta["id"]), index=pd.RangeIndex(n_time)
    )
    return dam_utilities.create_xarray_dataset(data, meta)


def test_datetime_pulse_gives_its_own_day_and_zt():
    # 2 days 12 h after ZT0: 0-based day 2 at ZT12 — not the last LD day.
    ds = dam_utilities.add_pulse_metadata(_build(["1/17/2025 21:00"]))
    assert ds["pulse_minute"].item() == 2 * 1440 + 12 * 60
    assert ds["pulse_time"].item() == "ZT12"
    np.testing.assert_allclose(dam_utilities.pulse_zt_hours(ds), [12.0])


def test_iso_and_spreadsheet_style_datetimes_read_the_same():
    a = dam_utilities.add_pulse_metadata(_build(["2025-01-17 21:00:00"]))
    b = dam_utilities.add_pulse_metadata(_build(["1/17/2025 21:00"]))
    assert a["pulse_minute"].item() == b["pulse_minute"].item()


def test_zt_shorthand_still_anchors_to_the_last_ld_day():
    ds = dam_utilities.add_pulse_metadata(_build(["ZT15"]))
    # first_DD_day is 4 days after start: split 5760, ZT15 the day before.
    assert ds["pulse_minute"].item() == 4 * 1440 - 1440 + 15 * 60
    assert "pulse_datetime" not in ds.coords


def test_mixed_cells_and_blank_controls():
    ds = dam_utilities.add_pulse_metadata(_build(["1/16/2025 9:00", "ZT15", ""]))
    minutes = ds["pulse_minute"].values
    assert minutes[0] == 1440
    assert minutes[1] == 3 * 1440 + 15 * 60
    assert np.isnan(minutes[2])
    assert list(ds["pulse_time"].values) == ["ZT0", "ZT15", ""]


def test_datetime_pulse_needs_no_first_dd_day():
    ds = _build(["1/17/2025 21:00"]).drop_vars("first_DD_day")
    assert dam_utilities.add_pulse_metadata(ds)["pulse_minute"].item() == 2 * 1440 + 720


def test_pulse_before_the_recording_is_refused():
    with pytest.raises(ValueError, match="before that row's start_datetime"):
        _build(["1/14/2025 21:00"])


def test_unparseable_cell_says_which_formats_work():
    with pytest.raises(ValueError, match="same format as start_datetime"):
        _build(["sometime"])


# ---------------------------------------------------------------------------
# pulse_datetime / pulse_duration: the two-column metadata contract
# ---------------------------------------------------------------------------


def _build_new(rows_in):
    """``rows_in``: ``[(pulse_datetime cell, pulse_duration cell), ...]``."""
    rows = [
        {
            "id": f"f{i}",
            "Monitor": 1,
            "genotype": "w",
            "sex": "M",
            "pulse_datetime": dt,
            "pulse_duration": dur,
            "start_datetime": pd.Timestamp("2025-01-15 09:00"),
            "stop_datetime": pd.Timestamp("2025-01-22 09:00"),
        }
        for i, (dt, dur) in enumerate(rows_in)
    ]
    meta = pd.DataFrame(rows)
    n_time = 7 * 1440
    data = pd.DataFrame(
        np.ones((n_time, len(rows))), columns=list(meta["id"]), index=pd.RangeIndex(n_time)
    )
    return dam_utilities.create_xarray_dataset(data, meta)


def test_the_zt_is_worked_out_from_pulse_datetime():
    ds = dam_utilities.add_pulse_metadata(_build_new([("1/17/2025 21:00", 10), ("", 0)]))
    assert list(ds["pulse_time"].values) == ["ZT12", ""]
    assert ds["pulse_minute"].values[0] == 2 * 1440 + 12 * 60
    assert np.isnan(ds["pulse_minute"].values[1])
    np.testing.assert_allclose(ds["pulse_duration_min"].values, [10, 0])
    assert "pulse_duration" not in ds.coords


def test_an_excel_timestamp_reads_like_text():
    a = _build_new([(pd.Timestamp("2025-01-17 21:00"), 10)])
    b = _build_new([("2025-01-17 21:00", 10)])
    assert a["pulse_datetime"].values[0] == b["pulse_datetime"].values[0]


def test_a_blank_control_duration_reads_as_zero():
    ds = _build_new([("1/17/2025 21:00", 10), (np.nan, np.nan)])
    np.testing.assert_allclose(ds["pulse_duration_min"].values, [10, 0])


@pytest.mark.parametrize(
    "rows, match",
    [
        ([("1/17/2025 21:00", 0)], "no 'pulse_duration' above 0"),
        ([("1/17/2025 21:00", np.nan)], "no 'pulse_duration' above 0"),
        ([("", 10)], "above 0 but no 'pulse_datetime'"),
        ([("ZT15", 10)], "could not be read as a date"),
        ([("1/17/2025 21:00", "ten")], "not a number of minutes"),
    ],
)
def test_contradictions_are_refused_by_row(rows, match):
    with pytest.raises(ValueError, match=match):
        _build_new(rows)


def test_the_error_names_the_spreadsheet_row():
    meta = pd.DataFrame({"pulse_datetime": ["1/17/2025 21:00", "", ""], "pulse_duration": [10, 0, 5]})
    with pytest.raises(ValueError, match=r"row\(s\) 4\b"):
        dam_utilities.normalize_pulse_columns(meta)


def test_both_spellings_in_one_file_are_refused():
    meta = pd.DataFrame({"pulse_datetime": ["1/17/2025 21:00"], "pulse_time": ["ZT15"]})
    with pytest.raises(ValueError, match="keep 'pulse_datetime'"):
        dam_utilities.normalize_pulse_columns(meta)


def test_older_columns_pass_through_unchanged():
    meta = pd.DataFrame({"pulse_time": ["ZT15", ""], "pulse_duration_min": [60, np.nan]})
    assert dam_utilities.normalize_pulse_columns(meta) is meta


def test_the_metadata_reader_reports_a_pulse_problem_as_a_metadata_error(tmp_path):
    from clockwork.core.dam_processor import MetadataProcessor
    from clockwork.core.import_diagnostics import REASON_METADATA_BAD_PULSE, MetadataError

    path = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "Monitor": [1, 2],
            "start_datetime": ["2025-01-15 09:00"] * 2,
            "stop_datetime": ["2025-01-22 09:00"] * 2,
            "genotype": ["w", "w"],
            "pulse_datetime": ["2025-01-17 21:00", ""],
            "pulse_duration": [10, 5],
        }
    ).to_csv(path, index=False)
    with pytest.raises(MetadataError) as err:
        MetadataProcessor(str(path), str(tmp_path)).expand_metadata()
    assert err.value.reason == REASON_METADATA_BAD_PULSE
    assert "row(s) 3" in str(err.value)
