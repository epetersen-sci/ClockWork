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
