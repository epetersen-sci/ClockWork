"""The DD heatmap is drawn in days since each fly's own DD start (commit 96c239a).

A ``select_phase`` DD view keeps every fly's whole time axis with the LD minutes
blanked, so flies that entered DD on different days have their DD data at
different x positions; ``_align_to_dd_start`` shifts each row so minute 0 is that
fly's first DD minute and the axis reads "Days in DD". The only heatmap tests
used a dataset with no ``phase_used``, where the function returns None, so the
shift itself was never exercised.

Real flies from example_data all entered DD on Jan 18. To make the staggering
the function exists for, Monitor 18's flies are given a DD start one day later
— the situation of combining two cohorts — and each row is checked to begin at
its own fly's first DD minute.
"""

import numpy as np
import pandas as pd
import pytest

from clockwork.core import dam_utilities, plotting

DAY = 1440


@pytest.fixture(scope="module")
def staggered(example_ds):
    ds = example_ds.copy()
    later = ds["Monitor"].values.astype(str) == "18"
    dd = pd.to_datetime(ds["first_DD_day"].values)
    dd = np.where(later, dd + pd.Timedelta(days=1), dd).astype("datetime64[ns]")
    return ds.assign_coords(first_DD_day=("id", dd)), later


def test_each_row_starts_at_its_own_dd_start(staggered):
    ds, later = staggered
    view, phase = dam_utilities.select_phase(ds, "DD")
    assert phase == "DD"
    aligned = plotting._align_to_dd_start(view["activity"], view)
    assert aligned is not None

    split = view["split_minute"].values.astype(int)
    assert set(split[later]) == {split[~later][0] + DAY}, "fixture should stagger by a day"
    raw = ds["activity"].transpose("time", "id").values
    for j in range(ds.sizes["id"]):
        n = raw.shape[0] - split[j]
        np.testing.assert_array_equal(aligned.values[:n, j], raw[split[j] :, j])
        assert np.isnan(aligned.values[n:, j]).all()


def test_discarding_the_first_dd_day_starts_a_day_later(staggered):
    ds, _ = staggered
    view, _ = dam_utilities.select_phase(ds, "DD", discard_first_dd_day=True)
    aligned = plotting._align_to_dd_start(view["activity"], view)
    split = view["split_minute"].values.astype(int)
    raw = ds["activity"].transpose("time", "id").values
    np.testing.assert_array_equal(aligned.values[:100, 0], raw[split[0] + DAY : split[0] + DAY + 100, 0])


def test_the_dd_heatmap_axis_is_days_in_dd_from_zero(staggered):
    ds, _ = staggered
    view, _ = dam_utilities.select_phase(ds, "DD")
    fig = plotting.dataset_to_heatmap(view, "activity", "DD")
    assert fig.layout.xaxis.title.text.startswith("Days in DD")
    x = np.asarray(fig.data[0].x, dtype=float)
    assert x.min() < 0.1


def test_ld_views_and_whole_datasets_are_not_shifted(staggered):
    ds, _ = staggered
    ld, _ = dam_utilities.select_phase(ds, "LD")
    assert plotting._align_to_dd_start(ld["activity"], ld) is None
    assert plotting._align_to_dd_start(ds["activity"], ds) is None
    title = plotting.dataset_to_heatmap(ld, "activity", "LD").layout.xaxis.title.text
    assert title.startswith("Days elapsed")
