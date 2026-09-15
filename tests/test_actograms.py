"""SCAMP actogram geometry: binning, double-plotting, stacking and the LD mask.

All pure numpy, so tested directly rather than through AppTest (see tests/README.md).
The one page-level check at the bottom is here rather than in test_page_behaviour.py
because what it exercises is this module's mask reaching the bars.

The invariant worth naming: **a bin nobody measured is NaN, never 0.** Zero is a
real reading meaning "the flies were still" (§2a), so the difference decides whether
a gap draws a flat floor or nothing at all.
"""

import json

import numpy as np
import pytest
import xarray as xr

import actograms as act

MPD = act.MINUTES_PER_DAY


def _ds(n_days=3, n_id=4, *, genotypes=None, first_dd_offsets=None, fill=1.0):
    """Minimal actogram input: integer-minute axis, (time, id) activity."""
    n_time = n_days * MPD
    genotypes = genotypes or ["a"] * (n_id // 2) + ["b"] * (n_id - n_id // 2)
    activity = np.full((n_time, n_id), fill, dtype=float)
    coords = {
        "id": [f"f{i}" for i in range(n_id)],
        "time": np.arange(n_time, dtype=np.int64),
        "genotype": ("id", np.array(genotypes)),
    }
    if first_dd_offsets is not None:
        coords["split_minute"] = ("id", np.asarray(first_dd_offsets, dtype=float))
    return xr.Dataset({"activity": (("time", "id"), activity)}, coords=coords)


# ---------------------------------------------------------------- binning


def test_bin_sums_within_bin_and_means_over_flies():
    """30-minute bins of activity 1.0 sum to 30; the group mean of equal flies is 30."""
    res = act.compute_group_actograms(_ds(), ("genotype",), bin_minutes=30)
    assert res["_params"] == {"bin_minutes": 30, "bins_per_day": 48, "group_by": ["genotype"]}
    assert res["a"]["n_flies"] == 2
    assert res["a"]["matrix"].shape == (3, 48)
    assert np.allclose(res["a"]["matrix"], 30.0)


def test_unmeasured_bin_stays_nan_not_zero():
    """A bin with no reading in ANY fly of the group is NaN — the §2a distinction."""
    ds = _ds(n_days=1, n_id=2)
    ds["activity"][60:120, :] = np.nan  # the whole second hour, both flies
    res = act.compute_group_actograms(ds, ("genotype",), bin_minutes=30)
    row = res["a"]["matrix"][0]
    assert np.isnan(row[2]) and np.isnan(row[3])  # bins 2,3 = minutes 60-120
    assert np.isfinite(row[0]) and np.isfinite(row[4])


def test_partially_measured_bin_averages_only_the_flies_measured():
    """One fly missing a bin must not drag the group mean toward zero."""
    ds = _ds(n_days=1, n_id=2, genotypes=["a", "a"])
    ds["activity"][0:30, 0] = np.nan  # fly 0 misses bin 0 entirely
    res = act.compute_group_actograms(ds, ("genotype",), bin_minutes=30)
    # Only fly 1 measured bin 0, and it summed to 30 — not 15, which is what
    # treating the missing fly as a zero would give.
    assert res["a"]["matrix"][0, 0] == pytest.approx(30.0)


def test_non_integer_time_axis_is_refused():
    ds = _ds(n_days=1)
    ds = ds.assign_coords(time=ds["time"].values.astype("datetime64[m]"))
    with pytest.raises(ValueError, match="relative-integer-minute"):
        act.compute_group_actograms(ds, ("genotype",))


def test_bin_width_must_divide_the_day():
    with pytest.raises(ValueError, match="must divide 1440"):
        act.compute_group_actograms(_ds(n_days=1), ("genotype",), bin_minutes=7)


def test_missing_activity_var_is_refused():
    with pytest.raises(KeyError):
        act.compute_group_actograms(_ds(n_days=1), ("genotype",), activity_var="nope")


def test_incomplete_final_day_is_dropped_not_padded():
    """SCAMP zero-pads a partial day; we drop it, because a padded day draws a floor."""
    ds = _ds(n_days=2, n_id=2)
    ds = ds.isel(time=slice(0, 2 * MPD - 100))  # 100 minutes short of day 2
    res = act.compute_group_actograms(ds, ("genotype",), bin_minutes=30)
    assert res["a"]["matrix"].shape[0] == 1


# ---------------------------------------------------------- double-plotting


def test_double_plot_puts_the_next_day_in_the_second_half():
    matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    rows = act.actogram_rows(matrix, reps=2)
    assert rows.shape == (3, 4)
    np.testing.assert_array_equal(rows[0], [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(rows[1], [3.0, 4.0, 5.0, 6.0])


def test_last_rows_tail_is_nan_because_no_day_follows():
    """SCAMP writes zeros here, which would imply measured inactivity."""
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    rows = act.actogram_rows(matrix, reps=2)
    np.testing.assert_array_equal(rows[1][:2], [3.0, 4.0])
    assert np.isnan(rows[1][2:]).all()


def test_reps_one_is_a_plain_single_plot():
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(act.actogram_rows(matrix, reps=1), matrix)


# --------------------------------------------------------------- stacking


def test_scale_is_scamps_ten_percent_headroom():
    matrix = np.array([[0.0, 10.0], [2.0, 8.0]])
    vmin, shift = act.actogram_scale(matrix)
    assert vmin == 0.0
    assert shift == pytest.approx(1.1 * 10.0)


def test_scale_ignores_nan_and_survives_a_flat_panel():
    vmin, shift = act.actogram_scale(np.array([[5.0, np.nan], [5.0, 5.0]]))
    assert vmin == 5.0
    assert shift == 1.0  # a zero range would stack every row on top of the last

    vmin, shift = act.actogram_scale(np.full((2, 2), np.nan))
    assert (vmin, shift) == (0.0, 1.0)


# --------------------------------------------------------------- LD mask


def test_ld_mask_marks_whole_days_before_the_release():
    """split at minute 1440 = every bin of day 0 is LD, none of day 1."""
    mask = act.ld_bin_mask(2, 48, 2, 30, split_minute=1440)
    assert mask.shape == (2, 96)
    assert mask[0, :48].all()  # row 0 first half is day 0
    assert not mask[0, 48:].any()  # row 0 second half is day 1, already DD
    assert not mask[1].any()


def test_ld_mask_leaves_the_straddling_bin_uncoloured():
    """A bin counts as LD only when its WHOLE span precedes the release."""
    # Release 15 minutes into day 1: the 1440-1470 bin contains it.
    mask = act.ld_bin_mask(2, 48, 2, 30, split_minute=1455)
    assert mask[0, 47]  # 1410-1440, wholly before
    assert not mask[0, 48]  # 1440-1470, contains the transition

    # A bin ending exactly at the release is wholly before it, so it IS LD.
    assert act.ld_bin_mask(2, 48, 2, 30, split_minute=1470)[0, 48]


def test_ld_mask_is_per_bin_so_one_row_can_straddle():
    """The reason the mask is not per row: with reps=2 a row spans two days."""
    mask = act.ld_bin_mask(3, 48, 2, 30, split_minute=2 * MPD)
    # Row 1 = day 1 (LD) then day 2 (DD) — mixed within a single row.
    assert mask[1, :48].all()
    assert not mask[1, 48:].any()


def test_ld_mask_none_and_inf_are_the_two_degenerate_partitions():
    assert not act.ld_bin_mask(2, 48, 2, 30, None).any()  # no boundary known
    assert act.ld_bin_mask(2, 48, 2, 30, np.inf).all()  # LD-only partition
    assert not act.ld_bin_mask(2, 48, 2, 30, np.nan).any()


# ------------------------------------------------- per-group boundaries


def test_group_split_takes_the_earliest_fly_and_flags_disagreement():
    """Conservative direction: an LD bin can be missed, never invented."""
    ds = _ds(
        n_days=3,
        n_id=4,
        genotypes=["a", "a", "b", "b"],
        first_dd_offsets=[1440, 2880, 1440, 1440],
    )
    splits, disagree = act.group_split_minutes(ds, ("genotype",))
    assert splits == {"a": 1440.0, "b": 1440.0}
    assert disagree == {"a"}  # a's two flies were released on different days


def test_group_split_is_empty_without_phase_metadata():
    splits, disagree = act.group_split_minutes(_ds(n_days=2), ("genotype",))
    assert (splits, disagree) == ({}, set())


def test_group_split_reports_none_for_a_group_with_no_finite_boundary():
    ds = _ds(
        n_days=2, n_id=2, genotypes=["a", "b"], first_dd_offsets=[np.nan, 1440]
    )
    splits, disagree = act.group_split_minutes(ds, ("genotype",))
    assert splits == {"a": None, "b": 1440.0}
    assert disagree == set()


# ------------------------------------------------------------ the page


def test_page_colours_ld_bins_when_asked(app, master_ds):
    """Ticking "Colour LD days" reaches the bars: the fixture's LD days come out
    in the LD colour, and the legend gains the two stand-in entries."""
    at = app(ds=master_ds, page="actograms")
    assert not at.exception
    at.session_state["actogram_mark_ld"] = True
    at = at.run()
    assert not at.exception

    colours = set()
    for el in at.get("plotly_chart"):
        for trace in json.loads(el.proto.spec).get("data", []):
            colour = trace.get("marker", {}).get("color")
            colours.update(colour if isinstance(colour, list) else [colour])
    assert "#FF8C00" in colours, "no bin was drawn in the LD colour"
    assert "#0000CD" in colours, "no bin was drawn in the DD colour"
