"""Bout-structure tables: binning, the Figure 6 definition ladder, and the seam.

``core/bout_spectrum`` re-reads the per-bout table ``sleep_analysis`` already wrote
rather than re-detecting anything, so what these tests pin is the bookkeeping on top
of it — and the bookkeeping is where the quiet mistakes live:

* a bout is credited to the day it STARTS in, so one running past midnight is
  counted once and never split;
* every fly gets a row for every bin, so a fly with no long bouts contributes a
  real 0 to the group mean instead of dropping out and biasing it upward;
* the ``>=X min`` definitions NEST, so they are not additive, and a percentage of
  them has to be taken against the fly's own total sleep;
* the denominator for "per day" is the fly's VALID record over the selected days
  only — not the length of the recording.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
import xarray as xr

import bout_spectrum as bs
import plotting

MPD = 1440


def _ds(durations, starts, *, groups=("a", "b"), n_days=3, invalid_minutes=0, relative=True):
    """Dataset carrying the per-bout table, shaped as ``sleep_analysis`` writes it.

    ``durations`` / ``starts`` are one list per fly; the rows are NaN-padded to the
    longest, which is what a rectangular ``(id, sleep_bout_number)`` array does to a
    fly with fewer bouts.
    """
    n_id = len(durations)
    width = max((len(d) for d in durations), default=0)
    dur = np.full((n_id, width), np.nan)
    sta = np.full((n_id, width), np.nan)
    for i, (d, s) in enumerate(zip(durations, starts)):
        dur[i, : len(d)] = d
        sta[i, : len(s)] = s

    n_time = n_days * MPD
    # sleep == -1 marks "no reading" — per_fly_recording_days must not count it.
    sleep = np.zeros((n_id, n_time), dtype=np.int8)
    if invalid_minutes:
        sleep[:, :invalid_minutes] = -1

    time = np.arange(n_time, dtype=np.int64)
    if not relative:
        time = np.datetime64("2025-01-15T09:00:00") + time.astype("timedelta64[m]")

    return xr.Dataset(
        {
            "duration": (("id", "sleep_bout_number"), dur),
            "start_time": (("id", "sleep_bout_number"), sta),
            "sleep": (("id", "time"), sleep),
        },
        coords={
            "id": [f"f{i}" for i in range(n_id)],
            "sleep_bout_number": np.arange(1, width + 1),
            "time": time,
            "group": ("id", np.array([groups[i % len(groups)] for i in range(n_id)])),
        },
        attrs={"sleep_threshold_seconds": 300},
    )


# --------------------------------------------------------------------- bins


def test_bins_are_half_open_with_an_unbounded_top():
    assert bs.make_bins([5, 10, 20]) == [(5.0, 10.0), (10.0, 20.0), (20.0, np.inf)]
    assert bs.make_bins([5]) == [(5.0, np.inf)]


def test_bin_edges_must_be_increasing_and_non_negative():
    for bad in ([10, 5], [5, 5], [-1, 5]):
        with pytest.raises(ValueError):
            bs.make_bins(bad)
    with pytest.raises(ValueError):
        bs.make_bins([])


def test_bin_labels_name_the_open_bin_differently():
    assert bs.bin_label(5, 10) == "5-10 min"
    assert bs.bin_label(240, np.inf) == "240+ min"
    assert bs.bin_label(240, None) == "240+ min"


def test_edges_are_parsed_from_commas_or_spaces():
    assert bs.parse_edges("5, 10 20,30") == [5.0, 10.0, 20.0, 30.0]
    with pytest.raises(ValueError, match="must be numbers"):
        bs.parse_edges("5, ten")
    with pytest.raises(ValueError, match="increasing"):
        bs.parse_edges("20, 10")
    with pytest.raises(ValueError, match="No bin edges"):
        bs.parse_edges("   ")


def test_the_definition_ladder_is_figure_sixes():
    defs = bs.default_definitions()
    labels = [lbl for lbl, _, _ in defs]
    assert labels[:7] == [">=5 min", ">=10 min", ">=20 min", ">=30 min",
                          ">=40 min", ">=50 min", ">=60 min"]
    assert labels[7:] == ["5-10 min", "5-20 min", "5-30 min",
                          "30-60 min", "30-120 min", "60-120 min"]
    # The minimums are open-topped and the intervals are not — the page splits
    # the two families on exactly this.
    assert all(not np.isfinite(hi) for _, _, hi in defs[:7])
    assert all(np.isfinite(hi) for _, _, hi in defs[7:])


def test_the_resolution_floor_comes_from_the_sleep_threshold():
    ds = _ds([[10.0]], [[0.0]])
    assert bs.min_definable_minutes(ds) == 5.0
    ds.attrs["sleep_threshold_seconds"] = 1200
    assert bs.min_definable_minutes(ds) == 20.0
    # A dataset written before the attr existed is assumed to be the standard 5.
    assert bs.min_definable_minutes(_ds([[10.0]], [[0.0]]).drop_attrs()) == 5.0


# --------------------------------------------------------------- bout table


def test_a_bout_is_credited_to_the_day_it_starts_in():
    """A bout beginning 10 minutes before midnight on day 0 and running into day 1
    belongs to day 0, once. Splitting it would double-count the bout and halve
    neither day's honestly."""
    ds = _ds([[30.0]], [[MPD - 10]])
    table = bs.bout_table(ds)
    assert list(table["day"]) == [0]
    assert list(bs.bout_table(ds, days=[0])["day"]) == [0]
    assert bs.bout_table(ds, days=[1]).empty


def test_the_day_filter_keeps_only_bouts_starting_on_those_days():
    ds = _ds([[10.0, 20.0, 30.0]], [[0.0, MPD + 5, 2 * MPD + 5]])
    assert sorted(bs.bout_table(ds, days=[0, 2])["duration"]) == [10.0, 30.0]
    assert len(bs.bout_table(ds)) == 3


def test_nan_padding_is_dropped_rather_than_counted_as_a_bout():
    """Two flies with different bout counts share one rectangular array."""
    ds = _ds([[10.0, 20.0], [30.0]], [[0.0, 100.0], [0.0]])
    table = bs.bout_table(ds)
    assert len(table) == 3
    assert sorted(table["duration"]) == [10.0, 20.0, 30.0]


def test_no_bout_table_gives_an_empty_frame_with_the_right_columns():
    ds = _ds([[10.0]], [[0.0]]).drop_vars("duration")
    out = bs.bout_table(ds)
    assert out.empty
    assert list(out.columns) == ["ID", "Group", "sleep_bout_number", "duration",
                                "start_time", "day"]


def test_day_selection_is_refused_on_a_datetime_axis():
    """Days cannot be numbered without the relative-minute axis, and quietly
    ignoring the request would return every bout under a day-filtered title."""
    ds = _ds([[10.0]], [[0.0]], relative=False)
    with pytest.raises(ValueError, match="relative-integer-minute"):
        bs.bout_table(ds, days=[0])
    assert np.isnan(bs.bout_table(ds)["day"]).all()


# ------------------------------------------------------- recording days


def test_recording_days_exclude_unmeasured_minutes():
    """sleep == -1 is "no reading", so it must not inflate the denominator that
    turns bout counts into bouts per day."""
    ds = _ds([[10.0]], [[0.0]], n_days=3)
    assert bs.per_fly_recording_days(ds)["f0"] == pytest.approx(3.0)

    ds = _ds([[10.0]], [[0.0]], n_days=3, invalid_minutes=MPD)
    assert bs.per_fly_recording_days(ds)["f0"] == pytest.approx(2.0)


def test_recording_days_can_be_restricted_to_the_selected_days():
    """Otherwise a two-day selection is still divided by the whole record."""
    ds = _ds([[10.0]], [[0.0]], n_days=5)
    assert bs.per_fly_recording_days(ds, days=[0, 1])["f0"] == pytest.approx(2.0)


# ------------------------------------------------------------ bout counts


def test_every_fly_gets_a_row_for_every_bin_including_zero():
    """A fly with no long bouts must contribute a real 0, not drop out."""
    ds = _ds([[6.0, 7.0], [300.0]], groups=("a", "a"), starts=[[0.0, 100.0], [0.0]])
    out = bs.per_fly_bout_counts(ds, bins=bs.make_bins([5, 10, 240]))
    assert len(out) == 2 * 3  # two flies x three bins
    f1_top = out[(out.ID == "f1") & (out.bin_label == "240+ min")]
    f0_top = out[(out.ID == "f0") & (out.bin_label == "240+ min")]
    assert f1_top["n_bouts"].iloc[0] == 1
    assert f0_top["n_bouts"].iloc[0] == 0, "the fly with no long bouts must still be here"


def test_bin_membership_is_lo_inclusive_hi_exclusive():
    ds = _ds([[10.0]], [[0.0]])
    out = bs.per_fly_bout_counts(ds, bins=bs.make_bins([5, 10, 20]))
    by_bin = dict(zip(out.bin_label, out.n_bouts))
    assert by_bin["5-10 min"] == 0
    assert by_bin["10-20 min"] == 1


def test_bout_counts_carry_both_the_count_and_the_minutes():
    ds = _ds([[12.0, 15.0]], [[0.0, 100.0]])
    out = bs.per_fly_bout_counts(ds, bins=bs.make_bins([10, 20]))
    row = out[out.bin_label == "10-20 min"].iloc[0]
    assert row["n_bouts"] == 2
    assert row["sleep_minutes"] == 27.0


def test_per_day_divides_by_the_flys_own_valid_record():
    ds = _ds([[12.0, 15.0]], [[0.0, 100.0]], n_days=3)
    out = bs.per_fly_bout_counts(ds, bins=bs.make_bins([10, 20]), per_day=True)
    row = out[out.bin_label == "10-20 min"].iloc[0]
    assert row["n_bouts"] == pytest.approx(2 / 3)
    assert row["sleep_minutes"] == pytest.approx(27.0 / 3)


def test_percent_of_total_sums_to_a_hundred_per_fly():
    """The duration bins partition the bouts exactly once, so they must."""
    ds = _ds([[6.0, 15.0, 300.0]], [[0.0, 100.0, 200.0]])
    out = bs.per_fly_bout_counts(ds, bins=bs.make_bins([5, 10, 240]), as_percent=True)
    assert out["n_bouts"].sum() == pytest.approx(100.0)
    assert out["sleep_minutes"].sum() == pytest.approx(100.0)


# -------------------------------------------------- sleep by definition


def test_minimum_definitions_nest_and_are_not_additive():
    """>=5 min contains >=30 min, so their sum means nothing — which is exactly
    why as_percent below divides by the fly's total rather than by this sum."""
    ds = _ds([[6.0, 40.0, 100.0]], [[0.0, 100.0, 200.0]])
    out = bs.per_fly_sleep_by_definition(ds)
    mins = dict(zip(out.definition, out.sleep_minutes))
    assert mins[">=5 min"] == 146.0
    assert mins[">=10 min"] == 140.0
    assert mins[">=30 min"] == 140.0
    assert mins[">=60 min"] == 100.0
    assert mins[">=5 min"] >= mins[">=10 min"] >= mins[">=30 min"] >= mins[">=60 min"]


def test_interval_definitions_are_the_disjoint_ones():
    ds = _ds([[6.0, 40.0, 100.0]], [[0.0, 100.0, 200.0]])
    out = bs.per_fly_sleep_by_definition(ds)
    mins = dict(zip(out.definition, out.sleep_minutes))
    assert mins["5-10 min"] == 6.0
    assert mins["30-60 min"] == 40.0
    assert mins["60-120 min"] == 100.0
    # 5-10, 30-60 and 60-120 do not overlap, and together cover every bout here.
    assert mins["5-10 min"] + mins["30-60 min"] + mins["60-120 min"] == 146.0


def test_percent_by_definition_is_against_the_flys_total_sleep():
    """Not against the sum over definitions, which the nesting makes meaningless.
    ``>=5 min`` keeps everything, so it must read exactly 100%."""
    ds = _ds([[6.0, 40.0, 100.0]], [[0.0, 100.0, 200.0]])
    out = bs.per_fly_sleep_by_definition(ds, as_percent=True)
    mins = dict(zip(out.definition, out.sleep_minutes))
    assert mins[">=5 min"] == pytest.approx(100.0)
    assert mins[">=60 min"] == pytest.approx(100.0 / 146.0 * 100.0)
    assert out["sleep_minutes"].sum() > 100.0, "nesting means these do not sum to 100"


# ----------------------------------------------------------- aggregation


def test_group_summary_reports_mean_sem_and_the_n_it_used():
    per_fly = pd.DataFrame({
        "ID": ["a1", "a2", "a3"],
        "Group": ["a", "a", "a"],
        "bin_label": ["5-10 min"] * 3,
        "n_bouts": [1.0, 2.0, 3.0],
    })
    stat = bs.summarize_by_group(per_fly, "n_bouts", "bin_label")
    assert stat["mean"].iloc[0] == pytest.approx(2.0)
    assert stat["n"].iloc[0] == 3
    assert stat["sem"].iloc[0] == pytest.approx(1.0 / np.sqrt(3))


def test_a_single_fly_gets_no_error_bar_rather_than_a_zero_one():
    """0 claims the flies agreed; NaN says one value has no standard error. Same
    rule as scamp_sleep.mean_sem, and §2a's "missing is never zero"."""
    per_fly = pd.DataFrame({
        "ID": ["a1"], "Group": ["a"], "bin_label": ["5-10 min"], "n_bouts": [4.0],
    })
    stat = bs.summarize_by_group(per_fly, "n_bouts", "bin_label")
    assert stat["n"].iloc[0] == 1
    assert np.isnan(stat["sem"].iloc[0])


def test_group_summary_honours_the_category_order():
    per_fly = pd.DataFrame({
        "ID": ["a1", "a1"], "Group": ["a", "a"],
        "bin_label": ["10-20 min", "5-10 min"], "n_bouts": [1.0, 2.0],
    })
    stat = bs.summarize_by_group(per_fly, "n_bouts", "bin_label", ["5-10 min", "10-20 min"])
    assert list(stat["bin_label"]) == ["5-10 min", "10-20 min"]


def test_empty_input_gives_the_declared_columns_not_a_crash():
    stat = bs.summarize_by_group(None, "n_bouts", "bin_label")
    assert stat.empty and list(stat.columns) == ["group", "bin_label", "mean", "sem", "n"]
    assert bs.wide_by_group(None, "n_bouts", "bin_label").empty


def test_wide_form_is_one_row_per_fly_in_category_order():
    per_fly = pd.DataFrame({
        "ID": ["a1", "a1", "a2", "a2"],
        "Group": ["a", "a", "a", "a"],
        "bin_label": ["5-10 min", "10-20 min"] * 2,
        "n_bouts": [1.0, 2.0, 3.0, 4.0],
    })
    wide = bs.wide_by_group(per_fly, "n_bouts", "bin_label", ["5-10 min", "10-20 min"])
    assert list(wide.columns) == ["ID", "Group", "5-10 min", "10-20 min"]
    assert len(wide) == 2


# ----------------------------------------------------- the renderer seam


class TestBarsTakeTheSummary:
    """It renders what it is given (backlog item 7).

    It arrived taking the long per-fly frame and calling summarize_by_group behind
    a deferred ``import bout_spectrum`` — the inversion item 7 removed from this
    module. The page now summarises once and hands the same object to the chart and
    to the CSV, so the two cannot be separate computations that disagree.
    """

    @staticmethod
    def _stat(sem=(0.5, 0.25)):
        return pd.DataFrame({
            "group": ["a", "a", "b", "b"],
            "bin_label": ["5-10 min", "10-20 min"] * 2,
            "mean": [1.0, 2.0, 3.0, 4.0],
            "sem": [sem[0], sem[1], sem[0], sem[1]],
            "n": [6, 6, 6, 6],
        })

    def test_signature_takes_no_per_fly_frame(self):
        import inspect

        params = list(inspect.signature(plotting.bout_spectrum_bars).parameters)
        assert params[0] == "stat_df"
        assert "value_col" not in params, (
            "a value column implies the renderer is aggregating"
        )

    def test_returns_only_a_figure(self):
        out = plotting.bout_spectrum_bars(self._stat(), "bin_label")
        assert isinstance(out, go.Figure), (
            "it returned (fig, summary) so the caller could export what it had "
            "computed internally; the caller now owns the summary"
        )

    def test_draws_one_trace_per_group_with_the_n_in_its_name(self):
        fig = plotting.bout_spectrum_bars(self._stat(), "bin_label")
        assert [t.name for t in fig.data] == ["a (n=6)", "b (n=6)"]

    def test_chart_style_switches_bars_for_lines(self):
        bars = plotting.bout_spectrum_bars(self._stat(), "bin_label", chart="bar")
        lines = plotting.bout_spectrum_bars(self._stat(), "bin_label", chart="line")
        assert all(t.type == "bar" for t in bars.data)
        assert all(t.type == "scatter" for t in lines.data)

    def test_a_nan_sem_passes_through_as_no_error_bar(self):
        """nan_to_num here would redraw the zero-length bar summarize_by_group
        deliberately stopped reporting."""
        fig = plotting.bout_spectrum_bars(self._stat(sem=(np.nan, 0.25)), "bin_label")
        assert np.isnan(fig.data[0].error_y["array"][0])

    def test_category_order_pins_the_axis_so_an_empty_slot_is_kept(self):
        """The page inserts a blank spacer between the nesting and disjoint
        definition families; it carries no rows but must keep its slot."""
        order = ["5-10 min", " ", "10-20 min"]
        fig = plotting.bout_spectrum_bars(self._stat(), "bin_label", category_order=order)
        assert list(fig.layout.xaxis.categoryarray) == order

    def test_the_axis_claims_the_margin_its_rotated_labels_need(self):
        """Fourteen definition labels make plotly turn the ticks vertical, and
        without automargin they are drawn into the existing bottom margin and
        overprint the axis title — which is what the exported PNG showed."""
        fig = plotting.bout_spectrum_bars(
            self._stat(), "bin_label",
            category_order=[lbl for lbl, _, _ in bs.default_definitions()],
        )
        assert fig.layout.xaxis.automargin is True
        assert fig.layout.xaxis.tickangle < 0

    def test_empty_and_none_give_the_placeholder_not_a_crash(self):
        for empty in (None, pd.DataFrame()):
            fig = plotting.bout_spectrum_bars(empty, "bin_label")
            assert not fig.data
            assert len(fig.layout.annotations) == 1


# --------------------------------------------------------------- the page


def test_page_draws_both_views_from_the_real_bout_table(app, states_ds):
    """states_ds runs the production sleep_analysis, so the bout table the page
    reads is the one the app actually writes."""
    at = app(ds=states_ds, page="sleep_bouts")
    assert not at.exception, f"raised: {at.exception}"
    assert len(at.get("plotly_chart")) == 2, "expected the spectrum and the ladder"
    assert not at.error, [e.value for e in at.error]


def test_page_stops_cleanly_when_there_are_no_bouts(app, master_ds):
    """master_ds has sleep masks but no per-bout table — a supported state that
    must warn and stop rather than draw empty axes."""
    at = app(ds=master_ds, page="sleep_bouts")
    assert not at.exception
    assert at.warning, "expected a warning naming what to run"
