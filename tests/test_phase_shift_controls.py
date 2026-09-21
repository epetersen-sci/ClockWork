"""Which control a group is measured against, and what goes wrong when it is the
wrong one.

The whole point of the matched pairing is that **genotypes differ in baseline
phase**. Referring every group to one cohort reports that genotype difference as
part of the shift, which is not a subtle error: on this fixture it turns a designed
1.5 h delay into 3.5 h, and gives an UNPULSED group a 2 h "shift".

``conftest._build_pulse_cohort`` therefore builds a cohort with a known right answer
— two genotypes 2 h apart in baseline phase, each with a pulsed and an unpulsed arm,
the pulsed arms delayed by 1.5 h from the first DD day — so these tests check a
number against the design rather than against the code's own output.

The other half is the day origin. A ZT pulse is given on the last entrained day, so
under ``day_origin="dd_onset"`` it always falls between day -1 and day 0 — which is
what lets two runs released into DD on different days of their own recordings be
compared at the same free-running age.
"""

import numpy as np
import pytest
from conftest import (
    PULSE_BASELINE_GAP_H,
    PULSE_DD_DAY,
    PULSE_SHIFT_H,
)

import phase_shift as ps

GROUP_BY = ("genotype", "condition")
TOL = 0.2  # hours; the peak is read off a 1-minute trace after a 12 h filter


def _diff(per_day, group, day, col="phase_difference_hours"):
    row = per_day[(per_day["group"] == group) & (per_day["day_index"] == day)]
    assert len(row) == 1, f"expected one row for {group} day {day}, got {len(row)}"
    return float(row[col].iloc[0])


# ------------------------------------------------------- the pairing itself


class TestMatchedControlMap:
    def test_each_group_is_paired_within_its_genotype(self, pulse_ds):
        cmap = ps.build_matched_control_map(pulse_ds, "noLP", group_by=GROUP_BY)
        assert cmap["A_LP"] == "A_noLP"
        assert cmap["B_LP"] == "B_noLP"

    def test_a_control_group_maps_to_itself(self, pulse_ds):
        """Its own difference is 0 by construction, and the page uses this to know
        which groups are references rather than results."""
        cmap = ps.build_matched_control_map(pulse_ds, "noLP", group_by=GROUP_BY)
        assert cmap["A_noLP"] == "A_noLP"
        assert cmap["B_noLP"] == "B_noLP"

    def test_a_group_with_no_matching_control_maps_to_none(self, pulse_ds):
        """Better than silently borrowing another genotype's control — which is the
        entire failure this function exists to prevent."""
        # Drop B's control arm, leaving B_LP unmatched.
        ds = pulse_ds.isel(
            id=[
                i
                for i, (g, c) in enumerate(
                    zip(pulse_ds["genotype"].values, pulse_ds["condition"].values)
                )
                if not (g == "B" and c == "noLP")
            ]
        )
        cmap = ps.build_matched_control_map(ds, "noLP", group_by=GROUP_BY)
        assert cmap["B_LP"] is None
        assert cmap["A_LP"] == "A_noLP"

    def test_two_controls_sharing_a_match_key_is_refused(self, pulse_ds):
        """The reference would be ambiguous, so it raises instead of picking one.
        Grouping on genotype alone makes both flyboxes' noLP arms one key."""
        ds = pulse_ds.assign_coords(
            condition=("id", np.where(pulse_ds["flybox"].values == "bun", "LP", "noLP")),
            block=("id", np.where(np.arange(pulse_ds.sizes["id"]) % 2 == 0, "one", "two")),
        )
        with pytest.raises(ValueError, match="ambiguous|More than one"):
            ps.build_matched_control_map(
                ds, "noLP", group_by=("genotype", "condition", "block"),
                match_on=["genotype"],
            )

    def test_control_on_must_be_a_grouping_column(self, pulse_ds):
        with pytest.raises(ValueError, match="control_on"):
            ps.build_matched_control_map(
                pulse_ds, "noLP", group_by=GROUP_BY, control_on="flybox"
            )

    def test_match_on_must_be_grouping_columns(self, pulse_ds):
        with pytest.raises(ValueError, match="match_on"):
            ps.build_matched_control_map(
                pulse_ds, "noLP", group_by=GROUP_BY, match_on=["flybox"]
            )

    def test_labels_containing_underscores_still_resolve(self, pulse_ds):
        """The pairing reads the coords, not a split of the label — so a genotype
        with a '_' in it cannot be mis-parsed."""
        ds = pulse_ds.assign_coords(
            genotype=("id", np.char.add("ds_Opa1_", pulse_ds["genotype"].values.astype(str)))
        )
        cmap = ps.build_matched_control_map(ds, "noLP", group_by=GROUP_BY)
        assert cmap["ds_Opa1_A_LP"] == "ds_Opa1_A_noLP"


# ----------------------------------------------- recovering a known shift


class TestTheDesignedShiftIsRecovered:
    def test_matched_pairing_recovers_the_pulse_and_nothing_else(self, pulse_ds):
        cmap = ps.build_matched_control_map(pulse_ds, "noLP", group_by=GROUP_BY)
        res = ps.compute_group_phase_difference(
            pulse_ds, cmap, group_by=GROUP_BY, day_origin="dd_onset"
        )
        per_day = res["per_day"]
        for group in ("A_LP", "B_LP"):
            # Before the pulse: the arms share a genotype, so no difference.
            assert abs(_diff(per_day, group, -2)) < TOL
            # After it: the designed delay, for BOTH genotypes.
            assert abs(_diff(per_day, group, 1) - PULSE_SHIFT_H) < TOL

    def test_a_single_control_folds_the_genotype_baseline_into_the_shift(self, pulse_ds):
        """The reason build_matched_control_map exists, measured.

        Against A's control, B's pulsed arm reads pulse + genotype gap, and B's
        UNPULSED arm reads the genotype gap as though it were a shift.
        """
        res = ps.compute_group_phase_difference(
            pulse_ds, "A_noLP", group_by=GROUP_BY, day_origin="dd_onset"
        )
        per_day = res["per_day"]
        assert abs(_diff(per_day, "A_LP", 1) - PULSE_SHIFT_H) < TOL
        assert (
            abs(_diff(per_day, "B_LP", 1) - (PULSE_SHIFT_H + PULSE_BASELINE_GAP_H)) < TOL
        )
        # An unpulsed group reporting a 2 h shift is the tell.
        assert abs(_diff(per_day, "B_noLP", 1) - PULSE_BASELINE_GAP_H) < TOL

    def test_rebasing_removes_the_offset_even_with_one_control(self, pulse_ds):
        """The second, independent defence: zeroing every group on the last day the
        pulse has not touched takes out any constant offset between the cohorts."""
        res = ps.compute_group_phase_difference(
            pulse_ds, "A_noLP", group_by=GROUP_BY, day_origin="dd_onset", baseline_day=-1
        )
        per_day = res["per_day"]
        col = "phase_difference_from_baseline_hours"
        assert abs(_diff(per_day, "B_LP", 1, col) - PULSE_SHIFT_H) < TOL
        assert abs(_diff(per_day, "B_noLP", 1, col)) < TOL
        # And the raw difference is kept alongside, still carrying the offset.
        assert abs(_diff(per_day, "B_noLP", 1) - PULSE_BASELINE_GAP_H) < TOL

    def test_the_baseline_day_itself_is_exactly_zero(self, pulse_ds):
        res = ps.compute_group_phase_difference(
            pulse_ds, "A_noLP", group_by=GROUP_BY, day_origin="dd_onset", baseline_day=-1
        )
        per_day = res["per_day"]
        for group in per_day["group"].unique():
            assert _diff(per_day, group, -1, "phase_difference_from_baseline_hours") == 0.0

    def test_without_a_baseline_day_the_rebased_column_is_all_nan(self, pulse_ds):
        res = ps.compute_group_phase_difference(
            pulse_ds, "A_noLP", group_by=GROUP_BY, day_origin="dd_onset"
        )
        assert res["per_day"]["phase_difference_from_baseline_hours"].isna().all()


class TestDifferenceSign:
    def test_control_minus_group_makes_a_delay_negative(self, pulse_ds):
        """The lab's plotting convention (noLP - LP)."""
        cmap = ps.build_matched_control_map(pulse_ds, "noLP", group_by=GROUP_BY)
        kw = dict(group_by=GROUP_BY, day_origin="dd_onset")
        fwd = ps.compute_group_phase_difference(
            pulse_ds, cmap, difference_sign="group_minus_control", **kw
        )
        rev = ps.compute_group_phase_difference(
            pulse_ds, cmap, difference_sign="control_minus_group", **kw
        )
        assert _diff(fwd["per_day"], "A_LP", 1) > 0
        assert _diff(rev["per_day"], "A_LP", 1) == pytest.approx(
            -_diff(fwd["per_day"], "A_LP", 1)
        )

    def test_an_unknown_sign_is_refused(self, pulse_ds):
        with pytest.raises(ValueError, match="difference_sign"):
            ps.compute_group_phase_difference(
                pulse_ds, "A_noLP", group_by=GROUP_BY, difference_sign="sideways"
            )


# ------------------------------------------------------------ day origin


class TestDayOrigin:
    def test_dd_onset_puts_the_pulse_between_day_minus_one_and_day_zero(self, pulse_ds):
        """A ZT pulse is given on the last entrained day, so under this origin it is
        always at -0.5 — which is why the page can draw the marker without being told
        where the pulse was."""
        res = ps.compute_group_phase_difference(
            pulse_ds, "A_noLP", group_by=GROUP_BY, day_origin="dd_onset"
        )
        days = sorted(res["per_day"]["day_index"].unique())
        assert min(days) == -PULSE_DD_DAY
        assert 0 in days and -1 in days

    def test_recording_start_leaves_the_days_as_recorded(self, pulse_ds):
        res = ps.compute_group_phase_difference(
            pulse_ds, "A_noLP", group_by=GROUP_BY, day_origin="recording_start"
        )
        assert min(res["per_day"]["day_index"]) == 0
        assert all(v == 0 for v in res["origin_day"].values())

    def test_the_two_origins_agree_once_the_offset_is_taken_out(self, pulse_ds):
        """Shifting the origin moves the day index AND the peak hours together. If it
        moved only the index, every difference would gain a whole 24 h."""
        kw = dict(group_by=GROUP_BY)
        start = ps.compute_group_phase_difference(
            pulse_ds, "A_noLP", day_origin="recording_start", **kw
        )
        onset = ps.compute_group_phase_difference(pulse_ds, "A_noLP", day_origin="dd_onset", **kw)
        a = _diff(start["per_day"], "B_LP", PULSE_DD_DAY + 1)
        b = _diff(onset["per_day"], "B_LP", 1)
        assert a == pytest.approx(b, abs=1e-9)

    def test_a_group_spanning_two_dd_release_days_is_refused(self, pulse_ds):
        """Averaged across two release days a group has no single phase to report,
        so it raises rather than picking one."""
        first_dd = pulse_ds["first_DD_day"].values.copy()
        first_dd[0] = first_dd[0] + np.timedelta64(1, "D")
        ds = pulse_ds.assign_coords(first_DD_day=("id", first_dd))
        with pytest.raises(ValueError, match="different days"):
            ps.compute_group_phase_difference(
                ds, "A_noLP", group_by=GROUP_BY, day_origin="dd_onset"
            )

    def test_an_unknown_origin_is_refused(self, pulse_ds):
        with pytest.raises(ValueError, match="day_origin"):
            ps.compute_group_phase_difference(
                pulse_ds, "A_noLP", group_by=GROUP_BY, day_origin="sideways"
            )


# --------------------------------------------------------- group_extras


def test_group_extras_lists_every_box_a_group_spans(pulse_ds):
    """A group pooling two flyboxes is pooling two sub-experiments, and that has to
    stay visible rather than collapse to whichever value came first."""
    labels, _cols = ps.group_labels(pulse_ds, GROUP_BY)
    groups = sorted(set(labels))
    extras = ps.group_extras(pulse_ds, labels, groups, ("flybox",))
    assert extras["A_LP"]["flybox"] == ["bun"]
    assert extras["A_noLP"]["flybox"] == ["pie"]

    pooled = pulse_ds.assign_coords(
        flybox=("id", np.where(np.arange(pulse_ds.sizes["id"]) % 2 == 0, "bun", "tart"))
    )
    extras = ps.group_extras(pooled, labels, groups, ("flybox",))
    assert extras["A_LP"]["flybox"] == ["bun", "tart"]


def test_group_values_records_the_coords_behind_each_label(pulse_ds):
    """So the page can facet on genotype without splitting a label on '_'."""
    res = ps.compute_group_phase_difference(pulse_ds, "A_noLP", group_by=GROUP_BY)
    assert res["group_values"]["A_LP"] == {"genotype": "A", "condition": "LP"}


# ------------------------------------------------------------ bootstrap


class TestBootstrap:
    """The plotted point is the peak of the group's MEAN trace, so there is no
    per-fly spread to turn into a SEM. Resampling the flies and re-running the whole
    pipeline is the only thing that describes the line actually drawn — and it
    carries the control arm's uncertainty as well as the pulsed arm's.
    """

    @staticmethod
    def _run(ds, **kw):
        cmap = ps.build_matched_control_map(ds, "noLP", group_by=GROUP_BY)
        return ps.bootstrap_group_phase_difference(
            ds, cmap, n_boot=12, seed=0, group_by=GROUP_BY, day_origin="dd_onset", **kw
        )

    def test_it_reports_an_interval_and_how_many_draws_worked(self, pulse_ds):
        boot = self._run(pulse_ds)
        assert {"group", "day_index", "n_boot_ok"} <= set(boot.columns)
        assert "phase_difference_boot_sd" in boot.columns
        row = boot[(boot["group"] == "A_LP") & (boot["day_index"] == 1)].iloc[0]
        assert row["n_boot_ok"] > 0
        assert row["phase_difference_lo"] <= row["phase_difference_hi"]

    def test_the_interval_brackets_the_point_estimate(self, pulse_ds):
        """A clean designed shift should sit inside its own bootstrap interval."""
        cmap = ps.build_matched_control_map(pulse_ds, "noLP", group_by=GROUP_BY)
        point = ps.compute_group_phase_difference(
            pulse_ds, cmap, group_by=GROUP_BY, day_origin="dd_onset"
        )
        boot = self._run(pulse_ds)
        est = _diff(point["per_day"], "A_LP", 1)
        row = boot[(boot["group"] == "A_LP") & (boot["day_index"] == 1)].iloc[0]
        assert row["phase_difference_lo"] - TOL <= est <= row["phase_difference_hi"] + TOL

    def test_the_seed_makes_it_reproducible(self, pulse_ds):
        a = self._run(pulse_ds)
        b = self._run(pulse_ds)
        assert a["phase_difference_boot_sd"].equals(b["phase_difference_boot_sd"])

    def test_the_settings_are_recorded_on_the_frame(self, pulse_ds):
        boot = self._run(pulse_ds)
        assert boot.attrs["n_boot"] == 12
        assert boot.attrs["seed"] == 0
        assert boot.attrs["ci"] == 95.0

    def test_one_resample_is_refused(self, pulse_ds):
        """An interval needs a spread; one draw has none."""
        with pytest.raises(ValueError, match="n_boot"):
            ps.bootstrap_group_phase_difference(
                pulse_ds, "A_noLP", n_boot=1, group_by=GROUP_BY
            )

    def test_it_covers_the_rebased_column_too(self, pulse_ds):
        boot = self._run(pulse_ds, baseline_day=-1)
        assert "phase_difference_from_baseline_boot_sd" in boot.columns
        row = boot[(boot["group"] == "A_LP") & (boot["day_index"] == 1)].iloc[0]
        assert np.isfinite(row["phase_difference_from_baseline_boot_sd"])


# --------------------------------------------------------------- the page


def test_the_page_runs_the_matched_comparison(app, pulse_ds):
    """End to end through the real page: the pairing picker defaults to matched (two
    genotypes are present), and Run produces the per-day table."""
    at = app(ds=pulse_ds, page="phase_shift")
    assert not at.exception, f"raised: {at.exception}"
    assert not at.error, [e.value for e in at.error]

    pairing = [r for r in at.radio if "control is each group compared against" in r.label]
    assert pairing, "the pairing picker is missing"
    assert pairing[0].value == "Unpulsed control of the same genotype", (
        "with two genotypes present the matched pairing must be preselected"
    )

    run = [b for b in at.button if "Run Group Phase Comparison" in b.label]
    assert run, "the run button is missing"
    at = run[0].click().run()
    assert not at.exception, f"run raised: {at.exception}"
    assert "phase_shift_group_results" in at.session_state

    res = at.session_state["phase_shift_group_results"]
    per_day = res["per_day"]
    assert res["control_map"]["A_LP"] == "A_noLP"

    # Every fly here shares one DD release day, so the page leaves the origin at the
    # recording start (it only preselects dd_onset for a combined dataset) and zeroes
    # on the pulse day. Post-pulse days are therefore PULSE_DD_DAY onward, not 1.
    params = res["params"]
    assert params["day_origin"] == "recording_start"
    assert params["baseline_day"] == PULSE_DD_DAY - 1
    # The page defaults to the lab's plotting convention, noLP - LP, in which a
    # DELAY READS NEGATIVE. Asserting it here rather than taking abs() keeps the
    # test honest about which way round the page presents the result.
    assert params["difference_sign"] == "control_minus_group"
    expected = -PULSE_SHIFT_H

    col = "phase_difference_from_baseline_hours"
    # The day before the pulse: nothing has happened yet.
    assert abs(_diff(per_day, "A_LP", PULSE_DD_DAY - 1, col)) < TOL
    # Two days after it: the designed delay, through the page's own defaults.
    assert abs(_diff(per_day, "A_LP", PULSE_DD_DAY + 1, col) - expected) < TOL
    assert abs(_diff(per_day, "B_LP", PULSE_DD_DAY + 1, col) - expected) < TOL
    # And the unpulsed arms stay flat, which is what the matched pairing buys.
    assert abs(_diff(per_day, "B_noLP", PULSE_DD_DAY + 1, col)) < TOL


def test_the_page_builds_the_phase_response_from_the_same_click(app, pulse_ds):
    """One button, two answers. The phase response curve is what a pulse experiment
    is run to produce, so it must not be behind a second button somebody has to find.

    This fixture has no pulse-duration column — its unpulsed arm is the string
    ``noLP`` — which is exactly the case that cannot identify a control by "duration
    is zero". It still gets a phase response, because the page hands over the
    pairing it already resolved instead of making core work it out again.
    """
    at = app(ds=pulse_ds, page="phase_shift")
    run = [b for b in at.button if "Run Group Phase Comparison" in b.label]
    at = run[0].click().run()
    assert not at.exception, f"run raised: {at.exception}"

    pres = at.session_state["phase_shift_response"]
    assert pres is not None, "the response was not computed"
    per_fly = pres["per_fly"]
    assert not per_fly.empty
    assert pres["control_map"]["A_LP"] == "A_noLP"

    # The fixture DELAYS the pulsed arms, and a delay reads negative.
    pulsed = per_fly[per_fly["zt"].notna()]
    assert len(pulsed) > 0
    assert pulsed["response_hours"].mean() == pytest.approx(-PULSE_SHIFT_H, abs=0.5)

    # Every fly in the cohort is accounted for: drawn, or counted as left out.
    assert int(pres["dropped"]["n_total"].sum()) == pulse_ds.sizes["id"]


def test_one_pulse_time_says_so_instead_of_drawing_a_curve(app, pulse_ds):
    """The lab's own data is currently single-timepoint, so this is the message most
    people will meet first. A one-point "curve" would imply a shape nobody measured."""
    at = app(ds=pulse_ds, page="phase_shift")
    run = [b for b in at.button if "Run Group Phase Comparison" in b.label]
    at = run[0].click().run()
    assert not at.exception
    assert any("one point is not a curve" in i.value for i in at.info), (
        [i.value for i in at.info]
    )


def test_the_page_stops_cleanly_without_a_pulse_column(app, master_ds):
    """No pulse_time in the metadata is a supported state: the page must say what to
    add rather than crash."""
    at = app(ds=master_ds, page="phase_shift")
    assert not at.exception
    assert any("pulse_time" in e.value for e in at.error)
