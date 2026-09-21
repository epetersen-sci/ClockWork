"""The phase response curve: one number per fly, and where it comes from.

The quantity, so the graphs cannot disagree about it::

    response(fly) = mean over the response days of
                    (that day's control-group mean peak time - this fly's peak time)
                    - the same difference on the baseline day

Control minus fly, so an **advance reads positive** and a delay negative — the
convention a PRC is drawn in.

Two properties make this worth computing per fly rather than per group, and both
are tested below: the mean of a group's per-fly responses is *exactly* the
group-level response, so a violin's centre is the group effect and its spread is
real between-fly variation; and the shared subtracted constant cancels when two
groups are compared, so comparing genotypes is sound.

What is NOT sound, and no test here implies it, is testing one group against zero
and reading that as a test against the control — that ignores the uncertainty in
the control mean itself.
"""

import numpy as np
import pandas as pd
import pytest
from conftest import (
    PULSE_DD_DAY,
    PULSE_SHIFT_H,
    _build_pulse_cohort,
)

import phase_shift as ps

GROUP_BY = ("genotype", "pulse_zt_hour", "pulse_duration_minutes")


@pytest.fixture(scope="module")
def prc_ds():
    """The known-design cohort, with the pulse columns a PRC needs.

    ``_build_pulse_cohort`` already encodes a designed advance; this adds the
    duration coord that marks the unpulsed arm, which is what identifies a control
    now that no string is matched against.
    """
    ds = _build_pulse_cohort()
    zt = np.asarray(ds["pulse_zt_hour"].values, dtype=float)
    # Duration 0 exactly where there is no pulse — that IS the control definition.
    return ds.assign_coords(
        pulse_duration_minutes=("id", np.where(np.isfinite(zt), 20.0, 0.0).astype("float32"))
    )


class TestControlsComeFromThePulse:
    def test_a_zero_duration_group_is_the_control(self, prc_ds):
        cmap, controls = ps.control_map_from_pulse(prc_ds, GROUP_BY)
        assert len(controls) == 2, "one control per genotype"
        assert all("0.0" in c for c in controls)

    def test_each_group_pairs_within_its_genotype(self, prc_ds):
        cmap, _ = ps.control_map_from_pulse(prc_ds, GROUP_BY)
        for grp, ctrl in cmap.items():
            assert ctrl is not None
            assert grp.split("_")[0] == ctrl.split("_")[0], f"{grp} -> {ctrl}"

    def test_a_control_maps_to_itself(self, prc_ds):
        cmap, controls = ps.control_map_from_pulse(prc_ds, GROUP_BY)
        for c in controls:
            assert cmap[c] == c

    def test_no_string_matching_is_involved(self, prc_ds):
        """The point of the rewrite. Renaming every condition label to nonsense must
        not change which groups are controls — only the duration decides."""
        renamed = prc_ds.assign_coords(
            condition=("id", np.array(["wibble"] * prc_ds.sizes["id"]))
        )
        before, _ = ps.control_map_from_pulse(prc_ds, GROUP_BY)
        after, _ = ps.control_map_from_pulse(renamed, GROUP_BY)
        assert before == after

    def test_one_control_serves_several_pulses_of_the_same_genotype(self, prc_ds):
        """Matching is on the group columns EXCEPT the pulse ones, so a genotype's
        20-minute and 60-minute arms share its single unpulsed cohort."""
        dur = np.asarray(prc_ds["pulse_duration_minutes"].values, dtype=float)
        gen = np.asarray(prc_ds["genotype"].values).astype(str)
        # Make half of each genotype's pulsed flies a 60-minute arm.
        new = dur.copy()
        pulsed = np.flatnonzero(dur > 0)
        new[pulsed[::2]] = 60.0
        ds = prc_ds.assign_coords(pulse_duration_minutes=("id", new.astype("float32")))
        cmap, controls = ps.control_map_from_pulse(ds, GROUP_BY)
        for gene in set(gen.tolist()):
            arms = [g for g in cmap if g.startswith(gene) and cmap[g] != g]
            assert len(arms) == 2, f"{gene} should have a 20 and a 60 arm: {arms}"
            assert len({cmap[a] for a in arms}) == 1, "both must share one control"

    def test_a_missing_duration_coord_says_what_to_add(self, prc_ds):
        with pytest.raises(ValueError, match="pulse-duration column"):
            ps.control_map_from_pulse(prc_ds.drop_vars("pulse_duration_minutes"), GROUP_BY)

    def test_a_group_with_no_control_maps_to_none(self, prc_ds):
        """Dropping one genotype's control must orphan that genotype, not let it
        borrow another's."""
        gen = np.asarray(prc_ds["genotype"].values).astype(str)
        dur = np.asarray(prc_ds["pulse_duration_minutes"].values, dtype=float)
        keep = ~((gen == "B") & (dur == 0))
        cmap, _ = ps.control_map_from_pulse(prc_ds.isel(id=np.flatnonzero(keep)), GROUP_BY)
        assert any(g.startswith("B") and c is None for g, c in cmap.items())


class TestTheResponseItself:
    def test_it_recovers_the_designed_shift(self, prc_ds):
        """The fixture delays the pulsed arms by PULSE_SHIFT_H. A delay is negative
        under this convention, so the response should be about -PULSE_SHIFT_H."""
        res = ps.compute_phase_response(prc_ds, group_by=GROUP_BY)
        summ = ps.summarize_phase_response(res["per_fly"])
        pulsed = summ[summ["zt"].notna()]
        assert len(pulsed) == 2
        for _, row in pulsed.iterrows():
            assert row["mean"] == pytest.approx(-PULSE_SHIFT_H, abs=0.4), row.to_dict()

    def test_controls_centre_on_zero_and_keep_their_spread(self, prc_ds):
        """Each control fly is measured against its own group's mean, so the control
        violin sits at zero by construction — and its SD is the honest noise floor
        for reading a per-fly phase at all."""
        res = ps.compute_phase_response(prc_ds, group_by=GROUP_BY)
        summ = ps.summarize_phase_response(res["per_fly"])
        ctrl = summ[summ["zt"].isna()]
        assert len(ctrl) == 2
        assert np.allclose(ctrl["mean"].to_numpy(), 0.0, atol=1e-9)
        assert (ctrl["n"] > 1).all()

    def test_the_group_mean_of_per_fly_values_is_the_group_response(self, prc_ds):
        """The property that lets one computation serve both graphs: the mean of the
        violin IS the point on the line."""
        res = ps.compute_phase_response(prc_ds, group_by=GROUP_BY)
        pf = res["per_fly"]
        summ = ps.summarize_phase_response(pf)
        for _, row in summ.iterrows():
            sel = pf[pf["group"] == row["group"]]["response_hours"]
            assert row["mean"] == pytest.approx(float(sel.mean()))

    def test_an_advance_is_positive(self, prc_ds):
        """Sign check against a cohort built to ADVANCE rather than delay."""
        import xarray as xr

        ds = prc_ds.copy(deep=True)
        # np.array, not asarray: asarray would hand back a view into the fixture.
        act = np.array(ds["activity"].transpose("time", "id").values, dtype=float)
        zt = np.asarray(ds["pulse_zt_hour"].values, dtype=float)
        # Roll the pulsed flies' post-pulse activity EARLIER by two hours.
        start = (PULSE_DD_DAY + 1) * 1440
        for j in np.flatnonzero(np.isfinite(zt)):
            act[start:, j] = np.roll(act[start:, j], -120)
        ds["activity"] = xr.DataArray(act, dims=("time", "id"), coords=ds["activity"].coords)
        res = ps.compute_phase_response(ds, group_by=GROUP_BY)
        summ = ps.summarize_phase_response(res["per_fly"])
        pulsed = summ[summ["zt"].notna()]
        assert (pulsed["mean"] > 0).all(), "an earlier peak must read positive"

    def test_flies_with_no_usable_peak_are_dropped_and_counted(self, prc_ds):
        """A group that loses flies must say so rather than look thin."""
        import xarray as xr

        ds = prc_ds.copy(deep=True)
        # np.array, not asarray: asarray would hand back a view into the fixture.
        act = np.array(ds["activity"].transpose("time", "id").values, dtype=float)
        act[:, 0] = np.nan  # one fly with nothing to detect
        ds["activity"] = xr.DataArray(act, dims=("time", "id"), coords=ds["activity"].coords)
        res = ps.compute_phase_response(ds, group_by=GROUP_BY)
        assert str(ds["id"].values[0]) not in set(res["per_fly"]["id"])
        assert int(res["dropped"]["n_dropped"].sum()) >= 1

    def test_an_orphaned_group_is_dropped_wholesale(self, prc_ds):
        gen = np.asarray(prc_ds["genotype"].values).astype(str)
        dur = np.asarray(prc_ds["pulse_duration_minutes"].values, dtype=float)
        ds = prc_ds.isel(id=np.flatnonzero(~((gen == "B") & (dur == 0))))
        res = ps.compute_phase_response(ds, group_by=GROUP_BY)
        assert not any(g.startswith("B_2") for g in res["per_fly"]["group"]), (
            "a group with no control must not appear with a borrowed reference"
        )
        orphan = res["dropped"][res["dropped"]["group"].str.startswith("B_2")]
        assert (orphan["n_dropped"] == orphan["n_total"]).all()

    def test_the_response_days_skip_the_transient(self, prc_ds):
        res = ps.compute_phase_response(prc_ds, group_by=GROUP_BY)
        assert res["params"]["response_days"] == [2, 3, 4], "DD day 1 is the transient"
        assert 1 not in res["params"]["response_days"]

    def test_no_pulse_anywhere_says_so(self, prc_ds):
        ds = prc_ds.assign_coords(
            pulse_zt_hour=("id", np.full(prc_ds.sizes["id"], np.nan, dtype="float32"))
        )
        with pytest.raises(ValueError, match="no phase response"):
            ps.compute_phase_response(ds, group_by=GROUP_BY)


class TestBaselineCorrection:
    def test_the_baseline_day_is_before_the_pulse_day(self):
        """Offset 0 finds no marker: the tracker leaves the pulse day out of both its
        pre- and post-pulse runs, so asking for it silently applies no correction —
        which is how this was found, with corrected and uncorrected output identical.
        """
        assert ps.BASELINE_DAYS_AFTER_PULSE == (-1,)

    def test_it_is_applied_to_every_fly_that_can_take_it(self, prc_ds):
        res = ps.compute_phase_response(prc_ds, group_by=GROUP_BY)
        pf = res["per_fly"]
        assert pf["baseline_used"].all(), "a silently skipped correction is the bug"

    def test_the_raw_value_is_kept_alongside(self, prc_ds):
        res = ps.compute_phase_response(prc_ds, group_by=GROUP_BY)
        pf = res["per_fly"]
        assert "response_hours_raw" in pf.columns

    def test_it_removes_a_pre_pulse_offset(self, prc_ds):
        """What the correction is FOR. Give the pulsed arm a constant one-hour offset
        from its control — a flybox difference present before the pulse as much as
        after — and the corrected answer should return to the designed shift while
        the uncorrected one carries the extra hour.

        On a cohort with no such offset the two agree, which is why this builds one
        rather than asserting they always differ."""
        import xarray as xr

        ds = prc_ds.copy(deep=True)
        # np.array, not asarray: asarray would hand back a view into the fixture.
        act = np.array(ds["activity"].transpose("time", "id").values, dtype=float)
        zt = np.asarray(ds["pulse_zt_hour"].values, dtype=float)
        for j in np.flatnonzero(np.isfinite(zt)):
            act[:, j] = np.roll(act[:, j], 60)  # a whole-record hour later
        ds["activity"] = xr.DataArray(act, dims=("time", "id"), coords=ds["activity"].coords)

        on = ps.summarize_phase_response(
            ps.compute_phase_response(ds, group_by=GROUP_BY)["per_fly"]
        )
        off = ps.summarize_phase_response(
            ps.compute_phase_response(ds, group_by=GROUP_BY, baseline_days=())["per_fly"]
        )
        on_p = on[on["zt"].notna()]["mean"].mean()
        off_p = off[off["zt"].notna()]["mean"].mean()
        # Uncorrected carries the extra hour as an apparent extra delay.
        assert off_p == pytest.approx(on_p - 1.0, abs=0.4), (on_p, off_p)
        assert on_p == pytest.approx(-PULSE_SHIFT_H, abs=0.4)


class TestSummary:
    def test_a_single_fly_has_no_error_bar(self):
        one = pd.DataFrame(
            {"zt": [21.0], "group": ["a"], "response_hours": [1.5]}
        )
        out = ps.summarize_phase_response(one)
        assert out["n"].iloc[0] == 1
        assert np.isnan(out["sd"].iloc[0]) and np.isnan(out["sem"].iloc[0])

    def test_empty_input_gives_the_declared_columns(self):
        out = ps.summarize_phase_response(pd.DataFrame())
        assert list(out.columns) == ["zt", "group", "mean", "sd", "sem", "n"]


def test_grouping_suggestion_names_real_coords(prc_ds):
    """The message shown when no control matched has to suggest columns this
    dataset actually has, or it is just noise."""
    import dam_utilities

    sugg = ps.grouping_suggestion(prc_ds)
    assert sugg, "expected a suggestion"
    available = set(dam_utilities.group_defining_coords(prc_ds))
    assert set(sugg) <= available
    assert "genotype" in sugg and "pulse_duration_minutes" in sugg
