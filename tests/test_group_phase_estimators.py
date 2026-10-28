"""How a group's daily peak is found for the per-day comparison (Phase over Time).

``fly_median`` (default) takes the median of the flies' own daily peaks; ``mean_trace`` takes
the peak of the group's averaged activity, as ``peakphaseplot.m`` does. Both must
recover the designed shift; ``fly_median`` must be exactly the clock-face median of
the per-fly peaks it reports, and must not be thrown by flies drifting apart.
"""

import numpy as np
import pytest
import xarray as xr

from clockwork.core import phase_shift as ps
from conftest import PULSE_DD_DAY, PULSE_SHIFT_H

GROUP_BY = ("genotype", "condition")
DAY = 1440


def _diff(per_day, group, day, col="phase_difference_hours"):
    row = per_day[(per_day["group"] == group) & (per_day["day_index"] == day)]
    assert len(row) == 1, f"expected one row for {group} day {day}, got {len(row)}"
    return float(row[col].iloc[0])


@pytest.mark.parametrize("estimator", ps.GROUP_ESTIMATORS)
def test_both_recover_the_designed_shift(pulse_ds, estimator):
    cmap = ps.build_matched_control_map(pulse_ds, "noLP", group_by=GROUP_BY)
    res = ps.compute_group_phase_difference(
        pulse_ds, cmap, group_by=GROUP_BY, estimator=estimator
    )
    for grp in ("A_LP", "B_LP"):
        for day in range(PULSE_DD_DAY, PULSE_DD_DAY + 3):
            assert _diff(res["per_day"], grp, day) == pytest.approx(PULSE_SHIFT_H, abs=0.2)
    assert res["params"]["estimator"] == estimator


def test_fly_median_is_the_median_of_the_flies_own_peaks(pulse_ds):
    markers = ps._group_fly_markers(pulse_ds)
    cmap = ps.build_matched_control_map(pulse_ds, "noLP", group_by=GROUP_BY)
    res = ps.compute_group_phase_difference(
        pulse_ds, cmap, group_by=GROUP_BY, fly_markers=markers
    )
    labels, _ = ps.group_labels(pulse_ds, GROUP_BY)
    ids = [str(i) for i in pulse_ds["id"].values]
    day = PULSE_DD_DAY + 1
    members = [f for f, lab in zip(ids, labels) if lab == "A_LP"]
    own = [(markers[f][day] - day * DAY) / 60.0 for f in members if day in markers[f]]
    row = res["per_day"].query("group == 'A_LP' and day_index == @day").iloc[0]
    assert row["peak_hours"] == pytest.approx(day * 24 + np.median(own))
    assert row["n_flies_with_peak"] == len(own)


def test_fly_median_has_a_value_on_the_pulse_day(pulse_ds):
    """The pulse day is the default zero point, so it must not go missing."""
    cmap = ps.build_matched_control_map(pulse_ds, "noLP", group_by=GROUP_BY)
    pd_ = ps.compute_group_phase_difference(pulse_ds, cmap, group_by=GROUP_BY)["per_day"]
    pulse_day = PULSE_DD_DAY - 1
    assert np.isfinite(_diff(pd_, "A_LP", pulse_day))


def _drifting_cohort(seed=3, n=12, n_days=10, ld_days=3):
    """Pulsed and control arms of one genotype, each half at tau 23.3 h and half at
    24.7 h in DD, so by the last days the halves are ~8 h apart. Hump heights are
    random per fly, which is what makes the averaged trace's tallest hump change
    from day to day. The pulsed arm is delayed 2 h from the first DD day."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_days * DAY, dtype=float)
    cols, ids, cond = [], [], []
    for arm, shift in (("LP", 2.0), ("noLP", 0.0)):
        for k in range(n):
            tau = 23.3 if k % 2 else 24.7
            amp = rng.uniform(10, 30)
            act = np.zeros(t.size)
            for d in range(n_days + 2):
                if d < ld_days:
                    c = d * DAY + 11 * 60
                else:
                    c = ld_days * DAY + 11 * 60 + (d - ld_days) * tau * 60 + shift * 60
                act += amp * np.exp(-0.5 * ((t - c) / 120.0) ** 2)
            cols.append(act + rng.normal(0, 0.05, t.size).clip(0))
            ids.append(f"{arm}_{k}")
            cond.append(arm)
    start = np.datetime64("2026-10-14T09:00:00")
    return xr.Dataset(
        {"activity": (("time", "id"), np.array(cols).T)},
        coords={
            "id": ids,
            "time": t.astype(np.int64),
            "genotype": ("id", np.array(["G"] * len(ids))),
            "condition": ("id", np.array(cond)),
            "pulse_time": ("id", np.array(["ZT17" if c == "LP" else "" for c in cond], dtype=object)),
            "start_datetime": ("id", np.array([start] * len(ids))),
            "first_DD_day": ("id", np.array([start + np.timedelta64(ld_days * DAY, "m")] * len(ids))),
        },
        attrs={"time_is_relative_minutes": 1, "phase": "full"},
    )


def test_fly_median_holds_the_shift_while_the_flies_drift_apart():
    ds = _drifting_cohort()
    res = ps.compute_group_phase_difference(ds, {"G_LP": "G_noLP"}, group_by=GROUP_BY)
    late = res["per_day"].query("group == 'G_LP' and day_index >= 3")
    assert np.allclose(late["phase_difference_hours"], 2.0, atol=0.5), late[
        ["day_index", "phase_difference_hours"]
    ]
    # The averaged trace on the same cohort loses the shift once the halves part.
    trace = ps.compute_group_phase_difference(
        ds, {"G_LP": "G_noLP"}, group_by=GROUP_BY, estimator="mean_trace"
    )["per_day"].query("group == 'G_LP' and day_index >= 3")
    assert (trace["phase_difference_hours"] - 2.0).abs().max() > 3.0


def test_a_difference_is_wrapped_into_half_a_day(pulse_ds):
    cmap = ps.build_matched_control_map(pulse_ds, "noLP", group_by=GROUP_BY)
    for est in ps.GROUP_ESTIMATORS:
        d = ps.compute_group_phase_difference(pulse_ds, cmap, group_by=GROUP_BY, estimator=est)
        vals = d["per_day"]["phase_difference_hours"].dropna()
        assert (vals.abs() <= 12).all()


def test_an_unknown_estimator_is_refused(pulse_ds):
    with pytest.raises(ValueError, match="estimator"):
        ps.compute_group_phase_difference(
            pulse_ds, "A_noLP", group_by=GROUP_BY, estimator="median"
        )


def test_the_bootstrap_runs_on_fly_median(pulse_ds):
    cmap = ps.build_matched_control_map(pulse_ds, "noLP", group_by=GROUP_BY)
    boot = ps.bootstrap_group_phase_difference(
        pulse_ds, cmap, n_boot=5, group_by=GROUP_BY, estimator="fly_median"
    )
    row = boot.query("group == 'A_LP'")
    assert len(row) and (row["n_boot_ok"] == 5).all()


def test_a_peak_drifting_past_midnight_is_followed():
    """A free-running peak that crosses midnight stays on its own cycle rather than
    being pinned to the edge of the calendar day."""
    t = np.arange(12 * DAY, dtype=float)
    tau = 25.5
    act = np.zeros(t.size)
    for k in range(13):
        act += 20 * np.exp(-0.5 * ((t - (18 * 60 + k * tau * 60)) / 120.0) ** 2)
    markers = ps.detect_peak_markers(t, act, pulse_minute=-DAY)
    days = sorted(d for d in markers if 1 <= d <= 9)
    slope = np.polyfit(days, [markers[d] for d in days], 1)[0] / 60.0
    assert slope == pytest.approx(tau, abs=0.1)
    assert all(np.isfinite(markers[d]) for d in days)
