"""The regression PRC: each fly's post-pulse peaks fitted with a line and read on
the day after the pulse, against its unpulsed control.

Two things the per-day estimator got wrong, each pinned here:

* **Cohorts released on different days of their recordings.** Days used to be
  counted from each fly's recording start with one pulse day for everyone, so a
  control run with one more LD day was compared a day off — and picked up a day of
  free-running drift as phase shift.
* **Periods that differ between flies.** Averaging days further from the pulse adds
  ``(tau - tau_control)`` per day; the line read at the first DD day does not.
"""

import numpy as np
import pytest
import xarray as xr

from clockwork.core import dam_utilities
from clockwork.core import phase_shift as ps

DAY = 1440
GROUP_BY = ("genotype", "pulse_time", "pulse_duration_min")
BASE_HOUR = 11.0
PULSE_ZT = 15.0


def _cohort(arms, n_days=12, seed=0):
    """Flies built from ``arms``: ``[(label, n, ld_days, shift_h, taus)]``.

    Each fly has one Gaussian bump per day at ``BASE_HOUR`` while entrained, then
    free-runs from its first DD day at its own period, displaced by ``shift_h``
    (positive = delay). ``ld_days`` is how many entrained days precede release; the
    pulse is on the last of them. Records past the arm's length stay NaN, the way a
    shorter run sits in a dataset.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_days * DAY, dtype=float)
    cols, ids, geno, zt, dur, dd_start, starts = [], [], [], [], [], [], []
    start0 = np.datetime64("2026-10-14T09:00:00")
    for label, n, ld_days, shift_h, taus in arms:
        pulsed = shift_h is not None
        for k in range(n):
            tau = float(taus[k % len(taus)])
            act = np.zeros(t.size)
            for d in range(n_days + 2):
                if d < ld_days:
                    centre = d * DAY + BASE_HOUR * 60
                else:
                    k_dd = d - ld_days
                    centre = (
                        ld_days * DAY + BASE_HOUR * 60 + k_dd * tau * 60
                        + (shift_h if pulsed else 0.0) * 60
                    )
                act += 20 * np.exp(-0.5 * ((t - centre) / 120.0) ** 2)
            act += rng.normal(0, 0.05, t.size).clip(0)
            cols.append(act)
            ids.append(f"{label}_{k}")
            geno.append("G")
            zt.append(f"ZT{PULSE_ZT:g}" if pulsed else "")
            dur.append(10.0 if pulsed else 0.0)
            dd_start.append(start0 + np.timedelta64(ld_days * DAY, "m"))
            starts.append(start0)
    return xr.Dataset(
        {"activity": (("time", "id"), np.array(cols).T)},
        coords={
            "id": ids,
            "time": t.astype(np.int64),
            "genotype": ("id", np.array(geno)),
            "pulse_time": ("id", np.array(zt, dtype=object)),
            "pulse_duration_min": ("id", np.array(dur, dtype="float32")),
            "start_datetime": ("id", np.array(starts)),
            "first_DD_day": ("id", np.array(dd_start)),
        },
        attrs={"time_is_relative_minutes": 1, "phase": "full"},
    )


def _pulsed_mean(res):
    pf = res["per_fly"]
    return float(pf[pf["zt"].notna()]["response_hours"].mean())


class TestCohortsReleasedOnDifferentDays:
    """A pulsed run with 2 LD days against a control run with 3, both at tau 23.5 h."""

    @pytest.fixture(scope="class")
    def ds(self):
        return _cohort(
            [("ctrl", 6, 3, None, [23.5]), ("lp", 6, 2, 1.5, [23.5])], n_days=9
        )

    @pytest.mark.parametrize("method", ps.PRC_METHODS)
    def test_the_designed_delay_is_recovered(self, ds, method):
        res = ps.compute_phase_response(ds, group_by=GROUP_BY, method=method)
        assert _pulsed_mean(res) == pytest.approx(-1.5, abs=0.2)

    def test_controls_are_split_on_their_own_last_ld_day(self, ds):
        full = dam_utilities.add_phase_metadata(dam_utilities.add_pulse_metadata(ds))
        pulse = np.asarray(full["pulse_minute"].values, float)
        ref = ps._reference_pulse_minutes(full, pulse)
        ctrl_days = np.floor(ref[:6] / DAY)
        lp_days = np.floor(ref[6:] / DAY)
        assert (ctrl_days == 2).all() and (lp_days == 1).all()


class TestPeriodsThatDifferBetweenFlies:
    """Flies free-running between 23.3 and 24.9 h, each pulsed fly delayed 2 h."""

    @pytest.fixture(scope="class")
    def ds(self):
        taus = [23.3, 23.7, 24.0, 24.4, 24.9]
        return _cohort(
            [("ctrl", 10, 3, None, taus), ("lp", 10, 3, 2.0, taus[::-1])], n_days=11
        )

    def test_regression_recovers_each_fly(self, ds):
        pf = ps.compute_phase_response(ds, group_by=GROUP_BY)["per_fly"]
        pulsed = pf[pf["zt"].notna()]
        assert np.allclose(pulsed["response_hours"], -2.0, atol=0.3)

    def test_the_fitted_slope_is_the_free_running_period(self, ds):
        pf = ps.compute_phase_response(ds, group_by=GROUP_BY)["per_fly"]
        assert pf["period_hours"].between(23.2, 25.0).all()


class TestKnownShifts:
    """Shifts across a delay-to-advance range, each recovered in sign and size."""

    @pytest.mark.parametrize("shift_h", [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0])
    @pytest.mark.parametrize("baseline", ps.PRC_BASELINES)
    def test_recovered(self, shift_h, baseline):
        ds = _cohort(
            [("ctrl", 6, 3, None, [24.2]), ("lp", 6, 3, shift_h, [24.2])], n_days=10
        )
        res = ps.compute_phase_response(ds, group_by=GROUP_BY, baseline=baseline)
        # A delay (positive shift) reads negative.
        assert _pulsed_mean(res) == pytest.approx(-shift_h, abs=0.25)


def test_controls_average_to_zero():
    ds = _cohort([("ctrl", 8, 3, None, [23.8, 24.3]), ("lp", 8, 3, 1.0, [24.0])])
    pf = ps.compute_phase_response(ds, group_by=GROUP_BY)["per_fly"]
    assert pf[pf["zt"].isna()]["response_hours"].mean() == pytest.approx(0.0, abs=1e-9)


def test_an_unknown_method_or_baseline_is_refused():
    ds = _cohort([("ctrl", 2, 3, None, [24.0]), ("lp", 2, 3, 1.0, [24.0])], n_days=8)
    with pytest.raises(ValueError, match="method"):
        ps.compute_phase_response(ds, group_by=GROUP_BY, method="cwt")
    with pytest.raises(ValueError, match="baseline"):
        ps.compute_phase_response(ds, group_by=GROUP_BY, baseline="day")
