"""Sleep-deprivation rebound, recomputed independently from the real sleep masks.

``compute_sd_analysis`` had no test; it sits behind the Sleep deprivation page's
Run button. example_data holds no deprivation, but it has three real LD days —
exactly the shape the analysis needs (baseline day 0, "SD" day 1, recovery day 2)
— and the quantity is simple enough to recompute directly from the ``sleep``
mask:

    rebound_pct(group, phase) = mean over flies of
        (recovery-day sleep minutes - baseline sleep minutes) / baseline x 100

Three things are checked against that: the numbers on the real flies, a planted
rebound whose size is known (every dsmcherry fly sleeps the whole recovery
night), and that the SD day itself enters neither side.
"""

import numpy as np
import pytest

import dam_utilities
import sleep_analysis
import sleep_deprivation

DAY = 1440
BASELINE, SD_DAY, RECOVERY = 0, 1, 2
SD_KW = {"sd_start_zt_minutes": 720, "sd_duration_minutes": 720, "sd_day_index": SD_DAY}


@pytest.fixture(scope="module")
def sleep_ds(example_ds):
    ds = dam_utilities._compute_moving(example_ds)
    return sleep_analysis.sleep_analysis(ds, phase="LD", sleep_threshold_sec=300)


def _minutes_asleep(ds, day, zt_from, zt_to):
    """Per-fly minutes asleep in [zt_from, zt_to) of ``day``, straight from the mask."""
    sleep = ds["sleep"].transpose("id", "time").values
    window = slice(day * DAY + zt_from, day * DAY + zt_to)
    return (sleep[:, window] == 1).sum(axis=1).astype(float)


def _expected_rebound(ds, zt_from, zt_to):
    base = _minutes_asleep(ds, BASELINE, zt_from, zt_to)
    rec = _minutes_asleep(ds, RECOVERY, zt_from, zt_to)
    out = {}
    for group in sorted(set(ds["group"].values)):
        sel = (ds["group"].values == group) & (base > 0)
        out[group] = float(np.mean((rec[sel] - base[sel]) / base[sel] * 100))
    return out


def _rebound(result, phase):
    df = result["rebound_pct"]
    df = df[(df["recovery_day"] == 1) & (df["phase"] == phase)]
    return dict(zip(df["group"], df["rebound_pct"]))


def test_the_example_has_three_ld_days(sleep_ds):
    n_days, bounds = sleep_deprivation.get_experiment_days(sleep_ds)
    assert n_days == 3
    assert bounds[0] == (0, DAY)


@pytest.mark.parametrize(
    "phase, zt_from, zt_to", [("Light (ZT0-12)", 0, 720), ("Dark (ZT12-24)", 720, DAY)]
)
def test_rebound_matches_a_direct_count_on_real_flies(sleep_ds, phase, zt_from, zt_to):
    result = sleep_deprivation.compute_sd_analysis(sleep_ds, **SD_KW)
    got = _rebound(result, phase)
    expected = _expected_rebound(sleep_ds, zt_from, zt_to)
    assert got.keys() == expected.keys()
    for group in expected:
        assert got[group] == pytest.approx(expected[group], rel=1e-9), group


def test_a_planted_rebound_is_recovered(sleep_ds):
    """Every dsmcherry fly sleeps the whole recovery night: its dark-phase rebound
    must be exactly (720 - baseline) / baseline, and the other group's unchanged."""
    ds = sleep_ds.copy(deep=True)
    sleep = ds["sleep"].transpose("id", "time").values.copy()
    planted = ds["genotype"].values == "dsmcherry"
    sleep[planted, RECOVERY * DAY + 720 : (RECOVERY + 1) * DAY] = 1
    ds["sleep"] = (("id", "time"), sleep)

    base = _minutes_asleep(ds, BASELINE, 720, DAY)
    group = str(ds["group"].values[planted][0])
    sel = planted & (base > 0)
    expected = float(np.mean((720 - base[sel]) / base[sel] * 100))

    got = _rebound(sleep_deprivation.compute_sd_analysis(ds, **SD_KW), "Dark (ZT12-24)")
    untouched = _rebound(sleep_deprivation.compute_sd_analysis(sleep_ds, **SD_KW), "Dark (ZT12-24)")
    assert got[group] == pytest.approx(expected, rel=1e-9)
    other = next(g for g in got if g != group)
    assert got[other] == pytest.approx(untouched[other], rel=1e-12)


def test_the_sd_day_enters_neither_baseline_nor_recovery(sleep_ds):
    ds = sleep_ds.copy(deep=True)
    sleep = ds["sleep"].transpose("id", "time").values.copy()
    sleep[:, SD_DAY * DAY : (SD_DAY + 1) * DAY] = 0  # deprived all day
    ds["sleep"] = (("id", "time"), sleep)
    before = sleep_deprivation.compute_sd_analysis(sleep_ds, **SD_KW)
    after = sleep_deprivation.compute_sd_analysis(ds, **SD_KW)
    for key in ("rebound_pct", "phase_totals", "baseline_profile"):
        assert after[key].equals(before[key]), key


def test_phase_totals_add_up(sleep_ds):
    totals = sleep_deprivation.compute_sd_analysis(sleep_ds, **SD_KW)["phase_totals"]
    np.testing.assert_allclose(
        totals["total_sleep_min"], totals["light_sleep_min"] + totals["dark_sleep_min"]
    )
    assert set(totals["day_label"]) == {"Baseline (avg)", "Recovery Day 1"}


@pytest.mark.parametrize("day", [0, 2])
def test_sd_on_the_first_or_last_day_is_refused(sleep_ds, day):
    with pytest.raises(ValueError):
        sleep_deprivation.compute_sd_analysis(sleep_ds, **{**SD_KW, "sd_day_index": day})
