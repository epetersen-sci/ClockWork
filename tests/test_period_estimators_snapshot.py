"""What the period estimators say about REAL flies, pinned (see snapshot_util.py).

Until this file, no test ever ran Lomb-Scargle, autocorrelation, MESA or the
circadian CWT. They sit behind the Period analysis page's Run buttons, which the
smoke tests never press, so an edit that shifted every fly's period by an hour
would have passed CI.

Synthetic sines were tried and are the wrong test for this: every estimator finds
the period of a clean sine, so they pass while saying nothing about the records
that are actually hard — weak rhythms, dead channels, phase drift. So this runs
the committed example_data (Monitors 17 and 18: 64 flies, 9 DD days) through the
same path the page takes:

    select_phase(ds, "DD") -> preprocess_activity(<method's default config>)
        -> <estimator>(phase="DD", page defaults) -> classify_all

and compares each fly's period, strength and rhythmic call with a snapshot a
person has reviewed. Periods are held to 0.05 h — well inside any biologically
meaningful difference, well outside floating-point noise.

The CWT is by far the slowest on CPU, so it runs on four flies; the other three
run on every fly. The whole file takes ~40 s on a laptop.
"""

import numpy as np
import pytest

from clockwork.core import dam_utilities, periodograms
from clockwork.core import rhythmicity_classification as rc
from clockwork.core.calibrations import (
    DEFAULT_CWT_MAX_PERIOD,
    DEFAULT_CWT_MIN_PERIOD,
    DEFAULT_MAX_BRIDGE_GAP_MINUTES,
    DEFAULT_MIN_DD_DAYS_FLOOR,
)
from clockwork.core.preprocessing import (
    ac_default_config,
    cwt_default_config,
    ls_default_config,
    preprocess_activity,
)
from conftest import requires_example_data
from snapshot_util import check_snapshot

pytestmark = requires_example_data

# The page's defaults: one period range for search and classification, the DD
# floor as a filter, the one gap-bridge ceiling.
MIN_P, MAX_P = DEFAULT_CWT_MIN_PERIOD, DEFAULT_CWT_MAX_PERIOD
PAGE_KW = {"min_period": MIN_P, "max_period": MAX_P, "min_num_days": DEFAULT_MIN_DD_DAYS_FLOOR}
BRIDGE_KW = {"max_bridge_gap_minutes": DEFAULT_MAX_BRIDGE_GAP_MINUTES}
N_PROCESSES = 2

# Two flies per monitor for the CWT: 17_2 and its neighbours include a clearly
# rhythmic and a clearly arrhythmic record, so both branches of the call are held.
CWT_FLIES = ["20250115_17_1", "20250115_17_2", "20250115_18_1", "20250115_18_2"]

PERIOD_ATOL = 0.05  # hours


@pytest.fixture(scope="module")
def dd_src(example_ds):
    src, phase = dam_utilities.select_phase(example_ds, "DD")
    assert phase == "DD"
    return src


@pytest.fixture(scope="module")
def ls_ds(dd_src):
    pp = preprocess_activity(dd_src, ls_default_config())
    return periodograms.lomb_scargle_analysis(pp, phase="DD", n_processes=N_PROCESSES, **PAGE_KW)


@pytest.fixture(scope="module")
def ac_ds(dd_src):
    pp = preprocess_activity(dd_src, ac_default_config())
    return periodograms.autocorrelation_analysis(
        pp, phase="DD", n_processes=N_PROCESSES, **PAGE_KW, **BRIDGE_KW
    )


@pytest.fixture(scope="module")
def mesa_ds(dd_src):
    pp = preprocess_activity(dd_src, ac_default_config())
    return periodograms.mesa_analysis(
        pp, phase="DD", n_processes=N_PROCESSES, **PAGE_KW, **BRIDGE_KW
    )


def _per_fly(ds, fields):
    """{fly: {field: value}} for the flies an estimator returned."""
    out = {}
    for i, fly in enumerate(ds["id"].values):
        out[str(fly)] = {f: ds[f].values[i] for f in fields if f in ds}
    return out


def _summary(ds, period_var, rhythmic_var=None):
    """What a reviewer should look at before accepting a snapshot."""
    lines = []
    for genotype in sorted(set(ds["genotype"].values)):
        sel = ds["genotype"].values == genotype
        periods = ds[period_var].values[sel].astype(float)
        finite = periods[np.isfinite(periods)]
        line = (
            f"{genotype}: {finite.size}/{sel.sum()} flies with a period, "
            f"median {np.median(finite):.2f} h" if finite.size else f"{genotype}: no periods"
        )
        if rhythmic_var and rhythmic_var in ds.coords:
            calls = ds[rhythmic_var].values[sel]
            line += f", {int(calls.sum())}/{sel.sum()} rhythmic"
        lines.append(line)
    return {"lines": lines}


def test_lomb_scargle(ls_ds, update_snapshots):
    classified = rc.classify_lomb_scargle(ls_ds, period_window=(MIN_P, MAX_P))
    records = _per_fly(ls_ds, ["ls_period", "ls_power", "ls_fap"])
    for fly, rhythmic in zip(classified["id"].values, classified["ls_rhythmic"].values):
        records[str(fly)]["ls_rhythmic"] = bool(rhythmic)
    check_snapshot(
        "ls_example",
        records,
        update_snapshots,
        atol={"ls_period": PERIOD_ATOL},
        summary=_summary(classified, "ls_period", "ls_rhythmic"),
    )


def test_autocorrelation(ac_ds, update_snapshots):
    classified = rc.classify_autocorrelation(ac_ds, period_window=(MIN_P, MAX_P))
    records = _per_fly(ac_ds, ["ac_period", "ac_power", "ac_rhythm_strength"])
    for fly, rhythmic in zip(classified["id"].values, classified["ac_rhythmic"].values):
        records[str(fly)]["ac_rhythmic"] = bool(rhythmic)
    check_snapshot(
        "ac_example",
        records,
        update_snapshots,
        atol={"ac_period": PERIOD_ATOL},
        summary=_summary(classified, "ac_period", "ac_rhythmic"),
    )


def test_mesa(mesa_ds, update_snapshots):
    # MESA is a period estimator only — it has no rhythmic call to pin.
    check_snapshot(
        "mesa_example",
        _per_fly(mesa_ds, ["mesa_period", "mesa_power", "mesa_order"]),
        update_snapshots,
        atol={"mesa_period": PERIOD_ATOL},
        summary=_summary(mesa_ds, "mesa_period"),
    )


def test_cwt(dd_src, update_snapshots):
    pp = preprocess_activity(dd_src.sel(id=CWT_FLIES), cwt_default_config())
    cwt_ds, _averages = periodograms.wavelet_analysis(
        pp, phase="DD", n_processes=N_PROCESSES, **PAGE_KW, **BRIDGE_KW
    )
    classified = rc.classify_cwt(cwt_ds)
    records = _per_fly(cwt_ds, ["cwt_period", "cwt_power", "cwt_rhythmicity"])
    for fly, rhythmic in zip(classified["id"].values, classified["cwt_rhythmic"].values):
        records[str(fly)]["cwt_rhythmic"] = bool(rhythmic)
    check_snapshot(
        "cwt_example",
        records,
        update_snapshots,
        atol={"cwt_period": PERIOD_ATOL},
        summary=_summary(classified, "cwt_period", "cwt_rhythmic"),
    )


def test_classify_all_agrees_with_the_individual_classifiers(ls_ds, ac_ds):
    """The Rhythmicity page calls ``classify_all``; the tests above call each
    classifier. The two must give the same calls on the same results."""
    merged = ls_ds.merge(
        ac_ds[["ac_period", "ac_power", "ac_rhythm_strength"]], compat="no_conflicts", join="outer"
    )
    both = rc.classify_all(merged, period_window=(MIN_P, MAX_P), run_cwt=False)
    ls_only = rc.classify_lomb_scargle(ls_ds, period_window=(MIN_P, MAX_P))
    ac_only = rc.classify_autocorrelation(ac_ds, period_window=(MIN_P, MAX_P))
    np.testing.assert_array_equal(
        both["ls_rhythmic"].sel(id=ls_only["id"]).values, ls_only["ls_rhythmic"].values
    )
    np.testing.assert_array_equal(
        both["ac_rhythmic"].sel(id=ac_only["id"]).values, ac_only["ac_rhythmic"].values
    )


def test_preprocessing_output_is_pinned(dd_src, update_snapshots):
    """A checksum of each method's preprocessed input, for four real flies.

    Every estimator reads the output of ``preprocess_activity``, so drift here
    moves every period at once. Pinning it separately says WHICH stage moved: a
    changed period with an unchanged checksum is the estimator; both changed is
    the preprocessing.
    """
    configs = {"ls": ls_default_config(), "ac": ac_default_config(), "cwt": cwt_default_config()}
    records = {}
    for name, cfg in configs.items():
        pp = preprocess_activity(dd_src.sel(id=CWT_FLIES), cfg)
        values = pp["activity"].transpose("id", "time").values.astype(np.float64)
        for fly, row in zip(CWT_FLIES, values):
            finite = np.isfinite(row)
            records[f"{name}:{fly}"] = {
                "n": int(row.size),
                "n_finite": int(finite.sum()),
                "first_finite": int(np.argmax(finite)) if finite.any() else None,
                "sum": float(row[finite].sum()),
                "sum_sq": float((row[finite] ** 2).sum()),
            }
    # The sum of a detrended series sits near zero, where a relative tolerance
    # alone means nothing; an absolute allowance scaled to ~13k samples covers it.
    check_snapshot(
        "preprocess_example",
        records,
        update_snapshots,
        atol={"sum": 1e-2},
        rtol=1e-5,
    )
