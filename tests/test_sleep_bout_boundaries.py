"""Where a sleep bout starts, ends, and which state it is — at the exact boundaries.

Every other sleep test runs a whole fixture through ``sleep_analysis`` and checks
that the result is consistent. None of them hands it a trace whose answer is
known to the minute, so none of them can say whether the thresholds sit where the
definition puts them. These do: one fly, one run of immobility of a chosen
length, and the bout that must come out.

**Found while writing this file — an off-by-one in bout duration.** The detector
measures a bout as ``end_time - start_time`` over the first and last immobile
minutes, so a run of N immobile 1-minute bins is recorded as N-1 minutes:

    immobile bins  sleep minutes in mask  bout duration  state
          5                  0                  -          -      (not sleep)
          6                  6                 5.0       short
         31                 31                30.0    intermediate
         61                 61                60.0       long

Under the standard definition (>= 5 min of inactivity, Shaw et al. 2000) five
immobile bins ARE a 5-minute bout. So today it takes six to make sleep, every
state boundary is a minute late, and summed bout durations fall one minute per
bout short of the sleep mask. The ``xfail(strict=True)`` tests below state the
standard definition; they fail now, and turn into an XPASS error the day the rule
is fixed, which is the reminder to delete the markers. Whether to fix it — it
changes every sleep total — is a decision for the lab, not for a test.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import sleep_analysis

N_TIME = 300
ONSET = 100


def _one_fly(immobile_runs, gaps=()):
    """One fly: moving throughout, except the given runs (start, length) are
    immobile (0) and the given gaps (start, length) are missing (-1)."""
    mov = np.ones(N_TIME)
    for start, length in immobile_runs:
        mov[start : start + length] = 0
    for start, length in gaps:
        mov[start : start + length] = -1
    return xr.Dataset(
        {"moving": (("time", "id"), mov[:, None])},
        coords={"time": np.arange(N_TIME, dtype=np.int64), "id": ["fly"]},
        attrs={"time_is_relative_minutes": 1},
    )


def _bouts(ds):
    out = sleep_analysis.sleep_analysis(ds, phase="both", sleep_threshold_sec=300)
    n_bouts = int(np.isfinite(out["duration"].values).sum()) if "duration" in out else 0
    durations = out["duration"].values.ravel()[:n_bouts].tolist() if n_bouts else []
    states = out["sleep_state"].values.ravel()[:n_bouts].tolist() if n_bouts else []
    return out, durations, states


# ---------------------------------------------------------------------------
# classify_sleep_bouts: the state cut on a duration (correct as it stands)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "duration, state",
    [
        (5, "short"),
        (29.99, "short"),
        (30, "intermediate"),  # the lower bound is inclusive
        (59.99, "intermediate"),
        (60, "long"),
        (600, "long"),
    ],
)
def test_state_boundaries_on_a_duration(duration, state):
    df = sleep_analysis.classify_sleep_bouts(pd.DataFrame({"duration": [duration]}))
    assert df["sleep_state"].iloc[0] == state


def test_custom_boundaries_move_the_cut():
    df = sleep_analysis.classify_sleep_bouts(
        pd.DataFrame({"duration": [19, 20, 44, 45]}), short_max_min=20, inter_max_min=45
    )
    assert df["sleep_state"].tolist() == ["short", "intermediate", "intermediate", "long"]


# ---------------------------------------------------------------------------
# Detection on a hand-built trace
# ---------------------------------------------------------------------------


def test_moving_throughout_is_never_asleep():
    out, durations, _ = _bouts(_one_fly([]))
    assert durations == []
    assert (out["sleep"].values == 0).all()


def test_the_sleep_mask_covers_exactly_the_immobile_run():
    out, _, _ = _bouts(_one_fly([(ONSET, 20)]))
    mask = out["sleep"].sel(id="fly").values
    assert (mask[ONSET : ONSET + 20] == 1).all()
    assert (mask[:ONSET] == 0).all() and (mask[ONSET + 20 :] == 0).all()


def test_two_runs_make_two_bouts_in_order():
    _, durations, _ = _bouts(_one_fly([(50, 20), (150, 40)]))
    assert len(durations) == 2
    assert durations[0] < durations[1]


def test_a_short_gap_inside_a_bout_does_not_split_it():
    """Gaps of up to 4 minutes flanked by immobility are bridged."""
    _, durations, _ = _bouts(_one_fly([(ONSET, 40)], gaps=[(ONSET + 15, 4)]))
    assert len(durations) == 1


def test_a_long_gap_inside_a_bout_splits_it():
    _, durations, _ = _bouts(_one_fly([(ONSET, 40)], gaps=[(ONSET + 15, 5)]))
    assert len(durations) == 2


def test_missing_minutes_stay_missing_in_the_mask():
    out, _, _ = _bouts(_one_fly([(ONSET, 40)], gaps=[(ONSET + 15, 3)]))
    mask = out["sleep"].sel(id="fly").values
    assert (mask[ONSET + 15 : ONSET + 18] == -1).all()


# ---------------------------------------------------------------------------
# The standard definition (Shaw et al. 2000) — see the module docstring
# ---------------------------------------------------------------------------

OFF_BY_ONE = pytest.mark.xfail(
    strict=True,
    reason="bout duration is end-start, so N immobile bins read as N-1 minutes "
    "(see module docstring); remove this marker when the rule is fixed",
)


@OFF_BY_ONE
def test_five_immobile_minutes_are_a_five_minute_bout():
    out, durations, _ = _bouts(_one_fly([(ONSET, 5)]))
    assert durations == [5.0]
    assert int((out["sleep"].values == 1).sum()) == 5


def test_four_immobile_minutes_are_not_sleep():
    out, durations, _ = _bouts(_one_fly([(ONSET, 4)]))
    assert durations == []
    assert (out["sleep"].values == 0).all()


@OFF_BY_ONE
@pytest.mark.parametrize(
    "n_immobile, state",
    [(29, "short"), (30, "intermediate"), (59, "intermediate"), (60, "long")],
)
def test_state_boundaries_on_immobile_minutes(n_immobile, state):
    _, durations, states = _bouts(_one_fly([(ONSET, n_immobile)]))
    assert durations == [float(n_immobile)]
    assert states == [state]


@OFF_BY_ONE
def test_bout_durations_add_up_to_the_sleep_mask():
    out, durations, _ = _bouts(_one_fly([(20, 12), (80, 33), (160, 70)]))
    assert sum(durations) == int((out["sleep"].values == 1).sum())
