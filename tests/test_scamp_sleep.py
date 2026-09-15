"""SCAMP's sleep definitions, pinned against hand-computed cases.

``core/scamp_sleep`` exists so a number can be put beside a SCAMP run, which makes
its value entirely a matter of matching ``sleepcalc3.m`` / ``fly_sleepthresh.m`` /
``fly_histo.m``. These tests therefore assert the MATLAB's semantics on inputs small
enough to work out by hand, rather than asserting that the code does what it does.

Three of them are the behaviours SCAMP is easy to get silently wrong about:

* a bout is credited to the bin it STARTS in, while its minutes fall in whichever
  bin they occur in — so ``sfreq`` and ``stdur`` disagree across a boundary, and
  they should;
* ``Pdoze`` and ``Pwake`` reset per bin and each skips its own first minute;
* a missing reading is scored as zero counts, so a dropout counts toward sleep.

The last one is a SCAMP convention that costs accuracy, and it is kept deliberately.
:func:`measured_fraction` is how the cost stays visible.
"""

import numpy as np
import pytest

import scamp_sleep as ss


def _counts(spec, n_fly=1):
    """Column vector of per-minute counts from a flat list."""
    col = np.asarray(spec, dtype=float)[:, None]
    return np.repeat(col, n_fly, axis=1)


# ------------------------------------------------- fly_sleepthresh / fly_histo


def test_sleep_needs_the_full_threshold_run():
    """A 4-minute run of zeros is not sleep at a 5-minute threshold."""
    sleep, _ = ss.per_minute(_counts([1, 0, 0, 0, 0, 1]), threshold=5)
    assert not sleep.any()

    sleep, _ = ss.per_minute(_counts([1, 0, 0, 0, 0, 0, 1]), threshold=5)
    assert sleep[1:6, 0].all()
    assert not sleep[0, 0] and not sleep[6, 0]


def test_every_minute_of_a_qualifying_run_is_asleep():
    sleep, _ = ss.per_minute(_counts([0] * 9 + [2]), threshold=5)
    assert sleep[:9, 0].sum() == 9


def test_no_gap_bridging():
    """SCAMP does not join two runs across a single active minute. Two 4-minute
    runs either side of one count stay awake, where a bridging definition would
    score all nine minutes as one 9-minute bout."""
    sleep, _ = ss.per_minute(_counts([0, 0, 0, 0, 1, 0, 0, 0, 0]), threshold=5)
    assert not sleep.any()


def test_bout_length_sits_at_the_first_minute_and_ignores_the_threshold():
    """fly_histo records every inactivity run, short ones included — the threshold
    is applied later, by bin_metrics."""
    _, bout = ss.per_minute(_counts([1, 0, 0, 1, 0, 0, 0, 0, 0, 0]), threshold=5)
    assert bout[1, 0] == 2  # a 2-minute run, below threshold, still recorded
    assert bout[4, 0] == 6
    assert np.isnan(bout[[0, 2, 3, 5, 6], 0]).all()


def test_a_missing_reading_is_scored_as_zero_so_it_counts_as_sleep():
    """SCAMP has no unmeasured minute. This is the cost, made explicit."""
    sleep, _ = ss.per_minute(_counts([1, np.nan, np.nan, np.nan, np.nan, np.nan, 1]))
    assert sleep[1:6, 0].all()


def test_measured_fraction_reports_what_was_really_recorded():
    counts = _counts([1, np.nan, 1, np.nan])
    assert ss.measured_fraction(counts, slice(0, 4))[0] == pytest.approx(0.5)
    assert ss.measured_fraction(counts, slice(0, 2))[0] == pytest.approx(0.5)
    assert ss.measured_fraction(counts, slice(0, 0))[0] == 0.0


def test_a_one_dimensional_trace_is_accepted():
    sleep, bout = ss.per_minute([0] * 6 + [1])
    assert sleep.shape == (7, 1) and bout.shape == (7, 1)


# ------------------------------------------------------------- sleepcalc3 bins


# minutes 0-3 active, 4-13 a ten-minute bout, 14-19 active again. Split at
# minute 10, so the bout starts in the first bin and ends in the second.
STRADDLE = [1] * 4 + [0] * 10 + [1] * 6
BIN_A, BIN_B = slice(0, 10), slice(10, 20)


def _metrics(window, spec=STRADDLE, threshold=5):
    counts = _counts(spec)
    sleep, bout = ss.per_minute(counts, threshold=threshold)
    return {k: v[0] for k, v in ss.bin_metrics(counts, sleep, bout, window, threshold).items()}


def test_a_bout_is_credited_to_the_bin_it_starts_in():
    """sleepcalc3.m says so in its header, and it is the behaviour most easily
    lost: the run is counted once, in the bin holding its first minute."""
    assert _metrics(BIN_A)["sfreq"] == 1
    assert _metrics(BIN_B)["sfreq"] == 0


def test_but_its_minutes_fall_in_the_bin_they_occur_in():
    """stdur and sfreq deliberately disagree across the boundary: six of the
    bout's minutes are in bin A and four in bin B, while the bout itself is
    bin A's. Reading stdur as "sfreq x smeandur" is the mistake this guards."""
    assert _metrics(BIN_A)["stdur"] == 6
    assert _metrics(BIN_B)["stdur"] == 4


def test_mean_duration_is_over_the_bouts_starting_here():
    assert _metrics(BIN_A)["smeandur"] == 10


def test_a_bin_with_no_bout_reports_zero_duration_not_nan():
    """SCAMP's convention, and it is load-bearing: NaN would drop the bin out of
    the group average, whereas 0 is the answer — no sleep episodes began here."""
    assert _metrics(BIN_B)["smeandur"] == 0.0


def test_short_runs_are_not_counted_as_episodes():
    """Below-threshold runs are in bout_len (fly_histo keeps them) but must not
    reach sfreq."""
    m = _metrics(slice(0, 10), spec=[1, 0, 0, 1, 0, 0, 1, 0, 0, 1])
    assert m["sfreq"] == 0
    assert m["stdur"] == 0


def test_activity_while_awake_divides_by_awake_minutes_only():
    """oamean is counts per AWAKE minute, so the sleeping minutes must not dilute
    it. Bin A has four awake minutes carrying one count each."""
    assert _metrics(BIN_A)["oamean"] == pytest.approx(1.0)


def test_a_bin_with_no_awake_minute_reports_zero_activity():
    assert _metrics(slice(0, 10), spec=[0] * 20)["oamean"] == 0.0


def test_pdoze_and_pwake_skip_their_own_first_minute():
    """Each is seeded per bin — inactive for Pdoze, active for Pwake — so minute 0
    contributes to neither numerator nor denominator.

    On [3,2,0,0,5] (no bout: the zero run is too short) the previous-minute states
    give two eligible minutes each way and one transition each way.
    """
    m = _metrics(slice(0, 5), spec=[3, 2, 0, 0, 5])
    assert m["Pdoze"] == pytest.approx(0.5)
    assert m["Pwake"] == pytest.approx(0.5)


def test_pdoze_and_pwake_reset_at_each_bin_boundary():
    """Bin A ends asleep and bin B begins asleep. Because the state is seeded
    rather than carried across, bin B sees no active previous minute until its own
    flies wake, so its Pdoze has an empty denominator and is NaN — not a value
    borrowed from the previous bin."""
    assert np.isnan(_metrics(BIN_B, spec=[1] * 4 + [0] * 16)["Pdoze"])


def test_an_all_missing_window_is_nan_across_every_metric():
    """The one departure from SCAMP. A fly that was not recorded in this window at
    all is absent, not still: zeros would fold a flat floor into the group mean,
    where NaN leaves the fly out of this window and keeps it in every other."""
    m = _metrics(BIN_B, spec=[1] * 10 + [np.nan] * 10)
    assert all(np.isnan(v) for v in m.values()), m


def test_a_gap_inside_a_measured_window_is_still_scored_as_zero():
    """Absence and a dropout are different: the window above is NaN, this one
    keeps SCAMP's convention because the fly was recorded here."""
    m = _metrics(slice(0, 10), spec=[1] + [np.nan] * 9 + [1] * 10)
    assert m["stdur"] == 9  # the gap read as nine still minutes
    assert m["sfreq"] == 1


# ---------------------------------------------------------------- profiles


def test_profiles_sum_counts_and_minutes_per_bin():
    counts = _counts([2] * 10 + [0] * 10)
    sleep, _ = ss.per_minute(counts)
    out = ss.profiles(counts, sleep, slice(0, 20), bin_minutes=10)
    np.testing.assert_allclose(out["amean"][:, 0], [20.0, 0.0])
    np.testing.assert_allclose(out["s30"][:, 0], [0.0, 10.0])


def test_a_profile_for_an_unrecorded_window_is_nan_not_a_row_of_zeros():
    counts = _counts([np.nan] * 20)
    sleep, _ = ss.per_minute(counts)
    out = ss.profiles(counts, sleep, slice(0, 20), bin_minutes=10)
    assert np.isnan(out["amean"][:, 0]).all()
    assert np.isnan(out["s30"][:, 0]).all()


def test_a_partial_final_bin_is_dropped():
    counts = _counts([1] * 25)
    sleep, _ = ss.per_minute(counts)
    out = ss.profiles(counts, sleep, slice(0, 25), bin_minutes=10)
    assert out["amean"].shape[0] == 2


# -------------------------------------------------------------- mean and SEM


def test_mean_sem_uses_the_sample_sd():
    mean, sem, n = ss.mean_sem(np.array([[1.0], [2.0], [3.0]]), axis=0)
    assert mean[0] == pytest.approx(2.0)
    assert n[0] == 3
    assert sem[0] == pytest.approx(1.0 / np.sqrt(3))  # ddof=1


def test_mean_sem_ignores_nan_and_has_no_error_bar_for_one_fly():
    mean, sem, n = ss.mean_sem(np.array([[5.0], [np.nan], [7.0]]), axis=0)
    assert mean[0] == pytest.approx(6.0) and n[0] == 2

    mean, sem, n = ss.mean_sem(np.array([[5.0], [np.nan]]), axis=0)
    assert mean[0] == pytest.approx(5.0) and n[0] == 1
    assert np.isnan(sem[0]), "one value cannot have a standard error"


# ------------------------------------------------------------------ Holm


def test_holm_is_step_down_and_monotone():
    """[.01, .04, .03] -> [.03, .06, .06]: the third comparison's own adjustment
    is 0.04, but Holm cannot report a smaller value than an earlier step, so it
    is raised to 0.06."""
    np.testing.assert_allclose(ss.holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_holm_caps_at_one():
    np.testing.assert_allclose(ss.holm_adjust([0.5, 0.6]), [1.0, 1.0])


def test_holm_passes_nan_through():
    out = ss.holm_adjust([0.01, np.nan, 0.02])
    assert np.isnan(out[1])
    # The family is the two real p-values, so the multiplier is 2 and not 3.
    assert out[0] == pytest.approx(0.02)
    assert ss.holm_adjust([np.nan, np.nan]).tolist() != [0.0, 0.0]
    assert np.isnan(ss.holm_adjust([np.nan, np.nan])).all()


def test_holm_of_a_single_comparison_is_unchanged():
    np.testing.assert_allclose(ss.holm_adjust([0.02]), [0.02])


# -------------------------------------------------------- pairwise tests


def test_pairwise_reports_both_tests_for_every_pair():
    rng = np.random.default_rng(0)
    per_fly = {
        "a": rng.normal(10, 1, 12),
        "b": rng.normal(14, 1, 12),
        "c": rng.normal(10.2, 1, 12),
    }
    rows = ss.pairwise_tests(per_fly)
    assert [(r["group_a"], r["group_b"]) for r in rows] == [
        ("a", "b"), ("a", "c"), ("b", "c")
    ]
    for r in rows:
        assert np.isfinite(r["p_welch"]) and np.isfinite(r["p_mannwhitney"])
        assert r["p_welch_holm"] >= r["p_welch"]  # correction never lowers a p
    # a-vs-b is the real difference; a-vs-c is not, and Holm keeps that ordering.
    by_pair = {(r["group_a"], r["group_b"]): r for r in rows}
    assert by_pair[("a", "b")]["p_welch_holm"] < 0.05
    assert by_pair[("a", "c")]["p_welch_holm"] > 0.05


def test_pairwise_drops_nan_flies_and_reports_the_n_it_used():
    rows = ss.pairwise_tests({"a": [1.0, 2.0, np.nan, 3.0], "b": [5.0, 6.0, 7.0]})
    assert rows[0]["n_a"] == 3 and rows[0]["n_b"] == 3


def test_pairwise_needs_two_flies_a_side():
    rows = ss.pairwise_tests({"a": [1.0], "b": [5.0, 6.0, 7.0]})
    assert rows[0]["n_a"] == 1
    assert np.isnan(rows[0]["p_welch"]) and np.isnan(rows[0]["p_mannwhitney"])


def test_pairwise_survives_two_identical_constant_samples():
    """Zero variance both sides. Welch is 0/0 and comes back NaN; Mann-Whitney is
    still defined and reports no difference. Neither may take the page down, and
    the NaN must stay NaN rather than being filled in as a significant result.

    (``pairwise_tests`` also catches ValueError around ``mannwhitneyu`` for this
    input. scipy 1.15 returns p=1.0 instead of raising, so that branch is now
    dead on this version — it is kept because older scipy did raise, and the cost
    of keeping it is nothing.)
    """
    row = ss.pairwise_tests({"a": [2.0, 2.0, 2.0], "b": [2.0, 2.0, 2.0]})[0]
    assert np.isnan(row["p_welch"]) and np.isnan(row["p_welch_holm"])
    assert row["p_mannwhitney"] == pytest.approx(1.0)
    assert row["mean_a"] == 2.0 and row["mean_b"] == 2.0
