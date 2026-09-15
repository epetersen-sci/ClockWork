"""Sleep and activity metrics computed exactly as SCAMP's ``sleepcalc3.m`` does.

Every definition here was read off the SCAMP sources rather than reimplemented from
the literature, so a number produced here can be compared with a SCAMP run:

``fly_sleepthresh.m``
    Sleep is a run of >= ``threshold`` consecutive minutes with zero beam crosses.
    Every minute of a qualifying run is marked asleep. No gap bridging.
``fly_histo.m``
    At the first minute of each inactivity run, the run's length in minutes; NaN
    everywhere else. The sleep threshold is NOT applied here.
``sleepcalc3.m``
    ``amean``    sum of counts in the bin. Named "mean" but summed since 2010.
    ``s30``      minutes asleep per 30-minute bin.
    ``stdur``    minutes asleep in the bin.
    ``sfreq``    number of inactivity runs >= threshold STARTING in the bin.
    ``smeandur`` mean length of those runs; a bin with no bout reports 0.
    ``oamean``   mean counts per awake minute; a bin with no awake minute reports 0.
    ``Pdoze``    P(inactive now | active in the previous minute), within the bin.
    ``Pwake``    P(active now | inactive in the previous minute), within the bin.

Two SCAMP behaviours that are easy to get wrong and are reproduced deliberately:

* A bout is credited to the bin it STARTS in, even if it runs past the boundary
  (``sleepcalc3.m`` says so in its header).
* ``Pdoze`` and ``Pwake`` reset their "previous minute" state at the start of each
  bin — ``active`` is initialised inside the per-bin loop, to 0 for Pdoze and 1 for
  Pwake, so each one ignores its own first minute.

Missing readings are scored as zero counts, matching SCAMP, which has no concept of
an unmeasured minute. That makes a dropout indistinguishable from stillness and so
counts toward sleep; :func:`measured_fraction` reports how much of a window was
really measured so the cost is visible rather than hidden.

The one departure: a fly with NO measured minute in a window gets NaN there, not a
zero. SCAMP never met that case because it reads one plate at a time; a ClockWork
dataset aligned on ``first_DD_day`` pads the boxes with a shorter LD run, so those
flies have days they were not recorded on at all. Scoring absence as "asleep, no
counts" would fold a flat zero into the group mean, whereas NaN leaves the fly out
of that day and keeps it in every other.
"""

import warnings

import numpy as np

SLEEP_THRESHOLD_MIN = 5
PROFILE_BIN_MINUTES = 30

# key -> (axis label, whether bigger bins make sense, number of decimals)
METRIC_LABELS = {
    "stdur": "Total Sleep Duration (min)",
    "sfreq": "Number of Sleep Episodes",
    "smeandur": "Mean Sleep Episode Duration (min)",
    "oamean": "Activity while Awake (beam crosses/min)",
    "Pdoze": "Pdoze",
    "Pwake": "Pwake",
}
PROFILE_LABELS = {
    "amean": "Activity Counts/30 Mins",
    "s30": "Sleep (min/30min)",
}


def _zero_runs(col):
    """Start and end index of every zero-run in ``col`` (end exclusive)."""
    zero = (col == 0).astype(np.int8)
    edges = np.diff(np.concatenate(([0], zero, [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def _runs_of_zero(col):
    """``fly_histo``: length of each zero-run at its first minute, NaN elsewhere."""
    out = np.full(col.size, np.nan)
    starts, ends = _zero_runs(col)
    out[starts] = ends - starts
    return out


def per_minute(counts, threshold=SLEEP_THRESHOLD_MIN):
    """Per-minute sleep mask and bout-start lengths for a (time, fly) count matrix.

    Returns ``(sleep, bout_len)``: ``sleep`` is bool (time, fly), ``bout_len`` holds
    each inactivity run's length at its starting minute and NaN elsewhere — the
    ``fly_histo`` matrix, before the threshold is applied.
    """
    counts = np.asarray(counts, dtype=float)
    if counts.ndim == 1:
        counts = counts[:, None]
    counts = np.nan_to_num(counts, nan=0.0)  # SCAMP has no missing-minute concept

    n_time, n_fly = counts.shape
    sleep = np.zeros((n_time, n_fly), dtype=bool)
    bout_len = np.full((n_time, n_fly), np.nan)
    for f in range(n_fly):
        starts, ends = _zero_runs(counts[:, f])
        bout_len[starts, f] = ends - starts
        long_enough = (ends - starts) >= threshold
        for s, e in zip(starts[long_enough], ends[long_enough]):
            sleep[s:e, f] = True
    return sleep, bout_len


def measured_fraction(counts, window):
    """Fraction of minutes in ``window`` that carried a real reading, per fly.

    SCAMP scores a missing minute as zero, i.e. as sleep. This is what that costs.
    """
    block = np.asarray(counts, dtype=float)[window]
    if block.ndim == 1:
        block = block[:, None]
    if block.shape[0] == 0:
        return np.zeros(block.shape[1])
    return np.isfinite(block).mean(axis=0)


def _pdoze_pwake(active):
    """``(Pdoze, Pwake)`` per fly for one bin. ``active`` is bool (time, fly).

    Mirrors the MATLAB loops: the state carried into minute k is minute k-1's, and
    the state is seeded per bin — inactive for Pdoze, active for Pwake — so each
    measure skips its own first minute.
    """
    n_time, n_fly = active.shape
    if n_time == 0:
        return np.full(n_fly, np.nan), np.full(n_fly, np.nan)

    prev_active = np.zeros_like(active)
    prev_active[1:] = active[:-1]  # seeded inactive: minute 0 contributes nothing

    prev_inactive = np.zeros_like(active)
    prev_inactive[1:] = ~active[:-1]  # seeded active: minute 0 contributes nothing

    doze_den = prev_active.sum(axis=0)
    doze_num = (prev_active & ~active).sum(axis=0)
    wake_den = prev_inactive.sum(axis=0)
    wake_num = (prev_inactive & active).sum(axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        pdoze = np.where(doze_den > 0, doze_num / np.maximum(doze_den, 1), np.nan)
        pwake = np.where(wake_den > 0, wake_num / np.maximum(wake_den, 1), np.nan)
    return pdoze, pwake


def bin_metrics(counts, sleep, bout_len, window, threshold=SLEEP_THRESHOLD_MIN):
    """The six per-bin metrics, per fly, over the minute range ``window``.

    ``window`` is a slice or index array over the time axis.

    A fly with NO measured minute in ``window`` gets NaN for all six metrics rather
    than the zeros that scoring missing minutes as "asleep, no counts" would produce.
    Scoring a gap that way is SCAMP's convention and stays; a window the fly was
    never recorded in at all is absence, not a gap, and has to drop out of the
    averages instead of entering them as a flat zero. This is what a cohort aligned
    on ``first_DD_day`` produces: the box with the shorter LD run has no data at all
    on the first LD day.
    """
    counts = np.asarray(counts, dtype=float)
    if counts.ndim == 1:
        counts = counts[:, None]
    measured = np.isfinite(counts[window]).any(axis=0)
    counts = np.nan_to_num(counts, nan=0.0)
    c = counts[window]
    s = sleep[window]
    b = bout_len[window]
    active = c > 0

    stdur = s.sum(axis=0).astype(float)

    # Only runs that both start in this bin AND reach the threshold count as bouts.
    is_bout = np.isfinite(b) & (b >= threshold)
    sfreq = is_bout.sum(axis=0).astype(float)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN bin, guarded below
        lengths = np.where(is_bout, b, np.nan)
        smeandur = np.where(
            is_bout.any(axis=0), np.nanmean(np.where(is_bout, lengths, np.nan), axis=0), 0.0
        )

    # Mean counts per AWAKE minute; SCAMP reports 0 for a bin with no awake minute.
    awake = ~s
    with np.errstate(invalid="ignore"):
        oamean = np.where(
            awake.any(axis=0),
            np.nansum(np.where(awake, c, 0.0), axis=0) / np.maximum(awake.sum(axis=0), 1),
            0.0,
        )

    pdoze, pwake = _pdoze_pwake(active)
    out = {
        "stdur": stdur,
        "sfreq": sfreq,
        "smeandur": np.nan_to_num(smeandur, nan=0.0),
        "oamean": np.nan_to_num(oamean, nan=0.0),
        "Pdoze": pdoze,
        "Pwake": pwake,
    }
    return {k: np.where(measured, v, np.nan) for k, v in out.items()}


def profiles(counts, sleep, window, bin_minutes=PROFILE_BIN_MINUTES):
    """``amean`` and ``s30`` across ``window``, shaped (n_bins, n_flies).

    ``amean`` is the SUM of counts per bin and ``s30`` the minutes asleep per bin —
    the two traces SCAMP plots against ZT/CT.

    Missing minutes inside ``window`` count as no beam breaks (NaN -> 0), SCAMP's
    convention, so a gap mid-record reads the same as it always did. A fly with NO
    measured minute in the whole window is the exception: its trace is NaN, not a
    row of zeros. That case only appears once flies share an axis — on a cohort
    aligned by ``first_DD_day`` the box with the shorter LD run has no data at all
    on the first LD day, and zero-filling it would draw a flat zero into that day's
    group mean instead of leaving the fly out of it (see :func:`bin_metrics`).
    """
    counts = np.asarray(counts, dtype=float)
    if counts.ndim == 1:
        counts = counts[:, None]
    c = counts[window]
    measured = np.isfinite(c).any(axis=0)  # per fly, over the whole window
    c = np.nan_to_num(c, nan=0.0)
    s = sleep[window].astype(float)
    usable = (c.shape[0] // bin_minutes) * bin_minutes
    c = c[:usable].reshape(-1, bin_minutes, c.shape[1])
    s = s[:usable].reshape(-1, bin_minutes, s.shape[1])
    return {
        "amean": np.where(measured, c.sum(axis=1), np.nan),
        "s30": np.where(measured, s.sum(axis=1), np.nan),
    }


def mean_sem(values, axis=0):
    """Group mean and standard error, ignoring NaN — the error bars SCAMP draws."""
    values = np.asarray(values, dtype=float)
    n = np.isfinite(values).sum(axis=axis)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(values, axis=axis)
        sd = np.nanstd(values, axis=axis, ddof=1)
        sem = np.where(n > 1, sd / np.sqrt(np.maximum(n, 1)), np.nan)
    return mean, sem, n


def holm_adjust(pvals):
    """Holm-Bonferroni step-down adjusted p-values, NaNs passed through."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    idx = np.flatnonzero(np.isfinite(p))
    if idx.size == 0:
        return out
    order = idx[np.argsort(p[idx])]
    m = order.size
    running = 0.0
    for rank, i in enumerate(order):
        adj = (m - rank) * p[i]
        running = max(running, adj)
        out[i] = min(1.0, running)
    return out


def pairwise_tests(per_fly):
    """All pairwise comparisons between groups for one metric in one bin.

    ``per_fly`` maps group label -> 1-D array of that group's per-fly values.
    Returns a list of dicts carrying both a Welch t-test (unequal variances, no
    assumption that the groups are the same size) and a Mann-Whitney U (no
    assumption of normality), each with its own Holm-corrected p across the
    comparisons in this family. Both are reported because which one is appropriate
    depends on the metric: Pdoze/Pwake are bounded proportions, sleep durations are
    not, and neither is reliably normal at n = 12.
    """
    from scipy import stats as _st

    groups = sorted(per_fly)
    rows = []
    for i, a in enumerate(groups):
        for b in groups[i + 1 :]:
            xa = np.asarray(per_fly[a], dtype=float)
            xb = np.asarray(per_fly[b], dtype=float)
            xa = xa[np.isfinite(xa)]
            xb = xb[np.isfinite(xb)]
            row = {
                "group_a": a,
                "group_b": b,
                "n_a": int(xa.size),
                "n_b": int(xb.size),
                "mean_a": float(np.mean(xa)) if xa.size else np.nan,
                "mean_b": float(np.mean(xb)) if xb.size else np.nan,
                "t": np.nan,
                "p_welch": np.nan,
                "U": np.nan,
                "p_mannwhitney": np.nan,
            }
            if xa.size >= 2 and xb.size >= 2:
                with np.errstate(invalid="ignore"):
                    t, p = _st.ttest_ind(xa, xb, equal_var=False)
                row["t"], row["p_welch"] = float(t), float(p)
                try:
                    u, pu = _st.mannwhitneyu(xa, xb, alternative="two-sided")
                    row["U"], row["p_mannwhitney"] = float(u), float(pu)
                except ValueError:
                    pass  # identical constant samples
            rows.append(row)

    for src, dst in (("p_welch", "p_welch_holm"), ("p_mannwhitney", "p_mannwhitney_holm")):
        adjusted = holm_adjust([r[src] for r in rows])
        for r, value in zip(rows, adjusted):
            r[dst] = float(value) if np.isfinite(value) else np.nan
    return rows
