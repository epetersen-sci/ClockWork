"""Group-mean actograms, following SCAMP's ``actogram2.m`` / ``dam_actogram2.m``.

The lab's conventions, taken from those two files so the output is comparable to a
SCAMP actogram rather than merely actogram-shaped:

* **Group mean, not per fly.** ``dam_actogram2`` does ``f = mean(f, 2)`` over the
  selected wells and plots the one averaged trace.
* **Binned first.** SCAMP hands the actogram ``s30`` — activity summed into 30-minute
  bins — and passes that bin width as ``o.int``. 30 minutes is the default here too.
* **Double-plotted.** With ``reps = 2`` each row shows its own day followed by the
  next one (``xx = [xx, y]`` where ``y`` is the day below). The last row's second
  half is empty because no day follows it.
* **One amplitude scale for the whole panel.** ``x = x - min``, then rows are stacked
  ``shift = 1.1 * (max - min)`` apart, so every day is drawn against the same scale
  and 10% of the row height is left as headroom.
* **Day 1 at the top**, ticks every 6 hours labelled modulo 24.

Beyond SCAMP: :func:`group_split_minutes` and :func:`ld_bin_mask` label which plotted
bins fall before each group's release into DD, so the entrained part of a recording can
be drawn in its own colour. The label is per bin rather than per row — a double-plotted
row can straddle the release.

What is deliberately NOT copied: ``actogram2`` pads a partial final day with zeros.
Zero means "no activity" while a missing reading means "we do not know" (§2a), so a
short final day is left NaN here and simply draws nothing.
"""

import numpy as np

MINUTES_PER_DAY = 1440
DEFAULT_BIN_MINUTES = 30  # SCAMP's s30
DEFAULT_REPS = 2  # double-plotted, actogram2's default


def _bin_sum(values, bins_per_day, bin_len):
    """Sum each fly's activity within a bin; a bin with no reading at all stays NaN."""
    usable = (values.shape[0] // bin_len) * bin_len
    block = values[:usable].reshape(-1, bin_len, values.shape[1])
    finite = np.isfinite(block)
    summed = np.where(finite, block, 0.0).sum(axis=1)
    return np.where(finite.any(axis=1), summed, np.nan)


def compute_group_actograms(
    ds,
    group_by,
    *,
    bin_minutes=DEFAULT_BIN_MINUTES,
    activity_var="activity",
):
    """Binned group-mean activity per group, shaped ``(n_days, bins_per_day)``.

    Parameters
    ----------
    ds : xr.Dataset
        Whole dataset on a relative-integer-minute time axis.
    group_by : sequence of str
        Coordinates whose combination defines a group.
    bin_minutes : int
        Bin width. Must divide 1440. SCAMP uses 30.

    Returns
    -------
    dict
        ``{group label: {"matrix": (n_days, bins_per_day) float array,
        "n_flies": int}}``, plus the key ``"_params"`` carrying ``bin_minutes``,
        ``bins_per_day`` and ``group_by``.
    """
    from phase_shift import group_labels

    if activity_var not in ds:
        raise KeyError(f"{activity_var!r} not in dataset")
    if MINUTES_PER_DAY % int(bin_minutes):
        raise ValueError(f"bin_minutes must divide {MINUTES_PER_DAY}, got {bin_minutes}")
    time_vals = np.asarray(ds["time"].values)
    if not np.issubdtype(time_vals.dtype, np.integer):
        raise ValueError(
            "compute_group_actograms requires a relative-integer-minute time axis; "
            "convert with dam_utilities.convert_to_relative_time first."
        )

    bin_len = int(bin_minutes)
    bins_per_day = MINUTES_PER_DAY // bin_len
    labels, cols = group_labels(ds, group_by)
    activity = np.asarray(ds[activity_var].transpose("time", "id").values, dtype=float)

    out = {"_params": {
        "bin_minutes": bin_len,
        "bins_per_day": bins_per_day,
        "group_by": list(cols),
    }}
    for grp in sorted(set(labels)):
        member = labels == grp
        binned = _bin_sum(activity[:, member], bins_per_day, bin_len)
        # Mean over the flies actually measured in each bin — a bin measured in no
        # fly stays NaN, never 0. Divided explicitly rather than through nanmean,
        # which warns "Mean of empty slice" on precisely the all-missing bins this
        # is built to keep, and would have to be silenced to stay quiet.
        measured = np.isfinite(binned)
        n_measured = measured.sum(axis=1)
        totals = np.where(measured, binned, 0.0).sum(axis=1)
        mean_trace = np.divide(
            totals,
            n_measured,
            out=np.full(totals.shape, np.nan),
            where=n_measured > 0,
        )
        n_days = mean_trace.shape[0] // bins_per_day
        matrix = mean_trace[: n_days * bins_per_day].reshape(n_days, bins_per_day)
        out[grp] = {"matrix": matrix, "n_flies": int(member.sum())}
    return out


def actogram_rows(matrix, reps=DEFAULT_REPS):
    """Double-plot: row *i* is day *i* followed by day *i+1* (``reps=2``).

    The trailing rows have no day after them, so their tail is NaN — SCAMP writes
    zeros there, which would draw a flat floor implying measured inactivity.
    """
    n_days, per_day = matrix.shape
    rows = np.full((n_days, per_day * reps), np.nan)
    for i in range(n_days):
        for r in range(reps):
            if i + r < n_days:
                rows[i, r * per_day : (r + 1) * per_day] = matrix[i + r]
    return rows


def group_split_minutes(ds, group_by):
    """Per-group LD→DD boundary in relative minutes.

    Reads the per-fly ``split_minute`` coordinate (deriving it from ``first_DD_day``
    if it has not been attached yet). A group is given the **earliest** boundary
    among its flies, so a bin is only ever called LD when every fly in the group was
    still entrained during it — the conservative direction: an LD bin can be missed,
    never invented.

    Returns
    -------
    (dict, set)
        ``{group label: split_minute (float) or None}`` and the set of groups whose
        flies do not share one boundary (the caller should say so — their highlight
        stops at the earliest fly's release).
        ``({}, set())`` when the dataset carries no phase metadata at all.
    """
    import dam_utilities
    from phase_shift import group_labels

    if "split_minute" in ds.coords:
        split = np.asarray(ds["split_minute"].values, dtype=float)
    elif "first_DD_day" in ds.coords:
        try:
            split = np.asarray(
                dam_utilities.add_phase_metadata(ds)["split_minute"].values, dtype=float
            )
        except Exception:
            return {}, set()
    else:
        return {}, set()

    labels, _ = group_labels(ds, group_by)
    out, disagreeing = {}, set()
    for grp in sorted(set(labels)):
        vals = split[labels == grp]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            out[grp] = None
            continue
        if np.unique(vals).size > 1:
            disagreeing.add(grp)
        out[grp] = float(vals.min())
    return out, disagreeing


def ld_bin_mask(n_days, bins_per_day, reps, bin_minutes, split_minute):
    """Boolean ``(n_days, bins_per_day * reps)`` — True where a plotted bin is LD.

    Shaped to match :func:`actogram_rows`, so element ``[i, j]`` labels exactly the
    bar that row *i*, bin *j* draws. Because the rows are double-plotted, one row can
    straddle the release into DD; the mask is therefore per bin, not per row.

    A bin counts as LD only when its **whole** span falls before ``split_minute``.
    The bin containing the transition is left unhighlighted rather than assigned to
    one epoch it only partly belongs to (the "mixed" case in ``phase_shift._day_phase``).

    ``split_minute=None`` returns an all-False mask; ``np.inf`` marks everything LD
    (an LD-only partition).
    """
    width = bins_per_day * reps
    if split_minute is None or not np.isfinite(split_minute):
        return np.full((n_days, width), bool(split_minute == np.inf))
    j = np.arange(width)
    starts = (np.arange(n_days)[:, None] + (j // bins_per_day)[None, :]) * MINUTES_PER_DAY + (
        j % bins_per_day
    )[None, :] * bin_minutes
    return (starts + bin_minutes) <= float(split_minute)


def actogram_scale(matrix):
    """``(vmin, shift)`` — SCAMP's stacking geometry for a whole panel.

    ``shift = 1.1 * (max - min)`` over every day at once, so the rows share one
    amplitude scale and cannot be compared misleadingly against each other.
    """
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = float(finite.min()), float(finite.max())
    shift = 1.1 * (vmax - vmin)
    return vmin, (shift if shift > 0 else 1.0)
