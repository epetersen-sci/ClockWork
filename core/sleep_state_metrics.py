"""Per-sleep-state metrics behind the Abhilash et al. 2026 figures.

Four docstrings in ``plotting.py`` already named this module as the source of
their input frames (``sleep_state_metrics.compute_normalized_waveforms`` and
friends) but it was never written, so every one of those renderers was
unreachable: there was no code anywhere that could build the frames they
describe. This is that layer.

Everything here reduces the per-minute state masks ``sleep`` / ``sleep_short``
/ ``sleep_intermediate`` / ``sleep_long`` written by :mod:`sleep_analysis`, plus
the per-bout table (``start_time`` / ``duration`` / ``sleep_state``), down to
the tidy frames the figures consume. Functions take an ALREADY PHASE-SLICED
dataset — the caller picks DD or LD (see ``dam_utilities.select_phase``) — for
the same reason ``group_ridge_density_plotly`` does: mixing a phase filter into
a compute helper hides which epoch a figure is actually showing.

Conventions taken from the paper's STAR Methods, and the places we knowingly
differ from it, are documented per function. Two run throughout:

- **ZT/CT minute is ``time % 1440``**, matching ``get_zt_binned_dataframe``.
  Relative minute 0 is each fly's ``start_datetime``, i.e. lights-on, so ZT00
  and (in DD, where the paper defines CT00 as the time lights *would* have come
  on) CT00 both land on the same clock. The two timescales differ in label
  only, which is why one set of helpers serves both.
- **The -1 missing sentinel is dropped, never counted.** A bin overlapping a
  data gap reports its value over measured minutes only.
"""

import numpy as np
import pandas as pd

from dam_utilities import get_zt_binned_dataframe

# The paper's colour assignments, read off Figures 1-6. These are load-bearing
# for a reader comparing our output to the printed figures, so they live here
# rather than being re-guessed per plot: activity is red in every panel it
# appears in, and short/intermediate/long are orange/green/blue throughout.
# The previous renderers used Plotly's default cycle, which put short sleep in
# blue and long sleep in red — the paper's colours for long sleep and activity
# respectively, i.e. exactly inverted for the two states a reader most wants to
# tell apart.
STATE_COLORS = {
    "activity": "#E03127",
    "standard": "#7F7F7F",
    "short": "#E8730C",
    "intermediate": "#2E9B47",
    "long": "#3B4DA0",
}

STATE_LABELS = {
    "activity": "Locomotor activity",
    "standard": "Standard sleep (>5-min)",
    "short": "Short sleep (5 to 30-min)",
    "intermediate": "Inter. sleep (30 to 60-min)",
    "long": "Long sleep (>60-min)",
}

# Order used for every state-faceted figure, matching the columns of Figure 3A.
STATE_ORDER = ("standard", "short", "intermediate", "long")

# 'standard' is the >=5-min mask, i.e. sleep by the field's unitary definition.
STATE_VARS = {
    "standard": "sleep",
    "short": "sleep_short",
    "intermediate": "sleep_intermediate",
    "long": "sleep_long",
}

MINUTES_PER_DAY = 1440


def available_states(ds, states=STATE_ORDER):
    """Which of ``states`` this dataset actually carries a mask for."""
    if ds is None:
        return []
    return [s for s in states if STATE_VARS.get(s) in ds.data_vars]


def _zt_minutes_of(ds, values):
    """ZT/CT minute-of-day for bout timestamps, in either time representation.

    ``start_time`` carries whatever the ``time`` coord uses — relative integer
    minutes for a normal import, datetime64 for an absolute-time dataset — so
    both have to reduce to the same 0-1439 minute-of-day used by the profiles.
    """
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.integer) or np.issubdtype(values.dtype, np.floating):
        return values.astype(float) % MINUTES_PER_DAY
    # Select by dimension name rather than a bare [0]: this is the first fly's
    # start, and `get_zt_binned_dataframe` uses the same reference so absolute-
    # time datasets bin identically here and there.
    ref_start = pd.to_datetime(ds["start_datetime"].isel(id=0).values)
    deltas = (pd.to_datetime(values) - ref_start).total_seconds() / 60.0
    return np.asarray(deltas, dtype=float) % MINUTES_PER_DAY


def _group_lookup(ds):
    """id -> group label, falling back to a single pooled group."""
    if "group" in ds.coords:
        return {str(i): str(g) for i, g in zip(ds["id"].values, ds["group"].values)}
    return {str(i): "All Flies" for i in ds["id"].values}


# ---------------------------------------------------------------------------
# Profiles (Figure 2, left columns; the input to the rose plots)
# ---------------------------------------------------------------------------


def state_profiles(ds, bin_size_min=30, states=STATE_ORDER, include_activity=True):
    """Per-fly daily profile of each sleep state, averaged over days.

    This is the quantity the paper's rose plots are built from: "We calculated
    sleep time series, binned at 30-minute intervals... These were then
    averaged over days and across flies." Averaging across flies is left to
    the caller (:func:`group_profiles`) so the per-fly rows stay available for
    the circular statistics, which need them.

    Units follow the paper's axes rather than the raw masks: sleep is
    **minutes per hour** (a bin whose flies slept throughout reads 60) and
    activity is **counts per hour**. Both come from a per-minute mean, so the
    figure is unaffected by the bin width chosen here.

    Parameters
    ----------
    ds : xr.Dataset
        Phase-sliced, with the state masks from ``sleep_analysis``.
    bin_size_min : int
        Profile bin width in minutes. Default 30, the paper's choice — "a
        reasonable compromise between visualizing temporal patterns... without
        losing nuance to smoothing".
    states : sequence of str
    include_activity : bool
        Add an ``'activity'`` state, which every rose-plot panel overlays.

    Returns
    -------
    pd.DataFrame
        Columns ``id``, ``group``, ``state``, ``zt_bin_minute``, ``value``.
    """
    empty = pd.DataFrame(columns=["id", "group", "state", "zt_bin_minute", "value"])
    if ds is None:
        return empty

    groups = _group_lookup(ds)
    wanted = list(available_states(ds, states))
    if include_activity and "activity" in ds.data_vars:
        wanted = ["activity"] + wanted
    if not wanted:
        return empty

    frames = []
    for state in wanted:
        var = "activity" if state == "activity" else STATE_VARS[state]
        binned = get_zt_binned_dataframe(
            ds, value_col=var, bin_size_minutes=bin_size_min, bin_function="mean"
        )
        if binned.empty:
            continue
        # get_zt_binned_dataframe returns a per-minute mean: for a 0/1 mask that
        # is the fraction of the bin spent in the state, and for activity it is
        # counts per minute. Both scale to per-hour the same way.
        binned = binned.rename(columns={var: "value"})
        binned["value"] = binned["value"].astype(float) * 60.0
        binned["state"] = state
        frames.append(binned[["id", "zt_bin_minute", "value", "state"]])

    if not frames:
        return empty
    out = pd.concat(frames, ignore_index=True)
    out["id"] = out["id"].astype(str)
    out["group"] = out["id"].map(groups).fillna("All Flies")
    return out[["id", "group", "state", "zt_bin_minute", "value"]]


def group_profiles(profile_df):
    """Mean +/- SEM across flies, per group / state / bin.

    The paper's Figure 2 left column and the rose plots both show this: the
    per-fly profiles averaged across flies. SEM is across flies.
    """
    empty = pd.DataFrame(columns=["group", "state", "zt_bin_minute", "mean", "sem", "n"])
    if profile_df is None or profile_df.empty:
        return empty
    agg = (
        profile_df.groupby(["group", "state", "zt_bin_minute"])["value"]
        .agg(["mean", "sem", "count"])
        .reset_index()
        .rename(columns={"count": "n"})
    )
    return agg


def compute_normalized_waveforms(ds, bin_size_min=30, states=STATE_ORDER):
    """Waveforms normalised to their own maximum, as in Figure 1B.

    The paper's recipe, verbatim: "For each sleep state and each experiment,
    the average trace across replicate flies was normalized to the maxima,
    such that all the waveforms range from 0 to 1. The normalized waveforms
    were then averaged across replicate experiments and errors computed."

    Note the ORDER — average across flies FIRST, then divide by the max of that
    averaged trace. Normalising each fly and then averaging is a different
    quantity (every fly's own peak becomes 1.0, which flattens between-fly
    differences in how sharply peaked the profile is), and it is the mistake
    this docstring exists to prevent.

    **Where we differ from the paper.** Its outer average is over three
    independent runs, so its error band is between-run SEM. A ClockWork dataset
    is normally one run, and the unit available here is the GROUP. So each
    group is normalised on its own averaged trace and the reported ``sem`` is
    between-FLY SEM propagated through that group's scale factor. It answers
    "how consistent are these flies" and not the paper's "how consistent are
    these runs"; treat the band as the narrower of the two claims.

    Returns
    -------
    pd.DataFrame
        Columns ``group``, ``state``, ``zt_bin_minute``, ``mean_normalized``,
        ``sem_normalized``, ``scale`` (the divisor, in min/h, so a reader can
        recover absolute values).
    """
    empty = pd.DataFrame(
        columns=[
            "group",
            "state",
            "zt_bin_minute",
            "mean_normalized",
            "sem_normalized",
            "scale",
        ]
    )
    profiles = state_profiles(ds, bin_size_min=bin_size_min, states=states, include_activity=False)
    if profiles.empty:
        return empty

    stats = group_profiles(profiles)
    rows = []
    for (group, state), sdf in stats.groupby(["group", "state"], sort=False):
        sdf = sdf.sort_values("zt_bin_minute")
        peak = float(np.nanmax(sdf["mean"].values)) if len(sdf) else np.nan
        if not np.isfinite(peak) or peak <= 0:
            # A state with no sleep at all has no maximum to normalise to.
            # Emitting zeros would draw a flat line at 0 that looks like data;
            # skipping the state leaves it out of the legend instead.
            continue
        rows.append(
            pd.DataFrame(
                {
                    "group": group,
                    "state": state,
                    "zt_bin_minute": sdf["zt_bin_minute"].values,
                    "mean_normalized": sdf["mean"].values / peak,
                    "sem_normalized": sdf["sem"].values / peak,
                    "scale": peak,
                }
            )
        )
    if not rows:
        return empty
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Probability of bout initiation (Figure 2, right columns)
# ---------------------------------------------------------------------------


def compute_initiation_probability(ds, bin_hours=1, states=STATE_ORDER):
    """Per-fly probability that a bout of each state is initiated in each hour.

    The paper: "a list of all the standard, short, intermediate, or long sleep
    bouts that were initiated at each minute was curated, and the CT/ZT at
    which sleep bouts were initiated was noted... we counted the number of
    sleep bouts that were initiated in every 1-h interval starting at CT/ZT01
    and proceeding until CT/ZT24. Then, the total number of bouts initiated in
    each 1-h window was divided by the total number of sleep bouts of the
    respective state that the fly showed."

    So the denominator is **per fly and per state** — each fly's own bout
    count, which is what makes it a probability and not a count. Every fly
    therefore contributes a curve summing to 1 regardless of how much it slept,
    and the between-fly SEM is comparable across states.

    Bins are labelled by their END hour, 1 through 24: bin 1 covers ZT/CT
    minutes 0-59. That is the paper's "starting at 01 and proceeding until 24"
    and matches the right-labelled bins the authors' own ``phase`` package uses
    for rose wedges.

    'standard' bouts are every bout >=5 min, so a bout is counted once under
    'standard' and once under whichever of short/intermediate/long it falls in.

    Returns
    -------
    pd.DataFrame
        Long, one row per (fly, state, bin): ``id``, ``group``, ``state``,
        ``bin_hour``, ``probability``, ``n_bouts`` (that fly's total for the
        state, i.e. the denominator).
    """
    empty = pd.DataFrame(
        columns=["id", "group", "state", "bin_hour", "probability", "n_bouts"]
    )
    if ds is None or "start_time" not in ds.data_vars or "sleep_state" not in ds.data_vars:
        return empty

    bouts = ds[["start_time", "sleep_state"]].to_dataframe().reset_index()
    bouts = bouts.dropna(subset=["start_time"])
    if bouts.empty:
        return empty

    bouts["id"] = bouts["id"].astype(str)
    bouts["zt_minute"] = _zt_minutes_of(ds, bouts["start_time"].values)
    n_bins = int(round(24 / bin_hours))
    bin_width = bin_hours * 60.0
    bouts["bin_hour"] = (
        np.clip((bouts["zt_minute"] // bin_width).astype(int), 0, n_bins - 1) + 1
    ) * bin_hours

    groups = _group_lookup(ds)
    wanted = [s for s in states if s in ("standard",) or s in available_states(ds, states)]
    bin_labels = (np.arange(n_bins) + 1) * bin_hours

    rows = []
    for fly, fdf in bouts.groupby("id", sort=False):
        for state in wanted:
            # 'standard' is every detected bout; the others select on the label.
            sdf = fdf if state == "standard" else fdf[fdf["sleep_state"] == state]
            total = len(sdf)
            if total == 0:
                # No bouts of this state means the probability is undefined for
                # this fly, not zero — a fly that never sleeps long must not
                # drag the group's P(long) toward zero. Left out of the mean.
                continue
            counts = (
                sdf.groupby("bin_hour").size().reindex(bin_labels, fill_value=0).values
            )
            rows.append(
                pd.DataFrame(
                    {
                        "id": fly,
                        "group": groups.get(fly, "All Flies"),
                        "state": state,
                        "bin_hour": bin_labels,
                        "probability": counts / float(total),
                        "n_bouts": total,
                    }
                )
            )
    if not rows:
        return empty
    return pd.concat(rows, ignore_index=True)


def group_initiation_probability(init_df):
    """Mean +/- SEM across flies of :func:`compute_initiation_probability`."""
    empty = pd.DataFrame(columns=["group", "state", "bin_hour", "mean", "sem", "n"])
    if init_df is None or init_df.empty:
        return empty
    return (
        init_df.groupby(["group", "state", "bin_hour"])["probability"]
        .agg(["mean", "sem", "count"])
        .reset_index()
        .rename(columns={"count": "n"})
    )


# ---------------------------------------------------------------------------
# Circular statistics (Figure 3C)
# ---------------------------------------------------------------------------


def _weighted_circular_mean(theta_deg, weights):
    """Centre of mass of a circular profile: mean angle and concentration.

    Ports the authors' own ``phase::CoM``, which is the standard weighted
    resultant vector:

        X = sum(w * cos(theta)) / sum(w)
        Y = sum(w * sin(theta)) / sum(w)
        theta_bar = atan2(Y, X);  r = hypot(X, Y)

    ``r`` is the paper's "concentration parameter" and is bounded 0-1: 1 means
    all the sleep in one bin, 0 means it is spread evenly around the clock.
    ``CoM.R`` branches on the sign of X to fix up ``atan``'s half-plane; a
    two-argument ``arctan2`` does that correctly for all four quadrants, which
    is why the branch is not reproduced here.
    """
    weights = np.asarray(weights, dtype=float)
    theta = np.deg2rad(np.asarray(theta_deg, dtype=float))
    good = np.isfinite(weights) & np.isfinite(theta) & (weights >= 0)
    if not np.any(good):
        return np.nan, np.nan
    weights, theta = weights[good], theta[good]
    total = weights.sum()
    if total <= 0:
        return np.nan, np.nan
    x = float((weights * np.cos(theta)).sum() / total)
    y = float((weights * np.sin(theta)).sum() / total)
    return float(np.rad2deg(np.arctan2(y, x)) % 360.0), float(np.hypot(x, y))


def _angular_deviation_deg(r):
    """Batschelet's angular deviation, ``s = sqrt(2 * (1 - r))`` radians.

    The paper uses this as its gate-width proxy: "we computed angular deviation
    (s)... Angular deviation is akin to standard deviation, but in polar
    coordinates. It measures the angular dispersion of activity and sleep
    states around their mean phase. We used the onset and offset of this
    angular deviation as proxies of the onset and offset of behavioral sleep
    states."

    It is not in the released ``phase`` package (the paper's circular figures
    came from custom scripts), so this is the standard definition rather than a
    port. It is bounded by sqrt(2) rad = 81.03 deg, so a gate can never exceed
    ~10.8 h either side of its mean phase.
    """
    if not np.isfinite(r):
        return np.nan
    return float(np.rad2deg(np.sqrt(2.0 * max(0.0, 1.0 - float(r)))))


def _is_bimodal(weights):
    """Two clear peaks around the clock, as judged for the angle-doubling step.

    The paper: "in case of clearly bimodal rhythms, as determined by visual
    inspection of the rose plots, an angle doubling transformation was carried
    out". Visual inspection is not available in a pipeline, so this stands in
    for it: count peaks that clear the halfway point between the profile's
    median and its maximum, treating the profile as circular, and call it
    bimodal at exactly two. Deliberately conservative — a fly whose profile is
    ambiguous keeps the untransformed mean, which is the safer default because
    doubling a unimodal profile corrupts its phase.
    """
    w = np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w), w, 0.0)
    if w.size < 6 or w.max() <= 0:
        return False
    height = np.median(w) + 0.5 * (w.max() - np.median(w))
    above = w >= height
    if not above.any():
        return False
    # Count runs of above-threshold bins, joining a run that wraps the origin.
    flips = np.diff(above.astype(int))
    n_runs = int((flips == 1).sum()) + (1 if above[0] else 0)
    if above[0] and above[-1] and n_runs > 1:
        n_runs -= 1
    return n_runs == 2


def circular_state_stats(
    ds, bin_size_min=30, states=STATE_ORDER, angle_doubling="auto"
):
    """Per-fly centre of mass, concentration, and circadian gate per state.

    One row per (fly, state) with the paper's Figure 3C quantities:
    ``mean_phase_h`` (theta of the centre of mass, in ZT/CT hours), ``r``,
    ``angular_deviation_h``, and the gate ``onset_h`` / ``offset_h`` =
    mean phase +/- angular deviation.

    Parameters
    ----------
    angle_doubling : {'auto', 'never', 'always'}
        Whether to apply the paper's angle-doubling transform for bimodal
        profiles. ``'auto'`` decides per (fly, state) via :func:`_is_bimodal`.
        Doubling maps theta -> 2*theta, takes the mean there, and halves it
        back, which is the standard fix for axial data — a profile with peaks
        near dawn and dusk otherwise averages to a mean phase in the middle of
        the day, where the fly is doing nothing at all. Halving leaves a
        180-degree ambiguity, resolved by picking whichever candidate the
        profile actually has more weight near.

    Returns
    -------
    pd.DataFrame
        ``id``, ``group``, ``state``, ``mean_phase_h``, ``r``,
        ``angular_deviation_h``, ``onset_h``, ``offset_h``, ``doubled``.
    """
    cols = [
        "id",
        "group",
        "state",
        "mean_phase_h",
        "r",
        "angular_deviation_h",
        "onset_h",
        "offset_h",
        "doubled",
    ]
    profiles = state_profiles(ds, bin_size_min=bin_size_min, states=states, include_activity=True)
    if profiles.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for (fly, group, state), sdf in profiles.groupby(["id", "group", "state"], sort=False):
        sdf = sdf.sort_values("zt_bin_minute")
        weights = sdf["value"].values
        # Bins are right-labelled for the wedges but their CENTRE is the correct
        # angle for a centre-of-mass calculation.
        centres = sdf["zt_bin_minute"].values + bin_size_min / 2.0
        theta = centres / MINUTES_PER_DAY * 360.0

        doubled = angle_doubling == "always" or (
            angle_doubling == "auto" and _is_bimodal(weights)
        )
        if doubled:
            mean_2, r = _weighted_circular_mean((2.0 * theta) % 360.0, weights)
            if np.isfinite(mean_2):
                candidates = np.array([mean_2 / 2.0, mean_2 / 2.0 + 180.0]) % 360.0
                # Resolve the halving ambiguity by weight near each candidate.
                scores = []
                for cand in candidates:
                    sep = np.abs((theta - cand + 180.0) % 360.0 - 180.0)
                    scores.append(float(np.nansum(np.where(sep <= 90.0, weights, 0.0))))
                mean_phase = float(candidates[int(np.argmax(scores))])
            else:
                mean_phase = np.nan
        else:
            mean_phase, r = _weighted_circular_mean(theta, weights)

        dev_deg = _angular_deviation_deg(r)
        to_h = 24.0 / 360.0
        rows.append(
            {
                "id": fly,
                "group": group,
                "state": state,
                "mean_phase_h": mean_phase * to_h if np.isfinite(mean_phase) else np.nan,
                "r": r,
                "angular_deviation_h": dev_deg * to_h if np.isfinite(dev_deg) else np.nan,
                "onset_h": ((mean_phase - dev_deg) % 360.0) * to_h
                if np.isfinite(mean_phase) and np.isfinite(dev_deg)
                else np.nan,
                "offset_h": ((mean_phase + dev_deg) % 360.0) * to_h
                if np.isfinite(mean_phase) and np.isfinite(dev_deg)
                else np.nan,
                "doubled": bool(doubled),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def group_gates(stats_df):
    """Group-mean gate per state, averaging phases CIRCULARLY.

    The mean of ZT23 and ZT01 is midnight, not noon. The previous
    ``polar_gating_plot`` took a plain arithmetic ``.mean()`` of onset and
    offset minutes, so any state whose gate straddles the origin — long sleep
    under DD, exactly the state the figure is about — got a mean gate on the
    opposite side of the clock. Here each fly's mean phase is averaged as a
    unit vector and the gate width is averaged as a scalar (it is a dispersion,
    not an angle), then the gate is rebuilt around the circular mean phase.
    """
    cols = ["group", "state", "mean_phase_h", "angular_deviation_h", "onset_h", "offset_h", "n"]
    if stats_df is None or stats_df.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for (group, state), sdf in stats_df.groupby(["group", "state"], sort=False):
        phases = sdf["mean_phase_h"].values.astype(float)
        widths = sdf["angular_deviation_h"].values.astype(float)
        good = np.isfinite(phases)
        if not good.any():
            continue
        ang = phases[good] / 24.0 * 2 * np.pi
        mean_phase = float(np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360.0)
        mean_phase_h = mean_phase * 24.0 / 360.0
        width = float(np.nanmean(widths)) if np.isfinite(widths).any() else np.nan
        rows.append(
            {
                "group": group,
                "state": state,
                "mean_phase_h": mean_phase_h,
                "angular_deviation_h": width,
                "onset_h": (mean_phase_h - width) % 24.0 if np.isfinite(width) else np.nan,
                "offset_h": (mean_phase_h + width) % 24.0 if np.isfinite(width) else np.nan,
                "n": int(good.sum()),
            }
        )
    return pd.DataFrame(rows, columns=cols)
