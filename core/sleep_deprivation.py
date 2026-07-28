"""
sleep_deprivation.py
====================
Core analysis functions for sleep deprivation (SD) experiments.

Supports experiments with baseline LD days, a single SD bout, and recovery
LD days.  Compares baseline-averaged sleep to each post-SD recovery day,
split by light/dark phase, across genotypes.

Scope of the "rebound" metric here (read before interpreting the numbers):
    ``rebound_pct`` is the descriptive Shaw et al. 2002 % — (recovery − baseline) /
    baseline × 100 on STANDARD (5-min-rule) sleep, per phase per recovery day. It is a
    within-cohort readout: there is NO undisturbed-control arm and NO significance test,
    so it describes the observed change, it does not test a hypothesis. Per Abhilash 2026
    Fig 4, a small standard-sleep rebound is the EXPECTED result, not a defect. If a
    control-cohort subtraction / hypothesis test is wanted, that is an additive feature
    (a shared undisturbed-cohort designation) — a user/data-model decision, not built here.
    NOTE: this is distinct from the per-sleep-STATE "rebound" in sleep_state_metrics.py
    (page 9), which is a different quantity (Abhilash "Metric 1", per state) — do not
    conflate the two.

Main entry point:
    compute_sd_analysis()  — runs the full SD comparison pipeline

Helper functions:
    get_experiment_days()          — enumerate LD days in the dataset
    get_day_sleep_traces()         — per-day ZT-binned sleep for the selector plot
    _get_fly_group()               — resolve fly → group label
    _zt_bin_sleep_for_day()        — bin one day of sleep data into ZT bins
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_fly_group(ds, fly_id):
    """Return the group label for a fly (genotype-temperature or 'All')."""
    fly_ds = ds.sel(id=fly_id)
    if "group" in ds.coords:
        return str(fly_ds["group"].item())
    elif "genotype" in ds.coords and "temperature" in ds.coords:
        return f"{fly_ds['genotype'].item()}-{fly_ds['temperature'].item()}"
    return "All"


def _fly_id_scalar(fly_id):
    """Normalise a fly ID to a plain Python scalar."""
    if isinstance(fly_id, np.ndarray):
        return fly_id.item() if fly_id.ndim == 0 else fly_id[0]
    if hasattr(fly_id, "item"):
        return fly_id.item()
    return fly_id


def _minutes_from_start(ds, fly_id):
    """
    Return a float array of minutes-from-experiment-start for every timepoint
    of *fly_id*, handling both relative-integer and absolute-datetime time.
    """
    fly_ds = ds.sel(id=fly_id)
    time_vals = fly_ds["time"].values
    if np.issubdtype(time_vals.dtype, np.integer):
        return time_vals.astype(float)
    start_dt = pd.to_datetime(fly_ds["start_datetime"].item())
    return (pd.to_datetime(time_vals) - start_dt).total_seconds() / 60.0


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_experiment_days(ds, phase="LD"):
    """
    Return the number of complete days in the requested phase and their boundaries.

    Each day spans 1440 minutes (ZT0 → ZT24).  Only complete days are counted.
    Pass the WHOLE dataset and let ``phase`` bound the count (Stage-2 phase API);
    the default ``"LD"`` counts only the entrained days before the LD→DD boundary.

    Parameters
    ----------
    ds : xr.Dataset
        Whole (or already single-phase) dataset on a relative-minute axis.
    phase : str
        ``"auto"`` (→ LD), ``"LD"``, ``"DD"``, or ``"both"``. Validated via the one
        phase policy. (Feeding an already-LD object with ``"LD"`` is a no-op mask.)

    Returns
    -------
    n_days : int
    day_boundaries : list[tuple[int, int]]
        (start_minute, end_minute) for each day, in the dataset's absolute
        relative-minute units (LD days start at 0; DD days start at the boundary).
    """
    from dam_utilities import add_phase_metadata, resolve_phase

    # Validate + resolve the phase (we slice by the boundary below, not mask).
    _, phase_used = resolve_phase(
        ds, phase, default="LD", allowed=("LD", "DD", "both"), caller="get_experiment_days"
    )
    fly_id = _fly_id_scalar(ds["id"].values[0])
    mins = _minutes_from_start(ds, fly_id)
    total_minutes = float(mins[-1] - mins[0]) + 1

    if phase_used == "both":
        start_min, end_min = 0.0, total_minutes
    else:
        # phase_used is LD/DD only when a boundary is derivable (else select_phase
        # falls back to "both"), so split_minute is safe here.
        ds2 = ds if "split_minute" in ds.coords else add_phase_metadata(ds)
        sm = float(int(ds2["split_minute"].min().item()))
        if phase_used == "LD":
            start_min, end_min = 0.0, min(sm, total_minutes)
        else:  # DD
            start_min, end_min = sm, total_minutes

    n_days = int((end_min - start_min) // 1440)
    base = int(start_min)
    day_boundaries = [(base + d * 1440, base + (d + 1) * 1440) for d in range(n_days)]
    return n_days, day_boundaries


def _build_minutes_array(ds):
    """
    Return a 1-D float array of minutes-from-experiment-start for the shared
    time axis, handling both relative-integer and absolute-datetime time.
    """
    time_vals = ds["time"].values
    if np.issubdtype(time_vals.dtype, np.integer):
        return time_vals.astype(float)
    # Absolute time: use first fly's start_datetime as reference
    start_dt = pd.to_datetime(ds["start_datetime"].values[0])
    return (pd.to_datetime(time_vals) - start_dt).total_seconds() / 60.0


def _bin_sleep_array(sleep_2d, mins, n_days, zt_bins, bin_size_minutes):
    """
    Bin a (time, n_flies) sleep array into (n_flies, n_days, n_bins).

    Parameters
    ----------
    sleep_2d : np.ndarray, shape (n_time, n_flies)
    mins : np.ndarray, shape (n_time,)
    n_days : int
    zt_bins : np.ndarray
    bin_size_minutes : int

    Returns
    -------
    np.ndarray, shape (n_flies, n_days, n_bins)
    """
    n_flies = sleep_2d.shape[1]
    n_bins = len(zt_bins)

    # Compute day index and ZT minute for every timepoint
    day_idx = (mins // 1440).astype(int)
    zt_minute = mins % 1440
    bin_idx = np.clip(np.digitize(zt_minute, zt_bins) - 1, 0, n_bins - 1)

    result = np.full((n_flies, n_days, n_bins), np.nan)
    for d in range(n_days):
        day_mask = day_idx == d
        if not day_mask.any():
            continue
        for b in range(n_bins):
            cell_mask = day_mask & (bin_idx == b)
            if cell_mask.any():
                result[:, d, b] = np.nanmean(sleep_2d[cell_mask, :], axis=0)

    return result


def get_day_sleep_traces(ds, bin_size_minutes=15):
    """
    Return per-day ZT-binned sleep traces averaged across all flies.

    Used by the Streamlit page to show the interactive SD-window selector.

    Parameters
    ----------
    ds : xr.Dataset
        Must contain 'sleep' variable.
    bin_size_minutes : int

    Returns
    -------
    pd.DataFrame
        Columns: day (int, 0-indexed), zt_bin_minute, sleep_mean, sleep_sem
    """
    if "sleep" not in ds.data_vars:
        raise ValueError("Dataset must contain 'sleep' variable. Run sleep analysis first.")

    n_days, _ = get_experiment_days(ds)
    zt_bins = np.arange(0, 1440, bin_size_minutes)

    # Extract full sleep array as (time, n_flies), replace -1 sentinel with NaN
    mins = _build_minutes_array(ds)
    sleep_da = ds["sleep"].transpose("time", "id")
    sleep_2d = sleep_da.values.astype(float)
    sleep_2d = np.where(sleep_2d < 0, np.nan, sleep_2d)

    # Bin into (n_flies, n_days, n_bins)
    binned = _bin_sleep_array(sleep_2d, mins, n_days, zt_bins, bin_size_minutes)

    # Average across flies → (n_days, n_bins) mean and SEM
    records = []
    for d in range(n_days):
        for b, zt in enumerate(zt_bins):
            vals = binned[:, d, b]
            valid = vals[~np.isnan(vals)]
            if len(valid) == 0:
                continue
            records.append(
                {
                    "day": d,
                    "zt_bin_minute": zt,
                    "sleep_mean": np.mean(valid),
                    "sleep_sem": valid.std(ddof=1) / np.sqrt(len(valid)) if len(valid) > 1 else 0.0,
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def compute_sd_analysis(
    ds, sd_start_zt_minutes, sd_duration_minutes, sd_day_index, bin_size_minutes=15, phase="auto"
):
    """
    Run the full sleep-deprivation comparison pipeline.

    Parameters
    ----------
    ds : xr.Dataset
        Must have 'sleep' variable (from sleep_analysis).
    sd_start_zt_minutes : int
        ZT time in minutes when SD begins (e.g. 720 for ZT12).
    sd_duration_minutes : int
        Duration of SD in minutes (e.g. 360 for 6 h).
    sd_day_index : int
        0-indexed day on which SD occurs.
    bin_size_minutes : int
        Bin size for ZT profiles (default 15).

    Returns
    -------
    dict with keys:
        baseline_profile    — DataFrame (zt_bin_minute, group, mean, sem, n)
        recovery_profiles   — dict[int, DataFrame] keyed by recovery day number (1, 2, …)
        difference_profiles — dict[int, DataFrame] (recovery − baseline per bin)
        cumulative_diff     — dict[int, DataFrame] (running sum of difference)
        phase_totals        — DataFrame (group, day_label, light_sleep_min, dark_sleep_min, total_sleep_min)
        rebound_pct         — DataFrame (group, recovery_day, phase, rebound_pct)
        sd_params           — dict of the SD parameters used
        n_baseline_days     — int
        n_recovery_days     — int
    """
    if "sleep" not in ds.data_vars:
        raise ValueError("Dataset must contain 'sleep' variable.")

    # Phase API (Stage-2): SD is an LD-structured protocol. Validate the requested
    # phase (LD only) and restrict to the LD calendar days so day indexing and ZT
    # binning cover only the SD epoch. We slice by the boundary rather than mask,
    # so the int8 'sleep' dtype is preserved and out-of-phase -1 minutes are simply
    # outside the analysed range. Feeding an already-LD object is a no-op slice.
    from dam_utilities import resolve_phase

    _, phase_used = resolve_phase(
        ds, phase, default="LD", allowed=("LD",), caller="compute_sd_analysis"
    )
    n_days, _ = get_experiment_days(ds, phase=phase_used)
    ds = ds.sel(time=(ds["time"] >= 0) & (ds["time"] < n_days * 1440))
    if sd_day_index < 0 or sd_day_index >= n_days:
        raise ValueError(f"sd_day_index={sd_day_index} is out of range (0–{n_days - 1}).")

    baseline_day_indices = list(range(0, sd_day_index))
    recovery_day_indices = list(range(sd_day_index + 1, n_days))

    if len(baseline_day_indices) == 0:
        raise ValueError("No baseline days before the SD day. SD must not be on the first day.")
    if len(recovery_day_indices) == 0:
        raise ValueError("No recovery days after the SD day. SD must not be on the last day.")

    zt_bins = np.arange(0, 1440, bin_size_minutes)
    n_bins = len(zt_bins)

    # ------------------------------------------------------------------
    # 1. Collect per-fly, per-day, per-bin sleep fractions (vectorized)
    # ------------------------------------------------------------------
    fly_ids = [_fly_id_scalar(fid) for fid in ds["id"].values]
    fly_groups = {fid: _get_fly_group(ds, fid) for fid in fly_ids}

    # Extract full sleep array and bin in one pass
    mins = _build_minutes_array(ds)
    sleep_2d = ds["sleep"].transpose("time", "id").values.astype(float)
    sleep_2d = np.where(sleep_2d < 0, np.nan, sleep_2d)

    # binned_3d shape: (n_flies, n_days, n_bins)
    binned_3d = _bin_sleep_array(sleep_2d, mins, n_days, zt_bins, bin_size_minutes)

    # Convert to the dict structure used by downstream code
    fly_day_bins = {}
    for i, fly_id in enumerate(fly_ids):
        fly_day_bins[fly_id] = {}
        for day_idx in range(n_days):
            arr = binned_3d[i, day_idx, :]
            if not np.all(np.isnan(arr)):
                fly_day_bins[fly_id][day_idx] = arr

    # ------------------------------------------------------------------
    # 2. Compute baseline average per fly (mean across baseline days)
    # ------------------------------------------------------------------
    fly_baseline = {}  # fly_id → array(n_bins,)
    for fly_id in fly_ids:
        day_arrays = [
            fly_day_bins[fly_id][d] for d in baseline_day_indices if d in fly_day_bins[fly_id]
        ]
        if day_arrays:
            fly_baseline[fly_id] = np.nanmean(np.stack(day_arrays), axis=0)

    # ------------------------------------------------------------------
    # 3. Build group-level profiles
    # ------------------------------------------------------------------
    groups = sorted(set(fly_groups.values()))

    def _group_profile(fly_bin_dict):
        """Aggregate per-fly bin arrays into a group-level DataFrame."""
        records = []
        for grp in groups:
            grp_flies = [fid for fid in fly_ids if fly_groups[fid] == grp and fid in fly_bin_dict]
            if not grp_flies:
                continue
            stacked = np.stack([fly_bin_dict[fid] for fid in grp_flies])
            mean_vals = np.nanmean(stacked, axis=0)
            n_vals = np.sum(~np.isnan(stacked), axis=0)
            sem_vals = np.where(
                n_vals > 1,
                np.nanstd(stacked, axis=0, ddof=1) / np.sqrt(n_vals),
                0.0,
            )
            for b in range(n_bins):
                records.append(
                    {
                        "zt_bin_minute": zt_bins[b],
                        "group": grp,
                        "mean": mean_vals[b],
                        "sem": sem_vals[b],
                        "n": int(n_vals[b]),
                    }
                )
        return pd.DataFrame(records)

    baseline_profile = _group_profile(fly_baseline)

    recovery_profiles = {}
    difference_profiles = {}
    cumulative_diffs = {}

    for rec_num, rec_day_idx in enumerate(recovery_day_indices, start=1):
        # Per-fly recovery day bins
        fly_rec = {
            fid: fly_day_bins[fid][rec_day_idx]
            for fid in fly_ids
            if rec_day_idx in fly_day_bins.get(fid, {})
        }
        recovery_profiles[rec_num] = _group_profile(fly_rec)

        # Per-fly difference (recovery − baseline)
        fly_diff = {}
        for fid in fly_ids:
            if fid in fly_rec and fid in fly_baseline:
                fly_diff[fid] = fly_rec[fid] - fly_baseline[fid]
        difference_profiles[rec_num] = _group_profile(fly_diff)

        # Cumulative difference (running sum across ZT bins)
        fly_cum = {}
        for fid in fly_ids:
            if fid in fly_diff:
                # Convert fraction-per-bin to minutes, then cumsum
                fly_cum[fid] = np.nancumsum(fly_diff[fid] * bin_size_minutes)
        cumulative_diffs[rec_num] = _group_profile(fly_cum)

    # ------------------------------------------------------------------
    # 4. Phase totals (light ZT0-12, dark ZT12-24) in minutes of sleep
    # ------------------------------------------------------------------
    light_mask = zt_bins < 720
    dark_mask = zt_bins >= 720

    phase_records = []
    for grp in groups:
        grp_flies = [fid for fid in fly_ids if fly_groups[fid] == grp]

        # Baseline phase totals (per fly, then average)
        bl_light, bl_dark = [], []
        for fid in grp_flies:
            if fid not in fly_baseline:
                continue
            arr = fly_baseline[fid]
            bl_light.append(np.nansum(arr[light_mask]) * bin_size_minutes)
            bl_dark.append(np.nansum(arr[dark_mask]) * bin_size_minutes)
        if bl_light:
            phase_records.append(
                {
                    "group": grp,
                    "day_label": "Baseline (avg)",
                    "light_sleep_min": np.mean(bl_light),
                    "dark_sleep_min": np.mean(bl_dark),
                    "total_sleep_min": np.mean(bl_light) + np.mean(bl_dark),
                }
            )

        # Recovery phase totals
        for rec_num, rec_day_idx in enumerate(recovery_day_indices, start=1):
            rec_light, rec_dark = [], []
            for fid in grp_flies:
                if rec_day_idx not in fly_day_bins.get(fid, {}):
                    continue
                arr = fly_day_bins[fid][rec_day_idx]
                rec_light.append(np.nansum(arr[light_mask]) * bin_size_minutes)
                rec_dark.append(np.nansum(arr[dark_mask]) * bin_size_minutes)
            if rec_light:
                phase_records.append(
                    {
                        "group": grp,
                        "day_label": f"Recovery Day {rec_num}",
                        "light_sleep_min": np.mean(rec_light),
                        "dark_sleep_min": np.mean(rec_dark),
                        "total_sleep_min": np.mean(rec_light) + np.mean(rec_dark),
                    }
                )

    phase_totals = pd.DataFrame(phase_records)

    # ------------------------------------------------------------------
    # 5. Sleep rebound % = (recovery − baseline) / baseline × 100
    #    (Shaw et al. 2002 metric, computed per phase per recovery day)
    # ------------------------------------------------------------------
    rebound_records = []
    for grp in groups:
        grp_flies = [fid for fid in fly_ids if fly_groups[fid] == grp]

        # Baseline phase totals per fly
        bl_light_vals = {}
        bl_dark_vals = {}
        for fid in grp_flies:
            if fid not in fly_baseline:
                continue
            arr = fly_baseline[fid]
            bl_light_vals[fid] = np.nansum(arr[light_mask]) * bin_size_minutes
            bl_dark_vals[fid] = np.nansum(arr[dark_mask]) * bin_size_minutes

        for rec_num, rec_day_idx in enumerate(recovery_day_indices, start=1):
            fly_rebound_light = []
            fly_rebound_dark = []
            for fid in grp_flies:
                if rec_day_idx not in fly_day_bins.get(fid, {}) or fid not in bl_light_vals:
                    continue
                arr = fly_day_bins[fid][rec_day_idx]
                rec_l = np.nansum(arr[light_mask]) * bin_size_minutes
                rec_d = np.nansum(arr[dark_mask]) * bin_size_minutes
                if bl_light_vals[fid] > 0:
                    fly_rebound_light.append(
                        (rec_l - bl_light_vals[fid]) / bl_light_vals[fid] * 100
                    )
                if bl_dark_vals[fid] > 0:
                    fly_rebound_dark.append((rec_d - bl_dark_vals[fid]) / bl_dark_vals[fid] * 100)

            if fly_rebound_light:
                rebound_records.append(
                    {
                        "group": grp,
                        "recovery_day": rec_num,
                        "phase": "Light (ZT0-12)",
                        "rebound_pct": np.mean(fly_rebound_light),
                    }
                )
            if fly_rebound_dark:
                rebound_records.append(
                    {
                        "group": grp,
                        "recovery_day": rec_num,
                        "phase": "Dark (ZT12-24)",
                        "rebound_pct": np.mean(fly_rebound_dark),
                    }
                )

    rebound_pct = pd.DataFrame(rebound_records)

    # ------------------------------------------------------------------
    # Package results
    # ------------------------------------------------------------------
    return {
        "baseline_profile": baseline_profile,
        "recovery_profiles": recovery_profiles,
        "difference_profiles": difference_profiles,
        "cumulative_diff": cumulative_diffs,
        "phase_totals": phase_totals,
        "rebound_pct": rebound_pct,
        "sd_params": {
            "sd_start_zt_minutes": sd_start_zt_minutes,
            "sd_duration_minutes": sd_duration_minutes,
            "sd_day_index": sd_day_index,
            "bin_size_minutes": bin_size_minutes,
        },
        "n_baseline_days": len(baseline_day_indices),
        "n_recovery_days": len(recovery_day_indices),
    }
