"""
sleep_analysis.py
=================
Sleep bout detection and classification for DAM activity data.

A fly is considered asleep during any consecutive immobile period whose
duration meets or exceeds the threshold (default 5 minutes, matching the
standard Drosophila sleep definition from Shaw et al. 2000).

The main function sleep_analysis() processes each fly independently,
adds a 'sleep' variable to the input dataset, and also returns a Dataset
containing per-bout statistics (start_time, end_time, duration,
sleep_state).

Time representation:
  Handles both relative integer-minute and absolute datetime time.
  The bout duration calculation is adapted for each case.
"""

import numpy as np
import pandas as pd
import xarray as xr


def _fill_short_gaps(series, max_gap=4):
    """
    Fill short runs of missing data (-1) before bout detection.

    If a gap of -1 values is surrounded on both sides by the same valid
    value (both 0 or both 1) and is no longer than max_gap samples, it
    is filled with that value.  This prevents artificially splitting
    sleep bouts at data dropouts.

    Parameters
    ----------
    series : pd.Series
        Movement data: 1=active, 0=immobile, -1=missing.
    max_gap : int
        Maximum gap length to fill (default 4 minutes).

    Returns
    -------
    pd.Series with short gaps filled.
    """
    filled = series.copy()
    is_gap = filled == -1
    if not is_gap.any():
        return filled

    gap_diff = is_gap.astype(int).diff().fillna(0)
    gap_starts = gap_diff[gap_diff == 1].index
    gap_ends = gap_diff[gap_diff == -1].index

    # Handle gaps at the very start or end of the series
    if is_gap.iloc[0]:
        gap_starts = gap_starts.insert(0, is_gap.index[0])
    if is_gap.iloc[-1]:
        gap_ends = gap_ends.append(pd.Index([is_gap.index[-1] + 1]))

    for start, end in zip(gap_starts, gap_ends):
        gap_length = end - start
        if gap_length <= max_gap:
            before_idx = start - 1
            after_idx = end
            if before_idx >= 0 and after_idx < len(filled):
                before_val = filled.iloc[before_idx]
                after_val = filled.iloc[after_idx]
                if before_val == after_val and before_val != -1:
                    filled.iloc[start:end] = before_val

    return filled


def classify_sleep_bouts(bouts_df, short_max_min=30, inter_max_min=60):
    """
    Classify sleep bouts into short, intermediate, or long states.

    Parameters
    ----------
    bouts_df : pd.DataFrame
        Must contain a 'duration' column in minutes.
    short_max_min : float
        Upper bound (exclusive) for short sleep in minutes. Default 30.
    inter_max_min : float
        Upper bound (exclusive) for intermediate sleep in minutes. Default 60.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with an added 'sleep_state' str column.
        Values: 'short' (5-30 min), 'intermediate' (30-60 min), 'long' (>=60 min).
    """
    df = bouts_df.copy()
    conditions = [
        df["duration"] < short_max_min,
        (df["duration"] >= short_max_min) & (df["duration"] < inter_max_min),
        df["duration"] >= inter_max_min,
    ]
    choices = ["short", "intermediate", "long"]
    df["sleep_state"] = np.select(conditions, choices, default="short")
    return df


def sleep_analysis(
    data,
    mov_column="moving",
    t_column="time",
    sleep_threshold_sec=300,
    short_max_min=30,
    inter_max_min=60,
    phase="auto",
):
    """
    Detect sleep bouts in DAM activity data and add results to the dataset.

    A sleep bout is a run of immobile time points (moving==0) whose total
    duration is >= sleep_threshold_sec seconds.  Short data gaps (-1) up to
    4 minutes are bridged before bout detection so they do not fragment bouts.

    Processing is per-fly; each fly's data is handled independently.

    Phase
    -----
    Pass ``data`` as the WHOLE dataset and let ``phase`` select the epoch — do not
    pre-slice. ``phase`` defaults to ``"LD"`` (the entrained day is the standard
    sleep reference); ``"DD"`` requests constant-darkness sleep; ``"both"`` runs on
    the full continuous recording (total sleep across LD+DD). Selection uses the
    one core selector :func:`dam_utilities.select_phase` (per-fly NaN mask, §2a:
    real 0s preserved, gaps stay missing). Out-of-phase minutes are recorded as
    ``-1`` (missing) in the ``sleep`` outputs so the result is dimensioned on the
    whole time axis and downstream selectors stay idempotent. The bridging /
    detrend-skip conventions are unchanged.

    After bout detection, bouts are classified into short, intermediate, or
    long sleep states based on duration thresholds (Abhilash et al. 2026).
    Per-minute binary masks are created for each sleep state.

    Parameters
    ----------
    data : xr.Dataset
        Must contain the movement variable (default 'moving') and time coordinate.
    mov_column : str
        Variable name for movement data (default 'moving').
    t_column : str
        Time coordinate name (default 'time').
    sleep_threshold_sec : int
        Minimum bout duration in seconds to classify as sleep (default 300 = 5 min).
    short_max_min : float
        Upper bound (exclusive) for short sleep in minutes (default 30).
    inter_max_min : float
        Upper bound (exclusive) for intermediate sleep in minutes (default 60).
    phase : str
        ``"auto"`` (→ ``"LD"``), ``"LD"``, ``"DD"``, or ``"both"``. Selects the
        epoch sleep is computed on. The whole dataset must be passed; selection is
        internal (see Phase note above).

    Returns
    -------
    xr.Dataset
        Original dataset merged with:
          - 'sleep'              : int8 variable (1=sleeping, 0=awake, -1=missing)
          - 'sleep_short'        : int8 variable (1=short sleep, 0=not, -1=missing)
          - 'sleep_intermediate' : int8 variable (1=intermediate sleep, 0=not, -1=missing)
          - 'sleep_long'         : int8 variable (1=long sleep, 0=not, -1=missing)
          - 'sleep_state'        : per-bout sleep state classification string
          - 'duration'           : bout duration in minutes, indexed by (id, sleep_bout_number)
          - 'start_time', 'end_time' : bout boundaries
    """
    if mov_column not in data.data_vars:
        raise KeyError(f"The movement column {mov_column} is not in the dataset")
    if t_column not in data.coords:
        raise KeyError(f"The time column {t_column} is not in the dataset")

    # Phase selection via the one core selector. `data` keeps the unmasked
    # vars (moving/activity int dtypes intact); `analysis_ds` is the per-fly
    # NaN-masked view the bout detector runs on. Results merge back onto
    # `data` so the output is dimensioned on the whole time axis.
    from dam_utilities import resolve_phase

    analysis_ds, phase_used = resolve_phase(
        data,
        phase,
        default="LD",
        allowed=("LD", "DD", "both"),
        caller="sleep_analysis",
    )

    def _wrapped_sleep_analysis(group_data, sleep_threshold, group_id):
        """Process a single fly and return its sleep mask, bout DataFrame, and state masks."""
        df = group_data.to_dataframe().reset_index()
        # select_phase() masks out-of-phase cells to NaN (when phase != 'both').
        # For sleep, out-of-phase = "not part of this epoch" → treat as missing
        # (-1), the int8 sentinel the detector already understands. In-phase real
        # 0s and true -1 gaps are untouched (§2a: out-of-phase NaN is never an
        # in-phase value, so this never fabricates or erases real behaviour).
        if df[mov_column].isna().any():
            df[mov_column] = df[mov_column].fillna(-1)
        is_missing = df[mov_column] == -1

        # Bridge short gaps for cleaner bout boundaries
        mov_for_bouts = _fill_short_gaps(df[mov_column], max_gap=4)
        df["potential_sleep"] = (mov_for_bouts == 0) & (~is_missing | (mov_for_bouts != -1))

        # Identify bout start and end time points
        df["bout_start"] = (~df["potential_sleep"].shift(1, fill_value=True)) & df[
            "potential_sleep"
        ]
        df["bout_end"] = df["potential_sleep"] & (~df["potential_sleep"].shift(-1, fill_value=True))

        if df["potential_sleep"].iloc[0]:
            df.loc[0, "bout_start"] = True
        if df["potential_sleep"].iloc[-1]:
            df.loc[df.index[-1], "bout_end"] = True

        bout_starts = df[df["bout_start"]][t_column]
        bout_ends = df[df["bout_end"]][t_column]

        # Ensure equal number of starts and ends
        min_length = min(len(bout_starts), len(bout_ends))
        bout_starts = bout_starts.iloc[:min_length]
        bout_ends = bout_ends.iloc[:min_length]

        # Calculate durations — handle both time representations
        raw_diffs = bout_ends.values - bout_starts.values
        if np.issubdtype(raw_diffs.dtype, np.integer):
            # Relative integer minutes: convert to seconds for threshold comparison
            bout_durations = raw_diffs * 60
        else:
            bout_durations = raw_diffs.astype("timedelta64[s]").astype(int)

        # Mark valid bouts (>= threshold)
        valid_bouts = bout_durations >= sleep_threshold

        # Build the per-sample sleep mask
        sleep_mask = np.zeros(len(df), dtype=np.int8)
        sleep_mask[is_missing.values] = -1
        for start, end, is_valid in zip(bout_starts.index, bout_ends.index, valid_bouts):
            if is_valid:
                for i in range(start, end + 1):
                    if sleep_mask[i] != -1:
                        sleep_mask[i] = 1

        # Build bout-level DataFrame
        valid_starts = bout_starts[valid_bouts]
        valid_ends = bout_ends[valid_bouts]
        valid_durations = bout_durations[valid_bouts]

        sleep_bouts_df = pd.DataFrame(
            {
                "start_time": valid_starts.values,
                "end_time": valid_ends.values,
                "duration": valid_durations.astype(float) / 60,  # seconds → minutes
            }
        )
        sleep_bouts_df["sleep_bout_number"] = range(1, len(sleep_bouts_df) + 1)
        sleep_bouts_df["id"] = group_id

        # Classify bouts into sleep states
        if len(sleep_bouts_df) > 0:
            sleep_bouts_df = classify_sleep_bouts(
                sleep_bouts_df, short_max_min=short_max_min, inter_max_min=inter_max_min
            )
        else:
            sleep_bouts_df["sleep_state"] = pd.Series(dtype=str)

        # Build per-minute binary masks for each sleep state
        # Use xr.where-style logic on numpy arrays (inside per-fly worker, numpy is appropriate)
        sleep_short_mask = np.zeros(len(df), dtype=np.int8)
        sleep_inter_mask = np.zeros(len(df), dtype=np.int8)
        sleep_long_mask = np.zeros(len(df), dtype=np.int8)
        sleep_short_mask[is_missing.values] = -1
        sleep_inter_mask[is_missing.values] = -1
        sleep_long_mask[is_missing.values] = -1

        for _, bout_row in sleep_bouts_df.iterrows():
            start_t = bout_row["start_time"]
            end_t = bout_row["end_time"]
            state = bout_row["sleep_state"]
            # Find time indices for this bout
            bout_idx = df.index[(df[t_column] >= start_t) & (df[t_column] <= end_t)]
            for i in bout_idx:
                if sleep_mask[i] == 1:  # only mark confirmed sleep minutes
                    if state == "short":
                        sleep_short_mask[i] = 1
                    elif state == "intermediate":
                        sleep_inter_mask[i] = 1
                    elif state == "long":
                        sleep_long_mask[i] = 1

        sleep_short_da = xr.DataArray(
            sleep_short_mask, coords={t_column: df[t_column]}, dims=[t_column]
        )
        sleep_inter_da = xr.DataArray(
            sleep_inter_mask, coords={t_column: df[t_column]}, dims=[t_column]
        )
        sleep_long_da = xr.DataArray(
            sleep_long_mask, coords={t_column: df[t_column]}, dims=[t_column]
        )

        sleep_mask_da = xr.DataArray(sleep_mask, coords={t_column: df[t_column]}, dims=[t_column])
        return sleep_mask_da, sleep_bouts_df, sleep_short_da, sleep_inter_da, sleep_long_da

    all_sleep_bouts_dfs = []
    all_sleep_masks = []
    all_short_masks = []
    all_inter_masks = []
    all_long_masks = []

    for group_id, group_data in analysis_ds.groupby("id", squeeze=False):
        sleep_mask_da, sleep_bouts_df, short_da, inter_da, long_da = _wrapped_sleep_analysis(
            group_data, sleep_threshold_sec, group_id
        )
        all_sleep_masks.append(sleep_mask_da.expand_dims({"id": [group_id]}))
        all_short_masks.append(short_da.expand_dims({"id": [group_id]}))
        all_inter_masks.append(inter_da.expand_dims({"id": [group_id]}))
        all_long_masks.append(long_da.expand_dims({"id": [group_id]}))
        all_sleep_bouts_dfs.append(sleep_bouts_df)

    combined_sleep_mask_da = xr.concat(all_sleep_masks, dim="id").rename("sleep")
    combined_short_da = xr.concat(all_short_masks, dim="id").rename("sleep_short")
    combined_inter_da = xr.concat(all_inter_masks, dim="id").rename("sleep_intermediate")
    combined_long_da = xr.concat(all_long_masks, dim="id").rename("sleep_long")
    combined_sleep_bouts_df = pd.concat(all_sleep_bouts_dfs, ignore_index=True)

    # Store bout data as a multi-indexed Dataset: dims (id, sleep_bout_number)
    # sleep_state column is included automatically since it is part of combined_sleep_bouts_df
    indexed_bouts = combined_sleep_bouts_df.set_index(["id", "sleep_bout_number"])
    bout_ds = xr.Dataset.from_dataframe(indexed_bouts)

    # Drop pre-existing sleep variables if re-running to avoid merge conflicts
    _sleep_vars = [
        "sleep",
        "sleep_short",
        "sleep_intermediate",
        "sleep_long",
        "sleep_state",
        "duration",
        "start_time",
        "end_time",
    ]
    _to_drop = [v for v in _sleep_vars if v in data.data_vars]
    if _to_drop:
        data = data.drop_vars(_to_drop)
    if "sleep_bout_number" in data.dims:
        _bout_dim_vars = [v for v in data.data_vars if "sleep_bout_number" in data[v].dims]
        if _bout_dim_vars:
            data = data.drop_vars(_bout_dim_vars)

    # Step 1: Merge time-series variables (all share id × time dims with data).
    # bout_ds has dims (id × sleep_bout_number) — merging it in the same call
    # causes xarray to broadcast across all four dims, allocating a massive array.
    merged_ds = xr.merge(
        [data, combined_sleep_mask_da, combined_short_da, combined_inter_da, combined_long_da]
    )

    # Step 2: Add bout-level variables individually so their (id, sleep_bout_number)
    # dims are never broadcast against the time dimension.
    for var_name in bout_ds.data_vars:
        merged_ds[var_name] = bout_ds[var_name]

    # Record parameters for reproducibility
    merged_ds.attrs["sleep_threshold_seconds"] = int(sleep_threshold_sec)
    merged_ds.attrs["sleep_short_max_min"] = float(short_max_min)
    merged_ds.attrs["sleep_inter_max_min"] = float(inter_max_min)
    # Phase the sleep outputs were computed on (out-of-phase minutes are -1).
    merged_ds.attrs["sleep_phase"] = phase_used

    return merged_ds
