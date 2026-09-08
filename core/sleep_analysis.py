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

    # Drop the previous run's results BEFORE anything else looks at `data`.
    #
    # This has to happen here rather than just before the merge below, because
    # `analysis_ds` is derived from `data` immediately after and the per-fly loop
    # calls `.to_dataframe()` on a slice of it. The bout variables are
    # (id, sleep_bout_number), so a slice that still carries them makes
    # to_dataframe take the cartesian product with `time`: on a reloaded .nc with
    # 134 bouts that is 2,315,654 rows per fly instead of 17,281, every timestamp
    # repeated 134 times. The duplicated time index then fails the merge with
    # "cannot reindex or align along dimension 'time'", after a long detour
    # building those frames — so re-running sleep analysis on a dataset loaded
    # from .nc was both very slow and guaranteed to fail.
    #
    # drop_dims (not drop_vars) for the bout dimension: it removes the dim, its
    # coordinate and every variable on it in one go. Dropping only the variables
    # leaves a dangling sleep_bout_number coord sized for the OLD run, which the
    # new bout results then have to align against.
    _sleep_mask_vars = ["sleep", "sleep_short", "sleep_intermediate", "sleep_long"]
    _to_drop = [v for v in _sleep_mask_vars if v in data.data_vars]
    if _to_drop:
        data = data.drop_vars(_to_drop)
    if "sleep_bout_number" in data.dims:
        data = data.drop_dims("sleep_bout_number")

    # Phase selection via the one core selector. `data` keeps the unmasked
    # vars (moving/activity int dtypes intact); `analysis_ds` is the per-fly
    # NaN-masked view the bout detector runs on. Results merge back onto
    # `data` so the output is dimensioned on the whole time axis.
    from dam_utilities import resolve_phase

    # ONLY the movement variable goes to the detector. `to_dataframe()`
    # broadcasts a Dataset over the UNION of its dimensions, so every variable
    # some OTHER analysis left on a dimension of its own multiplies the per-fly
    # frame. Dropping the previous sleep run's variables above is not enough:
    # on a dataset that had also been through the periodograms, one fly asked
    # for a (1, 17281, 600, 4321) float32 array — 167 GiB — and sleep analysis
    # could not be run at all. Where the product is small enough to allocate it
    # is worse than an error: every timestamp appears once per cell of the other
    # dimensions, so the bout detector sees a time axis with duplicates and
    # returns nonsense.
    #
    # Selecting the one variable needed is the form of this fix that does not
    # have to be revisited each time an analysis is added — a deny-list of
    # known dimensions would have to be.
    analysis_ds, phase_used = resolve_phase(
        data[[mov_column]],
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

    def _in_dataset_order(masks, name):
        """Concatenate the per-fly masks back into ``data``'s own ``id`` order.

        ``groupby("id")`` iterates in SORTED id order, and ``expand_dims`` builds
        each mask's id index from a Python list (object dtype). So the stacked
        masks disagree with ``data`` on both order and dtype whenever the
        dataset's ids are not already string-sorted — and ``xr.merge``'s current
        default, ``join="outer"``, reconciles that difference by silently
        re-sorting the merged dataset's fly dimension. Sleep analysis reordering
        the master's ids is a side effect nobody asked for, and it is what makes
        a later export list its flies in a different order.

        ``.sel`` (not ``.reindex``) on purpose: if a fly is genuinely missing it
        raises, where reindex would fill the row with NaN and hide it. It is also
        given plain label VALUES rather than ``data["id"]`` itself — passing the
        DataArray would carry every other id-dim coord (group, genotype,
        start_datetime, …) along with it and pull them into the merge's
        alignment, which is not what this is for.
        """
        stacked = xr.concat(masks, dim="id").rename(name)
        return stacked.sel(id=data["id"].values)

    combined_sleep_mask_da = _in_dataset_order(all_sleep_masks, "sleep")
    combined_short_da = _in_dataset_order(all_short_masks, "sleep_short")
    combined_inter_da = _in_dataset_order(all_inter_masks, "sleep_intermediate")
    combined_long_da = _in_dataset_order(all_long_masks, "sleep_long")
    combined_sleep_bouts_df = pd.concat(all_sleep_bouts_dfs, ignore_index=True)

    # Store bout data as a multi-indexed Dataset: dims (id, sleep_bout_number)
    # sleep_state column is included automatically since it is part of combined_sleep_bouts_df
    indexed_bouts = combined_sleep_bouts_df.set_index(["id", "sleep_bout_number"])
    bout_ds = xr.Dataset.from_dataframe(indexed_bouts)
    # from_dataframe can preserve pandas Arrow string dtypes; coerce so later
    # .sel/.isel on the merged dataset never hits ArrowStringArray.
    from dam_utilities import ensure_numpy_backed

    bout_ds = ensure_numpy_backed(bout_ds)

    # (The previous run's variables were dropped up front, before analysis_ds was
    # derived — see the comment there for why it cannot be done here.)

    # Step 1: Merge time-series variables (all share id × time dims with data).
    # bout_ds has dims (id × sleep_bout_number) — merging it in the same call
    # causes xarray to broadcast across all four dims, allocating a massive array.
    # join="exact": every input already carries data's exact id index (the helper
    # above) and its exact time index, so there is nothing to reconcile and any
    # future mismatch should fail loudly instead of being papered over.
    #
    # An earlier version of this comment claimed the join had to stay "outer"
    # because a phase selection left the masks with a SHORTER time axis. That was
    # wrong, and worth recording: select_phase never slices — it NaN-masks the
    # full axis (see its docstring) — so the masks always span data's whole time
    # range. The time-axis mismatch that seemed to prove otherwise was the
    # cartesian blow-up of the bout dimension, a separate bug fixed since.
    # Measured for LD, DD and both, on the fixture and on a real .nc: all five
    # merge inputs carry the identical time index.
    #
    # The id dimension needs the helper above regardless: an OUTER join over
    # equal-but-differently-ordered ids silently RE-SORTS the merged dataset's
    # flies, which is what made a later export list them in a different order.
    merged_ds = xr.merge(
        [data, combined_sleep_mask_da, combined_short_da, combined_inter_da, combined_long_da],
        join="exact",
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


def _filter_ds_by_group(ds, selected_genotypes=None, selected_temperatures=None):
    """Shared genotype/temperature filter used by the bout-duration helpers below.

    Mirrors the filter block duplicated across plotting.py's group-aware
    functions (e.g. daily_pattern_line). Returns the filtered dataset, or
    None if the filter leaves no flies.
    """
    if selected_genotypes is None and selected_temperatures is None:
        return ds
    filter_mask = True
    if "genotype" in ds.coords and selected_genotypes:
        filter_mask = filter_mask & ds["genotype"].isin(selected_genotypes)
    if "temperature" in ds.coords and selected_temperatures:
        filter_mask = filter_mask & ds["temperature"].isin(selected_temperatures)
    if not hasattr(filter_mask, "any"):
        return ds
    filtered_ids = ds["id"].where(filter_mask, drop=True)
    if len(filtered_ids) == 0:
        return None
    return ds.sel(id=filtered_ids)


def relative_minutes_of(ds, values):
    """Timestamps as elapsed minutes from the recording start, either dtype.

    ``start_time`` and the ``time`` coord carry whatever representation the
    import produced — relative integer minutes normally, datetime64 for an
    absolute-time dataset — so anything that compares them has to put both on
    one scale first.
    """
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.integer) or np.issubdtype(values.dtype, np.floating):
        return values.astype(float)
    # The FIRST fly's start, selected by dimension name rather than a bare [0]:
    # `get_zt_binned_dataframe` uses the same reference, so absolute-time
    # datasets land on the same scale here and there.
    ref_start = pd.to_datetime(ds["start_datetime"].isel(id=0).values)
    deltas = (pd.to_datetime(values) - ref_start).total_seconds() / 60.0
    return np.asarray(deltas, dtype=float)


def bouts_in_view(ds, bouts):
    """Drop bouts that start outside the epoch ``ds`` is a view of.

    Phase selection acts on the ``time`` axis. The bout table does not live on
    it — it is dimensioned on ``sleep_bout_number`` — so ``select_phase``
    leaves it completely untouched, and an LD view carries every DD bout as
    well. Any page that shows bouts beside a phase-selected profile will
    otherwise disagree with itself: the profile covers one epoch and the bout
    counts beside it cover the whole recording, under one epoch's label.

    The window comes from the standard sleep mask rather than from
    ``split_minute``, so it follows whatever the view actually masked —
    including ``discard_first_dd_day``, which moves the DD boundary a day
    later than ``split_minute`` records.

    A dataset that is not a phase view (nothing masked out) is returned
    untouched, so this is a no-op wherever there is no epoch to scope to.
    """
    if ds is None or "sleep" not in ds.data_vars or "time" not in ds.coords:
        return bouts
    if bouts is None or bouts.empty or "start_time" not in bouts.columns:
        return bouts
    mask = ds["sleep"]
    if set(mask.dims) != {"id", "time"}:
        return bouts
    # NaN >= 0 is False, so this rejects both representations of missing (the
    # -1 sentinel and a phase view's NaN) in one comparison.
    measured = np.asarray((mask >= 0).transpose("id", "time").values)
    if measured.all():
        return bouts  # nothing masked out: not a phase view

    times = relative_minutes_of(ds, ds["time"].values)
    lo, hi = {}, {}
    for row, fly in enumerate(str(i) for i in ds["id"].values):
        got = np.flatnonzero(measured[row])
        if got.size:
            # First and last MEASURED minute, so a real gap inside the epoch
            # does not shrink the window — a bout cannot start in a gap anyway.
            lo[fly], hi[fly] = times[got[0]], times[got[-1]]

    starts = relative_minutes_of(ds, bouts["start_time"].values)
    ids = bouts["id"].astype(str)
    low = ids.map(lo).to_numpy(dtype=float, na_value=np.nan)
    high = ids.map(hi).to_numpy(dtype=float, na_value=np.nan)
    with np.errstate(invalid="ignore"):
        keep = (starts >= low) & (starts <= high)
    return bouts[keep]


def raw_bout_dataframe(ds, selected_genotypes=None, selected_temperatures=None):
    """
    Tidy per-bout table: one row per detected sleep bout, across all flies.

    Columns: ``id``, ``sleep_bout_number``, ``group``, ``duration`` (minutes),
    plus ``sleep_state``/``start_time``/``end_time`` when present. This is the
    single source of truth behind the bout-duration CSV export and every
    bout-duration curve/summary calculation below, so the export and the
    plots can never drift apart.

    Requires that sleep_analysis() has already been run (adds 'duration').
    Returns an empty DataFrame (not an error) if 'duration' is absent, no
    fly survives the group filter, or every bout is NaN (ragged padding for
    flies with fewer bouts than the max — §2a: padding is never a real bout).
    """
    empty = pd.DataFrame(columns=["id", "sleep_bout_number", "group", "duration"])
    if ds is None or "duration" not in ds.data_vars:
        return empty

    ds = _filter_ds_by_group(ds, selected_genotypes, selected_temperatures)
    if ds is None:
        return empty

    bout_vars = [
        v for v in ("duration", "sleep_state", "start_time", "end_time") if v in ds.data_vars
    ]
    df = ds[bout_vars].to_dataframe().reset_index().dropna(subset=["duration"])
    # Scope to the epoch `ds` is a view of. This is the single source of truth
    # for every bout curve, summary and CSV on the Sleep & activity page, so
    # scoping here is what makes that page's phase selector reach the bout
    # tabs at all — the alternative is a Bouts tab silently covering the whole
    # recording while the profiles beside it cover one epoch.
    df = bouts_in_view(ds, df)
    if df.empty:
        return empty

    if "group" in ds.coords:
        group_dict = {
            id_val: ds["group"].sel(id=id_val).values.item() for id_val in df["id"].unique()
        }
        df["group"] = df["id"].map(group_dict)
    elif "genotype" in ds.coords and "temperature" in ds.coords:
        df["group"] = df["id"].apply(
            lambda x: f"{ds['genotype'].sel(id=x).values.item()}-{ds['temperature'].sel(id=x).values.item()}"
        )
    else:
        df["group"] = "All Flies"

    return df.reset_index(drop=True)


def per_fly_bout_duration_curves(
    ds,
    method="kde",
    selected_genotypes=None,
    selected_temperatures=None,
    n_grid=200,
    min_bouts=2,
    kde_bandwidth=0.3,
):
    """
    Per-fly sleep-bout-duration curve, one row per (fly, grid point).

    Computing ONE curve per fly (rather than pooling every bout across flies
    into a single distribution) means a fly with many bouts doesn't outweigh
    a fly with few — every fly counts once. All flies in the filtered set
    share the same x-grid, so their curves are directly comparable and can be
    pivoted straight into plotting.group_spectrum_plot's ``per_group_curves``
    (group -> (n_flies, n_grid) matrix).

    Parameters
    ----------
    ds : xr.Dataset
    method : {'kde', 'survival'}
        'kde': Gaussian KDE of log10(bout duration), evaluated on a
            log-spaced grid — a smooth density estimate. Right-skewed
            duration data is much better behaved log-transformed than raw.
        'survival': empirical P(bout duration > t), evaluated on a linear
            grid — the complementary CDF. No bandwidth or bin-width
            parameter at all; every bout contributes directly.
    n_grid : int
        Number of grid points spanning the pooled [min, max] bout duration.
    min_bouts : int
        Flies with fewer valid bouts than this are skipped (dropped, not
        fabricated) — a KDE/survival curve from 1 bout is not informative.
    kde_bandwidth : float
        Passed to ``scipy.stats.gaussian_kde`` as ``bw_method`` (method='kde' only).

    Returns
    -------
    pd.DataFrame
        Columns: ``id``, ``group``, ``x``, ``y``.
    """
    if method not in ("kde", "survival"):
        raise ValueError(f"Unknown method {method!r}; use 'kde' or 'survival'.")

    empty = pd.DataFrame(columns=["id", "group", "x", "y"])
    bout_df = raw_bout_dataframe(ds, selected_genotypes, selected_temperatures)
    if bout_df.empty:
        return empty

    lo = max(float(bout_df["duration"].min()), 1e-6)
    hi = float(bout_df["duration"].max())
    if hi <= lo:
        hi = lo * 1.5

    grid = (
        np.logspace(np.log10(lo), np.log10(hi), n_grid)
        if method == "kde"
        else np.linspace(lo, hi, n_grid)
    )

    rows = []
    for fly_id, fly_df in bout_df.groupby("id"):
        durations = fly_df["duration"].to_numpy(dtype=float)
        if len(durations) < min_bouts:
            continue
        group = fly_df["group"].iloc[0]

        if method == "kde":
            from scipy.stats import gaussian_kde

            try:
                kde = gaussian_kde(np.log10(durations), bw_method=kde_bandwidth)
                y = kde(np.log10(grid))
            except np.linalg.LinAlgError:
                # All bouts the same duration -> zero-variance input, KDE undefined.
                continue
        else:
            y = np.array([(durations > t).mean() for t in grid])

        rows.append(pd.DataFrame({"id": fly_id, "group": group, "x": grid, "y": y}))

    return pd.concat(rows, ignore_index=True) if rows else empty


def bout_duration_summary(ds, selected_genotypes=None, selected_temperatures=None):
    """
    Per-fly bout-duration summary — ONE row per fly.

    Columns: ``id``, ``group``, ``n_bouts``, ``median_duration_min``,
    ``log_mean_duration_min`` (geometric mean — the arithmetic mean in
    log10-duration space, back-transformed to minutes; more robust than the
    arithmetic mean for a strongly right-skewed bout-duration distribution).
    This is the per-fly table behind the group-comparison stats in
    :func:`bout_duration_group_stats` and the per-fly-summary CSV export —
    one computation, so the displayed stat and the export can't drift.

    Empty DataFrame if no fly survives the group filter.
    """
    cols = ["id", "group", "n_bouts", "median_duration_min", "log_mean_duration_min"]
    bout_df = raw_bout_dataframe(ds, selected_genotypes, selected_temperatures)
    if bout_df.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for fly_id, fly_df in bout_df.groupby("id"):
        durations = fly_df["duration"].to_numpy(dtype=float)
        rows.append(
            {
                "id": fly_id,
                "group": fly_df["group"].iloc[0],
                "n_bouts": len(durations),
                "median_duration_min": float(np.median(durations)),
                "log_mean_duration_min": float(10 ** np.mean(np.log10(durations))),
            }
        )
    return pd.DataFrame(rows, columns=cols).sort_values(["group", "id"]).reset_index(drop=True)


def bout_duration_group_stats(summary_df, value_col="log_mean_duration_min", alpha=0.05):
    """
    Omnibus + pairwise group comparison of a per-fly bout-duration summary
    statistic (from :func:`bout_duration_summary`).

    The comparison runs on log10(value_col) — bout-duration summaries are
    themselves right-skewed across flies, and testing in log space is what
    makes ``log_mean_duration_min`` (a geometric mean) the natural pairing:
    an arithmetic-mean/normality-based test on log10 values is exactly a
    test on that geometric mean.

    Groups with fewer than 2 flies are dropped from the comparison (a mean
    of one fly is a data point, not a distribution).

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of :func:`bout_duration_summary`; must have ``group`` and
        ``value_col`` columns.
    value_col : str
    alpha : float
        Significance threshold for the normality/variance pre-checks that
        decide ANOVA vs. Kruskal-Wallis, and for pairwise significance.

    Returns
    -------
    dict
        ``test`` ('anova'/'kruskal'/'none'), ``statistic``, ``pvalue``,
        ``pairwise`` (list of {group_a, group_b, pvalue, pvalue_adj,
        significant, stars}), ``n_per_group``, ``normality_passed``,
        ``equal_variance_passed``, ``notes``.
    """
    from scipy import stats as _stats

    result = {
        "test": "none",
        "statistic": float("nan"),
        "pvalue": float("nan"),
        "pairwise": [],
        "n_per_group": {},
        "normality_passed": False,
        "equal_variance_passed": False,
        "notes": "",
    }

    if summary_df is None or summary_df.empty or value_col not in summary_df.columns:
        result["notes"] = "No per-fly summary data available."
        return result

    df = summary_df.dropna(subset=[value_col])
    df = df[df[value_col] > 0]  # log10 requires positive values
    n_per_group = df.groupby("group")[value_col].size().to_dict()
    result["n_per_group"] = n_per_group

    valid_groups = sorted(g for g, n in n_per_group.items() if n >= 2)
    if len(valid_groups) < 2:
        result["notes"] = "Fewer than 2 groups with >=2 flies; no test run."
        return result

    values_by_group = {
        g: np.log10(df.loc[df["group"] == g, value_col].to_numpy()) for g in valid_groups
    }
    samples = [values_by_group[g] for g in valid_groups]

    normal = all(len(s) < 3 or _stats.shapiro(s)[1] > alpha for s in samples)
    equal_var = _stats.levene(*samples)[1] > alpha
    result["normality_passed"] = bool(normal)
    result["equal_variance_passed"] = bool(equal_var)

    if normal and equal_var:
        stat, p = _stats.f_oneway(*samples)
        test_name = "anova"
    else:
        stat, p = _stats.kruskal(*samples)
        test_name = "kruskal"
    result["test"] = test_name
    result["statistic"] = float(stat)
    result["pvalue"] = float(p)

    pair_keys, pair_pvals = [], []
    for i in range(len(valid_groups)):
        for j in range(i + 1, len(valid_groups)):
            ga, gb = valid_groups[i], valid_groups[j]
            if normal:
                _, p_pair = _stats.ttest_ind(
                    values_by_group[ga], values_by_group[gb], equal_var=equal_var
                )
            else:
                _, p_pair = _stats.mannwhitneyu(
                    values_by_group[ga], values_by_group[gb], alternative="two-sided"
                )
            pair_keys.append((ga, gb))
            pair_pvals.append(float(p_pair))

    if pair_pvals:
        if len(pair_pvals) > 1:
            from statsmodels.stats.multitest import multipletests

            _, p_adj, _, _ = multipletests(pair_pvals, alpha=alpha, method="holm")
        else:
            p_adj = pair_pvals
        for (ga, gb), p_raw, p_corr in zip(pair_keys, pair_pvals, p_adj):
            if p_corr < 0.001:
                stars = "***"
            elif p_corr < 0.01:
                stars = "**"
            elif p_corr < alpha:
                stars = "*"
            else:
                stars = "ns"
            result["pairwise"].append(
                {
                    "group_a": ga,
                    "group_b": gb,
                    "pvalue": float(p_raw),
                    "pvalue_adj": float(p_corr),
                    "significant": bool(p_corr < alpha),
                    "stars": stars,
                }
            )
    return result
