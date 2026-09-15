"""
bout_spectrum.py
================
Sleep-bout *structure*: how many bouts of each length a fly has, and how much
sleep each inactivity-duration definition of sleep accounts for.

This is the empirical half of Abhilash & Shafer (2024, *SLEEP* 47:zsad277)
Figure 6 — the panel that asks "which bout lengths actually carry the signal?" —
with no two-process model attached. Two views:

**Bout spectrum** — bouts binned by duration (5-10, 10-20, ... , 240+ min);
per fly you get the number of bouts in each bin and the total sleep minutes
those bouts account for.

**Sleep by definition** — the same bouts re-totalled under each candidate
definition of sleep: minimum definitions (any inactivity >= 5, 10, 20, 30, 40,
50, 60 min) and interval definitions (5-10, 5-20, 5-30, 30-60, 30-120, 60-120
min), the exact ladder used in Figure 6.

Both read the per-bout table that :func:`sleep_analysis.sleep_analysis` already
wrote onto the dataset (``duration``, dims ``id`` x ``sleep_bout_number``), so
nothing is re-detected here and the numbers cannot drift from the sleep masks.

Resolution limit
----------------
That table only holds bouts that met the sleep threshold, so the shortest
definition this module can honestly report is the threshold itself
(``ds.attrs['sleep_threshold_seconds']``, 300 s by default). Bins or definitions
reaching below it are unanswerable from the dataset — see
:func:`min_definable_minutes`.
"""

import numpy as np
import pandas as pd
import xarray as xr

MINUTES_PER_DAY = 1440

# Duration-bin edges (minutes). The last bin is open-ended: 240+.
DEFAULT_BIN_EDGES = [5, 10, 20, 30, 45, 60, 90, 120, 180, 240]

# Figure 6's definition ladder.
DEFAULT_MIN_DEFINITIONS = [5, 10, 20, 30, 40, 50, 60]
DEFAULT_INTERVAL_DEFINITIONS = [(5, 10), (5, 20), (5, 30), (30, 60), (30, 120), (60, 120)]


def min_definable_minutes(ds):
    """Shortest bout duration present in the dataset's bout table, in minutes.

    Bouts shorter than the sleep threshold were never recorded, so any bin or
    definition whose lower edge falls below this is unanswerable from this
    dataset (not empty — unmeasured). Defaults to 5 min if the attribute is
    absent (the standard threshold).
    """
    return float(ds.attrs.get("sleep_threshold_seconds", 300)) / 60.0


def bin_label(lo, hi):
    """Human label for a duration bin: ``'5-10 min'``, or ``'240+ min'`` when open."""
    if hi is None or not np.isfinite(hi):
        return f"{lo:g}+ min"
    return f"{lo:g}-{hi:g} min"


def make_bins(edges):
    """Turn ascending edge values into ``[(lo, hi), ...]`` with an open top bin.

    ``[5, 10, 20]`` -> ``[(5, 10), (10, 20), (20, inf)]``. Membership is
    ``lo <= duration < hi`` throughout, so every bout falls in exactly one bin.
    """
    edges = [float(e) for e in edges]
    if len(edges) < 1:
        raise ValueError("Need at least one bin edge.")
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError("Bin edges must be strictly increasing.")
    if edges[0] < 0:
        raise ValueError("Bin edges must be non-negative.")
    return [(lo, hi) for lo, hi in zip(edges, edges[1:])] + [(edges[-1], np.inf)]


def parse_edges(text):
    """Parse a comma/space-separated edge list typed by the user into floats.

    Raises ``ValueError`` with a readable message on anything unparseable, so
    the page can surface it verbatim.
    """
    tokens = [t for t in str(text).replace(",", " ").split() if t]
    if not tokens:
        raise ValueError("No bin edges given.")
    try:
        vals = [float(t) for t in tokens]
    except ValueError as exc:
        raise ValueError(f"Bin edges must be numbers: {exc}") from None
    if any(b <= a for a, b in zip(vals, vals[1:])):
        raise ValueError("Bin edges must be strictly increasing.")
    return vals


def definition_label(lo, hi):
    """Human label for a sleep definition: ``'>=30 min'`` or ``'30-60 min'``."""
    if hi is None or not np.isfinite(hi):
        return f">={lo:g} min"
    return f"{lo:g}-{hi:g} min"


def default_definitions():
    """The Figure 6 ladder as ``[(label, lo, hi), ...]`` — minimums then intervals."""
    defs = [(definition_label(lo, np.inf), float(lo), np.inf) for lo in DEFAULT_MIN_DEFINITIONS]
    defs += [
        (definition_label(lo, hi), float(lo), float(hi)) for lo, hi in DEFAULT_INTERVAL_DEFINITIONS
    ]
    return defs


def _group_labels(ds):
    """Map fly id -> group label, using the unified ``group`` coord when present.

    Falls back to ``genotype-temperature`` (the pre-unification pairing still
    honoured elsewhere in the codebase), then to ``'All'``.
    """
    ids = [str(i) for i in ds["id"].values]
    if "group" in ds.coords:
        vals = np.asarray(ds["group"].values, dtype=object)
        return {fid: str(v) for fid, v in zip(ids, vals)}
    if "genotype" in ds.coords and "temperature" in ds.coords:
        gen = np.asarray(ds["genotype"].values, dtype=object)
        tmp = np.asarray(ds["temperature"].values, dtype=object)
        return {fid: f"{g}-{t}" for fid, g, t in zip(ids, gen, tmp)}
    return dict.fromkeys(ids, "All")


def _step_minutes(ds, t_column="time"):
    """Sampling interval of the time axis in minutes (median of the diffs).

    Handles both time representations the pipeline uses: relative integer
    minutes and absolute datetimes.
    """
    tvals = np.asarray(ds[t_column].values)
    if tvals.size < 2:
        return 1.0
    diffs = np.diff(tvals)
    if np.issubdtype(diffs.dtype, np.timedelta64):
        step = float(np.median(diffs.astype("timedelta64[s]").astype(float))) / 60.0
    else:
        step = float(np.median(diffs.astype(float)))
    return step if step > 0 else 1.0


def _day_of(minutes):
    """Absolute 0-based day index for a relative-minute value (or array)."""
    return np.floor(np.asarray(minutes, dtype=float) / MINUTES_PER_DAY).astype(int)


def per_fly_recording_days(ds, t_column="time", days=None):
    """Days of *valid* record per fly, as a ``Series`` indexed by fly id (str).

    Counts only minutes where ``sleep != -1``, so out-of-phase minutes and data
    gaps are excluded rather than silently inflating the denominator (missing is
    never counted as 0). Used to turn raw bout counts into bouts/day, the only
    fair comparison when flies differ in how much usable record they contributed.

    ``days`` restricts the count to those absolute day indices, so the denominator
    covers exactly the days the bouts were taken from — otherwise a five-day
    selection would still be divided by twelve days of record.
    """
    if "sleep" not in ds.data_vars:
        return pd.Series(dtype=float)
    step = _step_minutes(ds, t_column)
    valid = ds["sleep"] != -1
    if days is not None:
        keep = np.isin(_day_of(ds[t_column].values), np.asarray(list(days), dtype=int))
        valid = valid & xr.DataArray(keep, coords={t_column: ds[t_column]}, dims=[t_column])
    counted = valid.sum(dim=t_column)
    out = np.asarray(counted.values, dtype=float) * step / 1440.0
    return pd.Series(out, index=[str(i) for i in ds["id"].values], dtype=float)


def bout_table(ds, days=None):
    """Tidy per-bout table: ``ID``, ``Group``, ``sleep_bout_number``, ``duration``,
    ``start_time``, ``day``.

    One row per detected sleep bout, NaN padding dropped. Empty DataFrame (with
    the right columns) when sleep analysis has not been run.

    ``days`` keeps only bouts that START on one of those absolute day indices. A
    bout is credited to the day it begins in — the same rule SCAMP's ``sleepcalc3``
    uses for its bins — so a bout running past midnight counts once, on its first
    day, and is never split or double-counted. ``day`` is only meaningful on a
    relative-integer-minute time axis; on a datetime axis it is left as NaN and
    ``days`` filtering is refused.
    """
    cols = ["ID", "Group", "sleep_bout_number", "duration", "start_time", "day"]
    if "duration" not in ds.data_vars or "sleep_bout_number" not in ds.dims:
        return pd.DataFrame(columns=cols)

    df = ds["duration"].to_dataframe(name="duration").reset_index()
    df = df.dropna(subset=["duration"])
    if df.empty:
        return pd.DataFrame(columns=cols)

    if "start_time" in ds.data_vars:
        starts = ds["start_time"].to_dataframe(name="start_time").reset_index()
        df = df.merge(starts, on=["id", "sleep_bout_number"], how="left")
    else:
        df["start_time"] = np.nan

    relative = np.issubdtype(np.asarray(ds["time"].values).dtype, np.integer)
    if relative:
        with np.errstate(invalid="ignore"):
            df["day"] = np.where(
                df["start_time"].notna(), _day_of(df["start_time"].fillna(0)), -1
            )
        df.loc[df["start_time"].isna(), "day"] = np.nan
    else:
        df["day"] = np.nan
        if days is not None:
            raise ValueError(
                "Day selection needs a relative-integer-minute time axis; this "
                "dataset uses absolute datetimes."
            )

    if days is not None:
        df = df[df["day"].isin([int(d) for d in days])]

    df["ID"] = df["id"].astype(str)
    df["Group"] = df["ID"].map(_group_labels(ds)).fillna("All")
    return df[cols].reset_index(drop=True)


def per_fly_bout_counts(ds, bins=None, per_day=False, as_percent=False, days=None):
    """Per-fly bout counts and sleep minutes in each duration bin — the raw
    numbers behind the bout-spectrum bars.

    Every fly gets a row for every bin (zero-filled), so a fly with no long
    bouts contributes a real 0 to the group mean instead of dropping out and
    biasing it upward.

    Parameters
    ----------
    ds : xr.Dataset
        Must carry the bout table written by ``sleep_analysis``.
    bins : list of (lo, hi), optional
        Duration bins in minutes; defaults to :data:`DEFAULT_BIN_EDGES`.
    per_day : bool
        Divide both counts and minutes by each fly's days of valid record.
    as_percent : bool
        Express each bin as a % of that fly's total bouts / total sleep minutes
        instead of an absolute. Applied instead of ``per_day`` (a percentage of
        a per-day value is the same percentage).

    Returns
    -------
    pd.DataFrame
        Long: ``ID``, ``Group``, ``bin_label``, ``bin_lo``, ``bin_hi``,
        ``n_bouts``, ``sleep_minutes``, ``recording_days``.
    """
    bins = bins if bins is not None else make_bins(DEFAULT_BIN_EDGES)
    cols = [
        "ID",
        "Group",
        "bin_label",
        "bin_lo",
        "bin_hi",
        "n_bouts",
        "sleep_minutes",
        "recording_days",
    ]
    bouts = bout_table(ds, days=days)
    if bouts.empty:
        return pd.DataFrame(columns=cols)

    rec_days = per_fly_recording_days(ds, days=days)
    groups = _group_labels(ds)
    by_fly = {fid: g["duration"].to_numpy(dtype=float) for fid, g in bouts.groupby("ID")}

    rows = []
    for fid in [str(i) for i in ds["id"].values]:
        sub = by_fly.get(fid, np.empty(0, dtype=float))
        fly_days = float(rec_days.get(fid, np.nan))
        for lo, hi in bins:
            in_bin = (sub >= lo) & (sub < hi)
            rows.append(
                {
                    "ID": fid,
                    "Group": groups.get(fid, "All"),
                    "bin_label": bin_label(lo, hi),
                    "bin_lo": float(lo),
                    "bin_hi": float(hi),
                    "n_bouts": float(in_bin.sum()),
                    "sleep_minutes": float(sub[in_bin].sum()),
                    "recording_days": fly_days,
                }
            )

    out = pd.DataFrame(rows, columns=cols)
    out = _normalize(out, ["n_bouts", "sleep_minutes"], per_day, as_percent)
    return out.sort_values(["Group", "ID", "bin_lo"]).reset_index(drop=True)


def per_fly_sleep_by_definition(
    ds, definitions=None, per_day=False, as_percent=False, days=None
):
    """Per-fly sleep totals under each candidate definition of sleep.

    For definition ``[lo, hi)``, a fly's sleep is the summed duration of every
    bout whose length falls in that window — i.e. what the fly's sleep record
    would look like if that were the rule. Minimum definitions nest (>=5 min
    contains >=30 min), so these rows are deliberately NOT additive; the
    interval definitions are the disjoint ones.

    Parameters
    ----------
    ds : xr.Dataset
    definitions : list of (label, lo, hi), optional
        Defaults to :func:`default_definitions` (the Figure 6 ladder).
    per_day, as_percent : bool
        As in :func:`per_fly_bout_counts`, except that ``as_percent`` here is
        relative to the fly's *total* recorded sleep (all bouts), not to the sum
        across definitions — which would be meaningless for the overlapping
        minimum definitions.

    Returns
    -------
    pd.DataFrame
        Long: ``ID``, ``Group``, ``definition``, ``def_lo``, ``def_hi``,
        ``sleep_minutes``, ``n_bouts``, ``recording_days``.
    """
    definitions = definitions if definitions is not None else default_definitions()
    cols = [
        "ID",
        "Group",
        "definition",
        "def_lo",
        "def_hi",
        "sleep_minutes",
        "n_bouts",
        "recording_days",
    ]
    bouts = bout_table(ds, days=days)
    if bouts.empty:
        return pd.DataFrame(columns=cols)

    rec_days = per_fly_recording_days(ds, days=days)
    groups = _group_labels(ds)
    by_fly = {fid: g["duration"].to_numpy(dtype=float) for fid, g in bouts.groupby("ID")}

    rows = []
    for fid in [str(i) for i in ds["id"].values]:
        sub = by_fly.get(fid, np.empty(0, dtype=float))
        fly_days = float(rec_days.get(fid, np.nan))
        fly_total_min = float(sub.sum())
        fly_total_n = float(sub.size)
        for label, lo, hi in definitions:
            sel = (sub >= lo) & (sub < hi)
            rows.append(
                {
                    "ID": fid,
                    "Group": groups.get(fid, "All"),
                    "definition": label,
                    "def_lo": float(lo),
                    "def_hi": float(hi),
                    "sleep_minutes": float(sub[sel].sum()),
                    "n_bouts": float(sel.sum()),
                    "recording_days": fly_days,
                    "_total_min": fly_total_min,
                    "_total_n": fly_total_n,
                }
            )

    out = pd.DataFrame(rows)
    if as_percent:
        # Denominator = the fly's total recorded sleep, so each value reads as
        # "% of this fly's sleep that this rule keeps".
        out["sleep_minutes"] = np.where(
            out["_total_min"] > 0, out["sleep_minutes"] / out["_total_min"] * 100.0, np.nan
        )
        out["n_bouts"] = np.where(
            out["_total_n"] > 0, out["n_bouts"] / out["_total_n"] * 100.0, np.nan
        )
    elif per_day:
        for col in ("sleep_minutes", "n_bouts"):
            out[col] = np.where(out["recording_days"] > 0, out[col] / out["recording_days"], np.nan)

    out = out[cols]
    return out.sort_values(["Group", "ID", "def_lo", "def_hi"]).reset_index(drop=True)


def _normalize(df, value_cols, per_day, as_percent):
    """Apply the per-day and/or per-fly-percent transform to ``value_cols``.

    ``as_percent`` wins over ``per_day``, and its denominator is the fly's own
    total across the rows present — correct for the duration bins, which
    partition the bouts exactly once.
    """
    if as_percent:
        for col in value_cols:
            totals = df.groupby("ID")[col].transform("sum")
            df[col] = np.where(totals > 0, df[col] / totals * 100.0, np.nan)
        return df
    if per_day:
        for col in value_cols:
            df[col] = np.where(df["recording_days"] > 0, df[col] / df["recording_days"], np.nan)
    return df


def summarize_by_group(per_fly_df, value_col, category_col, category_order=None):
    """Group mean +/- SEM of ``value_col`` for each (group, category) cell.

    The one aggregation both the bars and the exported CSV read, so the plotted
    numbers and the saved numbers are the same computation (no drift).

    Returns ``group``, ``<category_col>``, ``mean``, ``sem``, ``n`` — ``n`` is the
    number of flies contributing a non-NaN value.

    A cell with one fly gets ``sem = NaN``, not 0. One value has no standard error,
    and a zero-length error bar is a claim about variability rather than an admission
    of ignorance — the same distinction ``scamp_sleep.mean_sem`` makes, and §2a's
    rule that a missing quantity is never a zero.
    """
    if per_fly_df is None or per_fly_df.empty:
        return pd.DataFrame(columns=["group", category_col, "mean", "sem", "n"])

    stat = (
        per_fly_df.groupby(["Group", category_col], sort=False)[value_col]
        .agg(
            mean="mean",
            sem=lambda x: (x.std(ddof=1) / np.sqrt(x.notna().sum()))
            if x.notna().sum() > 1
            else np.nan,
            n=lambda x: int(x.notna().sum()),
        )
        .reset_index()
        .rename(columns={"Group": "group"})
    )
    if category_order is not None:
        order = {c: i for i, c in enumerate(category_order)}
        stat["_o"] = stat[category_col].map(order)
        stat = stat.sort_values(["group", "_o"]).drop(columns="_o")
    return stat.reset_index(drop=True)


def wide_by_group(per_fly_df, value_col, category_col, category_order=None):
    """Per-fly table pivoted wide — one row per fly, one column per category.

    The shape most stats packages want for a repeated-measures comparison across
    bins/definitions; the long form stays available for anything else.
    """
    if per_fly_df is None or per_fly_df.empty:
        return pd.DataFrame(columns=["ID", "Group"])
    wide = per_fly_df.pivot_table(
        index=["ID", "Group"], columns=category_col, values=value_col, aggfunc="first"
    ).reset_index()
    wide.columns.name = None
    if category_order is not None:
        keep = [c for c in category_order if c in wide.columns]
        wide = wide[["ID", "Group"] + keep]
    return wide.sort_values(["Group", "ID"]).reset_index(drop=True)
