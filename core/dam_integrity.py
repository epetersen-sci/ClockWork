"""
dam_integrity.py
================
Data-quality handling for raw Trikinetics DAM reads, applied during loading.

This module is the single home for the three load-time data-integrity pieces.
It is called from ``dam_processor.validate_files_and_dates`` while the raw
``monitor_status`` field is still available (it is dropped before the activity
columns reach the xarray builder). Keeping the logic here — not inline in the
loader — gives the three pieces one test surface and one §2a NaN invariant.

Vendor grounding (Trikinetics DAMSystem datasheet / user's guide, corroborated
on the web 2026-06): "any code other than 1 indicates that no data was received
from the monitor." So **only ``monitor_status == 1`` is real data.** Every other
code (24 power-on reset, 50 no PSU connection, 51 absent monitor / no response,
and the transient comm/CRC error codes 52/53/54/55) means the read FAILED and no
valid measurement exists for that minute — whatever bytes sit in the activity
columns of such a row are not a reading. Per §2a (zero ≠ NaN) those minutes must
become NaN on the time grid, never zero: a status-51 absent-monitor row is
all-zero, but that zero means "no reading," not "the fly did not move."

The three pieces (all operate on the per-monitor, in-window raw frame):
  1. ``resolve_status_and_duplicates`` — apply the status rule + de-duplicate the
     timestamp axis. At a timestamp slot a ``status == 1`` row wins; if only
     ``status != 1`` rows exist there, the slot survives as a single NaN row.
     This is what lets the downstream reindex (which raises on duplicate labels)
     succeed, and it subsumes the per-dataset status-50 placeholder doubling.
  2. ``scan_time_integrity`` — report (do not absorb) monotonicity, duplicate
     timestamps, gaps vs the reading interval, and interval mismatch, per monitor.
  3. ``classify_irregularities`` — split every irregular grid slot into COSMETIC
     (a real ``status == 1`` reading survived → quiet summary) vs DATA-LOSS (no
     ``status == 1`` at that slot → a genuine NaN hole → prominent warning).

All three share ONE handling: every data-loss slot is NaN-not-zero; the report
just makes the holes visible. They are built together so they cannot drift.
"""

import numpy as np
import pandas as pd

VALID_STATUS = 1  # the ONLY code that means "good data read" (vendor datasheet)

# The status codes whose meaning we have documented from the Trikinetics datasheet
# (1 good, 24 power-on reset, 50 no-PSU, 51 absent-monitor). Any OTHER non-1 code
# seen in the wild (e.g. the comm/CRC family 52/53/54/55, or a stray 49) is still
# handled identically — NaN, never zero (the rule is binary: ==1 is data) — but it
# is also REPORTED with a count so its frequency/pattern can be tracked over time.
# This is visibility, NOT an error: an undocumented code is not a failure, just a
# code we have not catalogued a meaning for.
DOCUMENTED_STATUS = frozenset({1, 24, 50, 51})

# Human-readable labels for the codes seen in the wild (for reporting only — the
# RULE is binary: ==1 is data, anything else is not).
STATUS_MEANINGS = {
    1: "good data read",
    24: "power-on reset (no data)",
    50: "no PSU/PSIU connection (no data)",
    51: "no monitor response / absent monitor (no data)",
    52: "monitor read error (no data)",
    53: "monitor read error (no data)",
    54: "monitor read error (no data)",
    55: "monitor read error (no data)",
}


def status_label(code):
    """Return a short human-readable label for a status code."""
    try:
        code = int(code)
    except (TypeError, ValueError):
        return str(code)
    return STATUS_MEANINGS.get(code, f"non-valid status {code} (no data)")


def resolve_status_and_duplicates(
    monitor_df, status_col="monitor_status", time_col="datetime", activity_cols=None
):
    """Apply the status rule and de-duplicate the timestamp axis for one monitor.

    Operates on a single monitor's raw frame (already sliced to its metadata
    window), BEFORE per-region columns are selected. Returns a frame with a
    UNIQUE ``time_col`` axis where:

      * activity at any ``status != 1`` row is set to NaN (§2a: no reading, not
        zero activity);
      * at a timestamp slot with both a ``status == 1`` row and ``status != 1``
        rows, the ``status == 1`` row WINS (the error rows were redundant);
      * at a slot with only ``status != 1`` rows, a single NaN row survives (the
        read was attempted and failed → a genuine hole, kept as NaN so the grid
        reindex does not raise on duplicate labels and the hole stays visible).

    The discriminator is ALWAYS the status field — never zero-ness or read order.
    A legitimate all-zero ``status == 1`` row (a fly that truly did not move) is
    real data and is preserved.

    Parameters
    ----------
    monitor_df : pd.DataFrame
        One monitor's rows; must contain ``status_col`` and ``time_col`` plus the
        ``channel_*`` activity columns.
    status_col, time_col : str
        Column names.
    activity_cols : list[str] or None
        Activity columns to NaN out for failed reads. If None, every column whose
        name starts with ``"channel_"`` is used.

    Returns
    -------
    (resolved_df, info) : (pd.DataFrame, dict)
        ``resolved_df`` has a unique ``time_col`` axis (still a column, not the
        index). ``info`` carries counts for reporting:
        ``n_rows_in``, ``n_status1``, ``n_status_bad``, ``bad_status_counts``,
        ``n_dup_slots``, ``n_rows_dropped`` (redundant rows removed by de-dup),
        ``n_slots_out`` (unique timestamps in the output).
    """
    df = monitor_df.copy()
    if activity_cols is None:
        activity_cols = [c for c in df.columns if str(c).startswith("channel_")]

    is_valid = df[status_col].astype("Int64") == VALID_STATUS
    n_rows_in = len(df)
    n_status1 = int(is_valid.sum())
    bad_mask = ~is_valid
    n_status_bad = int(bad_mask.sum())
    bad_status_counts = (
        df.loc[bad_mask, status_col].value_counts().to_dict() if n_status_bad else {}
    )

    # §2a: a failed read carries NO measurement → NaN, never the bytes in the row
    # (which may be all-zero for an absent monitor, or stale/garbage for a comm
    # error). Float so NaN can be held; downcast happens later in the xarray build.
    if n_status_bad:
        df[activity_cols] = df[activity_cols].astype("float64")
        df.loc[bad_mask.values, activity_cols] = np.nan

    # Mark validity per row so de-dup can prefer a real reading.
    df["_status_valid"] = is_valid.values

    # De-duplicate the timestamp axis: a status==1 row wins; among ties keep the
    # first. Sorting valid-first and then dropping duplicates keeps the real
    # reading where one exists, and a single NaN row where none does.
    n_dup_slots = int(
        df[time_col].duplicated(keep=False).sum() > 0 and (df[time_col].value_counts() > 1).sum()
    )
    df = df.sort_values([time_col, "_status_valid"], ascending=[True, False], kind="mergesort")
    before = len(df)
    df = df.drop_duplicates(subset=[time_col], keep="first")
    n_rows_dropped = before - len(df)

    df = df.drop(columns=["_status_valid"])
    bad_counts_int = {int(k): int(v) for k, v in bad_status_counts.items()}
    # Codes BEYOND the documented set {1, 24, 50, 51} (e.g. 52/53/55, or a stray
    # 49): handled identically (NaN, never zero) but tracked separately so their
    # frequency/pattern is visible over time. Not an error — just visibility.
    undocumented_counts = {c: n for c, n in bad_counts_int.items() if c not in DOCUMENTED_STATUS}
    info = {
        "n_rows_in": n_rows_in,
        "n_status1": n_status1,
        "n_status_bad": n_status_bad,
        "bad_status_counts": bad_counts_int,
        "undocumented_status_counts": undocumented_counts,
        "n_dup_slots": int(n_dup_slots),
        "n_rows_dropped": int(n_rows_dropped),
        "n_slots_out": int(len(df)),
    }
    return df, info


def _expected_interval_minutes(times):
    """Estimate the reading interval (minutes) as the modal positive diff."""
    if len(times) < 2:
        return 1.0
    diffs = pd.Series(times).diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return 1.0
    mode = diffs.mode()
    if len(mode):
        return mode.iloc[0].total_seconds() / 60.0
    return diffs.median().total_seconds() / 60.0


def scan_time_integrity(times, status1_times=None, gap_threshold=None, interval_minutes=None):
    """Scan a monitor's in-window time axis and REPORT (do not absorb).

    This is the FileScan validation role: it characterizes the time axis without
    changing data. It looks at the timestamps the monitor actually emitted (after
    de-dup) and, optionally, the subset of those that were real (``status == 1``)
    readings, so gaps reflect where genuine data is missing.

    Parameters
    ----------
    times : array-like of datetime64
        All de-duplicated in-window timestamps for the monitor.
    status1_times : array-like of datetime64 or None
        The subset of ``times`` that carried a ``status == 1`` reading. Gaps and
        coverage are measured against the EXPECTED grid using these when given
        (so a run of failed reads counts as missing real data).
    gap_threshold : pd.Timedelta or None
        Spacing strictly greater than this is reported as a significant gap.
        Defaults to ``max(2 * interval, 1h)`` — formalizes the loader's existing
        ``self.gap_threshold`` concept.
    interval_minutes : float or None
        Expected reading interval; inferred from ``times`` if None.

    Returns
    -------
    dict with keys:
        monotonic (bool), n_duplicates (int, residual — should be 0 after de-dup),
        interval_minutes (float), interval_mismatch (bool), n_expected_slots,
        n_present_slots (status1 if provided else all), n_missing_slots,
        gaps (list of {start, end, minutes}), coverage_fraction.
    """
    times = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(times)))).sort_values()
    monotonic = bool(pd.Series(times).is_monotonic_increasing)
    n_duplicates = int(pd.Series(times).duplicated().sum())

    if interval_minutes is None:
        interval_minutes = _expected_interval_minutes(times)
    interval = pd.Timedelta(minutes=interval_minutes)
    if gap_threshold is None:
        gap_threshold = max(interval * 2, pd.Timedelta(hours=1))

    # Real-data timestamps for coverage/gaps (fall back to all if not given).
    real = (
        pd.DatetimeIndex(pd.to_datetime(pd.Series(list(status1_times)))).sort_values()
        if status1_times is not None
        else times
    )

    # Expected grid spans the in-window extent at the reading interval.
    if len(times):
        grid = pd.date_range(times[0], times[-1], freq=interval)
        n_expected = len(grid)
    else:
        n_expected = 0
    n_present = len(real)
    n_missing = max(0, n_expected - n_present)

    # Gaps in the REAL-data series (consecutive real readings spaced > threshold).
    gaps = []
    if len(real) >= 2:
        rvals = real.values
        spans = np.diff(rvals)  # timedelta64 between consecutive real readings
        thresh = np.timedelta64(gap_threshold)
        for i in np.where(spans > thresh)[0]:
            gaps.append(
                {
                    "start": pd.Timestamp(rvals[i]),
                    "end": pd.Timestamp(rvals[i + 1]),
                    "minutes": float(spans[i] / np.timedelta64(1, "m")),
                }
            )

    # Interval mismatch: most-common spacing differs from the nominal interval.
    actual_interval = _expected_interval_minutes(times)
    interval_mismatch = abs(actual_interval - interval_minutes) > 1e-6

    return {
        "monotonic": monotonic,
        "n_duplicates": n_duplicates,
        "interval_minutes": float(interval_minutes),
        "actual_interval_minutes": float(actual_interval),
        "interval_mismatch": bool(interval_mismatch),
        "n_expected_slots": int(n_expected),
        "n_present_slots": int(n_present),
        "n_missing_slots": int(n_missing),
        "gaps": gaps,
        "coverage_fraction": float(n_present / n_expected) if n_expected else 1.0,
    }


def classify_irregularities(
    all_times,
    status1_times,
    error_times=None,
    interval_minutes=None,
    grid_start=None,
    grid_end=None,
):
    """Two-tier classification of irregular slots on the EXPECTED grid.

    Run AFTER status resolution + de-dup. Every interval that should carry a
    reading is examined:

      * COSMETIC: a grid slot that has a ``status == 1`` reading AND also had an
        error (``status != 1``) row originally — the error row was redundant, no
        data was lost. (Includes the de-dup'd status-50 placeholder doublings.)
      * DATA-LOSS: a grid slot with NO ``status == 1`` reading — whether an
        error-only row was present (read attempted, failed) OR no row at all
        (skipped reading). The recording does not exist there → NaN; the user
        must see it.

    The precise line (per the task): cosmetic = an error row *with* a surviving
    status-1 row at that slot; data-loss = no status-1 at that slot (error-only
    OR absent). A failed-read-with-no-real-reading is DATA-LOSS, never cosmetic.

    Parameters
    ----------
    all_times : array-like of datetime64
        De-duplicated timestamps the monitor emitted in window (one per slot it
        produced a row for, status-1 or not).
    status1_times : array-like of datetime64
        Timestamps carrying a real ``status == 1`` reading (subset of the grid).
        May contain duplicates (the raw, pre-dedup status-1 timestamps); that is
        fine, they are reduced to a set here.
    error_times : array-like of datetime64 or None
        Timestamps of ``status != 1`` (error/placeholder) rows, pre-dedup. Used
        to tell a cosmetic slot (status1 + a redundant error row) from a clean
        status1 slot. If None, no slot is marked cosmetic.
    interval_minutes : float or None
        Expected reading interval; inferred if None.
    grid_start, grid_end : pd.Timestamp or None
        Explicit bounds of the EXPECTED grid (the metadata window). When given,
        trailing/leading slots with no row at all (e.g. a file that ended before
        the window stop) are counted as DATA-LOSS-absent. If None, the grid spans
        the present rows only (cannot see absent slots beyond the last row).

    Returns
    -------
    dict with keys:
        n_expected_slots, n_status1_slots, n_cosmetic_slots, n_dataloss_slots,
        dataloss_error_only (slot had an error row but no status1),
        dataloss_absent (slot had no row at all),
        dataloss_spans (list of {start, end, n_slots}) — contiguous data-loss runs.
    """
    all_idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(all_times)))).sort_values()
    s1_idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(status1_times)))).sort_values()
    err_idx = (
        pd.DatetimeIndex(pd.to_datetime(pd.Series(list(error_times)))).sort_values()
        if error_times is not None
        else pd.DatetimeIndex([])
    )

    if interval_minutes is None:
        interval_minutes = _expected_interval_minutes(all_idx)
    interval = pd.Timedelta(minutes=interval_minutes)

    if len(all_idx) == 0 and grid_start is None:
        return {
            "n_expected_slots": 0,
            "n_status1_slots": 0,
            "n_cosmetic_slots": 0,
            "n_dataloss_slots": 0,
            "dataloss_error_only": 0,
            "dataloss_absent": 0,
            "dataloss_spans": [],
        }

    # Snap explicit bounds to the reading grid (raw DAM timestamps are
    # minute-aligned; metadata bounds can carry fractional seconds).
    g0 = pd.Timestamp(grid_start).ceil("min") if grid_start is not None else all_idx[0]
    g1 = pd.Timestamp(grid_end).floor("min") if grid_end is not None else all_idx[-1]
    grid = pd.date_range(g0, g1, freq=interval)
    s1_set = set(s1_idx)
    present_set = set(all_idx)  # any de-dup'd row (status1 or error)
    err_set = set(err_idx)  # slots that had an error row (pre-dedup)

    n_status1 = 0
    n_cosmetic = 0  # status1 slot that ALSO had an error row (redundant)
    dataloss_error_only = 0  # slot had an error row but no status1
    dataloss_absent = 0  # slot had no row at all
    dataloss_flags = np.zeros(len(grid), dtype=bool)

    for k, t in enumerate(grid):
        has_s1 = t in s1_set
        has_any = t in present_set
        if has_s1:
            n_status1 += 1
            # cosmetic iff a redundant error row coexisted at this real slot
            if t in err_set:
                n_cosmetic += 1
        else:
            dataloss_flags[k] = True
            if has_any or (t in err_set):
                dataloss_error_only += 1
            else:
                dataloss_absent += 1

    n_dataloss = int(dataloss_flags.sum())

    # Contiguous data-loss spans for reporting.
    spans = []
    k = 0
    while k < len(grid):
        if dataloss_flags[k]:
            j = k
            while j < len(grid) and dataloss_flags[j]:
                j += 1
            spans.append(
                {
                    "start": pd.Timestamp(grid[k]),
                    "end": pd.Timestamp(grid[j - 1]),
                    "n_slots": int(j - k),
                }
            )
            k = j
        else:
            k += 1

    return {
        "n_expected_slots": int(len(grid)),
        "n_status1_slots": int(n_status1),
        "n_cosmetic_slots": int(n_cosmetic),
        "n_dataloss_slots": int(n_dataloss),
        "dataloss_error_only": int(dataloss_error_only),
        "dataloss_absent": int(dataloss_absent),
        "dataloss_spans": spans,
    }


def format_monitor_report(
    monitor_id, status_info, scan, classification, significant_gap_minutes=60.0, max_spans=8
):
    """Build a per-monitor human-readable report string.

    COSMETIC irregularities → quiet one-line summary. DATA-LOSS → prominent
    warning lines (the user must see genuine NaN holes). To avoid drowning the
    user, only spans at/above ``significant_gap_minutes`` are listed individually
    (up to ``max_spans``); shorter intermittent failed-read holes are summarized
    as an aggregate count. Returns a multi-line str.
    """
    lines = []
    n_cos = classification.get("n_cosmetic_slots", 0)
    n_dl = classification.get("n_dataloss_slots", 0)

    # Quiet cosmetic summary
    if status_info.get("n_rows_dropped", 0) or n_cos:
        codes = status_info.get("bad_status_counts", {})
        codes_str = ", ".join(f"{status_label(c)} x{n}" for c, n in sorted(codes.items()))
        lines.append(
            f"  monitor {monitor_id}: {status_info.get('n_rows_dropped', 0)} "
            f"redundant non-status-1 rows removed at {n_cos} slots (cosmetic, "
            f"0 data lost){' - ' + codes_str if codes_str else ''}."
        )

    # Visibility for status codes BEYOND the documented set {1, 24, 50, 51}: same
    # NaN handling, but report each with its count so frequency/pattern is trackable
    # over time. NOT an error — just a code whose meaning we have not catalogued.
    undoc = status_info.get("undocumented_status_counts", {})
    if undoc:
        undoc_str = " + ".join(f"{n} status-{c}" for c, n in sorted(undoc.items()))
        lines.append(
            f"  monitor {monitor_id}: {undoc_str} rows -> NaN "
            f"(status code(s) beyond the documented set; handled as no-data, "
            f"tracked for visibility)."
        )

    if not scan.get("monotonic", True):
        lines.append(f"  [!] monitor {monitor_id}: time axis is NOT monotonic.")
    if scan.get("n_duplicates", 0):
        lines.append(
            f"  [!] monitor {monitor_id}: {scan['n_duplicates']} residual "
            f"duplicate timestamps after de-dup (unexpected)."
        )
    if scan.get("interval_mismatch", False):
        lines.append(
            f"  [!] monitor {monitor_id}: reading interval mismatch - actual "
            f"{scan.get('actual_interval_minutes')} min vs expected "
            f"{scan.get('interval_minutes')} min."
        )

    # Prominent data-loss warnings
    if n_dl:
        interval = scan.get("interval_minutes", 1.0)
        lines.append(
            f"  [DATA LOSS] monitor {monitor_id}: {n_dl} grid slots have NO "
            f"valid (status-1) reading -> stored as NaN "
            f"(error-only: {classification.get('dataloss_error_only', 0)}, "
            f"absent: {classification.get('dataloss_absent', 0)})."
        )
        spans = classification.get("dataloss_spans", [])
        sig = [s for s in spans if s["n_slots"] * interval >= significant_gap_minutes]
        small = [s for s in spans if s["n_slots"] * interval < significant_gap_minutes]
        for span in sig[:max_spans]:
            dur_h = span["n_slots"] * interval / 60.0
            lines.append(
                f"        gap {span['start']} -> {span['end']}: "
                f"{span['n_slots']} readings missing (~{dur_h:.1f} h, NaN)."
            )
        if len(sig) > max_spans:
            lines.append(
                f"        ... and {len(sig) - max_spans} more gaps "
                f">= {significant_gap_minutes:.0f} min."
            )
        if small:
            n_small_slots = sum(s["n_slots"] for s in small)
            lines.append(
                f"        + {len(small)} short intermittent failed-read holes "
                f"({n_small_slots} readings, < {significant_gap_minutes:.0f} min each), NaN."
            )
    return "\n".join(lines)
