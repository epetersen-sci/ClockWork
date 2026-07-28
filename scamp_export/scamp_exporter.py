"""
scamp_exporter.py
=================
Orchestrates writing a Pythomics xarray Dataset (after curation + LD/DD split)
into a SCAMP-loadable folder tree.

Output layout produced by :func:`export_dataset_to_scamp`:

    <out_root>/
        1min/   <prefix>M<board>C<channel>            (per-fly files)
        30min/  <prefix>M<board>C<channel>            (re-binned to 30-min sums)
        scamp_group_key.csv                            (board/channel ↔ fly_id ↔ group)
        export_manifest.json                           (params + counts)

See README.md in this folder for SCAMP format details and the manual MATLAB
acceptance procedure.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
import xarray as xr

from . import scamp_writer

# --- ID parsing -------------------------------------------------------------

_FLY_ID_PATTERN = re.compile(r"^(\d{8})_(.+)_(\d+)$")


def _parse_fly_id(fly_id: str):
    """
    Parse Pythomics fly ID ``YYYYMMDD_Monitor_Region``.

    Returns
    -------
    (date_str, monitor_str, region_id) : tuple
    """
    m = _FLY_ID_PATTERN.match(str(fly_id))
    if not m:
        raise ValueError(f"Fly id {fly_id!r} does not match YYYYMMDD_Monitor_Region.")
    date_str, monitor_str, region_str = m.group(1), m.group(2), m.group(3)
    return date_str, monitor_str, int(region_str)


def _sanitize_prefix(prefix: str) -> str:
    """
    Strip characters that would confuse SCAMP's filename parsing.

    SCAMP splits board IDs on ``C`` (`scamp.m`), so the prefix must contain
    neither ``C`` nor ``.``. We also strip whitespace.
    """
    if prefix is None:
        return ""
    s = str(prefix).strip()
    # SCAMP's dam_names.m scans BOTH cases — findstr(f,'C')/findstr(f,'c') for the
    # channel split and findstr(f,'M')/findstr(f,'m') for the board split — and its
    # lastC/lastM selection is order-fragile (concatenated, not sorted). So a lowercase
    # 'c' or 'm' anywhere in the prefix makes SCAMP mis-parse and SILENTLY DROP the
    # file. Forbid both cases of C and lowercase m; uppercase 'M' is only warned (the
    # real board 'M' is the last uppercase M, so it still wins).
    bad = [c for c in s if c in ("C", "c", "m", ".", "/", "\\", " ")]
    if bad:
        raise ValueError(
            f"Prefix {prefix!r} contains forbidden character(s) {sorted(set(bad))}. "
            "Avoid 'C'/'c', 'm', '.', '/', '\\', and spaces (SCAMP's dam_names.m "
            "parses board/channel on M/C in both cases)."
        )
    if "M" in s:
        # Tolerated, but warn — SCAMP picks the *last* M as the board separator,
        # so an embedded M in the prefix is fine as long as digits follow only
        # at the true M<board> boundary.
        warnings.warn(
            f"Prefix {prefix!r} contains 'M'. SCAMP uses the last 'M' as the "
            "board separator; verify your prefix doesn't break board grouping.",
            stacklevel=2,
        )
    return s


# --- Per-fly window selection ----------------------------------------------


def _fly_valid_span(activity_1d: np.ndarray):
    """
    Return ``(first_idx, last_idx, span)`` for a fly's longest contiguous-from-front
    valid stretch.

    Pythomics' :func:`split_xarray_dataset` already trims each fly to its
    longest continuous segment, so for typical data this is simply
    (first_finite, last_finite, count).

    A fly with no finite values returns ``(0, -1, 0)``.
    """
    valid = np.isfinite(activity_1d)
    if not valid.any():
        return 0, -1, 0
    first = int(np.argmax(valid))  # first True
    last = int(len(valid) - 1 - np.argmax(valid[::-1]))
    return first, last, last - first + 1


def _interpolate_interior_nan(values: np.ndarray) -> np.ndarray:
    """
    Linear-interpolate any NaN inside ``values`` (after slicing to the valid
    window). Endpoints are assumed already finite — caller ensures this by
    using the per-fly first/last valid indices.
    """
    arr = np.asarray(values, dtype=float).copy()
    n = arr.size
    if n == 0:
        return arr
    nan_mask = ~np.isfinite(arr)
    if not nan_mask.any():
        return arr
    idx = np.arange(n)
    arr[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], arr[~nan_mask])
    return arr


# --- Board labeling --------------------------------------------------------


def _board_label_for_fly(date_str: str, monitor_str: str, prefix: str, needs_date: bool):
    """
    Return ``(file_prefix, board_int)`` for a given fly.

    The filename is ``{file_prefix}M{board_int}C{channel}``. When monitor is a
    plain integer, ``board_int`` is that integer; otherwise a synthetic int is
    assigned in caller code via ``_assign_board_ints``.
    """
    file_prefix = f"{prefix}{date_str}" if needs_date else prefix
    return file_prefix, monitor_str


def _assign_board_ints(monitor_strs):
    """
    Map each distinct monitor string to a positive integer board ID.

    Numeric monitor strings pass through (``"42" -> 42``); non-numeric strings
    are assigned sequentially starting at 1001 to avoid colliding with numeric
    monitors.
    """
    mapping = OrderedDict()
    next_synth = 1001
    for m in monitor_strs:
        if m in mapping:
            continue
        if m.isdigit() and int(m) > 0:
            mapping[m] = int(m)
        else:
            mapping[m] = next_synth
            next_synth += 1
    return mapping


# --- Public API ------------------------------------------------------------


def export_dataset_to_scamp(
    ds: xr.Dataset,
    out_root: str,
    *,
    prefix: str = "PY",
    interval_set=(1, 30),
    lights_on_military: int = 900,
    min_days: float = 2.0,
    missing_sentinel: int = scamp_writer.MISSING_SENTINEL,
    interpolate_interior: bool = True,
    phase_label: str = "",
) -> dict:
    """
    Export one phase-split, curated Pythomics Dataset to a SCAMP folder tree.

    Per-board window selection (the equal-length constraint SCAMP enforces):

    1. Drop flies whose valid span ``< min_days * 1440`` minutes.
    2. ``window = min(valid_span)`` over the retained flies.
    3. Round **down** to whole days: ``window = floor(window / 1440) * 1440``.
       Files end up with len divisible by 30 and 1440 — clean for both 30-min
       binning and SCAMP day buttons.
    4. Slice each retained fly from its first valid index for ``window`` rows.

    Parameters
    ----------
    ds : xr.Dataset
        Curated + phase-split dataset (relative-minute time axis). Must have
        ``activity`` data variable and per-fly ``start_datetime`` coord.
    out_root : str
        Directory to write into (created if needed). ``1min/`` and ``30min/``
        subfolders are written underneath.
    prefix : str
        Filename prefix (no 'C', no '.'). When multiple recording dates are
        present, the per-fly date is appended to the prefix so each
        ``(date, monitor)`` pair lands on a distinct SCAMP "board".
    interval_set : iterable of int
        Sampling intervals (minutes) to write. ``(1, 30)`` matches what SCAMP
        prompts for.
    lights_on_military : int
        Single ``start`` (HHMM) written into every file's line-4. SCAMP uses
        this only for axis labelling; data are already ZT-aligned per fly via
        Pythomics' relative-time convention.
    min_days : float
        Minimum valid-span (days) required to keep a fly.
    missing_sentinel : int
        Negative integer written for missing samples (default -1).
    interpolate_interior : bool
        If True, linearly interpolate interior NaN inside each fly's window
        before writing. The split + longest-segment step has already guaranteed
        any interior NaN runs are below the configured gap threshold.
    phase_label : str
        Free-text label (e.g. ``"LD"`` or ``"DD"``) added to each file header
        and recorded in the manifest.

    Returns
    -------
    dict : summary with keys ``n_in``, ``n_out``, ``dropped``, ``window_days``,
    ``boards``, ``out_root``.
    """
    prefix = _sanitize_prefix(prefix)
    if "activity" not in ds.data_vars:
        raise ValueError("Dataset has no 'activity' variable.")
    if "id" not in ds.coords:
        raise ValueError("Dataset has no 'id' coord.")

    fly_ids = [str(x) for x in ds["id"].values]
    n_in = len(fly_ids)
    if n_in == 0:
        raise ValueError("Dataset has no flies.")

    # Parse Monitor/region_id from each fly_id
    parsed = [_parse_fly_id(fid) for fid in fly_ids]
    date_strs = [p[0] for p in parsed]
    monitor_strs = [p[1] for p in parsed]
    # TODO(confirm): _parse_fly_id also returns region_id (p[2]), but it is not
    # written to the SCAMP output here — verify the export isn't missing a region
    # column before relying on this.
    needs_date_in_prefix = len(set(date_strs)) > 1
    board_int_map = _assign_board_ints(monitor_strs)

    # Per-fly activity arrays on the shared time axis
    activity_2d = ds["activity"].transpose("time", "id").values.astype(float)
    n_time, n_flies = activity_2d.shape
    assert n_flies == n_in

    # --- 1. Per-fly span -> per-board grouping ----------------------------
    fly_records = []  # one per surviving fly: dict with slice info
    dropped = []  # (fly_id, reason)
    min_minutes = int(min_days * 1440)

    for i, fid in enumerate(fly_ids):
        first, last, span = _fly_valid_span(activity_2d[:, i])
        if span < min_minutes:
            dropped.append({"id": fid, "reason": f"valid_span={span} min < min_days={min_days}"})
            continue
        date_str, monitor_str, region = parsed[i]
        file_prefix = f"{prefix}{date_str}" if needs_date_in_prefix else prefix
        if not file_prefix:
            # SCAMP requires at least one non-C/non-digit char before M to anchor parsing
            file_prefix = "PY"
        board_int = board_int_map[monitor_str]
        fly_records.append(
            {
                "fly_id": fid,
                "date_str": date_str,
                "monitor": monitor_str,
                "region": region,
                "first_idx": first,
                "last_idx": last,
                "span": span,
                "file_prefix": file_prefix,
                "board_int": board_int,
                "_col_idx": i,
            }
        )

    if not fly_records:
        raise ValueError(f"All {n_in} flies dropped (none had ≥ {min_days} days valid data).")

    # Group by (file_prefix, board_int) — that's the SCAMP "board"
    boards = defaultdict(list)
    for rec in fly_records:
        boards[(rec["file_prefix"], rec["board_int"])].append(rec)

    # --- 2. Per-board common window ---------------------------------------
    board_summaries = []
    for (file_prefix, board_int), recs in boards.items():
        # Determine the common length: minimum span across retained flies,
        # floored to whole days.
        common_span = min(r["span"] for r in recs)
        window_days = common_span // 1440
        if window_days < 1:
            # Should not happen given min_days >= 1, but guard.
            for r in recs:
                dropped.append(
                    {
                        "id": r["fly_id"],
                        "reason": f"board {file_prefix}M{board_int} common window < 1 day",
                    }
                )
            continue
        window = int(window_days * 1440)
        for r in recs:
            r["slice_end"] = r["first_idx"] + window
        board_summaries.append(
            {
                "file_prefix": file_prefix,
                "board_int": board_int,
                "window_minutes": window,
                "window_days": int(window_days),
                "n_flies": len(recs),
            }
        )

    if not board_summaries:
        raise ValueError("No boards survived the window-selection step.")

    # --- 3. Write per-interval folders ------------------------------------
    os.makedirs(out_root, exist_ok=True)
    written_summary = {}
    group_key_rows = []

    for interval in interval_set:
        if int(interval) not in (1, 30):
            raise ValueError(f"interval_set may contain only 1 and 30; got {interval}.")
    has_1 = 1 in [int(x) for x in interval_set]
    has_30 = 30 in [int(x) for x in interval_set]

    dir_1 = os.path.join(out_root, "1min")
    dir_30 = os.path.join(out_root, "30min")
    if has_1:
        os.makedirs(dir_1, exist_ok=True)
    if has_30:
        os.makedirs(dir_30, exist_ok=True)

    n_written = 0
    for board in board_summaries:
        file_prefix = board["file_prefix"]
        board_int = board["board_int"]
        window = board["window_minutes"]
        recs = boards[(file_prefix, board_int)]

        for r in recs:
            i = r["_col_idx"]
            fid = r["fly_id"]
            region = r["region"]
            fly_slice = activity_2d[r["first_idx"] : r["slice_end"], i]
            if interpolate_interior:
                fly_slice = _interpolate_interior_nan(fly_slice)
            encoded = scamp_writer.encode_activity(fly_slice, missing_sentinel)

            filename = f"{file_prefix}M{board_int}C{region}"
            header = (
                f"Pythomics export | phase={phase_label} | id={fid} | "
                f"monitor={r['monitor']} | region={region} | date={r['date_str']}"
            )

            if has_1:
                scamp_writer.write_dam_file(
                    os.path.join(dir_1, filename),
                    header,
                    encoded,
                    interval_min=1,
                    start_military=lights_on_military,
                )
            if has_30:
                binned = scamp_writer.bin_to_30min(encoded, missing_sentinel)
                scamp_writer.write_dam_file(
                    os.path.join(dir_30, filename),
                    header,
                    binned,
                    interval_min=30,
                    start_military=lights_on_military,
                )
            n_written += 1

            # Group-key row
            row = {
                "filename": filename,
                "board_label": f"{file_prefix}M{board_int}",
                "board_int": board_int,
                "channel": region,
                "fly_id": fid,
                "monitor": r["monitor"],
                "region_id": region,
                "date": r["date_str"],
                "window_days": board["window_days"],
            }
            # Best-effort group/genotype lookup from coords
            for coord in ("group", "genotype", "temperature"):
                if coord in ds.coords:
                    try:
                        row[coord] = str(ds[coord].sel(id=fid).values)
                    except Exception:
                        row[coord] = ""
            group_key_rows.append(row)

    # --- 4. Group-key CSV --------------------------------------------------
    key_path = os.path.join(out_root, "scamp_group_key.csv")
    pd.DataFrame(group_key_rows).to_csv(key_path, index=False)

    # --- 5. Manifest -------------------------------------------------------
    manifest = {
        "phase_label": phase_label,
        "prefix": prefix,
        "needs_date_in_prefix": needs_date_in_prefix,
        "lights_on_military": lights_on_military,
        "min_days": min_days,
        "missing_sentinel": missing_sentinel,
        "interpolate_interior": interpolate_interior,
        "interval_set": [int(x) for x in interval_set],
        "n_input_flies": n_in,
        "n_exported_flies": n_written,
        "n_dropped": len(dropped),
        "dropped": dropped,
        "boards": board_summaries,
        "dataset_attrs": {k: str(v) for k, v in ds.attrs.items()},
    }
    manifest_path = os.path.join(out_root, "export_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    written_summary = {
        "n_in": n_in,
        "n_out": n_written,
        "dropped": dropped,
        "boards": board_summaries,
        "out_root": out_root,
        "group_key": key_path,
        "manifest": manifest_path,
    }
    return written_summary
