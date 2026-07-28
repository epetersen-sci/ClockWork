"""
scamp_writer.py
===============
Format primitives for writing SCAMP "luc"-format per-fly text files.

File layout (`dam_load/dam_file.m`):
    Line 1   : header (free text)
    Line 2   : len    (integer: number of data points)
    Line 3   : int    (integer: sampling interval in minutes)
    Line 4   : start  (integer: military time HHMM, e.g. 2330)
    Lines 5..: data   (one integer per line; negative = missing/error)

Filenames must match `<prefix>M<board>C<channel>` with no extension and no
extra `C` characters in `<prefix>` (SCAMP splits board IDs on `C`).
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

MISSING_SENTINEL = -1  # SCAMP treats any negative value as missing/error

FILENAME_PATTERN = re.compile(r"^[^.]*M\d+C\d+$")


def military_time(dt) -> int:
    """
    Return military-time integer HHMM from a datetime/Timestamp.

    Seconds are dropped (SCAMP samples at minute resolution).
    """
    ts = pd.Timestamp(dt)
    return int(ts.hour) * 100 + int(ts.minute)


def encode_activity(values, missing_sentinel: int = MISSING_SENTINEL) -> np.ndarray:
    """
    Round non-NaN counts to int; NaN → ``missing_sentinel`` (a negative int).

    SCAMP's `fscanf('%d')` requires integers and treats negatives as errors,
    which `dam_cleanup.m` then handles (interpolate interior, chop leading).
    """
    arr = np.asarray(values, dtype=float)
    out = np.empty(arr.shape, dtype=np.int64)
    valid = np.isfinite(arr)
    out[valid] = np.rint(arr[valid]).astype(np.int64)
    out[~valid] = int(missing_sentinel)
    return out


def bin_to_30min(activity_1min, missing_sentinel: int = MISSING_SENTINEL) -> np.ndarray:
    """
    Sum each contiguous 30-minute block of 1-minute counts.

    The input length must be a multiple of 30. If any of the 30 samples in a
    block are the missing sentinel (negative), the block is also written as
    the missing sentinel — SCAMP will then interpolate via `dam_cleanup.m`.

    Parameters
    ----------
    activity_1min : array-like of int
        1-minute integer counts (already encoded via :func:`encode_activity`).
    missing_sentinel : int
        Negative value used to mark missing data.

    Returns
    -------
    np.ndarray of int (length // 30,)
    """
    arr = np.asarray(activity_1min, dtype=np.int64)
    n = arr.size
    if n % 30 != 0:
        raise ValueError(
            f"bin_to_30min requires length divisible by 30; got {n}. "
            "Trim to whole days (multiples of 1440) before binning."
        )
    blocks = arr.reshape(-1, 30)
    any_missing = (blocks < 0).any(axis=1)
    sums = blocks.sum(axis=1)
    sums[any_missing] = missing_sentinel
    return sums.astype(np.int64)


def _validate_filename(name: str) -> None:
    """Raise ValueError if ``name`` is not a SCAMP-loadable filename."""
    if "." in name:
        raise ValueError(
            f"SCAMP filename must have no extension/dot: {name!r}. "
            "dam_names.m will otherwise rename the file in place."
        )
    if not FILENAME_PATTERN.match(name):
        raise ValueError(f"Filename {name!r} does not match <prefix>M<digits>C<digits>.")
    prefix = name.rsplit("M", 1)[0]
    bad = [c for c in prefix if c in ("C", "c", "m")]
    if bad:
        raise ValueError(
            f"Filename prefix {prefix!r} contains {sorted(set(bad))}. SCAMP's "
            "dam_names.m parses board/channel on M/C in BOTH cases with order-"
            "fragile logic, so a 'C'/'c'/'m' in the prefix makes it mis-parse and "
            "drop the file — keep the prefix free of those letters."
        )


def write_dam_file(
    path: str,
    header: str,
    values,
    interval_min: int,
    start_military: int,
) -> None:
    """
    Write one SCAMP-format per-fly text file.

    Parameters
    ----------
    path : str
        Output path. The basename must match ``<prefix>M\\d+C\\d+`` and have
        no dot.
    header : str
        Free-text header (single line; newlines are stripped).
    values : array-like of int
        Integer activity counts. Negative values mark missing/error.
    interval_min : int
        Sampling interval in minutes (e.g. 1 or 30).
    start_military : int
        Start clock time as HHMM (e.g. 900 for 09:00, 2330 for 23:30).
    """
    _validate_filename(os.path.basename(path))
    if interval_min <= 0:
        raise ValueError("interval_min must be a positive integer.")
    if not (0 <= int(start_military) <= 2359):
        raise ValueError(f"start_military must be HHMM in [0, 2359]; got {start_military}.")

    values = np.asarray(values, dtype=np.int64)
    header_clean = str(header).replace("\n", " ").replace("\r", " ").strip()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write(header_clean + "\n")
        fh.write(f"{int(values.size)}\n")
        fh.write(f"{int(interval_min)}\n")
        fh.write(f"{int(start_military)}\n")
        for v in values:
            fh.write(f"{int(v)}\n")


def read_dam_file(path: str):
    """
    Minimal Python re-implementation of `dam_load/dam_file.m` for tests.

    Returns
    -------
    dict with keys: header, len, int, start, values (np.ndarray of int).
    """
    with open(path, encoding="ascii") as fh:
        header = fh.readline().rstrip("\n").rstrip("\r")
        length = int(fh.readline().strip())
        interval = int(fh.readline().strip())
        start = int(fh.readline().strip())
        vals = np.fromstring(fh.read(), sep="\n", dtype=np.int64)
    return {
        "header": header,
        "len": length,
        "int": interval,
        "start": start,
        "values": vals,
    }


def write_board_folder(
    out_dir: str,
    board_files: dict,
    header_fmt: str,
    interval_min: int,
    start_military: int,
    length: int,
) -> list:
    """
    Write one board's per-fly files, asserting equal ``(start, int, len)``.

    Parameters
    ----------
    out_dir : str
        Directory to write into (created if needed).
    board_files : dict
        Mapping ``filename -> 1-D integer array of counts`` (already encoded).
        Filenames must validate via :func:`_validate_filename`.
    header_fmt : str
        Format string with ``{filename}`` (and optionally other fields) for the
        header line of each file.
    interval_min, start_military, length : int
        Must be identical across all files (SCAMP's dam_read_names.m enforces
        this and aborts the load otherwise).

    Returns
    -------
    list of str
        Paths written.
    """
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for filename, values in board_files.items():
        _validate_filename(filename)
        arr = np.asarray(values, dtype=np.int64)
        if arr.size != length:
            raise ValueError(
                f"{filename}: expected length {length}, got {arr.size}. "
                "SCAMP requires per-board identical len."
            )
        path = os.path.join(out_dir, filename)
        header = header_fmt.format(filename=filename) if "{filename}" in header_fmt else header_fmt
        write_dam_file(path, header, arr, interval_min, start_military)
        written.append(path)
    return written
