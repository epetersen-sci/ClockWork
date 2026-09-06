"""
dam_utilities.py
================
Core data transformation utilities for the Pythomics pipeline.

Handles the workflow from raw DAM DataFrames through to analysis-ready
xarray Datasets, and provides binning utilities used throughout the app.

Pipeline order:
  1. read_data_and_metadata()       — load raw DAM data via MetadataProcessor
  2. convert_to_relative_time()     — convert DatetimeIndex to integer minutes
  3. create_xarray_dataset()        — build the xarray Dataset with coordinates
  4. curate_dead_animals()          — identify and remove dead flies (computes 'moving' internally)
  5. split_xarray_dataset()         — separate LD / DD phases (after curation)

Legacy helpers (kept for compatibility):
  - split_activity_dataframe()     — separate LD/DD on raw pandas DataFrames
  - trim_first_dd_day()            — discard first DD calendar day (pandas)

Downstream helpers:
  - get_zt_binned_dataframe()       — ZT-bin data per fly for plotting/export

Time representation note:
  The dataset can store time as either absolute datetime64 or relative integer
  minutes (starting from 0).  The attribute 'time_is_relative_minutes' (1/0)
  flags which is active.  All functions here check this attribute and handle
  both cases.
"""

from collections import Counter

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

import dam_processor


def _as_numpy_array(values) -> np.ndarray:
    """Materialize values as a plain ``np.ndarray``.

    pandas 2.x with pyarrow can expose string columns as ``ArrowStringArray``
    (via ``.values`` / ExtensionArray). xarray cannot index those backends
    (``TypeError: Invalid array type`` on ``.sel`` / ``.isel``). Always convert
    pandas objects through ``to_numpy()`` so Dataset storage stays numpy-backed.
    """
    if isinstance(values, np.ndarray):
        return values
    if isinstance(values, (pd.Series, pd.Index)):
        return np.asarray(values.to_numpy())
    to_numpy = getattr(values, "to_numpy", None)
    if callable(to_numpy):
        return np.asarray(to_numpy())
    return np.asarray(values)


def _needs_numpy_materialization(data) -> bool:
    """True when xarray's indexer cannot wrap ``data`` (e.g. raw ArrowStringArray)."""
    if isinstance(data, np.ndarray):
        return False
    module = type(data).__module__ or ""
    if module.startswith(("dask.", "sparse", "cupy.", "jax.", "pint.")):
        return False
    try:
        from xarray.core.indexing import as_indexable

        as_indexable(data)
        return False
    except TypeError:
        return True
    except Exception:
        # Unknown backend — leave alone rather than force-convert.
        return False


def ensure_numpy_backed(ds: xr.Dataset) -> xr.Dataset:
    """Replace storage that xarray cannot index with plain numpy arrays.

    Fixes ``TypeError: Invalid array type: ArrowStringArray`` (and similar
    pandas ExtensionArrays) on ``.sel`` / ``.isel``. Idempotent: values that
    xarray can already index are left unchanged.
    """
    coord_updates = {}
    for name, da in ds.coords.items():
        if not _needs_numpy_materialization(da.variable._data):
            continue
        coord_updates[name] = (da.dims, _as_numpy_array(da.values))
    if coord_updates:
        ds = ds.assign_coords(coord_updates)

    var_updates = {}
    for name, da in ds.data_vars.items():
        if not _needs_numpy_materialization(da.variable._data):
            continue
        var_updates[name] = xr.Variable(da.dims, _as_numpy_array(da.values), attrs=da.attrs)
    if var_updates:
        ds = ds.assign(var_updates)
    return ds


def resolve_export_dir(ds=None, working_dir=None):
    """Return the canonical base directory for on-disk exports.

    Every export (NetCDF save, SCAMP export, group-averaged scalograms, …)
    should default to the user's WORKING FOLDER — the directory that holds the
    metadata file / the loaded ``.nc`` — so results land next to their
    experiment files, not in whatever directory Streamlit happened to be
    launched from (the old ``os.getcwd()`` fallback dumped a stray
    ``Averaged Scalograms/`` under ``app/``).

    Precedence (first that names an existing directory wins):
      1. ``ds.attrs['source_data_dir']`` — the working folder stamped at load
         (the metadata file's directory on a raw load). A plain string, so it
         survives the NetCDF round-trip: a reloaded ``.nc`` still exports to
         that folder when it exists.
      2. ``working_dir`` — the current session working directory (the metadata
         directory on a raw load, or the ``.nc``'s directory on a reload).

    The returned path is always ABSOLUTE, so a relative candidate can never be
    silently re-interpreted against the app's launch directory. If neither
    candidate names an existing directory, returns ``working_dir`` (absolute)
    when set, else the current working directory — so an export never silently
    vanishes.
    """
    import os

    _attrs = getattr(ds, "attrs", None) if ds is not None else None
    if _attrs:
        _sd = _attrs.get("source_data_dir")
        if isinstance(_sd, str) and _sd and os.path.isdir(_sd):
            return os.path.abspath(_sd)
    if working_dir and os.path.isdir(working_dir):
        return os.path.abspath(working_dir)
    return os.path.abspath(working_dir) if working_dir else os.getcwd()


def read_data_and_metadata(metadata_path, data_folder):
    """
    Convenience wrapper: load raw DAM data and metadata in one call.

    Parameters
    ----------
    metadata_path : str
        Path to CSV or Excel metadata file.
    data_folder : str
        Directory containing MonitorXXX.txt files.

    Returns
    -------
    metadata : pd.DataFrame
    all_data : pd.DataFrame  (DatetimeIndex, columns = fly IDs)
    """
    dp = dam_processor.MetadataProcessor(metadata_path, data_folder)
    metadata, all_data = dp.run()
    return metadata, all_data


def split_activity_dataframe(
    df_activity: pd.DataFrame, df_metadata: pd.DataFrame, progress_callback=None
) -> tuple:
    """
    Split activity data into LD and DD phases using the 'first_DD_day' in metadata.

    For each fly column the data is split at its individual first_DD_day so that
    experiments with different dark-cycle start dates are handled correctly.

    Parameters
    ----------
    df_activity : pd.DataFrame
        Activity data with DatetimeIndex; columns are fly IDs.
    df_metadata : pd.DataFrame
        Metadata containing 'first_DD_day' and 'id' columns.
    progress_callback : callable, optional
        Called as callback(completed, total) for UI progress bars.

    Returns
    -------
    ld_data : pd.DataFrame  — data from start up to (but not including) first_DD_day
    dd_data : pd.DataFrame  — data from first_DD_day to stop
    """
    if not isinstance(df_activity.index, pd.DatetimeIndex):
        try:
            df_activity.index = pd.to_datetime(df_activity.index)
        except Exception as e:
            raise ValueError(
                f"\nWARNING: Activity data must have datetime indices. Error: {e}"
            ) from e
    if "first_DD_day" not in df_metadata.columns or "id" not in df_metadata.columns:
        raise ValueError("df_metadata must contain 'first_DD_day' and 'id' columns.")

    df_metadata["first_DD_day"] = pd.to_datetime(df_metadata["first_DD_day"])
    id_metadata = df_metadata.set_index("id")

    ld_data = pd.DataFrame()
    dd_data = pd.DataFrame()

    _split_total = len(df_activity.columns)
    for _split_idx, fly_id in enumerate(
        tqdm(df_activity.columns, desc="Splitting Activity Data: ")
    ):
        current_column_data = df_activity[fly_id]

        try:
            start_time = id_metadata.loc[fly_id, "start_datetime"]
            stop_time = id_metadata.loc[fly_id, "stop_datetime"]
            DD_day = id_metadata.loc[fly_id, "first_DD_day"]
        except Exception as e:
            raise ValueError(f"\nWARNING: Time value not found for fly {fly_id}. Error: {e}") from e

        # LD phase: start up to (but not including) first dark day
        fly_ld = current_column_data[
            (current_column_data.index >= start_time) & (current_column_data.index < DD_day)
        ]

        # DD phase: first dark day to stop
        fly_dd = current_column_data[
            (current_column_data.index >= DD_day) & (current_column_data.index < stop_time)
        ]

        # Merge — pandas aligns by index and fills missing time points with NaN
        ld_data = ld_data.merge(fly_ld, left_index=True, right_index=True, how="outer")
        dd_data = dd_data.merge(fly_dd, left_index=True, right_index=True, how="outer")

        if progress_callback:
            progress_callback(_split_idx + 1, _split_total)

    # Drop rows where every column is NaN
    ld_data = ld_data.dropna(how="all")
    dd_data = dd_data.dropna(how="all")

    return ld_data, dd_data


def trim_first_dd_day(dd_data: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Remove the first calendar day of DD from each fly's data.

    The first day of constant darkness often carries lingering LD entrainment
    effects and is routinely discarded before circadian analysis.  For each
    fly, all timepoints on the calendar date of its ``first_DD_day`` are
    removed (i.e. up to but not including midnight of the following day).

    Parameters
    ----------
    dd_data : pd.DataFrame
        Activity data with DatetimeIndex; columns are fly IDs.
    metadata : pd.DataFrame
        Must contain 'id' and 'first_DD_day' columns.

    Returns
    -------
    pd.DataFrame
        Same structure with first-DD-day rows set to NaN per fly.
    """
    if not isinstance(dd_data.index, pd.DatetimeIndex):
        raise ValueError("dd_data must have a DatetimeIndex.")

    metadata = metadata.copy()
    metadata["first_DD_day"] = pd.to_datetime(metadata["first_DD_day"])
    id_meta = metadata.set_index("id")

    trimmed = dd_data.copy()
    for fly_id in trimmed.columns:
        if fly_id in id_meta.index:
            dd_start = id_meta.loc[fly_id, "first_DD_day"]
            dd_start = pd.Timestamp(dd_start)
            next_day = dd_start.normalize() + pd.Timedelta(days=1)
            mask = (trimmed.index >= dd_start) & (trimmed.index < next_day)
            trimmed.loc[mask, fly_id] = np.nan

    # Drop rows that are now entirely NaN
    trimmed = trimmed.dropna(how="all")
    return trimmed


def convert_to_relative_time(dam_data: pd.DataFrame, metadata: pd.DataFrame = None) -> pd.DataFrame:
    """
    Convert a DatetimeIndex DataFrame to a relative integer-minute index,
    aligning each fly to ITS OWN start_datetime so that every fly's first
    recorded minute lands at integer index 0.

    This avoids leading-NaN padding for flies whose recording started later
    than the earliest fly, and makes datasets from runs months apart directly
    comparable / combinable on a shared time axis.

    Each fly's column is sliced from its `start_datetime` to `stop_datetime`,
    reindexed to a contiguous 1-minute grid (NaN where samples are missing),
    then stacked into a (max_length, n_flies) matrix.  Shorter recordings are
    padded with NaN at the TAIL only.

    Parameters
    ----------
    dam_data : pd.DataFrame
        Activity data with DatetimeIndex.
    metadata : pd.DataFrame
        Must contain 'id', 'start_datetime', and 'stop_datetime' columns.
        Required — global-min alignment is no longer supported because it
        produces front-padding artifacts for late-starting flies.

    Returns
    -------
    pd.DataFrame
        Integer-indexed DataFrame [0, 1, ..., max_length-1]; columns are
        fly IDs in the same order as in `dam_data`.
    """
    if not isinstance(dam_data.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex to convert to relative time.")
    if metadata is None:
        raise ValueError(
            "convert_to_relative_time now requires a metadata DataFrame "
            "(with 'id', 'start_datetime', 'stop_datetime') to align each fly "
            "to its own start time."
        )

    meta = metadata.copy()
    meta["start_datetime"] = pd.to_datetime(meta["start_datetime"])
    meta["stop_datetime"] = pd.to_datetime(meta["stop_datetime"])
    id_meta = meta.set_index("id")

    per_fly_arrays = {}
    max_len = 0
    for fly_id in dam_data.columns:
        if fly_id not in id_meta.index:
            raise KeyError(f"Fly id {fly_id!r} missing from metadata.")

        # DAM monitor timestamps are minute-aligned (HH:MM:00), while metadata can
        # carry fractional-second values from spreadsheets (e.g. HH:MM:59.995).
        # Reindexing against an unaligned per-fly grid can turn valid series into
        # all-NaN columns. Snap bounds to minute resolution before building the grid.
        start_raw = pd.Timestamp(id_meta.loc[fly_id, "start_datetime"])
        stop_raw = pd.Timestamp(id_meta.loc[fly_id, "stop_datetime"])
        start = start_raw.ceil("min")
        stop = stop_raw.floor("min")
        if stop < start:
            raise ValueError(
                f"Invalid time window for fly {fly_id!r}: "
                f"start={start_raw}, stop={stop_raw} after minute alignment."
            )
        fly_grid = pd.date_range(start=start, end=stop, freq="1min")
        col = dam_data[fly_id].reindex(fly_grid)
        per_fly_arrays[fly_id] = col.values
        if len(col) > max_len:
            max_len = len(col)

    aligned = {}
    for fly_id, vals in per_fly_arrays.items():
        if len(vals) < max_len:
            padded = np.full(max_len, np.nan, dtype=float)
            padded[: len(vals)] = vals
            aligned[fly_id] = padded
        else:
            aligned[fly_id] = vals

    out = pd.DataFrame(
        aligned, index=pd.RangeIndex(max_len, name="time"), columns=list(dam_data.columns)
    )
    return out


# =============================================================================
# Group definition — which metadata columns define the comparison "group".
# Historically the group was hard-wired to genotype[-temperature]. These helpers
# let the group be defined by ANY chosen metadata columns (e.g. add sex, or use
# genotype alone). Every candidate column is also stored as its own per-id coord,
# so the group can be re-derived on an already-built dataset without re-ingesting.
# =============================================================================

# Columns that can NEVER define a group: per-fly ids / filenames / monitor+region
# numbers, and the experiment-timing columns. Datetime-typed columns are excluded
# additionally by dtype in group_defining_columns (robust to future column names).
GROUP_EXCLUDE_COLUMNS = (
    "file",
    "region_id",
    "Monitor",
    "id",
    "start_datetime",
    "stop_datetime",
    "first_DD_day",
)


def _parse_zt_hour(cell):
    """Parse a metadata ``pulse_time`` cell into a ZT hour, or NaN if blank.

    Accepts the lab's own shorthand and the bare number equally: ``"ZT15"``,
    ``"zt15"``, ``"ZT 15"``, ``"15"``, ``15``, ``"15.5"``. A blank cell means the
    cohort received no pulse and yields NaN — never 0, which is a real ZT.
    """
    if cell is None:
        return float("nan")
    if isinstance(cell, (int, float)) and not isinstance(cell, bool):
        return float("nan") if pd.isna(cell) else float(cell)
    text = str(cell).strip()
    if text == "" or text.lower() in ("nan", "none", "na"):
        return float("nan")
    if text.upper().startswith("ZT"):
        text = text[2:].strip()
    try:
        return float(text)
    except ValueError:
        raise ValueError(
            f"Unparseable pulse_time cell {cell!r}. Use a ZT hour like 'ZT15' or '15', "
            f"or leave it blank for a cohort that received no light pulse."
        ) from None


def group_defining_columns(metadata: pd.DataFrame):
    """Return the metadata columns eligible to define a comparison group: every
    column except the ids/filenames/monitor numbers/timing columns
    (GROUP_EXCLUDE_COLUMNS) and any datetime-typed column (caught by dtype, so a
    date column is excluded regardless of its name). Order follows the metadata."""
    import pandas.api.types as pdt

    out = []
    for col in metadata.columns:
        if col in GROUP_EXCLUDE_COLUMNS:
            continue
        if pdt.is_datetime64_any_dtype(metadata[col]):
            continue
        out.append(col)
    return out


def derive_group_labels(metadata: pd.DataFrame, group_columns):
    """Per-id group label = the chosen metadata columns joined by '-', as a Series
    indexed by id. With ``group_columns == ['genotype', 'temperature']`` this
    reproduces the historical label EXACTLY (byte-identical). Missing/NaN cells
    become the literal 'nan' in the joined label (kept explicit rather than dropped,
    so a partially-annotated fly stays visibly distinct). Returns None if none of the
    requested columns are present."""
    meta_idx = metadata.set_index("id") if "id" in metadata.columns else metadata
    cols = [c for c in group_columns if c in meta_idx.columns]
    if not cols:
        return None
    label = meta_idx[cols[0]].astype(str)
    for c in cols[1:]:
        label = label + "-" + meta_idx[c].astype(str)
    return label


def get_group_columns(ds):
    """Return the list of metadata columns currently defining ``ds['group']`` (read
    from ``ds.attrs['group_columns']``), robust to netCDF's single-element-list →
    scalar round-trip. Empty list if the dataset predates group-column tracking."""
    val = ds.attrs.get("group_columns")
    if val is None:
        return []
    return [str(v) for v in np.atleast_1d(val)]


def group_defining_coords(ds):
    """The :func:`group_defining_columns` question, asked of a BUILT dataset.

    Needed because re-grouping happens long after import, when the metadata frame
    is gone and only the dataset remains — a reloaded ``.nc`` never had one.

    A candidate is a per-``id`` coord that ``create_xarray_dataset`` stored FROM
    THE METADATA. That provenance is recoverable because that function also
    writes one attr per metadata property column, so "is a per-id coord AND an
    attr key" identifies exactly those, and excludes both the ids
    (``id``/``group``, which are outputs rather than inputs) and every coord a
    later analysis added (``split_minute``, ``pulse_minute``, the rhythmic
    flags), none of which is an attr.

    The same two filters as the metadata version then apply:
    :data:`GROUP_EXCLUDE_COLUMNS` and datetime dtype, so the timing columns do
    not offer themselves as grouping factors.
    """
    out = []
    for name, coord in ds.coords.items():
        col = str(name)
        if coord.dims != ("id",):
            continue
        if col in GROUP_EXCLUDE_COLUMNS or col == "group":
            continue
        if col not in ds.attrs:
            continue
        if np.issubdtype(coord.dtype, np.datetime64):
            continue
        out.append(col)
    return out


def regroup_dataset(ds, group_columns):
    """Re-derive the ``group`` coord on an ALREADY-built dataset from the chosen
    metadata columns (each is stored as a per-id coord). Returns a new dataset with
    the updated ``group`` coord and ``attrs['group_columns']`` — used to re-group a
    reloaded ``.nc`` without re-ingesting the raw monitor files. Columns not present
    as coords are skipped; if none remain, the dataset is returned unchanged."""
    chosen = [c for c in group_columns if c in ds.coords]
    if not chosen:
        return ds
    parts = [_as_numpy_array(ds[c].values).astype(str) for c in chosen]
    label = parts[0]
    for p in parts[1:]:
        label = np.char.add(np.char.add(label, "-"), p)
    out = ds.assign_coords(group=("id", _as_numpy_array(label)))
    out.attrs = dict(ds.attrs)
    out.attrs["group_columns"] = list(chosen)
    return out


def create_xarray_dataset(dam_data: pd.DataFrame, metadata: pd.DataFrame, group_columns=None):
    """
    Build an xarray Dataset from activity data and metadata.

    Preserves NaN values for missing time points (NaN = no measurement,
    0 = no activity).  Detects and regularizes time gaps by reindexing
    to a uniform 1-minute grid.

    Supports both relative integer-minute time (after convert_to_relative_time)
    and absolute datetime time.  The dataset attribute 'time_is_relative_minutes'
    records which mode is used.

    Parameters
    ----------
    dam_data : pd.DataFrame
        Activity data; index is either DatetimeIndex or integer minutes.
    metadata : pd.DataFrame
        Must include: id, start_datetime, stop_datetime, file, genotype,
        temperature, first_DD_day. Row ORDER does not matter — the rows are
        matched to ``dam_data``'s columns by ``id`` before any coord is built
        (see the alignment block below); passing them in a different order than
        the activity columns used to mislabel flies silently.

    Returns
    -------
    xr.Dataset or None
        None if the time dimension is empty after processing.
    """
    # ---------------------------------------------------------------------
    # ALIGN metadata TO the activity columns before anything reads either.
    #
    # Every per-fly coord below (id, genotype, start/stop, group, ...) is taken
    # from `metadata` in ROW order, while the activity matrix is taken from
    # `dam_data` in COLUMN order. Those two orders are not the same thing, and
    # when they diverged this function silently attached one fly's metadata to
    # another fly's trace — no error, because the shapes still matched.
    #
    # They diverge routinely: dam_processor builds the activity frame by
    # iterating `unique_combos.sort_values(["Monitor", "start_datetime"])` with
    # Monitor cast to str, so monitor "10" sorts before "2", while the metadata
    # keeps the spreadsheet's (numeric, natural) row order. Any experiment
    # mixing single- and double-digit monitor numbers was affected.
    #
    # Reindexing metadata onto dam_data.columns makes the pairing explicit and
    # order-independent: from here on, row i of metadata IS column i of the
    # activity matrix, by fly id rather than by luck.
    # ---------------------------------------------------------------------
    if "id" not in metadata.columns:
        raise ValueError(
            "metadata must carry an 'id' column to be aligned with the activity "
            "columns. It is built by MetadataProcessor.expand_metadata()."
        )
    # Plain Python strings on both sides: metadata['id'] can arrive as a pandas
    # Arrow-backed string column, which set_index rejects outright (and which
    # _as_numpy_array exists to defuse elsewhere in this module).
    data_ids = [str(c) for c in dam_data.columns]
    meta_ids = [str(v) for v in metadata["id"].tolist()]

    _counts = Counter(meta_ids)
    dup_ids = [i for i, n in _counts.items() if n > 1]
    if dup_ids:
        # With duplicates the reindex below would multiply rows instead of
        # selecting them, so refuse rather than guess which row owns the tube.
        raise ValueError(
            f"metadata contains {len(dup_ids)} duplicated fly id(s), so activity "
            f"columns cannot be matched to metadata rows unambiguously: "
            f"{', '.join(dup_ids[:10])}{' ...' if len(dup_ids) > 10 else ''}. "
            f"Overlapping region_id ranges on the same Monitor+start_datetime are "
            f"the usual cause."
        )

    _meta_id_set = set(meta_ids)
    missing_meta = [i for i in data_ids if i not in _meta_id_set]
    if missing_meta:
        raise ValueError(
            f"{len(missing_meta)} activity column(s) have no metadata row: "
            f"{', '.join(missing_meta[:10])}"
            f"{' ...' if len(missing_meta) > 10 else ''}."
        )

    _data_id_set = set(data_ids)
    extra_meta = [i for i in meta_ids if i not in _data_id_set]
    if extra_meta:
        # No activity column exists for these, so they cannot become flies. Loud,
        # because the flies the user asked for are not all here.
        print(
            f"\nWARNING: {len(extra_meta)} metadata row(s) have no activity data and "
            f"are dropped from the dataset: {', '.join(extra_meta[:10])}"
            f"{' ...' if len(extra_meta) > 10 else ''}."
        )

    metadata = metadata.set_index(pd.Index(meta_ids)).loc[data_ids].reset_index(drop=True)

    # The light-pulse columns are excluded here (and attached as explicit coords
    # below) because a blank cell — an unpulsed control cohort — makes their unique
    # list a mix of strings/numbers and NaN, which NetCDF cannot serialize as an
    # attribute. The per-fly coordinate is the durable record either way.
    exclude_columns = [
        "file",
        "region_id",
        "Monitor",
        "id",
        "pulse_time",
        "pulse_duration_min",
    ]
    properties = [col for col in metadata.columns.tolist() if col not in exclude_columns]
    attrs = {col: metadata[col].unique().tolist() for col in properties}

    # Detect time representation
    is_relative_time = not isinstance(dam_data.index, pd.DatetimeIndex)

    if is_relative_time:
        print("\n--- Using relative integer-minute time coordinates ---")
        time_values = dam_data.index.values
        expected_index = np.arange(time_values[0], time_values[-1] + 1)

        if len(expected_index) != len(time_values):
            print(f"Reindexing to fill {len(expected_index) - len(time_values)} gaps with NaN")
            dam_data = dam_data.reindex(index=expected_index)
            dam_data.index.name = "time"
            attrs["time_regularized"] = 1
            attrs["gaps_filled"] = len(expected_index) - len(time_values)
            attrs["gap_fill_value"] = "NaN"
        else:
            attrs["time_regularized"] = 0
            attrs["gaps_filled"] = 0
        attrs["time_is_relative_minutes"] = 1

    else:
        # Absolute datetime path — detect and fill gaps
        print("\n--- Checking for time gaps ---")
        datetime_original = dam_data.index
        time_diffs = pd.Series(datetime_original).diff().dropna()
        expected_diff = pd.Timedelta(minutes=1)
        gaps = time_diffs[time_diffs > expected_diff * 1.5]

        if len(gaps) > 0:
            total_gap_minutes = gaps.sum().total_seconds() / 60
            missing_percent = (
                total_gap_minutes
                / ((datetime_original[-1] - datetime_original[0]).total_seconds() / 60)
            ) * 100
            print(f"Detected {len(gaps)} gaps ({missing_percent:.2f}% missing)")

        first_time = datetime_original[0]
        last_time = datetime_original[-1]
        time_index = pd.date_range(start=first_time, end=last_time, freq="1min")

        if len(time_index) != len(datetime_original):
            print(
                f"Reindexing to regular 1-minute intervals: {len(datetime_original)} → {len(time_index)} points"
            )
            dam_data = dam_data.reindex(index=time_index)
            attrs["time_regularized"] = 1
            attrs["gaps_filled"] = len(time_index) - len(datetime_original)
            attrs["gap_fill_value"] = "NaN"
        else:
            attrs["time_regularized"] = 0
            attrs["gaps_filled"] = 0
        attrs["time_is_relative_minutes"] = 0

    # Keep as float32 to preserve NaN (zero means no activity, NaN means no data)
    activity_values = dam_data.values.astype("float32")
    datetime_idx = dam_data.index
    ids = dam_data.columns

    assert activity_values.shape == (len(datetime_idx), len(ids)), (
        f"Activity data shape {activity_values.shape} doesn't match expected ({len(datetime_idx)}, {len(ids)})"
    )

    data_vars = {"activity": (["time", "id"], activity_values)}
    # All coords go through _as_numpy_array so pandas Arrow/string dtypes never
    # land in the Dataset (xarray cannot .sel/.isel ArrowStringArray).
    coords = {
        "id": ("id", _as_numpy_array(metadata["id"])),
        "time": ("time", _as_numpy_array(dam_data.index)),
        "start_datetime": ("id", _as_numpy_array(metadata["start_datetime"])),
        "stop_datetime": ("id", _as_numpy_array(metadata["stop_datetime"])),
    }
    if "file" in metadata.columns:
        coords["file"] = ("id", _as_numpy_array(metadata["file"]))
    coords["genotype"] = ("id", _as_numpy_array(metadata["genotype"]))
    if "first_DD_day" in metadata.columns:
        coords["first_DD_day"] = ("id", _as_numpy_array(metadata["first_DD_day"]))
    if "pulse_time" in metadata.columns:
        coords["pulse_zt_hour"] = (
            "id",
            np.array([_parse_zt_hour(v) for v in metadata["pulse_time"].to_numpy()], dtype="float32"),
        )
    if "pulse_duration_min" in metadata.columns:
        coords["pulse_duration_minutes"] = (
            "id",
            np.asarray(metadata["pulse_duration_min"].to_numpy(), dtype="float32"),
        )

    # Add all remaining metadata columns as per-fly coordinates
    for col in properties:
        coords[col] = ("id", _as_numpy_array(metadata.set_index("id")[col]))

    # Combined group label for group-level analysis. ``group_columns`` chooses which
    # metadata factors define the comparison group; when None we reproduce the
    # historical default (genotype[-temperature]) exactly. The chosen columns are
    # recorded in attrs so the grouping is persistent and re-derivable.
    if group_columns:
        chosen = [c for c in group_columns if c in metadata.columns]
    else:
        chosen = [c for c in ("genotype", "temperature") if c in metadata.columns]
    group_series = derive_group_labels(metadata, chosen) if chosen else None
    if group_series is not None:
        coords["group"] = ("id", _as_numpy_array(group_series))
        attrs["group_columns"] = list(chosen)

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

    if ds.time.size == 0:
        return None

    # Ensure the time coordinate is contiguous after dataset creation
    if is_relative_time:
        first_t = int(ds.time.values[0])
        last_t = int(ds.time.values[-1])
        ds = ds.assign_coords(time=np.arange(first_t, last_t + 1))
    else:
        first_time_str = pd.to_datetime(ds.time.values[0]).strftime("%Y-%m-%d %H:%M:%S")
        last_time_str = pd.to_datetime(ds.time.values[-1]).strftime("%Y-%m-%d %H:%M:%S")
        time_index = pd.date_range(start=first_time_str, end=last_time_str, freq="min")
        ds = ds.assign_coords(time=time_index)

    return ensure_numpy_backed(ds)


def _compute_moving(ds, force=False):
    """
    Add the 'moving' variable to a dataset (1=active, 0=immobile, -1=missing data).

    E1 guard (2026-06-30): when it (re)computes, it does so on a COPY — it never
    mutates the caller's Dataset in place (a stale 'moving' silently carried into a
    re-curation could otherwise mask a death; see tests/test_simulated_death_curation).
    ``force=True`` recomputes 'moving' from the CURRENT activity even if a (possibly
    stale) 'moving' already exists; the default keeps the cheap no-op for the common
    repeat-call case where the existing 'moving' is known good.
    """
    if "moving" in ds.data_vars and not force:
        return ds
    moving = xr.where(ds["activity"].isnull(), -1, (ds["activity"] > 0).astype(int))
    ds = ds.copy()
    ds["moving"] = moving.astype(np.int8)
    return ds


def curate_dead_animals(
    data,
    mov_column="moving",
    t_column="time",
    time_window=24,
    prop_immobile=0.01,
    resolution=1,
    min_alive_days=2,
    progress_callback=None,
):
    """
    Identify and separate dead animals from the dataset.

    Computes the 'moving' variable automatically if not already present.

    Uses a rolling window to find the last time point at which a fly's
    activity exceeds a minimum threshold.  Flies that never exceed it,
    or that only exceed it for fewer than min_alive_days days, are
    classified as dead.

    Parameters
    ----------
    data : xr.Dataset
    mov_column : str
        Variable name for movement data (default 'moving').
    t_column : str
        Coordinate name for time (default 'time').
    time_window : int
        Rolling window size in hours (default 24).
    prop_immobile : float
        Activity threshold; values below this are "inactive" (default 0.01).
    resolution : int
        Data resolution in minutes per sample (default 1).
    min_alive_days : int
        Flies alive for fewer days are excluded (default 2).
    progress_callback : callable, optional
        Called as callback(completed, total).

    Returns
    -------
    tuple : (live_data, dead_data, error_ids, success_ids,
             total_before, total_after, removed_count,
             unchanged_count, trimmed_count)
    """
    # Compute 'moving' from the CURRENT activity (force=True): never trust a stale
    # 'moving' carried in (E1 guard) — a re-curation must re-derive it or it could
    # mask a death. _compute_moving copies, so `data` (the caller's) is not mutated.
    data = _compute_moving(data, force=True)

    if mov_column not in data.data_vars:
        raise KeyError(f"The movement column {mov_column} is not in the dataset")
    if t_column not in data.coords:
        raise KeyError(f"The time column {t_column} is not in the dataset")

    def _wrapped_curate_dead_animals(
        group_data, time_window, prop_immobile, resolution, min_alive_days
    ):
        window_size = int(time_window * 60 / resolution)
        mov_data = group_data[mov_column].astype(float)
        mov_data = mov_data.where(mov_data != -1)  # -1 = missing, exclude from mean
        rolling_activity = mov_data.rolling(time=window_size, center=True, min_periods=1).mean()

        valid_activity = rolling_activity.where(rolling_activity > prop_immobile, drop=True)
        mov_values = group_data[mov_column].values.flatten()
        is_missing = mov_values == -1

        if valid_activity.size == 0:
            is_alive = np.zeros(len(group_data[t_column]), dtype=np.int8)
            is_alive[is_missing] = -1
        else:
            last_active = valid_activity[t_column].max().values

            if pd.isnull(last_active):
                is_alive = np.zeros(len(group_data[t_column]), dtype=np.int8)
                is_alive[is_missing] = -1
            else:
                # Compare against the last non-missing time point, not the
                # absolute end of the time axis. Trailing NaN (monitor stopped,
                # incomplete bins) should not cause a fly to be marked dead.
                non_missing_times = group_data[t_column].values.flatten()[~is_missing]
                last_data_time = (
                    non_missing_times[-1]
                    if len(non_missing_times) > 0
                    else group_data[t_column].max().values
                )
                if last_active >= last_data_time:
                    is_alive = np.ones(len(group_data[t_column]), dtype=np.int8)
                else:
                    is_alive = (group_data[t_column] <= last_active).values.astype(np.int8)

                # Exclude flies alive for fewer than min_alive_days
                time_diff = last_active - group_data[t_column].min().values
                if np.issubdtype(type(time_diff), np.integer) or isinstance(
                    time_diff, (int, float)
                ):
                    alive_duration = float(time_diff) / (60 * 24)  # minutes → days
                else:
                    alive_duration = time_diff / np.timedelta64(1, "D")
                if alive_duration < min_alive_days:
                    is_alive[:] = 0
                    is_alive[is_missing] = -1
                else:
                    # Only flag missing samples AFTER last_active as -1, so
                    # missing samples within the live window [0, last_active]
                    # stay is_alive=1 and the front of a living fly's recording
                    # isn't dropped by `results.where(is_alive == 1, drop=True)`.
                    after_last_active = group_data[t_column].values > last_active
                    is_alive[is_missing & after_last_active] = -1
        is_alive_da = xr.DataArray(
            is_alive.astype(np.int8), coords=group_data[t_column].coords, dims=t_column
        )
        return group_data.assign(is_alive=is_alive_da)

    fly_ids = data["id"].values
    _cur_total = len(fly_ids)
    results_list = []
    for _cur_idx, fly_id in enumerate(fly_ids):
        group_data = data.sel(id=[fly_id])
        results_list.append(
            _wrapped_curate_dead_animals(
                group_data, time_window, prop_immobile, resolution, min_alive_days
            )
        )
        if progress_callback:
            progress_callback(_cur_idx + 1, _cur_total)

    results = xr.concat(results_list, dim="id")
    # Guarantee fly order matches input (defensive — concat should preserve it)
    results = results.sel(id=data["id"].values)

    live_data = results.where(results.is_alive == 1, drop=True)
    dead_data = results.where(results.is_alive == 0, drop=True)

    total_before = len(data["id"])
    total_after = len(live_data["id"])
    removed_count = total_before - total_after
    unchanged_count = len(results.where(results.is_alive.all("time"), drop=True)["id"])
    trimmed_count = total_after - unchanged_count
    success_ids = live_data["id"].to_series().to_list()

    # Record curation parameters for reproducibility
    live_data.attrs["curation_time_window_hours"] = time_window
    live_data.attrs["curation_prop_immobile_threshold"] = prop_immobile
    live_data.attrs["curation_min_alive_days"] = min_alive_days
    live_data.attrs["curation_resolution_minutes"] = resolution

    return (
        live_data,
        dead_data,
        [],
        success_ids,
        total_before,
        total_after,
        removed_count,
        unchanged_count,
        trimmed_count,
    )


def _longest_continuous_segment(activity_1d, time_vals, gap_threshold_minutes=60):
    """
    Find the longest stretch of non-NaN data for a single fly.

    A "gap" is any run of consecutive NaN values whose duration exceeds
    *gap_threshold_minutes*.  Short NaN runs (≤ threshold) are tolerated
    and considered part of the surrounding segment.

    Parameters
    ----------
    activity_1d : np.ndarray
        1-D activity array (one fly).
    time_vals : np.ndarray
        Corresponding time coordinate values (integer minutes or datetime64).
    gap_threshold_minutes : float
        Minimum NaN run duration (minutes) to be treated as a real gap.

    Returns
    -------
    (start_idx, end_idx) : tuple of int
        Indices into the arrays bounding the longest segment (inclusive).
        Returns (0, len-1) if no qualifying gaps are found.
    """
    n = len(activity_1d)
    if n == 0:
        return (0, 0)

    is_valid = np.isfinite(activity_1d)

    # Convert time to minutes-from-start for duration calculations
    if np.issubdtype(time_vals.dtype, np.datetime64):
        t_min = (time_vals - time_vals[0]) / np.timedelta64(1, "m")
    else:
        t_min = time_vals.astype(float) - float(time_vals[0])

    # Find gap boundaries: transitions from valid→NaN and NaN→valid
    # A gap is a contiguous block of NaN longer than the threshold
    gap_starts = []  # index where a qualifying gap begins
    gap_ends = []  # index where the gap ends (first valid after gap)

    i = 0
    while i < n:
        if not is_valid[i]:
            # Start of a NaN run
            j = i
            while j < n and not is_valid[j]:
                j += 1
            # Duration of this NaN run
            run_end = min(j, n - 1)
            duration = t_min[run_end] - t_min[i]
            if duration >= gap_threshold_minutes:
                gap_starts.append(i)
                gap_ends.append(j)
            i = j
        else:
            i += 1

    if not gap_starts:
        # No qualifying gaps — the whole recording is one segment
        return (0, n - 1)

    # Build segment list: data between (and around) the gaps
    segments = []
    # Before first gap
    if gap_starts[0] > 0:
        segments.append((0, gap_starts[0] - 1))
    # Between consecutive gaps
    for k in range(len(gap_starts)):
        seg_start = gap_ends[k]
        seg_end = gap_starts[k + 1] - 1 if k + 1 < len(gap_starts) else n - 1
        if seg_start <= seg_end:
            segments.append((seg_start, seg_end))

    if not segments:
        return (0, n - 1)

    # Pick the longest by duration (not index count, in case of uneven sampling)
    best = max(segments, key=lambda s: t_min[s[1]] - t_min[s[0]])
    return best


def _select_longest_segments(ds, gap_threshold_minutes=60):
    """
    For each fly, find the longest continuous segment and NaN-mask
    activity outside that segment.  Trims the shared time axis to
    remove any timepoints where ALL flies are NaN.

    Parameters
    ----------
    ds : xr.Dataset
        Phase-split dataset (LD or DD).
    gap_threshold_minutes : float
        Minimum NaN run to be considered a real gap.

    Returns
    -------
    ds : xr.Dataset
        Dataset with non-longest segments masked out.
    segment_info : list of dict
        Per-fly segment metadata (fly_id, segment_duration_days,
        n_gaps, original_duration_days).
    """
    fly_ids = ds["id"].values
    time_vals = ds["time"].values
    activity = ds["activity"].values  # shape: (time, id)
    segment_info = []

    for fly_idx, fid in enumerate(fly_ids):
        fly_act = activity[:, fly_idx]

        # Duration of full phase for this fly (before gap trimming)
        valid_mask = np.isfinite(fly_act)
        if valid_mask.any():
            if np.issubdtype(time_vals.dtype, np.datetime64):
                orig_dur = (
                    (time_vals[valid_mask][-1] - time_vals[valid_mask][0])
                    / np.timedelta64(1, "s")
                    / 86400.0
                )
            else:
                orig_dur = (
                    float(time_vals[valid_mask][-1]) - float(time_vals[valid_mask][0])
                ) / 1440.0
        else:
            orig_dur = 0.0

        start_idx, end_idx = _longest_continuous_segment(fly_act, time_vals, gap_threshold_minutes)

        if np.issubdtype(time_vals.dtype, np.datetime64):
            seg_dur = (time_vals[end_idx] - time_vals[start_idx]) / np.timedelta64(1, "s") / 86400.0
        else:
            seg_dur = (float(time_vals[end_idx]) - float(time_vals[start_idx])) / 1440.0

        # NaN-mask everything outside the longest segment
        fly_act[:start_idx] = np.nan
        if end_idx + 1 < len(fly_act):
            fly_act[end_idx + 1 :] = np.nan

        segment_info.append(
            {
                "fly_id": fid,
                "segment_duration_days": round(seg_dur, 2),
                "original_duration_days": round(orig_dur, 2),
                "segment_start_idx": int(start_idx),
                "segment_end_idx": int(end_idx),
            }
        )

    # Write modified activity back
    ds["activity"].values = activity

    # Also mask any other (time, id) data variables consistently
    for var in ds.data_vars:
        if var == "activity":
            continue
        if "time" in ds[var].dims and "id" in ds[var].dims:
            arr = ds[var].values
            # Only mask float-compatible vars (int vars like 'moving' need
            # conversion to float to support NaN)
            if np.issubdtype(arr.dtype, np.integer):
                arr = arr.astype(float)
                ds[var] = ds[var].astype(float)
                ds[var].values = arr
            for fly_idx, info in enumerate(segment_info):
                arr[: info["segment_start_idx"], fly_idx] = np.nan
                if info["segment_end_idx"] + 1 < arr.shape[0]:
                    arr[info["segment_end_idx"] + 1 :, fly_idx] = np.nan
            ds[var].values = arr

    # Trim shared time axis: drop timepoints where ALL flies are NaN
    any_valid = np.any(np.isfinite(ds["activity"].values), axis=1)
    if not np.all(any_valid):
        valid_times = ds["time"].values[any_valid]
        ds = ds.sel(time=np.isin(ds["time"].values, valid_times))

    # Record gap threshold used
    ds.attrs["gap_threshold_minutes"] = gap_threshold_minutes

    return ds, segment_info


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Derive-on-demand phase selection (Stage-1 refactor)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The legacy `split_xarray_dataset` physically slices the dataset into separate
# LD/DD copies, re-zeros time, and — on heterogeneous per-fly boundaries — falls
# back to conservative *global* bounds that DELETE the ragged region (data loss).
# These two helpers replace that model with a single whole dataset that carries a
# per-`id` boundary coordinate (`split_minute`) and one selector that derives a
# NaN-masked phase VIEW on demand. No physical separation, no re-normalization,
# no global-bounds collapse → no data loss on mixed-boundary cohorts.


def _derive_split_minute(ds):
    """Compute the per-`id` LD→DD boundary in the dataset's relative-minute units.

    boundary[id] = round((first_DD_day[id] - start_datetime[id]) / 1 minute)

    Requires the per-fly ``first_DD_day`` and ``start_datetime`` coordinates and a
    relative-minute time axis (``ds.attrs['time_is_relative_minutes'] == 1``).
    Returns an ``int32`` numpy array aligned to ``ds['id']``.
    """
    if "first_DD_day" not in ds.coords:
        raise ValueError("Cannot derive 'split_minute': dataset has no 'first_DD_day' coordinate.")
    if "start_datetime" not in ds.coords:
        raise ValueError(
            "Cannot derive 'split_minute': dataset has no 'start_datetime' coordinate."
        )
    if ds.attrs.get("time_is_relative_minutes", 0) != 1:
        raise ValueError(
            "add_phase_metadata/select_phase require a relative-integer-minute time "
            "axis (attrs['time_is_relative_minutes'] == 1). The current dataset uses "
            "absolute datetime time; convert with convert_to_relative_time first."
        )

    start_dts = ds["start_datetime"].values.astype("datetime64[s]")
    dd_dts = ds["first_DD_day"].values.astype("datetime64[s]")
    minutes = (dd_dts - start_dts) / np.timedelta64(1, "s") / 60.0
    # round() then cast — round((dd - start)/1min) is the spec.
    return np.rint(minutes).astype(np.int32)


def add_phase_metadata(ds):
    """Attach a per-`id` ``split_minute`` boundary coordinate to ``ds`` (idempotent).

    ``split_minute`` is the per-fly LD→DD boundary expressed in the dataset's
    relative-minute time units = ``round((first_DD_day - start_datetime) / 1 min)``,
    stored as an ``int32`` coordinate aligned to the ``id`` dimension. This is the
    lossless, single source of truth for phase membership — it round-trips through
    NetCDF as a coordinate (the loader's attr-datetime restore is buggy, so phase
    metadata must NOT live in attrs).

    Kept intentionally tiny: only the per-`id` boundary is stored. The big
    ``(id, time)`` phase mask is never materialized — it is derived on demand by
    :func:`select_phase`.

    Parameters
    ----------
    ds : xr.Dataset
        Whole (unsplit) dataset on a relative-minute time axis, carrying the
        per-fly ``first_DD_day`` and ``start_datetime`` coordinates.

    Returns
    -------
    xr.Dataset
        The same dataset with a ``split_minute`` ``(id,)`` coordinate. Idempotent:
        recomputes and overwrites if already present (cheap; keeps it consistent
        with the current ``first_DD_day``/``start_datetime``).

    Raises
    ------
    ValueError
        If ``first_DD_day`` / ``start_datetime`` are absent, or the time axis is
        not relative-integer-minute.
    """
    split_minute = _derive_split_minute(ds)
    return ds.assign_coords(split_minute=("id", split_minute))


def _derive_pulse_minute(ds):
    """Compute the per-`id` light-pulse onset in the dataset's relative-minute units.

    The pulse is recorded in the metadata as a ZT hour (``pulse_time``, e.g. ``ZT15``)
    with no date, because that is how the protocol is specified — the pulse is given
    at a fixed circadian time on the LAST ENTRAINED DAY, immediately before release
    into DD. So the day is not a free parameter: it is the day whose ZT0 is one day
    before ``first_DD_day``, which is why only the ZT needs writing down::

        pulse[id] = split_minute[id] - 1440 + pulse_zt_hour[id] * 60

    Example (the lab's own): ``first_DD_day`` 6/23 09:00 with ``start_datetime``
    6/20 09:00 gives ``split_minute`` 4320, so ZT15 resolves to minute 3780 — 9 h
    before the DD boundary, i.e. the next ZT0 is the first full DD day.

    Returns ``float32``, not ``int32``: ``pulse_time`` is legitimately blank for an
    unpulsed control cohort and that "no pulse" state must survive as NaN. An integer
    sentinel would read as a real minute to any caller that forgot to check.
    """
    if "pulse_zt_hour" not in ds.coords:
        raise ValueError(
            "Cannot derive 'pulse_minute': dataset has no 'pulse_zt_hour' coordinate. "
            "Add a 'pulse_time' column to the metadata (a ZT hour such as 'ZT15'; see "
            "metadata_template.csv) and reload the dataset."
        )
    if ds.attrs.get("time_is_relative_minutes", 0) != 1:
        raise ValueError(
            "add_pulse_metadata requires a relative-integer-minute time axis "
            "(attrs['time_is_relative_minutes'] == 1). The current dataset uses absolute "
            "datetime time; convert with convert_to_relative_time first."
        )
    if "split_minute" not in ds.coords:
        if "first_DD_day" not in ds.coords:
            raise ValueError(
                "Cannot derive 'pulse_minute': a ZT pulse time is anchored to the LD-DD "
                "boundary, so the metadata also needs a 'first_DD_day' column. Add it and "
                "reload the dataset."
            )
        ds = add_phase_metadata(ds)

    split = np.asarray(ds["split_minute"].values, dtype="float64")
    zt = np.asarray(ds["pulse_zt_hour"].values, dtype="float64")
    return np.rint(split - 1440.0 + zt * 60.0).astype(np.float32)


def add_pulse_metadata(ds):
    """Attach per-`id` light-pulse coordinates to ``ds`` (idempotent).

    Adds ``pulse_minute`` (float32, NaN where a cohort received no pulse) and passes
    ``pulse_duration_minutes`` through (float32, NaN meaning "duration unrecorded" —
    callers treat that as an instantaneous pulse). Structured exactly like
    :func:`add_phase_metadata`: the value lives in a coordinate (which round-trips
    through NetCDF losslessly) rather than in attrs.

    Raises
    ------
    ValueError
        If ``pulse_zt_hour`` / ``first_DD_day`` are absent, or the time axis is not
        relative-integer-minute.
    """
    pulse_minute = _derive_pulse_minute(ds)
    out = ds.assign_coords(pulse_minute=("id", pulse_minute))
    if "pulse_duration_minutes" in ds.coords:
        durations = np.asarray(ds["pulse_duration_minutes"].values, dtype="float32")
    else:
        durations = np.full(ds["id"].size, np.nan, dtype="float32")
    return out.assign_coords(pulse_duration_minutes=("id", durations))


def select_phase(ds, phase="auto", discard_first_dd_day=False):
    """Return a per-fly NaN-masked phase VIEW of the WHOLE dataset (derive-on-demand).

    This is the one core selector for LD/DD phase. Unlike the legacy
    :func:`split_xarray_dataset`, it never physically slices, never re-normalizes
    time, and never collapses heterogeneous per-fly boundaries to global bounds.
    It returns the full time axis with the out-of-phase cells of every
    ``(id, time)`` data_var masked to **NaN**, honoring each fly's own boundary.

    Phase resolution
    ----------------
    - ``"auto"`` → ``"DD"`` if a boundary is derivable (``first_DD_day`` or an
      existing ``split_minute`` present), else ``"both"``.
    - ``"DD"`` / ``"LD"`` on a WHOLE/unsplit dataset (canonical phase ``'full'``) →
      that phase, **re-derived** from ``split_minute`` and masked. There is no
      *silent* "already split, pass through" on a full dataset.
    - ``"DD"`` / ``"LD"`` on a dataset **already stamped a single phase** (LD/DD —
      the B4 guard): if it IS the requested phase, return it as-is (safe no-op, no
      re-derive/re-mask — re-masking a physically-sliced, re-zeroed object was the
      Stage-1 foot-gun that dropped the first epoch); if a DIFFERENT phase is
      requested, **raise** (that epoch isn't present). This is the LOUD replacement
      for the old pass-through (which silently returned the wrong phase).
    - ``"both"`` → the whole dataset, returned UNCHANGED.

    Masking rule (per fly, broadcast over ``(id, time)`` via ``split_minute``)
    -------------------------------------------------------------------------
    - DD keeps ``time >= split_minute`` (or ``>= split_minute + 1440`` when
      ``discard_first_dd_day`` — drops the first DD calendar day).
    - LD keeps ``time < split_minute``.
    - Out-of-phase cells become **NaN** (§2a: NEVER 0). In-phase real ``0`` values
      are preserved as ``0``; true gaps (already NaN) stay NaN. ``split_minute``
      remains on the returned view as the lossless phase source-of-truth, so a
      caller can always re-distinguish "out-of-phase" from "true gap".

    Dtype note (§2b)
    ----------------
    ``activity`` is masked with :func:`xr.where`, which preserves its ``float32``
    dtype (the NaN fill stays float32 — no float64 upcast). Integer ``(id, time)``
    vars (e.g. ``moving``, ``sleep``) cannot hold NaN, so to mask them this VIEW
    upcasts them to float. That is transient (a returned view, never stored back to
    the NetCDF master) and acceptable; ``split_minute`` is the durable phase record.

    Parameters
    ----------
    ds : xr.Dataset
        Whole dataset on a relative-minute time axis. ``split_minute`` is derived
        if absent (requires ``first_DD_day`` + ``start_datetime``).
    phase : str
        ``"auto"`` (default), ``"DD"``, ``"LD"``, or ``"both"``.
    discard_first_dd_day : bool
        If True (DD phase only), also drop each fly's first DD calendar day
        (1440 minutes after its boundary).

    Returns
    -------
    (analysis_ds, phase_used) : tuple[xr.Dataset, str]
        The masked phase view (or the whole dataset for ``"both"``) and the phase
        that was actually applied (``"DD"``, ``"LD"``, or ``"both"``).
    """
    requested = phase.lower()
    has_boundary = ("split_minute" in ds.coords) or ("first_DD_day" in ds.coords)

    if requested == "auto":
        phase_used = "DD" if has_boundary else "both"
    elif requested in ("dd", "ld"):
        phase_used = requested.upper()
    elif requested == "both":
        phase_used = "both"
    else:
        raise ValueError(f"Unknown phase {phase!r}; expected 'auto', 'DD', 'LD', or 'both'.")

    if phase_used == "both":
        # Whole dataset, returned unchanged. Stamp provenance on a shallow copy so
        # the caller's dataset attrs are not mutated as a side effect.
        out = ds.copy()
        out.attrs["phase_used"] = "both"
        return out, "both"

    # If ``ds`` is ALREADY a single-phase partition, never re-derive the boundary
    # and re-mask it — that was a real foot-gun:
    # a physically-sliced, re-zeroed DD/LD object still carries the original
    # ``first_DD_day``, so re-deriving ``split_minute`` and masking the re-zeroed
    # axis silently dropped the first epoch (or emptied a mismatched partition).
    #   - already the requested phase  → return as-is (safe no-op; no data loss);
    #   - a DIFFERENT phase requested  → that epoch isn't present in this partition;
    #     fail LOUD rather than truncate/empty.
    # The WHOLE master is stamped ``phase='full'`` (not LD/DD) so it is never
    # guarded here and masks normally; a masked VIEW is stamped with its own phase,
    # so re-selecting the same phase is the idempotent no-op above.
    from dataset_meta import dataset_phase as _dataset_phase

    _stamp = _dataset_phase(ds)
    if _stamp in ("LD", "DD"):
        if _stamp == phase_used:
            out = ds.copy()
            out.attrs["phase_used"] = phase_used
            return out, phase_used
        raise ValueError(
            f"select_phase: dataset is already the {_stamp} partition; cannot "
            f"select phase {phase_used!r} from it (that epoch is not present). "
            f"Pass the WHOLE unsplit dataset to select a phase, or request "
            f"{_stamp!r}."
        )

    # DD / LD requested but no boundary available → cannot derive; fall back to the
    # whole dataset (mirrors the legacy "no first_DD_day" behavior of _select_phase).
    if not has_boundary:
        print(
            f"  No boundary ('split_minute'/'first_DD_day') in dataset — "
            f"using full dataset instead of {phase_used}"
        )
        out = ds.copy()
        out.attrs["phase_used"] = "both"
        return out, "both"

    # Ensure the per-`id` boundary coordinate exists (derive on demand, idempotent).
    if "split_minute" not in ds.coords:
        ds = add_phase_metadata(ds)

    split_minute = ds["split_minute"]  # (id,) int32, broadcasts against (id, time)
    time = ds["time"]

    if phase_used == "DD":
        boundary = split_minute + 1440 if discard_first_dd_day else split_minute
        keep = time >= boundary
    else:  # "LD"
        keep = time < split_minute

    # Mask each (id, time) data_var to NaN where out of phase. xr.where preserves
    # the in-phase values verbatim (real 0s stay 0; existing NaN gaps stay NaN).
    masked = ds.copy()
    for var in ds.data_vars:
        da = ds[var]
        if "id" in da.dims and "time" in da.dims:
            if np.issubdtype(da.dtype, np.floating):
                # float32 stays float32; float64 stays float64 (NaN fill is dtype-safe).
                m = xr.where(keep, da, np.nan).astype(da.dtype)
            else:
                # Integer/bool vars can't hold NaN → upcast to float for the masked
                # VIEW only (transient; never written back to the master).
                m = xr.where(keep, da, np.nan)
            # xr.where may reorder dims when broadcasting; restore the original
            # dim order so downstream `.transpose('time','id')` callers are unaffected.
            masked[var] = m.transpose(*da.dims)

    masked.attrs["phase"] = phase_used
    masked.attrs["phase_used"] = phase_used
    masked.attrs["phase_discard_first_dd_day"] = int(discard_first_dd_day)
    return masked, phase_used


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Uniform phase API — per-function policy layer over the one selector
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# `select_phase` is the single masking MECHANISM (LD/DD/both). `resolve_phase`
# is the thin POLICY layer every analysis entry point calls: it resolves the
# per-function default for ``phase="auto"`` and rejects a phase the analysis does
# not accept (fail loud — e.g. a periodogram over combined LD+DD mixes entrained
# and free-running rhythms and must NOT silently compute). This keeps one
# selection mechanism while letting each analysis declare its own valid set and
# default at its call site.
#
# The canonical per-function default table (single source of truth for the
# documented defaults). Each value is (default_phase, allowed_phases).
# Defaults are small scientific decisions —
# kept explicit here, not buried in signatures.
PHASE_POLICY = {
    # Period / rhythm analyses → DD (free-running circadian period); reject
    # "both" (combining LD-entrained + DD free-running rhythms is meaningless).
    "lomb_scargle_analysis": ("DD", ("LD", "DD")),
    "wavelet_analysis": ("DD", ("LD", "DD")),
    "autocorrelation_analysis": ("DD", ("LD", "DD")),
    "sleep_cwt_analysis": ("DD", ("LD", "DD")),
    # (ultradian_rhythmicity_ls takes no phase — it runs on the already
    #  phase-selected sleep_cwt_* amplitude series, inheriting that phase.)
    # Sleep / activity → LD (the entrained day is the standard sleep reference);
    # "both" is allowed (total sleep across the whole continuous recording).
    "sleep_analysis": ("LD", ("LD", "DD", "both")),
    # Sleep deprivation is an LD-structured protocol (light/dark baseline +
    # recovery days). LD only — DD/both have no SD interpretation here.
    "compute_sd_analysis": ("LD", ("LD",)),
    "get_experiment_days": ("LD", ("LD", "DD", "both")),
    # Visualisation / continuity → both (the continuous LD+DD trace actograms
    # want); all three are valid views.
    "actogram": ("both", ("LD", "DD", "both")),
}

_PHASE_CANON = {"ld": "LD", "dd": "DD", "both": "both"}


def resolve_phase(
    ds, phase="auto", *, default, allowed, caller="analysis", discard_first_dd_day=False
):
    """Resolve ``phase`` against one analysis's policy, then apply :func:`select_phase`.

    This is the single per-function policy gate. It (1) resolves ``"auto"`` to the
    caller's ``default``, (2) validates the resolved phase against the caller's
    ``allowed`` set and raises a clear error if the analysis does not accept it
    (fail loud), and (3) delegates the actual masking to :func:`select_phase` — the
    one selection mechanism. No analysis re-implements slicing or default logic.

    Parameters
    ----------
    ds : xr.Dataset
        Whole dataset on a relative-minute time axis (carries ``split_minute`` or
        ``first_DD_day`` so a boundary is derivable for LD/DD).
    phase : str
        ``"auto"`` (use ``default``), ``"LD"``, ``"DD"``, or ``"both"`` (case-insensitive).
    default : str
        The phase used when ``phase="auto"`` — this analysis's documented default
        (see :data:`PHASE_POLICY`).
    allowed : tuple[str]
        The phases this analysis accepts. A request outside this set raises
        ``ValueError`` rather than silently computing on the wrong epoch.
    caller : str
        Name used in error messages (the analysis function).
    discard_first_dd_day : bool
        Forwarded to :func:`select_phase` (DD only).

    Returns
    -------
    (analysis_ds, phase_used) : tuple[xr.Dataset, str]
        The masked phase view (or the whole dataset for ``"both"``) and the phase
        actually applied.

    Raises
    ------
    ValueError
        If ``phase`` is unrecognised, or the resolved phase is not in ``allowed``.
    """
    req = (phase or "auto").lower()
    if req == "auto":
        resolved = _PHASE_CANON.get(default.lower())
        if resolved is None:
            raise ValueError(f"{caller}: invalid default phase {default!r}.")
    else:
        resolved = _PHASE_CANON.get(req)
        if resolved is None:
            raise ValueError(
                f"{caller}: unknown phase {phase!r} (expected 'auto', 'LD', 'DD', or 'both')."
            )

    allowed_canon = tuple(_PHASE_CANON.get(a.lower(), a) for a in allowed)
    if resolved not in allowed_canon:
        hint = ""
        if "both" not in allowed_canon and resolved == "both":
            hint = (
                " A periodogram over combined LD-entrained + DD free-running "
                "data mixes two rhythms; request 'LD' or 'DD' explicitly."
            )
        raise ValueError(
            f"{caller}: phase {resolved!r} is not valid for this analysis "
            f"(accepts {list(allowed_canon)}).{hint}"
        )

    return select_phase(ds, phase=resolved, discard_first_dd_day=discard_first_dd_day)


def split_xarray_dataset(ds, phase="LD", discard_first_dd_day=False, gap_threshold_minutes=60):
    """
    Split a curated xarray Dataset into LD or DD phase.

    Designed to be called AFTER curate_dead_animals() so that dead-fly
    detection uses the complete recording.  Uses the per-fly
    ``start_datetime`` and ``first_DD_day`` coordinates already stored
    in the dataset to compute the split point in relative-minute time.

    After slicing to the requested phase, detects large gaps (≥
    *gap_threshold_minutes* of consecutive NaN) per fly and keeps only
    the longest continuous segment for each fly.  This prevents
    phase-discontinuous data from corrupting autocorrelation and CWT
    analyses while preserving the maximum amount of usable data.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with relative-integer-minute time axis and per-fly
        ``start_datetime`` / ``first_DD_day`` coordinates.
    phase : str
        ``"LD"`` to keep light-dark data, ``"DD"`` for constant darkness,
        or ``"both"`` to return the dataset unchanged.
    discard_first_dd_day : bool
        If True and phase includes DD, remove the first 1440 minutes
        (one calendar day) of DD data per fly.
    gap_threshold_minutes : float
        Minimum consecutive-NaN duration (minutes) to treat as a real
        gap.  Set to 0 to disable gap detection.

    Returns
    -------
    xr.Dataset
        Subset of the input with time re-normalised to start at 0 and
        each fly trimmed to its longest continuous segment.
    """
    if phase.lower() == "both" and not discard_first_dd_day:
        return ds

    if "first_DD_day" not in ds.coords:
        raise ValueError("Dataset has no 'first_DD_day' coordinate — cannot split LD/DD.")

    is_relative = ds.attrs.get("time_is_relative_minutes", 0) == 1

    # --- compute per-fly split point in the dataset's time units ----------
    if is_relative:
        start_dts = ds["start_datetime"].values.astype("datetime64[s]")
        dd_dts = ds["first_DD_day"].values.astype("datetime64[s]")
        split_points = ((dd_dts - start_dts) / np.timedelta64(1, "s") / 60.0).tolist()
    else:
        split_points = [pd.Timestamp(v) for v in ds["first_DD_day"].values]

    unique_splits = sorted(set(split_points))
    time = ds["time"]

    # --- select time slices using xarray boolean indexing -----------------
    if len(unique_splits) == 1:
        # Common case: all flies share the same split point
        sp = unique_splits[0]

        if phase.upper() == "LD":
            result = ds.sel(time=time < sp)
        elif phase.upper() == "DD":
            if is_relative:
                dd_start = sp + 1440 if discard_first_dd_day else sp
            else:
                dd_start = sp + pd.Timedelta(days=1) if discard_first_dd_day else sp
            result = ds.sel(time=time >= dd_start)
        else:  # "both" with discard_first_dd_day
            if is_relative:
                dd_end = sp + 1440
            else:
                dd_end = sp + pd.Timedelta(days=1)
            result = ds.sel(time=(time < sp) | (time >= dd_end))
    else:
        # Different per-fly split points: use conservative global bounds
        min_sp = min(split_points)
        max_sp = max(split_points)

        if phase.upper() == "LD":
            result = ds.sel(time=time < min_sp)
        elif phase.upper() == "DD":
            if is_relative:
                dd_start = max_sp + 1440 if discard_first_dd_day else max_sp
            else:
                dd_start = max_sp + pd.Timedelta(days=1) if discard_first_dd_day else max_sp
            result = ds.sel(time=time >= dd_start)
        else:  # "both" with discard_first_dd_day
            if is_relative:
                discard_end = max_sp + 1440
            else:
                discard_end = max_sp + pd.Timedelta(days=1)
            result = ds.sel(time=(time < min_sp) | (time >= discard_end))

    # --- re-normalise time to start at 0 ----------------------------------
    if is_relative and len(result["time"]) > 0:
        new_origin = int(result["time"].values[0])
        if new_origin != 0:
            result = result.assign_coords(time=result["time"].values - new_origin)

    # --- per-fly longest-segment selection -----------------------------------
    segment_info = None
    if gap_threshold_minutes > 0 and len(result["time"]) > 0:
        result, segment_info = _select_longest_segments(
            result, gap_threshold_minutes=gap_threshold_minutes
        )
        # Re-normalise time again after trimming
        if is_relative and len(result["time"]) > 0:
            new_origin2 = int(result["time"].values[0])
            if new_origin2 != 0:
                result = result.assign_coords(time=result["time"].values - new_origin2)

        # Print summary
        if segment_info:
            durs = [s["segment_duration_days"] for s in segment_info]
            orig_durs = [s["original_duration_days"] for s in segment_info]
            n_trimmed = sum(1 for s, o in zip(durs, orig_durs) if s < o)
            if n_trimmed > 0:
                print(
                    f"\n  Gap detection (threshold={gap_threshold_minutes} min): "
                    f"{n_trimmed}/{len(durs)} flies trimmed to longest segment"
                )
                print(
                    f"  Segment durations — min: {min(durs):.1f}d, "
                    f"avg: {np.mean(durs):.1f}d, max: {max(durs):.1f}d"
                )

    # Record what was done. `phase` and `split_applied` are the canonical attrs
    # (see core/dataset_meta.py). The legacy `split_phase` alias is deliberately
    # NOT written any more: it duplicated `phase` on every new file while the
    # readers already fall back to it, so writing it only grew the set of files
    # carrying two sources of truth. dataset_meta still READS it, for `.nc`
    # files saved before this change — see the note in its module docstring.
    result.attrs["phase"] = phase
    result.attrs["split_applied"] = True
    result.attrs["split_discard_first_dd_day"] = int(discard_first_dd_day)
    result.attrs["gap_threshold_minutes"] = gap_threshold_minutes
    if segment_info:
        result.attrs["segment_info"] = str(segment_info)

    return result


# int8 state masks that use -1 as the "missing / no measurement" sentinel (§2a):
# a -1 means no measurement existed and MUST be excluded from means/sums — never
# averaged in as if it were data (that fabricates negative "sleep" in ZT bins that
# overlap a gap). Real 0 (measured, not asleep/not moving) is kept.
MASK_MISSING_SENTINEL = -1
MASK_VARS_WITH_MISSING_SENTINEL = frozenset(
    {
        "moving",
        "sleep",
        "sleep_short",
        "sleep_intermediate",
        "sleep_long",
        "hmm_state",
        "hmm_sleep",
    }
)


def get_zt_binned_dataframe(
    ds: xr.Dataset, value_col: str = "activity", bin_size_minutes: int = 30, bin_function="mean"
) -> pd.DataFrame:
    """
    Bin per-fly data by Zeitgeber Time (ZT) using each fly's individual start time.

    ZT is the time elapsed since that fly's lights-on (start_datetime), wrapped
    modulo 24 hours.  Bins are [0, bin_size), [bin_size, 2*bin_size), …, [1440-bin_size, 1440).

    The original dataset is NOT modified.

    Parameters
    ----------
    ds : xr.Dataset
        Must contain the requested variable and 'start_datetime'/'stop_datetime' coords.
    value_col : str
        Variable to bin (e.g., 'activity', 'sleep').
    bin_size_minutes : int
        Width of each ZT bin in minutes (default 30).
    bin_function : str
        Aggregation function: 'mean' or 'sum'.

    Returns
    -------
    pd.DataFrame
        Columns: 'id', 'zt_bin_minute', value_col.
        One row per (fly, ZT bin).
    """
    if value_col not in ds.data_vars:
        raise ValueError(f"Variable '{value_col}' not found in the dataset.")
    if "start_datetime" not in ds.coords or "stop_datetime" not in ds.coords:
        raise ValueError("Dataset must have 'start_datetime' and 'stop_datetime' as coordinates.")

    if bin_function not in ("mean", "sum"):
        raise ValueError(f"Invalid bin function: {bin_function!r}. Use 'mean' or 'sum'.")

    fly_ids = ds["id"].values
    zt_minute_bins = np.arange(0, 1440 + bin_size_minutes, bin_size_minutes)
    zt_bin_labels = zt_minute_bins[:-1]  # bin starts: 0, 30, 60, …
    n_bins = len(zt_bin_labels)

    time_vals = ds["time"].values
    is_relative = np.issubdtype(time_vals.dtype, np.integer)

    # Compute ZT minute for every timepoint (shared across all flies in relative mode)
    if is_relative:
        zt_minutes = time_vals.astype(float) % 1440
    else:
        # Absolute time: ZT is relative to each fly's start_datetime, but all flies
        # in the same experiment typically share the same start. Use first fly's start
        # as reference (same as relative mode offset).
        ref_start = pd.to_datetime(ds["start_datetime"].values[0])
        zt_minutes = ((pd.to_datetime(time_vals) - ref_start).total_seconds() / 60) % 1440

    bin_idx = np.clip(np.digitize(zt_minutes, zt_minute_bins) - 1, 0, n_bins - 1)

    # Extract full data array: (time, n_flies)
    data_2d = ds[value_col].transpose("time", "id").values.astype(float)

    # For the int8 state masks, -1 is the missing sentinel (§2a) — exclude it from
    # the per-bin aggregate exactly as NaN is excluded, so a bin overlapping a gap
    # reports the fraction over MEASURED minutes (never a negative fabricated value).
    drop_sentinel = value_col in MASK_VARS_WITH_MISSING_SENTINEL

    # Bin across all flies at once
    binned_data_list = []
    for i, fly_id in enumerate(fly_ids):
        # Normalize fly_id to scalar
        if isinstance(fly_id, np.ndarray):
            fly_id = fly_id.item() if fly_id.ndim == 0 else fly_id[0]
        elif hasattr(fly_id, "item"):
            fly_id = fly_id.item()

        fly_vals = data_2d[:, i]

        binned_vals = np.full(n_bins, np.nan)
        for b in range(n_bins):
            b_mask = bin_idx == b
            valid = fly_vals[b_mask]
            valid = valid[~np.isnan(valid)]
            if drop_sentinel:
                valid = valid[valid != MASK_MISSING_SENTINEL]  # -1 = missing (§2a)
            if len(valid) > 0:
                binned_vals[b] = np.mean(valid) if bin_function == "mean" else np.sum(valid)

        binned_fly = pd.DataFrame(
            {
                "id": fly_id,
                "zt_bin_minute": zt_bin_labels,
                value_col: binned_vals,
            }
        )
        binned_data_list.append(binned_fly)

    if not binned_data_list:
        print("Warning: No data was binned for any fly within their specified active periods.")
        return pd.DataFrame(columns=["id", "zt_bin_minute", value_col])

    result_df = pd.concat(binned_data_list, ignore_index=True)
    result_df["zt_bin_minute"] = pd.to_numeric(result_df["zt_bin_minute"], errors="coerce")
    return result_df


# --- ZT bin -> plotted x-coordinate convention (single source of truth) ---------
# A ZT bin spans [zt_bin_minute, zt_bin_minute + bin_size).  Every daily-profile line
# plot and its matching export places the bin's single marker at the bin's CLOSING
# edge (offset fraction 1.0), NOT its center.  Rationale, verified against the LD
# activity profile: the evening anticipation peak crests in the last half-hour before
# lights-off — the [11.5h, 12.0h) bin — so the closing-edge label lands that peak
# exactly on the ZT12 gridline (center put it at 11.75; left edge at 11.5).  The
# morning startle is reactive (just after lights-on, in [0, 0.5h)); the closing edge
# places it at ZT0.5 — the deliberate tradeoff of prioritising ZT12 alignment.
#
# This is a DISPLAY convention ONLY.  Day/Night partitioning must use the bin's actual
# content (which side of ZT12 = 720 min the bin's data lies), never this label — see
# plotting.per_fly_summary_table.  Change the fraction here to shift every profile plot
# AND its export together (0.0 = left edge / bin start, 0.5 = center, 1.0 = closing).
ZT_LABEL_OFFSET_FRACTION = 1.0


def zt_bin_to_hours(zt_bin_minute, bin_size_minutes):
    """Convert a ZT bin-start (minutes) to its plotted x-coordinate in hours.

    Applies the shared closing-edge convention (:data:`ZT_LABEL_OFFSET_FRACTION`) so
    the daily profile's evening peak aligns to ZT12.  Accepts a scalar, numpy array,
    or pandas Series/Index and returns the same shape, in hours.
    """
    return (zt_bin_minute + ZT_LABEL_OFFSET_FRACTION * bin_size_minutes) / 60.0
