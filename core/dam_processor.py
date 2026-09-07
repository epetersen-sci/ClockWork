"""
dam_processor.py
================
Loads, expands, and validates raw Drosophila Activity Monitor (DAM) data files.

Primary entry point for the data loading pipeline. Reads metadata (CSV or Excel),
expands it to per-region rows if needed, then validates that matching monitor files
exist and that the requested date ranges are present and contiguous.

Typical usage (also called from dam_utilities.read_data_and_metadata):
    processor = MetadataProcessor(metadata_path, data_folder)
    validated_metadata, all_monitor_data = processor.run()

Output:
    - validated_metadata : pd.DataFrame — one row per (monitor, region, start_datetime)
    - all_monitor_data   : pd.DataFrame — DatetimeIndex, columns = fly IDs
      (format: YYYYMMDD_Monitor_Region)

Flies that could not be loaded are never dropped silently. Every exclusion is
recorded on ``self.import_issues`` and rendered by ``import_report_lines`` /
``_print_import_report`` (see ``import_diagnostics``), so an import that yields
nothing can still say whether the monitor file was missing, the metadata window
fell outside the recording, or the tubes are not in the file. Problems with the
metadata file itself are raised as ``MetadataError`` before any monitor is read.
"""

import os
import sys
from datetime import timedelta

import pandas as pd
from tqdm import tqdm

import dam_integrity
import import_diagnostics
from import_diagnostics import ImportIssue, MetadataError

# Columns without which the pipeline cannot build a dataset at all. `genotype` is
# here because create_xarray_dataset attaches it unconditionally as a coordinate;
# omitting it used to surface as a bare KeyError three steps downstream.
REQUIRED_METADATA_COLUMNS = ("Monitor", "start_datetime", "stop_datetime", "genotype")


def _parse_region_ids(cell, n_channels=32):
    """Expand one metadata ``region_id`` cell into the list of DAM tube/channel
    numbers it selects.

    * blank / NaN / empty  -> every tube ``1..n_channels`` (the whole monitor)
    * a single value ``5`` / ``"5"`` / ``5.0``  -> ``[5]``
    * an inclusive hyphen range ``"1-16"``  -> ``[1, 2, ..., 16]``
    * a comma list ``"1,3,5"`` or mixed ``"1-4,17-20"``  -> the de-duplicated union

    Expansion is therefore PER ROW: a monitor split across genotypes can be
    written as two range rows (``1-16`` / ``17-32``) while every other monitor
    stays a single blank-region row that still auto-expands to all tubes. A cell
    that cannot be parsed raises ``ValueError`` naming the offending value rather
    than silently dropping flies.
    """
    if cell is None or (not isinstance(cell, str) and pd.isna(cell)):
        return list(range(1, n_channels + 1))
    s = str(cell).strip()
    if s == "" or s.lower() == "nan":
        return list(range(1, n_channels + 1))
    ids = []
    try:
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo_s, hi_s = part.split("-", 1)
                lo, hi = int(float(lo_s)), int(float(hi_s))
                if hi < lo:
                    lo, hi = hi, lo
                ids.extend(range(lo, hi + 1))
            else:
                ids.append(int(float(part)))
    except ValueError as e:
        raise ValueError(
            f"Unparseable region_id cell {cell!r}: {e}. Use a single number "
            f"('5'), an inclusive range ('1-16'), or a comma list ('1,3,5')."
        ) from e
    # de-duplicate, preserving first-seen order
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


class MetadataProcessor:
    """
    Handles the initial loading, expansion, and validation of DAM metadata.

    Parameters
    ----------
    metadata_path : str
        Path to a CSV or Excel metadata file.
    data_folder : str
        Directory containing MonitorXXX.txt raw DAM files.
    gap_threshold_hours : float
        Gaps larger than this in the DAM file will generate a warning (default 1 h).
    interactive : bool
        If True, prompt on stdin when validation failures occur (legacy CLI
        behavior). If False (default), run unattended: never call ``input()``
        and never ``sys.exit()`` — failed combos are auto-excluded with a
        warning. The default is non-interactive so headless/Streamlit/test runs
        cannot hang on a stdin prompt.
    strict : bool
        If True, any validation failure raises ``ValueError`` instead of being
        excluded. Useful for tests/CI that want a hard failure on bad input.
        Mutually exclusive in effect with ``interactive`` (strict wins).
    write_expanded_csv : bool
        If True, ``expand_metadata`` writes the expanded metadata to
        ``expanded_csv_path``. Default False so loading does not pollute the
        current working directory with ``metadata_all.csv``.
    expanded_csv_path : str or None
        Destination for the expanded-metadata CSV when ``write_expanded_csv`` is
        True. Defaults to ``"metadata_all.csv"`` in the cwd (legacy path) only
        when the write is explicitly requested.
    """

    def __init__(
        self,
        metadata_path,
        data_folder,
        gap_threshold_hours=1.0,
        interactive=False,
        strict=False,
        write_expanded_csv=False,
        expanded_csv_path=None,
    ):
        self.metadata_path = metadata_path
        self.data_folder = data_folder
        self.gap_threshold = timedelta(hours=gap_threshold_hours)
        self.interactive = interactive
        self.strict = strict
        self.write_expanded_csv = write_expanded_csv
        self.expanded_csv_path = expanded_csv_path
        # Per-monitor data-integrity findings (dam_integrity), populated during
        # validate_files_and_dates and surfaced to the user. Keyed by monitor id.
        self.integrity_report = {}
        # Why flies did not make it in (import_diagnostics.ImportIssue), populated
        # during validate_files_and_dates. An import that ends with 0 flies must be
        # able to say which of file-missing / window-wrong / no-data caused it.
        self.import_issues = []
        # Flies the metadata asked for, before any exclusion — the denominator of
        # the "imported N of M" headline.
        self.n_flies_requested = 0
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found at: {self.metadata_path}")
        if not os.path.isdir(self.data_folder):
            raise FileNotFoundError(f"Data folder not found at: {self.data_folder}")

    @staticmethod
    def _check_metadata_shape(meta_df):
        """Fail fast, and by name, on a metadata file the pipeline cannot use.

        Runs on the file as written — BEFORE region expansion — so every row
        number quoted back to the user is the row they can go and look at in
        their spreadsheet (row 1 = the header, matching what Excel shows).

        Each of these used to surface much later and much less legibly: a missing
        column as a bare ``KeyError`` from whichever step first touched it, an
        unparseable date as a pandas parser traceback, and a reversed window as an
        import that quietly produced nothing.
        """
        if meta_df.empty:
            raise MetadataError(
                import_diagnostics.REASON_METADATA_EMPTY,
                "the file parsed successfully but contains no data rows.",
            )

        missing = [c for c in REQUIRED_METADATA_COLUMNS if c not in meta_df.columns]
        if missing:
            raise MetadataError(
                import_diagnostics.REASON_METADATA_MISSING_COLUMNS,
                f"missing {', '.join(repr(c) for c in missing)}. "
                f"Columns found: {', '.join(map(str, meta_df.columns))}.",
            )

        # Datetime columns. start/stop must parse on every row (a blank cell is a
        # failure); first_DD_day is optional, so only non-blank cells are checked.
        for col, allow_blank in (
            ("start_datetime", False),
            ("stop_datetime", False),
            ("first_DD_day", True),
        ):
            if col not in meta_df.columns:
                continue
            parsed = pd.to_datetime(meta_df[col], errors="coerce")
            bad = parsed.isna()
            if allow_blank:
                bad &= meta_df[col].notna() & (meta_df[col].astype(str).str.strip() != "")
            if bad.any():
                rows = [int(i) + 2 for i in meta_df.index[bad][:10]]
                values = [repr(v) for v in meta_df.loc[bad, col].head(5).tolist()]
                raise MetadataError(
                    import_diagnostics.REASON_METADATA_BAD_DATETIME,
                    f"{int(bad.sum())} row(s) have a '{col}' value that could not be "
                    f"read as a date/time — spreadsheet row(s) "
                    f"{', '.join(map(str, rows))}{' ...' if int(bad.sum()) > 10 else ''}; "
                    f"value(s): {', '.join(values)}.",
                )

        starts = pd.to_datetime(meta_df["start_datetime"])
        stops = pd.to_datetime(meta_df["stop_datetime"])
        reversed_rows = stops <= starts
        if reversed_rows.any():
            rows = [int(i) + 2 for i in meta_df.index[reversed_rows][:10]]
            first = meta_df.index[reversed_rows][0]
            raise MetadataError(
                import_diagnostics.REASON_STOP_BEFORE_START,
                f"{int(reversed_rows.sum())} row(s) have stop_datetime at or before "
                f"start_datetime — spreadsheet row(s) {', '.join(map(str, rows))}; "
                f"e.g. start {starts[first]} -> stop {stops[first]}.",
            )

    def expand_metadata(self):
        """
        Read the metadata file and, if no 'region_id' column is present,
        expand each row to 32 rows (one per DAM channel).

        Also computes the fly 'id' column as YYYYMMDD_Monitor_Region, which
        uniquely identifies each fly across experiments.

        Returns
        -------
        pd.DataFrame
            Expanded metadata with one row per fly.
        """
        print("--- Step 1: Loading and Expanding Metadata ---")

        # Support both CSV and Excel formats. A read failure here is reported as a
        # metadata problem rather than a raw pandas traceback: from the user's side
        # "the metadata file is not readable" is the actionable fact, and the
        # underlying error is kept as the detail.
        if self.metadata_path.endswith(".csv"):
            reader, kind = pd.read_csv, "CSV"
        elif self.metadata_path.endswith((".xlsx", ".xls")):
            reader, kind = pd.read_excel, "Excel"
        else:
            raise MetadataError(
                import_diagnostics.REASON_METADATA_UNREADABLE,
                f"'{os.path.basename(self.metadata_path)}' is not a .csv or .xlsx/.xls file.",
            )
        try:
            meta_df = reader(self.metadata_path)
        except Exception as e:
            raise MetadataError(
                import_diagnostics.REASON_METADATA_UNREADABLE,
                f"pandas could not read '{os.path.basename(self.metadata_path)}' as {kind}: {e}",
            ) from e

        self._check_metadata_shape(meta_df)

        # Region expansion is PER ROW, not per file (see _parse_region_ids):
        #   * a blank / missing region_id  -> the whole monitor (tubes 1-32)
        #   * a number / range / list      -> exactly those tubes
        # So you can leave most monitors as one blank-region row (auto-expanded)
        # and split ONE monitor across genotypes with a couple of range rows
        # (e.g. region_id '1-16' vs '17-32') — no need to hand-list every other
        # monitor's 32 tubes. A file with NO region_id column still expands every
        # row to 32 tubes (legacy behavior); a file with every region_id filled to
        # a single integer still yields one row per tube (also unchanged).
        has_region_col = "region_id" in meta_df.columns
        if has_region_col:
            print(
                "INFO: 'region_id' column found — expanding per row (blank -> all "
                "32 tubes; a number/range like '1-16' -> those tubes)."
            )
        else:
            print("INFO: 'region_id' column not found. Expanding each row to 32 regions.")
        expanded_rows = []
        for _, row in meta_df.iterrows():
            cell = row["region_id"] if has_region_col else None
            for tube in _parse_region_ids(cell):
                new_row = row.copy()
                new_row["region_id"] = tube
                expanded_rows.append(new_row)
        meta_df = pd.DataFrame(expanded_rows).reset_index(drop=True)
        # Denominator for the import report: what the metadata asked for, before
        # any combo is excluded.
        self.n_flies_requested = len(meta_df)
        print(f"INFO: Will analyze {len(meta_df)} fly/channel combinations.")

        # Ensure correct column types
        meta_df["start_datetime"] = pd.to_datetime(meta_df["start_datetime"])
        meta_df["stop_datetime"] = pd.to_datetime(meta_df["stop_datetime"])
        if "first_DD_day" in meta_df.columns:
            meta_df["first_DD_day"] = pd.to_datetime(meta_df["first_DD_day"])
        # Light-pulse (phase-shift) columns are optional; a blank cell means "this
        # group got no pulse" and must stay NaN rather than becoming a real time.
        if "pulse_duration_min" in meta_df.columns:
            meta_df["pulse_duration_min"] = pd.to_numeric(
                meta_df["pulse_duration_min"], errors="coerce"
            ).astype("float32")
        meta_df["Monitor"] = meta_df["Monitor"].astype(str)
        meta_df["region_id"] = meta_df["region_id"].astype(int)

        # Build unique fly ID: date prefix prevents ID collisions across experiments
        date_prefix = meta_df["start_datetime"].dt.strftime("%Y%m%d")
        meta_df["id"] = (
            date_prefix + "_" + meta_df["Monitor"] + "_" + meta_df["region_id"].astype(str)
        )

        # Guard: with per-row expansion a tube could be claimed by two rows (e.g.
        # overlapping split ranges on the same Monitor+start). That collides on
        # `id` and one row would silently overwrite the other downstream — surface
        # it loudly instead so the metadata gets fixed.
        _dups = meta_df.loc[meta_df["id"].duplicated(keep=False), "id"].unique()
        if len(_dups):
            _shown = ", ".join(map(str, _dups[:10])) + (" ..." if len(_dups) > 10 else "")
            print(
                f"\nWARNING: {len(_dups)} fly id(s) are assigned by more than one "
                f"metadata row (overlapping region_id ranges on the same "
                f"Monitor+start?): {_shown}. Fix the overlap so each tube maps to "
                f"exactly one genotype/region row."
            )

        # Save expanded metadata for review — opt-in only. The legacy behavior
        # wrote 'metadata_all.csv' to the cwd on EVERY load, polluting the tree
        # (e.g. app/metadata_all.csv). Now gated behind write_expanded_csv so
        # the canonical load is side-effect-free; the return value is unchanged.
        if self.write_expanded_csv:
            output_path = self.expanded_csv_path or "metadata_all.csv"
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            meta_df.to_csv(output_path, index=False)
            print(f"INFO: Metadata has been saved to '{output_path}'.")

        return meta_df

    def validate_files_and_dates(self, metadata_df, progress_callback=None):
        """
        For each unique (Monitor, start_datetime) combination in the metadata,
        validate that:
          1. The MonitorXXX.txt file exists.
          2. The requested start/stop range falls within the file's data range.
          3. There are no data gaps exceeding gap_threshold_hours.

        Caches raw file reads so each file is loaded only once, even if it
        appears in multiple metadata rows.

        Parameters
        ----------
        metadata_df : pd.DataFrame
            Output of expand_metadata().
        progress_callback : callable, optional
            Called as callback(completed, total) after each combo is processed.

        Returns
        -------
        validated_metadata : pd.DataFrame
            Metadata with failed combos removed (user is prompted if failures occur).
        all_monitor_data : pd.DataFrame
            DatetimeIndex DataFrame; columns are fly IDs from metadata.

        Side effect: populates ``self.import_issues`` with one entry per reason
        flies were excluded, for ``import_report_lines``.
        """
        print("\n--- Step 2: Validating Monitor Files and Data Integrity ---")
        failed_combos = []
        self.integrity_report = {}
        self.import_issues = []
        dropped_ids = []

        def _fail(reason, combo_row_meta, monitor_id, start_dt, detail, hint=None):
            """Record WHY a combo was excluded, then exclude it.

            Every ``failed_combos.append`` goes through here so a silent drop is
            impossible: the reason, the monitor, and the number of flies it cost
            are captured at the point where the facts are still in hand.
            """
            failed_combos.append((monitor_id, start_dt))
            self.import_issues.append(
                ImportIssue(
                    reason=reason,
                    detail=detail,
                    monitor=monitor_id,
                    start_datetime=start_dt,
                    n_flies=len(combo_row_meta),
                    excluded=True,
                    hint=hint,
                )
            )

        metadata_df["start_datetime"] = pd.to_datetime(metadata_df["start_datetime"])
        metadata_df["stop_datetime"] = pd.to_datetime(metadata_df["stop_datetime"])
        metadata_df["region_id"] = metadata_df["region_id"].astype(int)

        # Iterate over unique (Monitor, start_datetime) pairs
        unique_combos = (
            metadata_df[["Monitor", "start_datetime"]]
            .drop_duplicates()
            .sort_values(["Monitor", "start_datetime"])
        )

        # Cache raw monitor reads to avoid redundant disk I/O
        monitor_file_cache = {}
        all_monitor_data = pd.DataFrame()

        _combo_total = len(unique_combos)
        for _combo_idx, (_, combo_row) in enumerate(
            tqdm(
                unique_combos.iterrows(), total=_combo_total, desc="Validating Monitor/Start combos"
            )
        ):
            monitor_id = combo_row["Monitor"]
            start_dt = combo_row["start_datetime"]
            if progress_callback:
                progress_callback(_combo_idx + 1, _combo_total)

            combo_meta = metadata_df[
                (metadata_df["Monitor"] == monitor_id) & (metadata_df["start_datetime"] == start_dt)
            ]
            stop_dt = combo_meta["stop_datetime"].max()
            regions_of_interest = sorted(combo_meta["region_id"].unique().tolist())
            combo_label = f"Monitor {monitor_id} / {start_dt}"

            # --- Load (or retrieve from cache) the raw monitor file ---
            if monitor_id not in monitor_file_cache:
                monitor_filename = f"Monitor{monitor_id}.txt"
                monitor_filepath = os.path.join(self.data_folder, monitor_filename)

                if not os.path.exists(monitor_filepath):
                    print(f"\nWARNING: {combo_label}: File not found at '{monitor_filepath}'.")
                    # List what IS in the folder: a Monitor number typo and a
                    # wrong data directory look identical until you see this.
                    present = sorted(
                        f
                        for f in os.listdir(self.data_folder)
                        if f.lower().startswith("monitor") and f.lower().endswith(".txt")
                    )
                    if present:
                        shown = ", ".join(present[:12]) + (" ..." if len(present) > 12 else "")
                        found = f"Files present in that folder: {shown}."
                    else:
                        found = "That folder contains no Monitor*.txt files at all."
                    _fail(
                        import_diagnostics.REASON_FILE_MISSING,
                        combo_meta,
                        monitor_id,
                        start_dt,
                        f"expected '{monitor_filename}' in '{self.data_folder}'. {found}",
                    )
                    continue

                try:
                    raw_data = pd.read_csv(monitor_filepath, sep="\t", header=None)
                    fixed_columns = [
                        "index",
                        "date",
                        "time",
                        "monitor_status",
                        "extras",
                        "monitor_number",
                        "tube_number",
                        "data_type",
                        "unused",
                        "light_status",
                    ]
                    num_channels = raw_data.shape[1] - 10
                    all_activity_columns = [f"channel_{i}" for i in range(1, num_channels + 1)]
                    raw_data.columns = fixed_columns + all_activity_columns
                    raw_data["datetime"] = pd.to_datetime(
                        raw_data["date"] + " " + raw_data["time"], format="%d %b %y %H:%M:%S"
                    )
                    # Snap reading timestamps to the 1-minute grid. Trikinetics DAM
                    # files are already minute-aligned (HH:MM:00), so this is a no-op
                    # for them. Some recorders (e.g. the FlyBox 96-well monitor) write
                    # a drifting sub-minute seconds field (HH:MM:SS with SS creeping
                    # 06 -> 07 -> ... -> 59 -> 00 across the record); left unrounded,
                    # those timestamps never match the :00 minute grid that
                    # convert_to_relative_time and dam_integrity build by exact label,
                    # so nearly every reading would reindex to NaN (fake data loss).
                    # The whole pipeline operates on a 1-minute grid (freq="1min"
                    # throughout), so rounding to the nearest minute is the correct
                    # normalization. A rare collision from two readings rounding to the
                    # same minute is resolved by resolve_status_and_duplicates (keep
                    # the status==1 row, first wins).
                    raw_data["datetime"] = raw_data["datetime"].dt.round("min")
                    monitor_file_cache[monitor_id] = raw_data
                    print(f"  Monitor {monitor_id}: Loaded file with {num_channels} channels")
                except Exception as e:
                    print(f"\nWARNING: {combo_label}: Failed to read or parse file. Error: {e}")
                    _fail(
                        import_diagnostics.REASON_FILE_UNREADABLE,
                        combo_meta,
                        monitor_id,
                        start_dt,
                        f"'{monitor_filename}' exists but could not be parsed as a raw "
                        f"DAM file: {type(e).__name__}: {e}",
                    )
                    continue

            monitor_data = monitor_file_cache[monitor_id].copy()
            num_channels = sum(1 for c in monitor_data.columns if c.startswith("channel_"))

            # --- Check that start/stop datetimes are within the file's data range ---
            min_date = monitor_data["datetime"].min()
            max_date = monitor_data["datetime"].max()

            starts_early = start_dt < min_date
            ends_late = stop_dt > max_date
            if starts_early or ends_late:
                print(
                    f"\nWARNING: {combo_label}: start/stop datetimes are outside the available data range."
                )
                print(f"  Metadata Range: {start_dt} to {stop_dt}")
                print(f"  Available Range: {min_date} to {max_date}")

                # Name WHICH bound is wrong and BY HOW MUCH. The motivating case
                # was a file truncated to 09:01 against a 09:00 start — a
                # one-minute overhang that silently cost the whole monitor, and
                # is indistinguishable from a wrong-year typo in a report that
                # only says "outside the available data range".
                file_range = f"the file covers {min_date} -> {max_date}"
                if starts_early and ends_late:
                    reason = import_diagnostics.REASON_WINDOW_OUTSIDE
                    detail = (
                        f"metadata asks for {start_dt} -> {stop_dt}, which overhangs "
                        f"the file at both ends by "
                        f"{import_diagnostics.format_timedelta(min_date - start_dt)} at the "
                        f"start and {import_diagnostics.format_timedelta(stop_dt - max_date)} "
                        f"at the end; {file_range}."
                    )
                    hint = None
                elif starts_early:
                    short_by = min_date - start_dt
                    reason = import_diagnostics.REASON_WINDOW_STARTS_EARLY
                    detail = (
                        f"metadata start_datetime {start_dt} is "
                        f"{import_diagnostics.format_timedelta(short_by)} before the "
                        f"file's first reading ({min_date}); {file_range}."
                    )
                    if short_by < timedelta(days=1):
                        # A small overhang is almost always a truncated export.
                        # The fix is NOT to snap start_datetime to the file's
                        # first reading: start_datetime defines ZT0, so moving it
                        # by a few minutes re-bases every ZT bin in the analysis.
                        hint = (
                            f"The file is missing only the first "
                            f"{import_diagnostics.format_timedelta(short_by)} of the "
                            f"requested window — a truncated export usually explains a "
                            f"gap this small. Either re-export Monitor{monitor_id}.txt so "
                            f"it reaches back to {start_dt}, or move start_datetime "
                            f"forward. Do NOT simply set it to {min_date}: "
                            f"start_datetime is ZT0 (lights-on) and shifting it by "
                            f"minutes re-bases every ZT bin — move it a whole day, to "
                            f"{start_dt + timedelta(days=1)}."
                        )
                    else:
                        hint = (
                            f"The requested start is {import_diagnostics.format_timedelta(short_by)} "
                            f"before anything in this file — too far to be a truncated "
                            f"export. Check the year and month in start_datetime, and "
                            f"that Monitor {monitor_id} is the intended file for this "
                            f"experiment."
                        )
                else:
                    over_by = stop_dt - max_date
                    reason = import_diagnostics.REASON_WINDOW_ENDS_LATE
                    detail = (
                        f"metadata stop_datetime {stop_dt} is "
                        f"{import_diagnostics.format_timedelta(over_by)} after the file's "
                        f"last reading ({max_date}); {file_range}."
                    )
                    hint = (
                        f"The recording ends {import_diagnostics.format_timedelta(over_by)} "
                        f"short of the requested window. Set stop_datetime to {max_date} "
                        f"or earlier, or re-export the file so it runs to {stop_dt}."
                    )
                _fail(reason, combo_meta, monitor_id, start_dt, detail, hint=hint)
                continue

            # Slice to the relevant time window
            monitor_data = monitor_data[
                (monitor_data["datetime"] >= start_dt) & (monitor_data["datetime"] <= stop_dt)
            ]

            # The window is inside the file's overall span but holds no rows —
            # the file jumps straight across it. Caught here so it cannot reach
            # the integrity scan as an all-NaN frame.
            if monitor_data.empty:
                print(f"\nWARNING: {combo_label}: no readings inside the requested window.")
                _fail(
                    import_diagnostics.REASON_NO_ROWS_IN_WINDOW,
                    combo_meta,
                    monitor_id,
                    start_dt,
                    f"the file covers {min_date} -> {max_date} but contains no rows at "
                    f"all between {start_dt} and {stop_dt}.",
                )
                continue

            # --- Data-quality handling (dam_integrity, §2a) ---------------------
            # Capture the real-data (status==1) timestamps BEFORE resolution so the
            # cosmetic/data-loss classification and the time-integrity scan can tell
            # "a real reading survived here" from "this slot is a NaN hole".
            activity_cols = [c for c in monitor_data.columns if str(c).startswith("channel_")]
            _status_is_valid = (
                monitor_data["monitor_status"].astype("Int64") == dam_integrity.VALID_STATUS
            )
            status1_times = monitor_data.loc[_status_is_valid, "datetime"].values
            error_times = monitor_data.loc[~_status_is_valid, "datetime"].values

            # PIECE 1: status rule + de-duplicate the timestamp axis. Any row with
            # monitor_status != 1 is NOT real data → its activity becomes NaN, never
            # zero (a status-51 absent monitor reads all-zero, but that 0 means
            # "no reading", not "no movement"). At a timestamp a status==1 row wins;
            # a slot with only status!=1 rows survives as one NaN row. This resolves
            # the duplicate-label reindex downstream and subsumes the status-50
            # placeholder doubling (no raw-file edit needed).
            monitor_data, _status_info = dam_integrity.resolve_status_and_duplicates(
                monitor_data,
                status_col="monitor_status",
                time_col="datetime",
                activity_cols=activity_cols,
            )

            all_dedup_times = monitor_data["datetime"].values

            # PIECE 2: time-integrity scan (report, do not absorb). Formalizes the
            # existing self.gap_threshold concept.
            _scan = dam_integrity.scan_time_integrity(
                all_dedup_times,
                status1_times=status1_times,
                gap_threshold=self.gap_threshold,
            )

            # PIECE 3: cosmetic vs data-loss classification on the expected grid.
            # The grid spans the metadata window so a file that ended before the
            # window stop has its trailing absent slots counted as data-loss.
            _classification = dam_integrity.classify_irregularities(
                all_dedup_times,
                status1_times,
                error_times=error_times,
                interval_minutes=_scan.get("interval_minutes"),
                grid_start=start_dt,
                grid_end=stop_dt,
            )

            # Accumulate per-monitor integrity for the surfaced report.
            self.integrity_report[monitor_id] = {
                "status": _status_info,
                "scan": _scan,
                "classification": _classification,
            }
            _report_text = dam_integrity.format_monitor_report(
                monitor_id, _status_info, _scan, _classification
            )
            if _report_text:
                print(_report_text)
            # -------------------------------------------------------------------

            # --- Extract columns for the regions of interest, named by fly ID ---
            selected_columns = []
            selected_regions = []
            missing_regions = []
            for region in regions_of_interest:
                source_col = f"channel_{region}"
                # Use the metadata 'id' as the column name for uniqueness
                row_id = combo_meta.loc[combo_meta["region_id"] == region, "id"].values[0]

                if source_col in monitor_data.columns:
                    monitor_data[row_id] = monitor_data[source_col]
                    selected_columns.append(row_id)
                    selected_regions.append(region)
                else:
                    print(f"\nWARNING: {combo_label}: Region {region} not found in data file.")
                    missing_regions.append(region)
                    # Drop the metadata row too. Left in, it made the metadata one
                    # row longer than the activity matrix, and the xarray build
                    # then failed with a 'conflicting sizes for dimension id'
                    # message that named neither the monitor nor the tube.
                    dropped_ids.append(row_id)

            # ONE issue per monitor, not one per tube: a 96-well metadata row
            # against a 32-channel file would otherwise emit 64 near-identical
            # lines. The tubes are named as ranges so the entry still says
            # exactly which ones.
            if missing_regions:
                self.import_issues.append(
                    ImportIssue(
                        reason=import_diagnostics.REASON_REGION_NOT_IN_FILE,
                        detail=(
                            f"region_id "
                            f"{import_diagnostics.format_number_ranges(missing_regions)} "
                            f"requested, but 'Monitor{monitor_id}.txt' has only "
                            f"{num_channels} channels."
                        ),
                        monitor=monitor_id,
                        start_datetime=start_dt,
                        n_flies=len(missing_regions),
                        excluded=True,
                    )
                )

            monitor_data = monitor_data[["datetime"] + selected_columns].copy()
            monitor_data.set_index("datetime", inplace=True)
            print(
                f"  {combo_label}: Selected {len(selected_columns)} channels out of {num_channels} total"
            )

            # A channel that is entirely NaN across the window carries no data at
            # all (every read failed, or the tube was never populated). The flies
            # are still imported — that is the existing behaviour and curation
            # handles them — but the import must say so, because otherwise the
            # symptom is an import that "worked" and analyses that come out empty.
            # Grouped per monitor, with the dead tubes named as ranges: a whole
            # monitor that never powered on is one line reading "tubes 1-32",
            # not 32 lines, while two dead tubes still read "tubes 5, 19".
            if selected_columns:
                empty_regions = [
                    region
                    for col, region in zip(selected_columns, selected_regions)
                    if monitor_data[col].isna().all()
                ]
                if empty_regions:
                    self.import_issues.append(
                        ImportIssue(
                            reason=import_diagnostics.REASON_NO_USABLE_DATA,
                            detail=(
                                f"{len(empty_regions)} of {len(selected_columns)} selected "
                                f"channels are entirely NaN over {start_dt} -> {stop_dt} "
                                f"(no valid reading at any minute) — tubes "
                                f"{import_diagnostics.format_number_ranges(empty_regions)}."
                            ),
                            monitor=monitor_id,
                            start_datetime=start_dt,
                            n_flies=len(empty_regions),
                            excluded=False,
                        )
                    )

            if all_monitor_data.empty:
                all_monitor_data = monitor_data
            else:
                all_monitor_data = all_monitor_data.merge(
                    monitor_data, left_index=True, right_index=True, how="outer"
                )

        # --- Handle validation failures ---
        if failed_combos:
            print("\nValidation finished with failures.")
            print("Failed (Monitor, start_datetime) combos:")
            for monitor_id, start_dt in sorted(set(failed_combos)):
                print(f"  Monitor {monitor_id} / {start_dt}")

            def _exclude_failed(meta):
                """Drop the failed (Monitor, start_datetime) combos from metadata."""
                failed_df = pd.DataFrame(failed_combos, columns=["Monitor", "start_datetime"])
                merged = meta.merge(
                    failed_df.assign(_fail=True), on=["Monitor", "start_datetime"], how="left"
                )
                return merged[merged["_fail"].isna()].drop(columns="_fail")

            if self.strict:
                # Hard failure for tests/CI: never silently drop data. The reasons
                # travel with the exception so a strict caller gets the same
                # diagnosis the UI does, not just a list of combo tuples.
                raise ValueError(
                    "Validation failed for combos: "
                    f"{sorted(set(failed_combos))}. "
                    "Construct MetadataProcessor(strict=False) to auto-exclude "
                    "them, or interactive=True to be prompted.\n"
                    + import_diagnostics.format_report(self.import_issues)
                )
            elif self.interactive:
                # Legacy CLI behavior: prompt on stdin. Only reachable when a
                # caller explicitly opts in (never the headless default), so it
                # cannot hang an unattended run.
                while True:
                    choice = input(
                        "Do you want to continue with the analysis, excluding failed combos? (y/n): "
                    ).lower()
                    if choice in ["y", "yes"]:
                        print("Continuing analysis...")
                        metadata_df = _exclude_failed(metadata_df)
                        break
                    elif choice in ["n", "no"]:
                        print("Analysis terminated by user.")
                        sys.exit()
            else:
                # Non-interactive default: auto-exclude failed combos and proceed.
                # No input(), no sys.exit() — safe for headless / Streamlit / CI.
                print(
                    "Running non-interactively: excluding the failed combos and "
                    "continuing. Pass interactive=True to be prompted, or "
                    "strict=True to raise instead."
                )
                metadata_df = _exclude_failed(metadata_df)

        # Tubes that the file does not have. Dropped from the metadata as well as
        # from the data so the two stay the same length (see the note at the
        # missing-region branch above).
        if dropped_ids:
            metadata_df = metadata_df[~metadata_df["id"].isin(dropped_ids)]

        required_columns = ["start_datetime", "stop_datetime", "Monitor", "region_id"]
        for col in required_columns:
            if col not in metadata_df.columns:
                raise ValueError(f"Missing required column in metadata: {col}")

        if self.import_issues:
            print("\nValidation finished with issues — see the import report below.")
        else:
            print("\nValidation successful. Files are present and date ranges are valid.")
        print(f"Total channels selected: {len(all_monitor_data.columns)}")
        self._print_integrity_summary()
        self._print_import_report(n_imported=len(all_monitor_data.columns))

        return metadata_df, all_monitor_data

    def import_report_lines(self, n_imported=None):
        """Return why flies were dropped, as ``(severity, text)`` tuples.

        Same shape as ``integrity_summary_lines`` so the UI renders both with one
        loop. The two reports answer different questions and are deliberately
        separate: integrity is about holes INSIDE data that loaded, this is about
        flies that never loaded at all.

        Parameters
        ----------
        n_imported : int, optional
            Channels actually imported. Supplied by the caller (the loader knows
            it, the processor does not keep the frame around) so the report can
            open with "imported N of M".
        """
        return import_diagnostics.summary_lines(
            self.import_issues,
            n_requested=self.n_flies_requested,
            n_imported=n_imported,
        )

    def _print_import_report(self, n_imported=None):
        """Print the import report to the console."""
        text = import_diagnostics.format_report(
            self.import_issues,
            n_requested=self.n_flies_requested,
            n_imported=n_imported,
        )
        if text:
            print("\n" + text)

    def integrity_scalars(self):
        """The aggregate integrity counters as plain ints, for stamping onto
        ``ds.attrs`` so a reloaded ``.nc`` can still report its own quality.

        Deliberately only scalars. The per-monitor structure is nested and its
        spans are only actionable at import time, while the raw files are still
        in hand; NetCDF attributes handle plain numbers cleanly and nested blobs
        (a JSON string, say) badly. Follows the precedent set by
        ``time_regularized`` / ``gaps_filled`` in ``dam_utilities``.

        Returns ``{}`` when no integrity report exists, so a caller can
        ``attrs.update(...)`` unconditionally.
        """
        if not self.integrity_report:
            return {}
        rows = self.integrity_report.values()
        return {
            "integrity_n_status_bad": int(sum(r["status"]["n_status_bad"] for r in rows)),
            "integrity_n_cosmetic_slots": int(
                sum(r["classification"]["n_cosmetic_slots"] for r in rows)
            ),
            "integrity_n_dataloss_slots": int(
                sum(r["classification"]["n_dataloss_slots"] for r in rows)
            ),
            "integrity_n_monitors": int(len(self.integrity_report)),
        }

    def integrity_monitor_reports(self):
        """Per-monitor detail as ``(monitor_id, severity, text)``, sorted by monitor.

        ``dam_integrity.format_monitor_report`` already builds this text; until
        now it was printed to the console and nowhere else, so anyone not running
        the app from a terminal never saw which monitor a gap was in. Monitors
        with nothing to report are omitted rather than listed as clean, so the
        expander shows only what needs attention.

        severity is ``"warning"`` when the monitor lost real data and ``"info"``
        when its irregularities were cosmetic — the same distinction
        :meth:`integrity_summary_lines` draws in aggregate.
        """
        out = []
        # NUMERIC where possible, falling back to string. `key=str` would order
        # monitors 2 and 10 as "10" < "2" — cosmetic here, but it is the same
        # string-vs-numeric trap that caused the metadata mispairing this
        # codebase already had (see tests/test_monitor_label_pairing.py), and a
        # report the user scans monitor-by-monitor should not reintroduce it.
        # The (kind, value) tuple keeps ints and non-numeric ids from comparing
        # against each other, which would raise.
        def _monitor_sort_key(monitor_id):
            try:
                return (0, int(monitor_id), "")
            except (TypeError, ValueError):
                return (1, 0, str(monitor_id))

        for monitor_id in sorted(self.integrity_report, key=_monitor_sort_key):
            r = self.integrity_report[monitor_id]
            text = dam_integrity.format_monitor_report(
                monitor_id, r["status"], r["scan"], r["classification"]
            )
            if not text:
                continue
            severity = (
                "warning" if r["classification"].get("n_dataloss_slots", 0) else "info"
            )
            out.append((monitor_id, severity, text))
        return out

    def integrity_summary_lines(self):
        """Return the aggregate data-integrity summary as ``(severity, text)``
        tuples for both console and UI consumption (the FileScan role).

        severity is one of ``"info"`` / ``"warning"`` / ``"error"``: cosmetic
        irregularities (a real status-1 reading survived) are info; DATA-LOSS
        holes (no status-1 at a slot → NaN) are warnings the user must see.
        """
        if not self.integrity_report:
            return []
        total_bad = sum(r["status"]["n_status_bad"] for r in self.integrity_report.values())
        total_cos = sum(
            r["classification"]["n_cosmetic_slots"] for r in self.integrity_report.values()
        )
        total_dl = sum(
            r["classification"]["n_dataloss_slots"] for r in self.integrity_report.values()
        )
        lines = []
        if total_bad == 0 and total_dl == 0:
            lines.append(("info", "Data integrity: all monitors clean (no failed reads, no gaps)."))
            return lines
        lines.append(
            (
                "info",
                "Data integrity (status rule: only status==1 is real data; "
                "every other code -> NaN, never zero):",
            )
        )
        lines.append(("info", f"  Non-status-1 rows handled (->NaN): {total_bad}"))
        lines.append(
            ("info", f"  Cosmetic slots (real reading survived, 0 data lost): {total_cos}")
        )
        # Aggregate codes beyond the documented set {1, 24, 50, 51}, summed across
        # monitors, with the per-monitor breakdown. Same NaN handling — surfaced so
        # the frequency/pattern of undocumented codes (52/53/55, stray 49, ...) is
        # trackable over time. Not an error, just visibility.
        undoc_total = {}
        undoc_by_mon = {}
        for mon, r in self.integrity_report.items():
            u = r["status"].get("undocumented_status_counts", {})
            if u:
                undoc_by_mon[mon] = u
                for c, n in u.items():
                    undoc_total[c] = undoc_total.get(c, 0) + n
        if undoc_total:
            tot_str = ", ".join(f"status-{c} x{n}" for c, n in sorted(undoc_total.items()))
            lines.append(
                (
                    "info",
                    f"  Undocumented status codes (beyond 1/24/50/51), "
                    f"handled as no-data -> NaN, tracked for visibility: {tot_str}",
                )
            )
            for mon, u in undoc_by_mon.items():
                mon_str = " + ".join(f"{n} status-{c}" for c, n in sorted(u.items()))
                lines.append(("info", f"    monitor {mon}: {mon_str} rows -> NaN"))
        if total_dl:
            lines.append(
                (
                    "warning",
                    f"DATA-LOSS: {total_dl} grid slots have no "
                    f"valid reading -> stored as NaN (genuine holes).",
                )
            )
            for mon, r in self.integrity_report.items():
                ndl = r["classification"]["n_dataloss_slots"]
                if ndl:
                    interval = r["scan"].get("interval_minutes", 1.0)
                    spans = [
                        s
                        for s in r["classification"]["dataloss_spans"]
                        if s["n_slots"] * interval >= 60.0
                    ]
                    lines.append(("warning", f"  monitor {mon}: {ndl} missing readings (NaN)"))
                    for s in spans[:8]:
                        dur_h = s["n_slots"] * interval / 60.0
                        lines.append(
                            (
                                "warning",
                                f"    {s['start']} -> {s['end']}: "
                                f"{s['n_slots']} missing (~{dur_h:.1f} h)",
                            )
                        )
        return lines

    def _print_integrity_summary(self):
        """Print the aggregate data-integrity summary to the console."""
        lines = self.integrity_summary_lines()
        if not lines:
            return
        print("\n--- Data integrity summary ---")
        for severity, text in lines:
            prefix = "  [!] " if severity == "warning" else "  "
            print(f"{prefix}{text.strip()}")

    def run(self, progress_callback=None):
        """
        Execute the full metadata processing and validation workflow.

        Parameters
        ----------
        progress_callback : callable, optional
            Passed through to validate_files_and_dates for UI progress reporting.

        Returns
        -------
        validated_metadata : pd.DataFrame
        all_monitor_data : pd.DataFrame
        """
        expanded_df = self.expand_metadata()
        validated_df, all_monitor_data = self.validate_files_and_dates(
            expanded_df, progress_callback=progress_callback
        )
        return validated_df, all_monitor_data
