"""
curate_and_export.py
====================
Standalone CLI: raw Trikinetics MonitorXXX.txt + metadata → curated, LD/DD-split,
SCAMP-loadable folder trees (one per phase).

Usage
-----
    python -m scamp_export.curate_and_export \\
        --metadata path/to/metadata.csv \\
        --data-folder path/to/monitors/ \\
        --out path/to/scamp_ready/ \\
        --prefix EXP240115 \\
        --min-alive-days 2 --time-window 24 --prop-immobile 0.01 \\
        --gap-threshold 60 --discard-first-dd-day \\
        --ld-min-days 2 --dd-min-days 3 \\
        --lights-on-military 900

Alternative: ``--nc path/to/saved.nc`` to skip raw loading and re-curation,
and export an already-prepared phase dataset (e.g. one saved by the Streamlit
app). When ``--nc`` is given the dataset is assumed to be already curated and
phase-split; the phase label is read from the dataset attrs (``phase``) or
overridden via ``--phase``.
"""

from __future__ import annotations

import argparse
import os
import sys

# Make ``core/`` importable when this script is run as a module from the
# project root (``python -m scamp_export.curate_and_export ...``).
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)  # repository root
_CORE = os.path.join(_PROJECT_ROOT, "core")
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)

from scamp_export.scamp_exporter import export_dataset_to_scamp  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Curate, split, and export DAM data into SCAMP-loadable form."
    )
    src = p.add_argument_group("input source")
    src.add_argument(
        "--metadata",
        help="Path to metadata CSV/XLSX (with id, Monitor, "
        "region_id, start_datetime, stop_datetime, "
        "first_DD_day, genotype, …).",
    )
    src.add_argument("--data-folder", help="Folder containing MonitorXXX.txt files.")
    src.add_argument(
        "--nc", help="Optional: path to a saved .nc to skip raw loading (assumed curated + split)."
    )

    p.add_argument(
        "--out", required=True, help="Output root. LD/ and DD/ subfolders are written underneath."
    )
    p.add_argument(
        "--prefix",
        default="PY",
        help="Filename prefix. Must not contain 'C', '.', '/', '\\', or spaces.",
    )
    p.add_argument(
        "--lights-on-military",
        type=int,
        default=900,
        help="Single start (HHMM) written into every file. Default 900 "
        "(09:00). Used by SCAMP only for axis labels.",
    )
    p.add_argument(
        "--phase", choices=("LD", "DD", "both"), default="both", help="Which phase(s) to export."
    )

    cur = p.add_argument_group("curation (ignored when --nc is given)")
    cur.add_argument(
        "--time-window",
        type=float,
        default=24,
        help="Rolling immobility window (hours). Default 24.",
    )
    cur.add_argument(
        "--prop-immobile",
        type=float,
        default=0.01,
        help="Activity threshold for 'alive'. Default 0.01.",
    )
    cur.add_argument(
        "--min-alive-days",
        type=float,
        default=2,
        help="Minimum days alive to retain a fly. Default 2.",
    )

    spl = p.add_argument_group("LD/DD split (ignored when --nc is given)")
    spl.add_argument(
        "--gap-threshold",
        type=float,
        default=60,
        help="Gap threshold (minutes) for longest-segment trim. Default 60.",
    )
    spl.add_argument(
        "--discard-first-dd-day",
        action="store_true",
        help="Discard first 1440 min of DD (handles LD aftereffect).",
    )

    exp = p.add_argument_group("per-phase export")
    exp.add_argument(
        "--ld-min-days", type=float, default=2, help="Min valid span (days) for LD export window."
    )
    exp.add_argument(
        "--dd-min-days",
        type=float,
        default=2,
        help="Min valid span (days) for DD export window. Counted after --discard-first-dd-day.",
    )
    exp.add_argument(
        "--no-interpolate-interior",
        action="store_true",
        help="Write the missing sentinel (-1) for interior NaN rather "
        "than linear-interpolating before writing.",
    )
    exp.add_argument(
        "--intervals",
        default="1,30",
        help="Comma-separated intervals to write. Default '1,30' "
        "matches SCAMP's two-folder prompt.",
    )
    return p


def _print_summary(phase: str, summary: dict) -> None:
    print(f"\n=== {phase} export summary ===")
    print(f"  flies in:      {summary['n_in']}")
    print(f"  flies written: {summary['n_out']}")
    print(f"  dropped:       {len(summary['dropped'])}")
    for d in summary["dropped"]:
        print(f"     - {d['id']}: {d['reason']}")
    print(f"  boards: {len(summary['boards'])}")
    for b in summary["boards"]:
        print(
            f"     {b['file_prefix']}M{b['board_int']}  "
            f"n_flies={b['n_flies']}  window_days={b['window_days']}"
        )
    print(f"  written to:    {summary['out_root']}")
    print(f"  group key:     {summary['group_key']}")
    print(f"  manifest:      {summary['manifest']}")


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    intervals = tuple(int(x) for x in str(args.intervals).split(",") if x.strip())

    if args.nc:
        # Import lazily so the CLI can be used without netCDF deps installed
        from load_and_save_datasets import load_dataset_from_netcdf

        ds = load_dataset_from_netcdf(args.nc)
        phase = ds.attrs.get("phase", args.phase)
        out_dir = os.path.join(args.out, str(phase).upper())
        summary = export_dataset_to_scamp(
            ds,
            out_dir,
            prefix=args.prefix,
            interval_set=intervals,
            lights_on_military=args.lights_on_military,
            min_days=args.ld_min_days if str(phase).upper() == "LD" else args.dd_min_days,
            interpolate_interior=not args.no_interpolate_interior,
            phase_label=str(phase).upper(),
        )
        _print_summary(str(phase).upper(), summary)
        return 0

    if not args.metadata or not args.data_folder:
        print(
            "ERROR: --metadata and --data-folder are required when --nc is not given.",
            file=sys.stderr,
        )
        return 2

    from dam_utilities import (
        convert_to_relative_time,
        create_xarray_dataset,
        curate_dead_animals,
        read_data_and_metadata,
        select_phase,
    )

    print(f"--- Loading raw data from {args.data_folder} ---")
    metadata, raw_df = read_data_and_metadata(args.metadata, args.data_folder)

    print("--- Aligning per-fly to relative-minute axis ---")
    rel_df = convert_to_relative_time(raw_df, metadata)

    print("--- Building xarray dataset ---")
    ds = create_xarray_dataset(rel_df, metadata)
    if ds is None:
        print("ERROR: dataset is empty after time alignment.", file=sys.stderr)
        return 1

    print(
        f"--- Curating dead flies (window={args.time_window}h, "
        f"thresh={args.prop_immobile}, min_alive_days={args.min_alive_days}) ---"
    )
    live, _dead, _err, _ok, n_before, n_after, removed, _unch, _trim = curate_dead_animals(
        ds,
        time_window=args.time_window,
        prop_immobile=args.prop_immobile,
        min_alive_days=args.min_alive_days,
    )
    print(f"    {n_before} → {n_after} flies (removed {removed})")

    phases = ["LD", "DD"] if args.phase == "both" else [args.phase]
    rc = 0
    for phase in phases:
        print(f"\n--- Selecting + exporting {phase} ---")
        # Phase API (Stage-2): use the one core selector (per-fly NaN-masked view
        # of the whole dataset) instead of the legacy physical slicer.
        # export_dataset_to_scamp self-extracts each fly's first-to-last finite
        # block (np.isfinite) and interpolates interior gaps, so the masked view
        # produces the same SCAMP files as the old split for clean data. The
        # longest-segment trim (--gap-threshold) no longer pre-trims: a fly with a
        # large intra-phase gap now exports its full finite span with the gap
        # interpolated, rather than being cut to its longest contiguous segment.
        ds_phase, _ = select_phase(
            live,
            phase=phase,
            discard_first_dd_day=args.discard_first_dd_day and phase == "DD",
        )
        if ds_phase is None or ds_phase["id"].size == 0:
            print(f"  no flies in {phase}; skipping.")
            continue

        out_dir = os.path.join(args.out, phase)
        min_days = args.ld_min_days if phase == "LD" else args.dd_min_days
        try:
            summary = export_dataset_to_scamp(
                ds_phase,
                out_dir,
                prefix=args.prefix,
                interval_set=intervals,
                lights_on_military=args.lights_on_military,
                min_days=min_days,
                interpolate_interior=not args.no_interpolate_interior,
                phase_label=phase,
            )
        except ValueError as e:
            print(f"  ERROR exporting {phase}: {e}", file=sys.stderr)
            rc = 1
            continue
        _print_summary(phase, summary)

    return rc


if __name__ == "__main__":
    sys.exit(main())
