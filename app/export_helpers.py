"""Shared export helpers: the 'Save to working folder' buttons, and the one
canonical layout for the wide ZT summary table.

Every line/bar graph's underlying data (the numbers actually plotted) is written
as a CSV into the working folder — ``dam_utilities.resolve_export_dir`` (the
metadata file's directory on a raw load, or the ``.nc``'s directory on a reload,
so exports land next to the user's experiment files and survive a ``.nc`` reload)
— under a ``Graph Exports/`` subfolder, instead of a browser download. One button
per graph; a confirmation shows the written path.

Used by the Periodograms, Period Analysis, Sleep and Activity, and HMM pages so the
export mechanism is defined ONCE (no per-page drift). ``streamlit`` and
``dam_utilities`` are imported INSIDE the functions so this module always uses
whichever streamlit the calling page is running under, rather than binding one at
import time.

(This paragraph used to describe "the headless stub the page-smoke tests swap in".
There was no such stub and no such tests — the suite it referred to had been gone
long enough that only the comment remained. ``tests/`` now drives the real
streamlit through ``st.testing.v1.AppTest``, which needs no stub: under AppTest a
button is False until a test clicks it, so no file is written by accident.)

:func:`zt_group_summary_table` lives here for the same reason: Export data and
Sleep & activity both ship a group Mean/SD/N-per-ZT-bin CSV, and they had drifted
apart on both the casing and the column order of that header (see below). One
builder means they cannot drift again.
"""

import os

#: ``(id, time)`` vars stored as int8 with ``-1`` for "no data". Slicing pads with
#: NaN, which forces a float upcast, so they are restored after a phase slice.
_INT8_SLEEP_VARS = ("sleep", "sleep_short", "sleep_intermediate", "sleep_long")


def phase_slice(ds, phase):
    """Physically slice ``ds`` into an ``'LD'`` or ``'DD'`` dataset, using the
    split parameters recorded on ``ds`` itself.

    This is what the export pages use instead of the old
    ``st.session_state.dataset_LD`` / ``dataset_DD`` caches. Those were physical
    slices kept alongside the master, which meant four files had to keep them in
    sync and Sleep analysis had to regenerate them on every run. Re-slicing on
    demand costs a few seconds per export and cannot go stale.

    :func:`dam_utilities.select_phase` is NOT a substitute here. It returns a
    NaN-masked view over the *full* time axis, and both consumers need real
    equal-length per-board files: SCAMP's loader requires it, and a per-phase
    ``.nc`` is supposed to contain only that phase's timepoints.

    ``discard_first_dd_day`` is passed for DD only, matching how the split was
    originally applied on the Curate & split page — the LD epoch has no first-DD
    day to drop, and passing it there would be meaningless rather than harmless.
    """
    import dam_utilities

    kwargs = {
        "phase": phase,
        "gap_threshold_minutes": int(ds.attrs.get("gap_threshold_minutes", 60)),
    }
    if phase == "DD":
        kwargs["discard_first_dd_day"] = bool(ds.attrs.get("split_discard_first_dd_day", 0))
    sliced = dam_utilities.split_xarray_dataset(ds, **kwargs)

    # §2b: the trim pads with NaN, which upcasts the int8 sleep masks to float.
    # Restore int8 (padding/missing → -1) so a sliced dataset carries the same
    # dtypes as the master. This used to live in sleep_detection.py, where it ran
    # on every sleep computation to keep the caches consistent; it belongs at the
    # point of slicing, which is now here. Dropping it would silently change the
    # dtype of every per-phase .nc from int8 to float64.
    for var in _INT8_SLEEP_VARS:
        if var in sliced.data_vars:
            sliced[var] = sliced[var].fillna(-1).astype("int8")
    return sliced


# Stat order inside each group's column block. GraphPad's grouped-table layout is
# Mean, SD, N — NOT the alphabetical Mean, N, SD that sorting the pivot's columns
# would give — so the CSV pastes straight into a Prism grouped table.
ZT_STAT_ORDER = ("Mean", "SD", "N")


def zt_group_summary_table(binned_df, value_col, bin_size, *, group_col="group"):
    """Group Mean/SD/N per ZT bin, as the wide ``(group, stat)`` table the ZT
    exports ship.

    ``binned_df`` is the long per-fly frame from the ZT binner: one row per fly
    per bin, carrying ``zt_bin_minute``, ``group_col`` and ``value_col``. The
    result is indexed by ``zt_bin_minute``, has a leading ``('zt_hours', '')``
    column, and orders the rest by group ALPHABETICALLY, then by
    :data:`ZT_STAT_ORDER` within each group.

    Both the casing and that ordering are load-bearing: a downstream GraphPad
    template reads columns positionally, so a header row that says ``mean, n,
    sd`` where the last one said ``Mean, SD, N`` silently pastes the wrong
    numbers into the wrong columns.
    """
    import pandas as pd

    import dam_utilities

    agg = (
        binned_df.groupby(["zt_bin_minute", group_col])[value_col]
        .agg(
            Mean="mean",
            SD=lambda x: x.std(ddof=1),
            N=lambda x: x.notna().sum(),
        )
        .reset_index()
    )
    pivot = agg.pivot_table(
        index="zt_bin_minute", columns=group_col, values=list(ZT_STAT_ORDER)
    )
    # pivot_table nests (stat, group); the export wants (group, stat), so flip the
    # tuples before reindexing onto the explicit order.
    pivot.columns = pd.MultiIndex.from_tuples(
        [(grp, stat) for stat, grp in pivot.columns], names=["group", "stat"]
    )
    pivot = pivot.reindex(
        columns=pd.MultiIndex.from_tuples(
            [(g, s) for g in sorted(agg[group_col].unique()) for s in ZT_STAT_ORDER],
            names=["group", "stat"],
        )
    )
    pivot.insert(0, ("zt_hours", ""), dam_utilities.zt_bin_to_hours(pivot.index, bin_size))
    return pivot


def _export_dir(ds):
    import streamlit as st

    import dam_utilities

    base = dam_utilities.resolve_export_dir(ds, st.session_state.get("working_dir"))
    out = os.path.join(base, "Graph Exports")
    os.makedirs(out, exist_ok=True)
    return out


def save_df_button(label, df, ds, filename, key, *, index=False, help=None):
    """Render a 'Save to working folder' button. On click, write ``df`` as CSV into
    the working data folder's ``Graph Exports/`` and confirm the path. Returns the
    written path, or None (not clicked / empty df)."""
    import streamlit as st

    disabled = df is None or (hasattr(df, "empty") and df.empty)
    if st.button(label, key=key, help=help, disabled=disabled):
        try:
            path = os.path.join(_export_dir(ds), filename)
            df.to_csv(path, index=index)
            st.success(f"Saved to `{path}`")
            return path
        except Exception as e:
            st.error(f"Export failed: {e}")
    return None


def save_multi_df_button(label, items, ds, key, *, index=False, help=None):
    """Render ONE button that writes SEVERAL CSVs into the working folder's
    ``Graph Exports/`` in a single click. ``items`` is an iterable of
    ``(filename, df)``; empty/None dataframes are skipped (e.g. an analysis that
    was not run, so no file is written for it). On success a confirmation lists
    every written path. Returns the list of written paths, or None (not clicked)."""
    import streamlit as st

    items = [
        (fn, df) for fn, df in items if df is not None and not (hasattr(df, "empty") and df.empty)
    ]
    disabled = not items
    if st.button(label, key=key, help=help, disabled=disabled):
        try:
            out = _export_dir(ds)
            written = []
            for fn, df in items:
                path = os.path.join(out, fn)
                df.to_csv(path, index=index)
                written.append(path)
            if written:
                st.success(
                    "Saved {} file(s):\n{}".format(
                        len(written), "\n".join(f"- `{p}`" for p in written)
                    )
                )
            else:
                st.info("Nothing to export (no analyses run).")
            return written
        except Exception as e:
            st.error(f"Export failed: {e}")
    return None


def save_csv_button(label, csv_text, ds, filename, key, *, help=None):
    """Same as :func:`save_df_button` but for pre-rendered CSV text (a ``to_csv()``
    string). Returns the written path, or None."""
    import streamlit as st

    disabled = not csv_text
    if st.button(label, key=key, help=help, disabled=disabled):
        try:
            path = os.path.join(_export_dir(ds), filename)
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(csv_text)
            st.success(f"Saved to `{path}`")
            return path
        except Exception as e:
            st.error(f"Export failed: {e}")
    return None


def save_group_average_scalograms(group_averages, out_dir, ds=None):
    """Write one PNG + CSV per group-averaged scalogram, and return the manifest.

    ``periodograms.wavelet_analysis`` used to do this itself, which meant an
    analysis function could not be called without also deciding where files go —
    and it had to reach into ``plotting`` to do the rendering, a deferred import
    whose only purpose was dodging a circular one. The analysis now returns the
    arrays and this writes them, so the dependency runs app → core rather than
    core → core.

    ``ds``, if given, gets the manifest stamped onto its attrs the way the
    analysis used to, so the record still travels with the dataset.
    """
    import datetime
    import json
    import re

    import pandas as pd

    import plotting

    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    saved = []
    for avg in group_averages:
        # Group labels come from user metadata and end up in filenames.
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(avg["group"])).strip("_") or "group"
        stem = f"averaged_scalogram_{safe}_{avg['phase_label']}_{timestamp}"
        png_path = os.path.join(out_dir, stem + ".png")
        csv_path = os.path.join(out_dir, stem + ".csv")

        plotting.save_group_average_scalogram_png(
            avg["mean_power"],
            avg["period_axis"],
            avg["time_h"],
            group_label=str(avg["group"]),
            n_flies=avg["n_flies"],
            out_png_path=png_path,
            period_range=avg["period_range"],
            phase_label=avg["phase_label"],
        )
        # CSV: rows are periods (h), columns are time (h).
        frame = pd.DataFrame(
            avg["mean_power"], index=avg["period_axis"], columns=avg["time_h"]
        )
        frame.index.name = "period_h"
        frame.columns.name = "time_h"
        frame.to_csv(csv_path)

        saved.append(
            {
                "group": str(avg["group"]),
                "n": avg["n_flies"],
                "png": png_path,
                "csv": csv_path,
                "phase": avg["phase_label"],
            }
        )

    if saved and ds is not None:
        ds.attrs["cwt_group_average_paths"] = json.dumps(saved)
        ds.attrs["cwt_group_average_dir"] = out_dir
    return saved
