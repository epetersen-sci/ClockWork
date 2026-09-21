"""Shared export helpers: the 'Save to working folder' buttons, and the one
canonical layout for the wide ZT summary table.

Every line/bar graph's underlying data (the numbers actually plotted) is written
as a CSV into the working folder — ``dam_utilities.resolve_export_dir`` (the
metadata file's directory on a raw load, or the ``.nc``'s directory on a reload,
so exports land next to the user's experiment files and survive a ``.nc`` reload)
— under a ``Graph Exports_<experiment>/`` subfolder, instead of a browser download.
The experiment name comes from the metadata file, so two runs sharing a working
folder do not overwrite each other. One button per graph; a confirmation shows the
written path.

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

    # No dtype repair needed. The trim used to cast integer vars to float so it
    # could write NaN sentinels, and this function cast them back — a round trip
    # that existed only because the masking could not express "missing" for an
    # int. _select_longest_segments now masks by dimension name and writes each
    # dtype's own sentinel (-1 for the int8 sleep masks), so they arrive here
    # already int8. Backlog item 16.
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


def _export_dir(ds, subfolder=None):
    """``Graph Exports_<experiment>/`` in the working folder, optionally one deeper.

    The folder carries the experiment's name (from the metadata file — see
    ``dam_utilities.experiment_suffix``) because a paired run keeps both metadata
    files in ONE working folder, so an unsuffixed ``Graph Exports/`` had the second
    experiment's figures overwriting the first's. A dataset with no experiment name
    still gets the plain ``Graph Exports/``.

    ``subfolder`` keeps one page's output together (``"Sleep_bouts"``) instead of
    every page emptying into one flat directory.
    """
    import streamlit as st

    import dam_utilities

    base = dam_utilities.resolve_export_dir(ds, st.session_state.get("working_dir"))
    out = os.path.join(base, f"Graph Exports{dam_utilities.experiment_suffix(ds)}")
    if subfolder:
        out = os.path.join(out, subfolder)
    os.makedirs(out, exist_ok=True)
    return out


def save_df_button(label, df, ds, filename, key, *, index=False, help=None, subfolder=None):
    """Render a 'Save to working folder' button. On click, write ``df`` as CSV into
    the working data folder's ``Graph Exports/`` and confirm the path. Returns the
    written path, or None (not clicked / empty df)."""
    import streamlit as st

    disabled = df is None or (hasattr(df, "empty") and df.empty)
    if st.button(label, key=key, help=help, disabled=disabled):
        try:
            path = os.path.join(_export_dir(ds, subfolder), filename)
            df.to_csv(path, index=index)
            st.success(f"Saved to `{path}`")
            return path
        except Exception as e:
            st.error(f"Export failed: {e}")
    return None


def save_multi_df_button(label, items, ds, key, *, index=False, help=None, subfolder=None):
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
            out = _export_dir(ds, subfolder)
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


def save_csv_button(label, csv_text, ds, filename, key, *, help=None, subfolder=None):
    """Same as :func:`save_df_button` but for pre-rendered CSV text (a ``to_csv()``
    string). Returns the written path, or None."""
    import streamlit as st

    disabled = not csv_text
    if st.button(label, key=key, help=help, disabled=disabled):
        try:
            path = os.path.join(_export_dir(ds, subfolder), filename)
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(csv_text)
            st.success(f"Saved to `{path}`")
            return path
        except Exception as e:
            st.error(f"Export failed: {e}")
    return None


def _remember(key, message):
    """Keep a "saved" confirmation alive past the rerun that follows the click.

    ``st.success`` inside a button branch is drawn once and gone on the next
    rerun — and a rerun is exactly what a Streamlit button causes. For a CSV the
    message is a nicety; for a button that writes a dozen PNGs it is the only
    report of where they went, so it has to survive.
    """
    import streamlit as st

    st.session_state[f"_saved_note_{key}"] = message


def _show_remembered(key):
    """Redraw the remembered confirmation, if this button has one."""
    import streamlit as st

    note = st.session_state.get(f"_saved_note_{key}")
    if note:
        st.success(note)


def save_figures_png_button(label, figures, ds, key, *, scale=2, subfolder=None, help=None):
    """Write every ``(filename, plotly figure)`` in ``figures`` as a PNG, in one click.

    The modebar's own camera button downloads ONE figure into the browser's
    download folder, which no page can redirect (``ui.charts`` configures its name
    and resolution, which is as far as a web page may go). This is the other half:
    every figure on the page at once, written into the working folder beside the
    experiment's data like every other export here.

    Needs ``kaleido``, plotly's static image backend. It is in requirements.txt, so
    a missing one means a broken environment rather than an optional extra — the
    error says which package and stays on screen.
    """
    import streamlit as st

    figures = list(figures or [])
    if st.button(label, key=key, help=help, disabled=not figures):
        try:
            out = _export_dir(ds, subfolder)
            paths = []
            for filename, fig in figures:
                path = os.path.join(out, filename)
                with open(path, "wb") as fh:
                    fh.write(fig.to_image(format="png", scale=scale))
                paths.append(path)
            _remember(key, f"Saved {len(paths)} PNG file(s) to `{out}`")
            _show_remembered(key)
            return paths
        except Exception as e:
            st.error(
                f"PNG export failed: {e}. Static image export needs the "
                "`kaleido` package (`pip install kaleido`)."
            )
            return None
    _show_remembered(key)
    return None


def save_excel_button(label, sheets, ds, filename, key, *, help=None, subfolder=None):
    """Write a multi-sheet ``.xlsx`` into the working folder, in one click.

    ``filename`` is a STEM: ``.xlsx`` is appended when it carries no extension, so
    no caller has to remember the format it asked for.

    ``sheets`` is an iterable of ``(sheet_name, df)``, or a zero-argument callable
    returning one. An empty or None frame is skipped, so a sheet whose analysis
    produced nothing is simply absent rather than present and blank. Returns the
    written path, or None (not clicked / no sheets).

    **Pass a callable when a sheet is expensive to build.** The iterable form is
    evaluated on every rerun, because the argument is built before the button is
    rendered; the callable form runs only on the click. The sleep-SCAMP workbook's
    test sheet is the case that matters â€” it is a fifth of that page's cost, and
    computing it just to have a button sit unclicked is a fifth of every rerun.

    A workbook rather than a folder of CSVs because the sheets are read together â€”
    a summary, the per-fly values behind it, and the tests over those â€” and because
    a reader who opens one of three CSVs cannot tell it apart from the others. The
    confirmation is remembered like the PNG button's, since a click reruns the page
    and would otherwise take the only record of where the file went with it.

    Needs ``openpyxl``, which is in requirements.txt: a missing one is a broken
    environment, not an optional extra, so the error names the package.
    """
    import pandas as pd
    import streamlit as st

    def _resolve():
        built = sheets() if callable(sheets) else sheets
        return [
            (name, df)
            for name, df in (built or [])
            if df is not None and not (hasattr(df, "empty") and df.empty)
        ]

    # A callable is assumed to have something to write, since asking would mean
    # building it. The iterable form can be checked, so an empty one disables.
    has_sheets = True if callable(sheets) else bool(_resolve())
    if st.button(label, key=key, help=help, disabled=not has_sheets):
        try:
            resolved = _resolve()
            if not resolved:
                st.info("Nothing to export.")
                return None
            path = os.path.join(_export_dir(ds, subfolder), filename)
            # Callers name the workbook, not the format — they pass "sleep_totals",
            # not "sleep_totals.xlsx". Without this the file lands with no
            # extension, which Windows will not open and which looks for all the
            # world like a successful export: the button even reports where it
            # went.
            if not os.path.splitext(path)[1]:
                path += ".xlsx"
            with pd.ExcelWriter(path, engine="openpyxl") as xl:
                for name, df in resolved:
                    # Keep the index when it carries something. A ZT table is
                    # indexed by the bin and has (group, stat) MultiIndex columns
                    # — pandas REFUSES index=False for those outright — while a
                    # per-fly table is a plain RangeIndex whose numbers are noise
                    # in a spreadsheet.
                    keep_index = (
                        isinstance(df.columns, pd.MultiIndex)
                        or isinstance(df.index, pd.MultiIndex)
                        or df.index.name is not None
                    )
                    # Excel's own limit, which openpyxl raises on rather than
                    # trimming. Callers here pass short names, so this is a guard
                    # against a crash, not a naming policy.
                    df.to_excel(xl, sheet_name=str(name)[:31], index=keep_index)
            _remember(key, f"Saved {len(resolved)} sheet(s) to `{path}`")
            _show_remembered(key)
            return path
        except ImportError as e:
            # The one failure the package really explains. openpyxl is in
            # requirements.txt, so a missing one is a broken environment rather
            # than an optional extra, and the error names it.
            st.error(
                f"Excel export failed: {e}. Writing .xlsx needs the `openpyxl` "
                "package (`pip install openpyxl`)."
            )
            return None
        except Exception as e:
            # Everything else reports ITSELF. This used to blame openpyxl for any
            # exception at all, so a pandas refusal to write MultiIndex columns
            # came out as "install openpyxl" — with openpyxl installed, and the
            # real cause nowhere on screen.
            st.error(f"Excel export failed: {e}")
            return None
    _show_remembered(key)
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


def with_group_means(df, *, group_col="Group", id_col="ID", label="Mean"):
    """One ``Mean`` row after each group's block, so the sheet reads by eye.

    A per-fly sheet is a wall of numbers with no landmark in it. A mean row
    closes each genotype's section and answers the first question anyone asks of
    the block above it, without opening a second file to find out.

    The label goes in ``id_col`` rather than in a column of its own, so the sheet
    still loads as a table: a reader filtering ``ID != "Mean"`` gets exactly the
    flies back. Non-numeric columns are left blank on the mean row rather than
    carrying the first fly's value, which would read as data.
    """
    import pandas as pd

    if df is None or df.empty or group_col not in df.columns:
        return df
    numeric = [c for c in df.columns if c not in (group_col, id_col)]
    blocks = []
    for grp, block in df.groupby(group_col, sort=True):
        blocks.append(block)
        row = {group_col: grp, id_col: label}
        for c in numeric:
            vals = pd.to_numeric(block[c], errors="coerce")
            row[c] = vals.mean() if vals.notna().any() else ""
        blocks.append(pd.DataFrame([row], columns=df.columns))
    return pd.concat(blocks, ignore_index=True)


def per_fly_wide(long_df, *, value_col, bin_col="zt_hours", id_col="ID", group_col="Group"):
    """A long per-fly time course as ONE ROW PER FLY, one column per bin.

    The long form — a row per fly per ZT bin — is the right shape for a stats
    package and the wrong one for a person: a 36-fly recording at 30-minute bins
    is 1,728 rows in a single column of numbers, and reading one fly's day means
    scrolling past every other fly's.

    Wide, with a mean row closing each group (see :func:`with_group_means`), is
    the shape you can actually look at. The long form is not lost — it is what
    the group summary sheet beside it is built from.
    """

    if long_df is None or long_df.empty:
        return long_df
    wide = long_df.pivot_table(
        index=[group_col, id_col], columns=bin_col, values=value_col, aggfunc="mean"
    )
    # ZT-labelled headers: a bare 7.5 in a header row is not obviously an hour.
    wide.columns = [f"ZT{c:g}" if isinstance(c, (int, float)) else str(c) for c in wide.columns]
    wide = wide.reset_index()
    wide.columns.name = None
    return with_group_means(wide, group_col=group_col, id_col=id_col)
