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
``dam_utilities`` are imported INSIDE the functions so this module always uses the
page's active streamlit (the real one, or the headless stub the page-smoke tests
swap in — under which ``st.button`` is False, so no file is ever written in a test).

:func:`zt_group_summary_table` lives here for the same reason: Export data and
Sleep & activity both ship a group Mean/SD/N-per-ZT-bin CSV, and they had drifted
apart on both the casing and the column order of that header (see below). One
builder means they cannot drift again.
"""

import os

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
