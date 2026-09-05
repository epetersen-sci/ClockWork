"""
Sleep & activity — the descriptive plots, in three tabs.

Daily profiles (activity and sleep patterns), Bouts & states (per-fly bout
duration curves and short/intermediate/long totals), and Day/night totals.
Every chart has a matching group-level and per-fly CSV export.

Read-only with respect to the dataset: everything here renders results that the
**Sleep analysis** page computed. That page is where the 5-minute rule runs.
"""


import numpy as np
import pandas as pd
import streamlit as st

import dam_utilities
import export_helpers as ex
import plotting
import sleep_analysis
from analysis_detection import detect_analyses
from dataset_meta import (
    PHASE_DD,
    PHASE_LD,
    dataset_fingerprint,
    dataset_phase,
)
from ui.filters import DISPLAY_GROUPS_KEY, bin_size_sidebar, group_filter_sidebar
from ui.guards import require_dataset

ds = require_dataset()
# ----------------------------------------------------------------
# Cached helpers — heavy plot/aggregation calls are wrapped so a
# Streamlit rerun (sidebar toggle, widget change) doesn't recompute
# them. The first arg of every cached function is a fingerprint tuple
# (see core/dataset_meta.dataset_fingerprint); Streamlit hashes that
# tuple while the leading-underscore ``_ds`` arg is NOT hashed (per
# Streamlit's caching convention). Returning plotly figures from a
# cached function is supported — figures pickle cleanly.
#
# The fingerprint parameter is named ``fp``, NOT ``_fp``. Streamlit's
# rule is purely syntactic — ANY leading-underscore parameter is left
# out of the cache key, not just the dataset one — so naming it ``_fp``
# excluded the very thing it exists to key on, and every one of these
# caches then ignored which flies were in ``ds``. The visible symptom
# was the sidebar group filter appearing to do nothing: narrow the
# groups and the plots kept their old traces until some other cache
# input (e.g. bin size) happened to change.
# ----------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _cached_zt_binned(fp, _ds, value_col, bin_size_minutes):
    """Wrap dam_utilities.get_zt_binned_dataframe with a fingerprint key."""
    return dam_utilities.get_zt_binned_dataframe(_ds, value_col, bin_size_minutes)

@st.cache_data(show_spinner=False)
def _cached_summary_bars(
    fp, _ds, variable, selected_genotypes, selected_temperatures, bin_size_minutes, phase_label
):
    """Cache the per-fly summary computation that feeds the bars figure.
    ``phase_label`` is included in the cache key so DD vs LD relabeling
    invalidates correctly."""
    return plotting.summary_bars(
        _ds,
        variable,
        selected_genotypes=list(selected_genotypes) if selected_genotypes else None,
        selected_temperatures=list(selected_temperatures) if selected_temperatures else None,
        bin_size_minutes=bin_size_minutes,
        phase_label=phase_label,
    )

@st.cache_data(show_spinner=False)
def _cached_daily_pattern(
    fp,
    _ds,
    variable,
    title,
    selected_genotypes,
    selected_temperatures,
    phase_label,
    bin_size_minutes,
):
    """Cache the daily-pattern line plot. ``phase_label`` and
    ``bin_size_minutes`` are part of the cache key, so both DD/LD relabeling and
    the sidebar Bin size control correctly invalidate the cached figure."""
    return plotting.daily_pattern_line(
        _ds,
        variable,
        title,
        selected_genotypes=list(selected_genotypes) if selected_genotypes else None,
        selected_temperatures=list(selected_temperatures) if selected_temperatures else None,
        phase_label=phase_label,
        bin_size_minutes=bin_size_minutes,
    )

@st.cache_data(show_spinner=False)
def _cached_summary_table(
    fp, _ds, variable, selected_genotypes, selected_temperatures, bin_size_minutes, phase_label
):
    """Per-group summary table (mean + SEM per period) for CSV export — the
    SAME numbers plotting.summary_bars draws (one computation, no drift)."""
    return plotting.summary_table(
        _ds,
        variable,
        selected_genotypes=list(selected_genotypes) if selected_genotypes else None,
        selected_temperatures=list(selected_temperatures) if selected_temperatures else None,
        bin_size_minutes=bin_size_minutes,
        phase_label=phase_label,
    )

@st.cache_data(show_spinner=False)
def _cached_bout_duration_lines(
    fp, _ds, method, selected_genotypes, selected_temperatures, show_individual
):
    """Cache the per-fly bout-duration curve computation (KDE/survival curves
    + per-fly summary stats + the group-comparison test) — the same numbers
    drive the plot and both CSV exports below, so they can't drift apart."""
    return plotting.sleep_bout_duration_lines(
        _ds,
        method=method,
        selected_genotypes=list(selected_genotypes) if selected_genotypes else None,
        selected_temperatures=list(selected_temperatures) if selected_temperatures else None,
        show_individual=show_individual,
    )

def _summary_csv(tbl):
    """Friendly-header CSV of the per-group summary table (mean + SEM per period)."""
    return tbl.rename(
        columns={
            "group": "Group",
            "n": "n_flies",
            "All Day": "All Day mean (min)",
            "All Day_sem": "All Day SEM (min)",
            "Day Only": "Day mean (min)",
            "Day Only_sem": "Day SEM (min)",
            "Night Only": "Night mean (min)",
            "Night Only_sem": "Night SEM (min)",
        }
    ).to_csv(index=False)



# The sidebar filter narrows a page-LOCAL view for plotting only. It used to
# be a hazard: sleep analysis lived above it on this same page and wrote back
# to the master dataset, so running it from below the filter would have
# dropped deselected flies for good. That block is now its own page.
analyses = detect_analyses(ds)

# The unified group axis: filter on the single `group` coord defined by the
# metadata columns chosen at import, not on separate per-genotype /
# per-temperature axes. Selecting a subset narrows every plot below.
_grp_cols = dam_utilities.get_group_columns(ds)
_group_vals, _all_groups, selected_groups, ds = group_filter_sidebar(
    ds,
    key=DISPLAY_GROUPS_KEY,
    label="Groups",
    subset=True,
    help=(
        "Groups are defined by the metadata column(s) chosen at import"
        + (f": {', '.join(_grp_cols)}." if _grp_cols else ".")
    ),
)
# The unified group axis supersedes the old per-genotype / per-temperature filters;
# plots receive the already-group-filtered dataset with no further per-column slicing.
selected_genotypes = None
selected_temperatures = None

bin_size = bin_size_sidebar(key="viz_bin_size")

tab_profiles, tab_bouts, tab_totals = st.tabs(
    ["Daily profiles", "Bouts & states", "Day/night totals"]
)

with tab_profiles:
    # ============================================================
    # Daily Activity Pattern
    # ============================================================
    st.subheader("Daily Activity Pattern")
    if "activity" in ds.data_vars:
        _ds_fp = dataset_fingerprint(ds)
        fig = _cached_daily_pattern(
            _ds_fp,
            ds,
            "activity",
            "Daily Activity Pattern",
            tuple(selected_genotypes) if selected_genotypes else None,
            tuple(selected_temperatures) if selected_temperatures else None,
            dataset_phase(ds),
            bin_size,
        )
        # theme=None: let the figure's own styling (black text, transparent bg) drive both
        # the on-screen chart and the "Download plot as PNG" export (see daily_pattern_line).
        st.plotly_chart(fig, width="stretch", theme=None)

        # CSV download of binned data (grouped: mean, SD, n per condition)
        try:
            binned_df = _cached_zt_binned(_ds_fp, ds, "activity", bin_size)
            _id_to_group = {}
            for _fid in binned_df["id"].unique():
                try:
                    _fly = ds.sel(id=_fid)
                    if "group" in ds.coords:
                        _id_to_group[_fid] = str(_fly["group"].item())
                    elif "genotype" in ds.coords and "temperature" in ds.coords:
                        _id_to_group[_fid] = f"{_fly['genotype'].item()}-{_fly['temperature'].item()}"
                    else:
                        _id_to_group[_fid] = "All"
                except Exception:
                    _id_to_group[_fid] = "All"
            binned_df["group"] = binned_df["id"].map(_id_to_group)
            # Same builder as the Export page's ZT table: Mean/SD/N per group, in
            # GraphPad's grouped-table order. This used to sort_index the columns
            # instead, which gave the alphabetical Mean/N/SD — a different header
            # row for the same quantity.
            _pivot = ex.zt_group_summary_table(binned_df, "activity", bin_size)
            csv_act = _pivot.to_csv()
            ex.save_csv_button(
                "Save Binned Activity (group mean±SD±N) to working folder",
                csv_act,
                ds,
                "activity_binned.csv",
                key="dl_act",
            )
            # Per-fly binned time course (long) so other stats can be computed: one row
            # per fly per ZT bin (ID, Group, zt_bin_minute, zt_hours, activity).
            _act_pf = binned_df.rename(columns={"id": "ID", "group": "Group"}).copy()
            _act_pf["zt_hours"] = dam_utilities.zt_bin_to_hours(_act_pf["zt_bin_minute"], bin_size)
            _act_pf = (
                _act_pf[["ID", "Group", "zt_bin_minute", "zt_hours", "activity"]]
                .sort_values(["Group", "ID", "zt_bin_minute"])
                .reset_index(drop=True)
            )
            ex.save_df_button(
                "Save per-fly Binned Activity (for stats) to working folder",
                _act_pf,
                ds,
                "activity_binned_per_fly.csv",
                key="dl_act_perfly",
            )
        except Exception:
            pass

    st.divider()

    if analyses["sleep"]:
        st.subheader("Daily Sleep Pattern")
        _ds_fp_sl = dataset_fingerprint(ds)
        fig = _cached_daily_pattern(
            _ds_fp_sl,
            ds,
            "sleep",
            "Daily Sleep Pattern",
            tuple(selected_genotypes) if selected_genotypes else None,
            tuple(selected_temperatures) if selected_temperatures else None,
            dataset_phase(ds),
            bin_size,
        )
        # theme=None: the figure's black-text / transparent-bg styling drives screen + PNG.
        st.plotly_chart(fig, width="stretch", theme=None)

        try:
            binned_sleep = _cached_zt_binned(_ds_fp_sl, ds, "sleep", bin_size)
            _id_to_group_sl = {}
            for _fid in binned_sleep["id"].unique():
                try:
                    _fly = ds.sel(id=_fid)
                    if "group" in ds.coords:
                        _id_to_group_sl[_fid] = str(_fly["group"].item())
                    elif "genotype" in ds.coords and "temperature" in ds.coords:
                        _id_to_group_sl[_fid] = (
                            f"{_fly['genotype'].item()}-{_fly['temperature'].item()}"
                        )
                    else:
                        _id_to_group_sl[_fid] = "All"
                except Exception:
                    _id_to_group_sl[_fid] = "All"
            binned_sleep["group"] = binned_sleep["id"].map(_id_to_group_sl)
            # Shared builder — see the activity block above.
            _pivot_sl = ex.zt_group_summary_table(binned_sleep, "sleep", bin_size)
            csv_sleep = _pivot_sl.to_csv()
            ex.save_csv_button(
                "Save Binned Sleep (group mean±SD±N) to working folder",
                csv_sleep,
                ds,
                "sleep_binned.csv",
                key="dl_sleep",
            )
            # Per-fly binned time course (long) so other stats can be computed.
            _sl_pf = binned_sleep.rename(columns={"id": "ID", "group": "Group"}).copy()
            _sl_pf["zt_hours"] = dam_utilities.zt_bin_to_hours(_sl_pf["zt_bin_minute"], bin_size)
            _sl_pf = (
                _sl_pf[["ID", "Group", "zt_bin_minute", "zt_hours", "sleep"]]
                .sort_values(["Group", "ID", "zt_bin_minute"])
                .reset_index(drop=True)
            )
            ex.save_df_button(
                "Save per-fly Binned Sleep (for stats) to working folder",
                _sl_pf,
                ds,
                "sleep_binned_per_fly.csv",
                key="dl_sleep_perfly",
            )
        except Exception:
            pass

        st.divider()

with tab_bouts:
    if analyses["sleep"]:
        # Sleep Bout Duration
        st.subheader("Sleep Bout Duration")
        if "duration" in ds.data_vars:
            st.caption(
                "One curve per fly (not one pooled histogram) — a fly with many bouts no "
                "longer outweighs a fly with few, so genotypes overlay cleanly as lines."
            )
            _bd_col1, _bd_col2 = st.columns([2, 1])
            with _bd_col1:
                _bd_method_label = st.radio(
                    "Curve type",
                    ["KDE (log-duration)", "Survival curve (CCDF)"],
                    index=0,
                    horizontal=True,
                    key="bout_curve_method",
                )
            with _bd_col2:
                bout_show_individual = st.checkbox(
                    "Show individual flies", value=True, key="bout_show_individual"
                )
            bout_method = "kde" if _bd_method_label.startswith("KDE") else "survival"

            _ds_fp_bout = dataset_fingerprint(ds)
            bout_fig, bout_curves_df, bout_summary_df, bout_stats = _cached_bout_duration_lines(
                _ds_fp_bout,
                ds,
                bout_method,
                tuple(selected_genotypes) if selected_genotypes else None,
                tuple(selected_temperatures) if selected_temperatures else None,
                bout_show_individual,
            )
            st.plotly_chart(bout_fig, width="stretch", theme=None)

            if bout_stats and np.isfinite(bout_stats.get("pvalue", float("nan"))):
                st.caption(
                    f"{bout_stats['test'].upper()} across groups on per-fly "
                    f"log-mean bout duration: p={bout_stats['pvalue']:.4f} "
                    f"(normality {'passed' if bout_stats['normality_passed'] else 'failed'}, "
                    f"equal variance {'passed' if bout_stats['equal_variance_passed'] else 'failed'})."
                )
                if bout_stats["pairwise"]:
                    st.dataframe(pd.DataFrame(bout_stats["pairwise"]), width="stretch")

            raw_bout_df = sleep_analysis.raw_bout_dataframe(
                ds, selected_genotypes=selected_genotypes, selected_temperatures=selected_temperatures
            )
            # Named apart from the Export page's sleep_bouts.csv on purpose. Both
            # come from raw_bout_dataframe, but this one is filtered to the group
            # selection in the sidebar while that one is every fly — under one
            # filename, whichever the user opened last silently won.
            ex.save_df_button(
                "Save Sleep Bout Data (current group selection) to working folder",
                raw_bout_df,
                ds,
                "sleep_bouts_filtered.csv",
                key="dl_bouts",
            )
            ex.save_df_button(
                "Save per-fly Bout Duration Summary (for stats) to working folder",
                bout_summary_df,
                ds,
                "sleep_bout_duration_summary_per_fly.csv",
                key="dl_bouts_summary",
            )

        st.divider()

        # ============================================================
        # Sleep-State Totals (Abhilash short / intermediate / long)
        # ============================================================
        if any(v in ds.data_vars for v in ("sleep_short", "sleep_intermediate", "sleep_long")):
            st.subheader("Sleep-State Totals by Genotype")
            st.caption(
                "Total time in each sleep state — **short / intermediate / long** bouts "
                "(Abhilash et al. 2026) — per genotype, from the **Sleep State Thresholds** "
                "set in Sleep Analysis above. Group mean per fly ± SEM."
            )
            ss_pct = st.checkbox(
                "Show as % of each fly's classified sleep", value=False, key="ss_state_pct"
            )
            ss_fig, ss_df = plotting.sleep_state_totals_bars(
                ds,
                selected_genotypes=selected_genotypes,
                selected_temperatures=selected_temperatures,
                as_percent=ss_pct,
            )
            if ss_fig is not None:
                st.plotly_chart(ss_fig, width="stretch")
            if ss_df is not None and not ss_df.empty:
                ex.save_df_button(
                    "Save Sleep-State Totals (group mean±SEM) to working folder",
                    ss_df,
                    ds,
                    "sleep_state_totals.csv",
                    key="dl_sleep_states",
                )
                # Per-fly totals (the raw values behind the group bars) for your own stats:
                # one row per fly, one column per state (Short / Intermediate / Long).
                _ss_perfly = plotting.per_fly_sleep_state_totals(
                    ds, selected_genotypes, selected_temperatures, as_percent=ss_pct
                )
                ex.save_df_button(
                    "Save per-fly Sleep-State totals (for stats) to working folder",
                    _ss_perfly,
                    ds,
                    "sleep_state_totals_per_fly.csv",
                    key="dl_sleep_states_perfly",
                )
            st.divider()
    else:
        st.info("Run **Sleep analysis** first — these views read its bout output.")

with tab_totals:
    # ============================================================
    # Summary Bars
    # ============================================================
    _ds_phase_label = dataset_phase(ds)
    _summary_subhead_suffix = (
        " (DD — subjective time)"
        if _ds_phase_label == PHASE_DD
        else (" (LD — Day/Night)" if _ds_phase_label == PHASE_LD else " (Day/Night)")
    )
    st.subheader(f"Activity Summary{_summary_subhead_suffix}")
    if _ds_phase_label == PHASE_DD:
        st.caption(
            "**DD note:** the bin labels are subjective time relative to the "
            "last lights-on transition (CT). Anchoring is reliable when the "
            "DD split was applied with a clean discard-first-DD-day boundary. "
            "If your recording had large data gaps at the DD start, the CT "
            "alignment may drift — verify the actogram before publishing."
        )
    if "activity" in ds.data_vars:
        fig = _cached_summary_bars(
            dataset_fingerprint(ds),
            ds,
            "activity",
            tuple(selected_genotypes) if selected_genotypes else None,
            tuple(selected_temperatures) if selected_temperatures else None,
            bin_size,
            _ds_phase_label,
        )
        st.plotly_chart(fig, width="stretch")
        _act_tbl = _cached_summary_table(
            dataset_fingerprint(ds),
            ds,
            "activity",
            tuple(selected_genotypes) if selected_genotypes else None,
            tuple(selected_temperatures) if selected_temperatures else None,
            bin_size,
            _ds_phase_label,
        )
        if not _act_tbl.empty:
            ex.save_csv_button(
                "Save Activity Summary (group mean±SEM) to working folder",
                _summary_csv(_act_tbl),
                ds,
                "activity_summary.csv",
                key="dl_act_summary",
            )
            # Per-fly totals (the raw values behind the group summary) so you can run
            # your own stats — one row per fly: ID, Group, All Day / Day / Night totals.
            _act_perfly = plotting.per_fly_summary_table(
                ds,
                "activity",
                tuple(selected_genotypes) if selected_genotypes else None,
                tuple(selected_temperatures) if selected_temperatures else None,
                bin_size,
            )
            ex.save_df_button(
                "Save per-fly Activity totals (for stats) to working folder",
                _act_perfly,
                ds,
                "activity_totals_per_fly.csv",
                key="dl_act_summary_perfly",
            )

    if analyses["sleep"]:
        st.subheader(f"Sleep Summary{_summary_subhead_suffix}")
        fig = _cached_summary_bars(
            dataset_fingerprint(ds),
            ds,
            "sleep",
            tuple(selected_genotypes) if selected_genotypes else None,
            tuple(selected_temperatures) if selected_temperatures else None,
            bin_size,
            _ds_phase_label,
        )
        st.plotly_chart(fig, width="stretch")
        _sleep_tbl = _cached_summary_table(
            dataset_fingerprint(ds),
            ds,
            "sleep",
            tuple(selected_genotypes) if selected_genotypes else None,
            tuple(selected_temperatures) if selected_temperatures else None,
            bin_size,
            _ds_phase_label,
        )
        if not _sleep_tbl.empty:
            ex.save_csv_button(
                "Save Sleep Summary (group mean±SEM) to working folder",
                _summary_csv(_sleep_tbl),
                ds,
                "sleep_summary.csv",
                key="dl_sleep_summary",
            )
            # Per-fly totals (the raw values behind the group summary) for your own stats.
            _sleep_perfly = plotting.per_fly_summary_table(
                ds,
                "sleep",
                tuple(selected_genotypes) if selected_genotypes else None,
                tuple(selected_temperatures) if selected_temperatures else None,
                bin_size,
            )
            ex.save_df_button(
                "Save per-fly Sleep totals (for stats) to working folder",
                _sleep_perfly,
                ds,
                "sleep_totals_per_fly.csv",
                key="dl_sleep_summary_perfly",
            )
