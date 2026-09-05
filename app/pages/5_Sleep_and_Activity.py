"""
Sleep and Activity Page - sleep analysis, LD activity & sleep patterns, per-fly sleep bout
duration curves, per-group activity/sleep summaries, and sleep-state
(short/intermediate/long) totals (interactive Plotly plots +
save-to-working-folder CSV export).
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

# Add core/ (analysis modules) and app/ to the import path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORE_DIR = os.path.join(PROJECT_ROOT, "core")
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [CORE_DIR, APP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

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
    is_split_applied,
)

st.header("Sleep and Activity")

if st.session_state.get("dataset") is None:
    st.warning("No dataset loaded. Go to **Data Loading** first.")
    st.stop()


# ----------------------------------------------------------------
# Cached helpers — heavy plot/aggregation calls are wrapped so a
# Streamlit rerun (sidebar toggle, widget change) doesn't recompute
# them. The first arg of every cached function is a fingerprint tuple
# (see core/dataset_meta.dataset_fingerprint); Streamlit hashes that
# tuple while the leading-underscore ``_ds`` arg is NOT hashed (per
# Streamlit's caching convention). Returning plotly figures from a
# cached function is supported — figures pickle cleanly.
# ----------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _cached_zt_binned(_fp, _ds, value_col, bin_size_minutes):
    """Wrap dam_utilities.get_zt_binned_dataframe with a fingerprint key."""
    return dam_utilities.get_zt_binned_dataframe(_ds, value_col, bin_size_minutes)


@st.cache_data(show_spinner=False)
def _cached_summary_bars(
    _fp, _ds, variable, selected_genotypes, selected_temperatures, bin_size_minutes, phase_label
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
    _fp,
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
    _fp, _ds, variable, selected_genotypes, selected_temperatures, bin_size_minutes, phase_label
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
    _fp, _ds, method, selected_genotypes, selected_temperatures, show_individual
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


# ============================================================
# Sleep Analysis
# ============================================================
# Computes the sleep variables every sleep plot below depends on, and that the
# downstream Sleep Deprivation / HMM / Export pages read off the master dataset.
#
# This deliberately sits ABOVE the sidebar group filter. That filter narrows the
# local ``ds`` for plotting only, while this block writes its result back to
# ``session_state.dataset`` — run from below the filter, deselecting a group and
# hitting Run would silently drop those flies from the master dataset.


@st.fragment
def _sleep_analysis_fragment():
    st.subheader("Sleep Analysis")

    # Outcome of the last successful run, stashed just before the app-scoped rerun
    # below — that rerun discards whatever this fragment had already drawn, so a
    # plain st.success() at the call site would never be seen.
    _last_msg = st.session_state.pop("_sleep_run_message", None)
    if _last_msg:
        st.success(_last_msg)

    ds = st.session_state.dataset
    analyses = detect_analyses(ds)

    if "moving" not in ds.data_vars:
        st.warning(
            "Curate dead animals on the **Preprocessing** page first — curation "
            "computes the movement data that sleep analysis needs."
        )
        return

    # Phase selection for sleep analysis. Read canonical phase metadata
    # from the dataset itself (core/dataset_meta.py) so a DD-only or
    # LD-only NetCDF doesn't trigger the "split not applied" prompt.
    _ds_phase = dataset_phase(ds)
    _has_split_datasets = (
        st.session_state.get("dataset_DD") is not None
        and st.session_state.get("dataset_LD") is not None
    )
    _has_dd_coord = "first_DD_day" in ds.coords

    # Phase API (Stage-2): feed the WHOLE dataset and pass an explicit phase to
    # core sleep_analysis (dam_utilities.select_phase derives the epoch per fly).
    # NEVER feed a pre-sliced/re-zeroed object — the selector would re-mask it.
    # `_sleep_ds` is the dataset handed to sleep_analysis; `_sleep_phase_arg` is
    # the phase argument ("LD"/"DD"/"both"); `_sleep_phase` is the display label.
    if _ds_phase in (PHASE_LD, PHASE_DD):
        # The loaded file is itself a single-phase partition; analyse it as-is
        # (no further masking — the file already IS the phase).
        _sleep_ds = ds
        _sleep_phase = _ds_phase
        _sleep_phase_arg = "both"
        st.info(
            f"Using the loaded **{_sleep_phase}** dataset for sleep analysis "
            f"({len(_sleep_ds['id'])} flies, {len(_sleep_ds['time'])} timepoints)."
        )
    elif _has_split_datasets:
        sleep_phase = st.radio(
            "Data phase for sleep analysis",
            ["LD (recommended)", "DD"],
            index=0,
            horizontal=True,
            help="Sleep analysis is typically performed on LD data where "
            "the light-dark cycle drives consolidated sleep/wake patterns. "
            "DD is an explicit request for constant-darkness sleep.",
            key="sleep_phase_radio",
        )
        _sleep_phase = "LD" if "LD" in sleep_phase else "DD"
        # Feed the whole (unsplit) master + explicit phase — not dataset_LD/DD.
        _sleep_ds = ds
        _sleep_phase_arg = _sleep_phase
        st.info(
            f"Computing **{_sleep_phase}** sleep from the full dataset "
            f"({len(ds['id'])} flies, {len(ds['time'])} timepoints)."
        )
    elif _has_dd_coord and not is_split_applied(ds):
        st.warning(
            "LD/DD split has not been applied yet — it lives on the **Preprocessing** "
            "page. Sleep analysis will run on the **full unsplit dataset** (LD+DD). "
            "For best results, apply the split first so sleep analysis can use LD "
            "data only."
        )
        _sleep_ds = ds
        _sleep_phase = "unsplit"
        _sleep_phase_arg = "both"
    else:
        _sleep_ds = ds
        _sleep_phase = "full"
        _sleep_phase_arg = "both"

    # Check if sleep analysis was already done (on the target dataset)
    if "sleep" in _sleep_ds.data_vars:
        _sleep_sec = _sleep_ds.attrs.get("sleep_threshold_seconds")
        _short_max = _sleep_ds.attrs.get("sleep_short_max_min")
        _inter_max = _sleep_ds.attrs.get("sleep_inter_max_min")
        if _sleep_sec is not None:
            _state_info = ""
            if _short_max is not None and _inter_max is not None:
                _state_info = (
                    f", state thresholds: short <{int(_short_max)} min, "
                    f"intermediate <{int(_inter_max)} min"
                )
            st.info(
                f"Sleep analysis already completed on this dataset — "
                f"**{int(_sleep_sec)}s** threshold ({int(_sleep_sec) // 60} min)"
                f"{_state_info}"
            )
        else:
            st.info("Sleep analysis already completed (parameters not recorded).")
        rerun_sleep = st.checkbox("Re-run sleep analysis with different parameters", value=False)
        if not rerun_sleep:
            return
    elif analyses["sleep"]:
        # Sleep exists on master but not on the phase dataset
        st.info(
            "Sleep analysis was previously run but not on this phase dataset. "
            "Run it below to compute sleep for the selected phase."
        )

    sleep_threshold = st.number_input(
        "Sleep threshold (seconds) - minimum immobility duration to classify as sleep",
        min_value=60,
        max_value=1800,
        value=300,
        step=60,
    )

    with st.expander("Sleep State Thresholds (Abhilash et al. 2026)"):
        short_max_min = st.number_input(
            "Short sleep upper bound (minutes)",
            min_value=5,
            max_value=120,
            value=30,
            step=5,
            help="Bouts 5–N min = short sleep (paper default: 30 min, DAM system).",
            key="short_max_min_input",
        )
        inter_max_min = st.number_input(
            "Intermediate sleep upper bound (minutes)",
            min_value=short_max_min + 1,
            max_value=360,
            value=max(60, short_max_min + 1),
            step=5,
            help="Bouts N–M min = intermediate sleep; >M min = long sleep (paper default: 60 min).",
            key="inter_max_min_input",
        )
        st.caption(
            "These thresholds are provisional and DAM-system specific. "
            "Adjust based on your experimental context and the paper's supplemental methods."
        )

    if st.button("Run Sleep Analysis", key="run_sleep"):
        _run_msg = None
        with st.spinner(f"Running sleep analysis on {_sleep_phase} data..."):
            try:
                _sleep_ds = sleep_analysis.sleep_analysis(
                    _sleep_ds,
                    sleep_threshold_sec=sleep_threshold,
                    short_max_min=short_max_min,
                    inter_max_min=inter_max_min,
                    phase=_sleep_phase_arg,
                )
                # _sleep_ds is now the WHOLE dataset with phase-masked sleep
                # (out-of-phase minutes are -1). The master always carries it.
                st.session_state.dataset = _sleep_ds
                # TRANSITIONAL (retires with the dataset_LD/DD sweep): unmigrated
                # downstream pages still read the pre-sliced dataset_LD/DD, so
                # regenerate the one for the phase just computed from the new sleep
                # result. Same slice params as the Preprocessing split, so
                # activity/moving are identical and now carry the correct per-phase
                # sleep.
                if _has_split_datasets and _sleep_phase in ("LD", "DD"):
                    _gap = int(_sleep_ds.attrs.get("gap_threshold_minutes", 60))
                    if _sleep_phase == "LD":
                        _sliced = dam_utilities.split_xarray_dataset(
                            _sleep_ds, phase="LD", gap_threshold_minutes=_gap
                        )
                    else:
                        _disc = bool(_sleep_ds.attrs.get("split_discard_first_dd_day", 0))
                        _sliced = dam_utilities.split_xarray_dataset(
                            _sleep_ds,
                            phase="DD",
                            discard_first_dd_day=_disc,
                            gap_threshold_minutes=_gap,
                        )
                    # §2b: slicing upcasts the int8 sleep masks to float (NaN trim
                    # padding). Restore int8 (padding/missing → -1) so the sliced
                    # object keeps the efficient dtype the masks had on the master.
                    for _sv in ("sleep", "sleep_short", "sleep_intermediate", "sleep_long"):
                        if _sv in _sliced.data_vars:
                            _sliced[_sv] = _sliced[_sv].fillna(-1).astype("int8")
                    if _sleep_phase == "LD":
                        st.session_state.dataset_LD = _sliced
                    else:
                        st.session_state.dataset_DD = _sliced
                st.session_state.analyses = detect_analyses(_sleep_ds)
                ds = _sleep_ds

                if "duration" in ds.data_vars:
                    bout_df = (
                        ds["duration"].to_dataframe().reset_index().dropna(subset=["duration"])
                    )
                    n_bouts = len(bout_df)
                    mean_dur = bout_df["duration"].mean()

                    # Build per-state counts if sleep_state variable is present
                    _state_msg = ""
                    if "sleep_state" in ds.data_vars:
                        try:
                            state_counts = (
                                ds["sleep_state"]
                                .to_dataframe()
                                .reset_index()
                                .dropna()["sleep_state"]
                                .value_counts()
                            )
                            _state_parts = [f"{k}: {v}" for k, v in state_counts.items()]
                            _state_msg = f" | States — {', '.join(_state_parts)}"
                        except Exception:
                            pass

                    _run_msg = (
                        f"Sleep analysis complete. "
                        f"Detected {n_bouts} sleep bouts (mean duration: {mean_dur:.1f} min)."
                        f"{_state_msg}"
                    )
                else:
                    _run_msg = "Sleep analysis complete."
            except Exception as e:
                st.error(f"Error during sleep analysis: {e}")

        # Outside the try on purpose: st.rerun raises RerunException, an Exception
        # subclass the handler above would swallow and report as a failure. The
        # rerun is app-scoped rather than fragment-scoped because a fragment rerun
        # re-runs only this function, leaving every plot below still rendering the
        # pre-sleep dataset. The message rides across in session_state.
        if _run_msg:
            st.session_state["_sleep_run_message"] = _run_msg
            st.rerun(scope="app")


_sleep_analysis_fragment()

st.divider()

ds = st.session_state.dataset
analyses = detect_analyses(ds)

# Group filter sidebar — UNIFIED group axis: filter on the single `group` coord
# (defined by the metadata columns chosen on the Data Loading page), not on separate
# per-genotype / per-temperature axes. Selecting a subset of groups subsets the
# dataset here, so every plot below sees the same group-filtered cohort.
st.sidebar.subheader("Filter Groups")
if "group" in ds.coords:
    _grp_cols = dam_utilities.get_group_columns(ds)
    _all_groups = sorted({str(v) for v in ds["group"].values})
    selected_groups = st.sidebar.multiselect(
        "Groups",
        _all_groups,
        default=_all_groups,
        key="viz_groups",
        help=(
            "Groups are defined by the metadata column(s) chosen on the Data "
            "Loading page" + (f": {', '.join(_grp_cols)}." if _grp_cols else ".")
        ),
    )
    if selected_groups and len(selected_groups) < len(_all_groups):
        _sel = set(selected_groups)
        _keep = [str(i) for i in ds["id"].values if str(ds["group"].sel(id=i).item()) in _sel]
        if _keep:
            ds = ds.sel(id=_keep)
# The unified group axis supersedes the old per-genotype / per-temperature filters;
# plots receive the already-group-filtered dataset with no further per-column slicing.
selected_genotypes = None
selected_temperatures = None

bin_size = st.sidebar.slider("Bin size (minutes)", 5, 60, 30, step=5, key="viz_bin_size")

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
        _agg = (
            binned_df.groupby(["zt_bin_minute", "group"])["activity"]
            .agg(
                mean="mean",
                sd=lambda x: x.std(ddof=1),
                n=lambda x: x.notna().sum(),
            )
            .reset_index()
        )
        _pivot = _agg.pivot_table(
            index="zt_bin_minute", columns="group", values=["mean", "sd", "n"]
        )
        _pivot.columns = pd.MultiIndex.from_tuples(
            [(grp, stat) for stat, grp in _pivot.columns], names=["group", "stat"]
        )
        _pivot = _pivot.sort_index(axis=1, level=0)
        _pivot.insert(0, ("zt_hours", ""), dam_utilities.zt_bin_to_hours(_pivot.index, bin_size))
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

# ============================================================
# Sleep Pattern
# ============================================================
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
        _agg_sl = (
            binned_sleep.groupby(["zt_bin_minute", "group"])["sleep"]
            .agg(
                mean="mean",
                sd=lambda x: x.std(ddof=1),
                n=lambda x: x.notna().sum(),
            )
            .reset_index()
        )
        _pivot_sl = _agg_sl.pivot_table(
            index="zt_bin_minute", columns="group", values=["mean", "sd", "n"]
        )
        _pivot_sl.columns = pd.MultiIndex.from_tuples(
            [(grp, stat) for stat, grp in _pivot_sl.columns], names=["group", "stat"]
        )
        _pivot_sl = _pivot_sl.sort_index(axis=1, level=0)
        _pivot_sl.insert(
            0, ("zt_hours", ""), dam_utilities.zt_bin_to_hours(_pivot_sl.index, bin_size)
        )
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
        ex.save_df_button(
            "Save Sleep Bout Data to working folder",
            raw_bout_df,
            ds,
            "sleep_bouts.csv",
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
