"""
Preprocessing Page - Dead animal curation, LD/DD split and activity heatmap.
"""

import os
import sys

import streamlit as st

# Add core/ (analysis modules) and app/ to the import path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORE_DIR = os.path.join(PROJECT_ROOT, "core")
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [CORE_DIR, APP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import dam_utilities
import plotting
from analysis_detection import detect_analyses

st.header("Preprocessing")

if st.session_state.get("dataset") is None:
    st.warning("No dataset loaded. Go to **Data Loading** first.")
    st.stop()

ds = st.session_state.dataset

# ============================================================
# Step 1: Curate Dead Animals
# ============================================================
st.subheader("1. Curate Dead Animals")

if ds.attrs.get("curation_time_window_hours") is not None:
    st.info(
        f"Previous curation applied — "
        f"**{ds.attrs['curation_min_alive_days']}** day min alive, "
        f"**{ds.attrs['curation_time_window_hours']}h** window, "
        f"**{ds.attrs['curation_prop_immobile_threshold']}** activity threshold"
    )
elif "is_alive" in ds.data_vars:
    st.info(
        "This dataset was curated with an older algorithm version. "
        "Re-running curation is recommended for consistency with current settings."
    )

col1, col2 = st.columns(2)
with col1:
    min_alive_days = st.slider(
        "Minimum alive days",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5,
        help="Flies alive for fewer days than this are excluded.",
    )
    time_window = st.number_input(
        "Rolling window (hours)",
        min_value=1,
        max_value=72,
        value=24,
        help="Size of rolling window for activity assessment.",
    )
with col2:
    prop_immobile = st.number_input(
        "Immobility proportion threshold",
        min_value=0.001,
        max_value=0.1,
        value=0.01,
        step=0.005,
        format="%.3f",
        help="Activity threshold below which a fly is considered dead.",
    )

if st.button("Run Curation", key="run_curation"):
    curate_progress = st.progress(0, text="Starting curation...")
    with st.spinner("Curating dead animals..."):
        try:

            def _curate_cb(completed, total):
                curate_progress.progress(
                    completed / total, text=f"Curation: fly {completed}/{total}"
                )

            result = dam_utilities.curate_dead_animals(
                ds,
                time_window=time_window,
                prop_immobile=prop_immobile,
                min_alive_days=min_alive_days,
                progress_callback=_curate_cb,
            )
            (
                live_data,
                dead_data,
                error_ids,
                success_ids,
                total_before,
                total_after,
                removed,
                unchanged,
                trimmed,
            ) = result

            st.session_state.dataset = live_data
            st.session_state.curated_dead_data = dead_data
            st.session_state.analyses = detect_analyses(live_data)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Before", total_before)
            col2.metric("After", total_after)
            col3.metric("Removed", removed)
            col4.metric("Trimmed", trimmed)

            if removed > 0:
                st.success(
                    f"Curation complete. Kept {total_after}/{total_before} flies "
                    f"({removed} removed, {trimmed} trimmed)."
                )
            else:
                st.success("All flies passed curation criteria.")
            st.rerun()
        except Exception as e:
            st.error(f"Error during curation: {e}")
        finally:
            curate_progress.empty()

st.divider()

# ============================================================
# Step 2: Apply LD/DD Split (after curation)
# ============================================================
st.subheader("2. Apply LD/DD Split")

_already_split = ds.attrs.get("split_phase") is not None
_has_dd_coord = "first_DD_day" in ds.coords

if _already_split:
    _prev_discard = bool(ds.attrs.get("split_discard_first_dd_day", 0))
    # Show info about existing split
    _has_dd_ds = st.session_state.get("dataset_DD") is not None
    _has_ld_ds = st.session_state.get("dataset_LD") is not None
    if _has_dd_ds and _has_ld_ds:
        _dd_ds = st.session_state.dataset_DD
        _ld_ds = st.session_state.dataset_LD
        st.info(
            f"LD/DD split applied — discard first DD day: **{_prev_discard}**\n\n"
            f"- **DD dataset:** {len(_dd_ds['id'])} flies, {len(_dd_ds['time'])} timepoints\n"
            f"- **LD dataset:** {len(_ld_ds['id'])} flies, {len(_ld_ds['time'])} timepoints"
        )
    else:
        st.info(f"LD/DD split already applied — discard first DD day: **{_prev_discard}**")
    _resplit = st.checkbox(
        "Re-run split with different settings", value=False, key="resplit_checkbox"
    )
    if _resplit:
        st.warning(
            "Re-splitting requires the original (unsplit) dataset. "
            "Please reload your data from the Data Loading page."
        )
elif not _has_dd_coord:
    st.info(
        "No `first_DD_day` coordinate in this dataset — only one lighting phase "
        "is present, so no LD/DD split is needed. All analyses will use the full dataset."
    )
else:
    st.markdown(
        "Split the recording into separate **LD** and **DD** datasets. "
        "Downstream analysis pages will let you choose which phase to use "
        "(e.g. DD for circadian period, LD for sleep)."
    )

    if ds.attrs.get("curation_time_window_hours") is None:
        st.warning(
            "Curation has not been run yet. It is recommended to curate dead animals "
            "(Step 1) **before** splitting so that dead-fly detection uses the complete recording."
        )

    discard_first_dd_day = st.checkbox(
        "Discard first day of DD (removes lingering LD effects)",
        value=False,
        key="discard_first_dd_checkbox",
    )

    gap_threshold = st.number_input(
        "Gap threshold (minutes)",
        min_value=0,
        max_value=1440,
        value=60,
        step=15,
        help="Consecutive NaN runs longer than this are treated as real gaps "
        "(e.g. monitor power loss). Each fly keeps only its longest "
        "continuous segment within the selected phase. Set to 0 to disable.",
        key="gap_threshold_input",
    )

    if st.button("Apply LD/DD Split", key="apply_split"):
        with st.spinner("Splitting dataset into LD and DD..."):
            try:
                import ast

                import numpy as np

                # Create both DD and LD datasets
                ds_dd = dam_utilities.split_xarray_dataset(
                    ds,
                    phase="DD",
                    discard_first_dd_day=discard_first_dd_day,
                    gap_threshold_minutes=gap_threshold,
                )
                ds_ld = dam_utilities.split_xarray_dataset(
                    ds,
                    phase="LD",
                    gap_threshold_minutes=gap_threshold,
                )

                # Store both in session state — master dataset stays unsplit
                st.session_state.dataset_DD = ds_dd
                st.session_state.dataset_LD = ds_ld
                # Mark master as split-applied (but keep all timepoints).
                # Canonical phase stays 'full' since the master spans both
                # epochs; split_applied=True records that LD/DD partitions
                # are available via session_state. See core/dataset_meta.py.
                from dataset_meta import PHASE_FULL, stamp_phase

                stamp_phase(ds, PHASE_FULL, split_applied=True)
                ds.attrs["split_phase"] = "both"  # legacy alias
                ds.attrs["split_discard_first_dd_day"] = int(discard_first_dd_day)
                ds.attrs["gap_threshold_minutes"] = gap_threshold
                st.session_state.dataset = ds
                st.session_state.analyses = detect_analyses(ds)

                # Show DD stats
                _dd_seg_str = ds_dd.attrs.get("segment_info", None)
                if _dd_seg_str:
                    _dd_seg = ast.literal_eval(_dd_seg_str)
                    _dd_durs = [s["segment_duration_days"] for s in _dd_seg]
                    _dd_orig = [s["original_duration_days"] for s in _dd_seg]
                    _dd_trimmed = sum(1 for d, o in zip(_dd_durs, _dd_orig) if d < o)
                    st.info(
                        f"**DD** — {len(ds_dd['id'])} flies, "
                        f"min: **{min(_dd_durs):.1f}** days, "
                        f"avg: **{np.mean(_dd_durs):.1f}** days, "
                        f"max: **{max(_dd_durs):.1f}** days"
                    )
                    if _dd_trimmed > 0:
                        st.warning(
                            f"DD: **{_dd_trimmed}/{len(_dd_durs)}** flies had gaps "
                            f"and were trimmed to their longest continuous segment."
                        )
                else:
                    st.info(f"**DD** — {len(ds_dd['id'])} flies, {len(ds_dd['time'])} timepoints")

                # Show LD stats
                _ld_seg_str = ds_ld.attrs.get("segment_info", None)
                if _ld_seg_str:
                    _ld_seg = ast.literal_eval(_ld_seg_str)
                    _ld_durs = [s["segment_duration_days"] for s in _ld_seg]
                    st.info(
                        f"**LD** — {len(ds_ld['id'])} flies, "
                        f"min: **{min(_ld_durs):.1f}** days, "
                        f"avg: **{np.mean(_ld_durs):.1f}** days, "
                        f"max: **{max(_ld_durs):.1f}** days"
                    )
                else:
                    st.info(f"**LD** — {len(ds_ld['id'])} flies, {len(ds_ld['time'])} timepoints")

                st.success("Split complete. LD and DD datasets created separately.")
                st.rerun()
            except Exception as e:
                st.error(f"Error during split: {e}")

st.divider()

# ============================================================
# Step 2b: Activity / Movement Heatmap
# ============================================================
st.subheader("Activity / Movement Heatmap")

# Re-read ds in case split was applied above
ds = st.session_state.dataset

_heatmap_vars = [v for v in ("moving", "activity") if v in ds.data_vars]
if not _heatmap_vars:
    st.info("No data variables available for heatmap.")
else:
    heatmap_var = st.radio(
        "Variable to display",
        _heatmap_vars,
        index=0,
        horizontal=True,
        key="heatmap_var_choice",
        help="`moving` shows binary activity (0/1) and makes circadian rhythms easier to see. "
        "`activity` shows raw beam-break counts.",
    )

    # Option to show curated/removed flies
    _has_dead_data = (
        st.session_state.get("curated_dead_data") is not None
        and len(st.session_state.curated_dead_data.get("id", [])) > 0
    )
    show_curated = False
    if _has_dead_data:
        show_curated = st.checkbox(
            "Also show flies removed during curation",
            value=False,
            key="show_curated_heatmap",
        )

    # Clear cached heatmap if the user switches variable selection
    if st.session_state.get("_heatmap_var_last") != heatmap_var:
        st.session_state.pop("show_preprocessing_heatmap", None)
        st.session_state["_heatmap_var_last"] = heatmap_var

    if st.button("Generate Heatmap", key="gen_heatmap_preproc"):
        st.session_state["show_preprocessing_heatmap"] = True

    if st.session_state.get("show_preprocessing_heatmap"):
        _var_label = "Movement (moving)" if heatmap_var == "moving" else "Activity"
        _split_phase = ds.attrs.get("split_phase", None)
        _has_phases = "first_DD_day" in ds.coords and _split_phase in (None, "both")

        with st.spinner("Generating heatmap..."):
            try:
                if _has_phases:
                    # Dataset has both LD and DD — show them as separate heatmaps.
                    # Render via the per-fly NaN-masked selector (select_phase) on
                    # the WHOLE master, NOT the legacy split_xarray_dataset: the
                    # legacy global-bounds branch collapses heterogeneous per-fly
                    # boundaries to global min/max (cropping LD to the MIN boundary
                    # and DD to >= MAX), which masks the real per-group LD lengths
                    # and truncates DD. select_phase honors each fly's own boundary
                    # and keeps the full time axis with out-of-phase cells = NaN, so
                    # the heatmap shows every group's true LD window and real gaps
                    # (NaN) as blanks, not stitched closed (§2a). The plotter is
                    # faithful — it pivots (id, time) and renders NaN as gaps.
                    _discard = bool(ds.attrs.get("split_discard_first_dd_day", 0))
                    ds_ld, _ = dam_utilities.select_phase(ds, phase="LD")
                    ds_dd, _ = dam_utilities.select_phase(
                        ds, phase="DD", discard_first_dd_day=_discard
                    )

                    st.markdown(f"**LD phase — {_var_label}**")
                    fig_ld = plotting.dataset_to_heatmap(
                        ds_ld, heatmap_var, f"{_var_label} — LD Phase"
                    )
                    st.plotly_chart(fig_ld, width="stretch")

                    st.markdown(f"**DD phase — {_var_label}**")
                    fig_dd = plotting.dataset_to_heatmap(
                        ds_dd, heatmap_var, f"{_var_label} — DD Phase"
                    )
                    st.plotly_chart(fig_dd, width="stretch")
                else:
                    # Single phase (already split, or only LD data)
                    _phase_label = ds.attrs.get("split_phase", "")
                    _title_suffix = f" — {_phase_label} Phase" if _phase_label else ""
                    title = f"{_var_label}{_title_suffix}"
                    fig = plotting.dataset_to_heatmap(ds, heatmap_var, title)
                    st.plotly_chart(fig, width="stretch")

                # Show curated/removed flies if requested
                if show_curated and _has_dead_data:
                    dead_ds = st.session_state.curated_dead_data
                    if heatmap_var in dead_ds.data_vars:
                        st.markdown(f"**Removed during curation — {_var_label}**")
                        fig_dead = plotting.dataset_to_heatmap(
                            dead_ds, heatmap_var, f"{_var_label} — Removed Flies (Curation)"
                        )
                        st.plotly_chart(fig_dead, width="stretch")

                # Download button for main dataset
                df_heat = ds[heatmap_var].to_pandas()
                st.download_button(
                    f"Download {_var_label} Heatmap Data (CSV)",
                    df_heat.to_csv(),
                    f"{heatmap_var}_heatmap.csv",
                    "text/csv",
                    key="dl_heat_preproc",
                )
            except Exception as e:
                st.error(f"Heatmap error: {e}")

st.divider()
st.caption(
    "**Sleep analysis has moved** to the Sleep and Activity page. Run it there once "
    "curation and the LD/DD split are done."
)

# ============================================================
# Current dataset summary
# ============================================================
st.subheader("Current Dataset")
ds = st.session_state.dataset
st.write(f"**Flies:** {len(ds['id'])} | **Timepoints:** {len(ds['time'])}")
st.write(f"**Variables:** {', '.join(sorted(ds.data_vars))}")
