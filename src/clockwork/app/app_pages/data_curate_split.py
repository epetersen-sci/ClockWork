"""
Preprocessing Page - Dead animal curation, LD/DD split and activity heatmap.
"""


import streamlit as st

from clockwork import pipeline
from clockwork.app.analysis_detection import detect_analyses
from clockwork.app.ui import charts
from clockwork.app.ui.guards import require_dataset
from clockwork.core import dam_utilities, plotting
from clockwork.core.dataset_meta import PHASE_FULL, dataset_phase, is_split_applied

ds = require_dataset()

# ============================================================
# Step 1: Curate Dead Animals
# ============================================================
st.subheader("1. Curate dead animals")

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

            result = pipeline.curate(
                ds,
                pipeline.CurationConfig(
                    min_alive_days=min_alive_days,
                    rolling_window_hours=time_window,
                    immobility_proportion=prop_immobile,
                ),
                progress=_curate_cb,
            )

            st.session_state.dataset = result.live
            st.session_state.curated_dead_data = result.dead
            st.session_state.analyses = detect_analyses(result.live)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Before", result.total_before)
            col2.metric("After", result.total_after)
            col3.metric("Removed", result.removed)
            col4.metric("Trimmed", result.trimmed)

            if result.removed > 0:
                st.success(
                    f"Curation complete. Kept {result.total_after}/{result.total_before} flies "
                    f"({result.removed} removed, {result.trimmed} trimmed)."
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
st.subheader("2. Apply the LD/DD split")

_already_split = is_split_applied(ds)
_has_dd_coord = "first_DD_day" in ds.coords

if _already_split:
    _prev_discard = bool(ds.attrs.get("split_discard_first_dd_day", 0))
    # Split state comes from the master's own canonical attrs, not from pre-sliced
    # session copies. There are no per-phase fly/timepoint counts to show here any
    # more, because there is no stored per-phase dataset to count — consumers slice
    # on demand. What persists is the split PARAMETERS, and they live on the master,
    # so they survive a .nc round-trip in a way the session caches never did.
    st.info(
        f"LD/DD split applied — discard first DD day: **{_prev_discard}**, "
        f"gap threshold: **{int(ds.attrs.get('gap_threshold_minutes', 60))}** min.\n\n"
        "LD and DD views are derived when needed: analysis pages use "
        "`select_phase`, the export pages re-slice at export time."
    )
    _resplit = st.checkbox(
        "Re-run split with different settings", value=False, key="resplit_checkbox"
    )
    if _resplit:
        st.warning(
            "Re-splitting requires the original (unsplit) dataset. "
            "Please reload your data from the **Data → Import** page."
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
                config = pipeline.SplitConfig(
                    discard_first_dd_day=discard_first_dd_day,
                    gap_threshold_minutes=gap_threshold,
                )
                # The LD and DD views are built only to report on, then dropped.
                # They used to be cached in session_state, which meant four files
                # had to keep that copy in sync with the master. The split
                # PARAMETERS are recorded on the master instead (pipeline.split),
                # so any consumer reproduces the same slice with select_phase.
                report = pipeline.split_report(ds, config)
                # The MASTER is marked split-applied and keeps every timepoint;
                # its canonical phase stays 'full' since it spans both epochs.
                ds = pipeline.split(ds, config)
                st.session_state.dataset = ds
                st.session_state.analyses = detect_analyses(ds)

                for phase in ("DD", "LD"):
                    entry = report[phase]
                    if "days_min" in entry:
                        st.info(
                            f"**{phase}** — {entry['n_flies']} flies, "
                            f"min: **{entry['days_min']:.1f}** days, "
                            f"avg: **{entry['days_mean']:.1f}** days, "
                            f"max: **{entry['days_max']:.1f}** days"
                        )
                        if phase == "DD" and entry["n_trimmed"] > 0:
                            st.warning(
                                f"DD: **{entry['n_trimmed']}/{entry['n_flies']}** flies had "
                                f"gaps and were trimmed to their longest continuous segment."
                            )
                    else:
                        st.info(
                            f"**{phase}** — {entry['n_flies']} flies, "
                            f"{entry['n_timepoints']} timepoints"
                        )

                st.success("Split complete. LD and DD datasets created separately.")
                st.rerun()
            except Exception as e:
                st.error(f"Error during split: {e}")

st.divider()

# ============================================================
# Step 2b: Activity / Movement Heatmap
# ============================================================
st.subheader("3. Inspect the result")

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
        # Canonical phase, not the legacy split_phase alias. 'full' covers both
        # the never-split case (which the alias left as None) and the post-split
        # master (which it set to "both") — those are the two states in which this
        # dataset still spans LD and DD and can be shown as two heatmaps.
        _has_phases = "first_DD_day" in ds.coords and dataset_phase(ds) == PHASE_FULL

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
                    charts.plotly_chart(fig_ld, width="stretch")

                    st.markdown(f"**DD phase — {_var_label}**")
                    fig_dd = plotting.dataset_to_heatmap(
                        ds_dd, heatmap_var, f"{_var_label} — DD Phase"
                    )
                    charts.plotly_chart(fig_dd, width="stretch")
                else:
                    # Single phase (already split, or only LD data)
                    # dataset_phase() always returns a label, so 'full' (an
                    # unsplit dataset with no DD coord) has to be mapped back to
                    # no suffix — the alias returned "" for that case.
                    _phase = dataset_phase(ds)
                    _phase_label = "" if _phase == PHASE_FULL else _phase
                    _title_suffix = f" — {_phase_label} Phase" if _phase_label else ""
                    title = f"{_var_label}{_title_suffix}"
                    fig = plotting.dataset_to_heatmap(ds, heatmap_var, title)
                    charts.plotly_chart(fig, width="stretch")

                # Show curated/removed flies if requested
                if show_curated and _has_dead_data:
                    dead_ds = st.session_state.curated_dead_data
                    if heatmap_var in dead_ds.data_vars:
                        st.markdown(f"**Removed during curation — {_var_label}**")
                        fig_dead = plotting.dataset_to_heatmap(
                            dead_ds, heatmap_var, f"{_var_label} — Removed Flies (Curation)"
                        )
                        charts.plotly_chart(fig_dead, width="stretch")

                # Download button for main dataset: the full per-minute data, not
                # the binned view. Handed over as a callable so the CSV is built
                # only when the button is clicked; building it eagerly re-ran a
                # flies x minutes to_csv on every rerun of this page.
                _heat_da = ds[heatmap_var]
                st.download_button(
                    f"Download {_var_label} Heatmap Data (CSV)",
                    lambda: _heat_da.to_pandas().to_csv(),
                    f"{heatmap_var}_heatmap.csv",
                    "text/csv",
                    key="dl_heat_preproc",
                )
            except Exception as e:
                st.error(f"Heatmap error: {e}")
