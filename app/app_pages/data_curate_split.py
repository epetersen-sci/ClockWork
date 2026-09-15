"""
Preprocessing Page - Dead animal curation, LD/DD split and activity heatmap.
"""


import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import dam_utilities
import plotting
from analysis_detection import detect_analyses
from dataset_meta import PHASE_FULL, dataset_phase, is_split_applied
from ui import charts
from ui.guards import require_dataset

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

# ============================================================
# Step 1b: Inspect the individual flies
# ============================================================
with st.expander("Inspect individual flies (kept and dropped)", expanded=False):
    st.caption(
        "Every fly's own activity trace. After curation the dropped flies are shown "
        "alongside the kept ones, so a borderline call can be checked by eye rather "
        "than taken on trust."
    )

    _live = st.session_state.get("dataset")
    _dead = st.session_state.get("curated_dead_data")
    # Three verdicts, and the kept/trimmed/dropped reconciliation is in core because
    # it is the part that is easy to get wrong — see curation_verdict_table.
    _tbl = dam_utilities.curation_verdict_table(_live, _dead)

    if _tbl.empty:
        st.info("Load a dataset to inspect its flies.")
    else:
        _n = _tbl.verdict.value_counts()
        if _dead is None:
            st.write(f"**{int(_n.get('kept', 0))} flies** — curation not run yet")
        else:
            st.write(
                f"**{int(_n.get('kept', 0))} kept** · "
                f"**{int(_n.get('trimmed', 0))} trimmed** (kept, dead tail cut) · "
                f"**{int(_n.get('dropped', 0))} dropped** "
                f"— {len(_tbl)} flies in total"
            )

        def _select_all_when_options_change(label, options, key):
            """A multiselect defaulting to every option, that notices new ones.

            ``default=`` is ignored once a keyed widget has state, so this cannot
            be written as ``st.multiselect(..., default=options, key=key)``: the
            selection made BEFORE curation — when "kept" was the only verdict —
            would survive it and go on hiding the trimmed and dropped flies this
            expander exists to show. (Backlog item 11 is the same rule elsewhere:
            sharing or defaulting a key is never enough on its own.)

            So the key is seeded directly, and re-seeded only when the set of
            options changes. A selection the user narrowed by hand is therefore
            left alone, while curation appearing — or a different "Group by"
            column — starts from everything again rather than from a value that
            no longer means anything.
            """
            seen_key = f"_{key}_options"
            if st.session_state.get(seen_key) != options:
                st.session_state[seen_key] = options
                st.session_state[key] = list(options)
            return st.multiselect(label, options, key=key)

        _cols = [c for c in dam_utilities.VERDICT_META_COORDS if c in _tbl.columns]
        f1, f2 = st.columns([2, 3])
        with f1:
            # Only the verdicts actually present: a run where nothing was dropped
            # should not offer "dropped" as something to look at.
            _verdicts = [v for v in ("kept", "trimmed", "dropped") if (_tbl.verdict == v).any()]
            _show = _select_all_when_options_change("Show", _verdicts, "flyview_verdict")
        with f2:
            _by = st.selectbox("Group by", ["(none)", *_cols], key="flyview_groupby")

        _sel = _tbl[_tbl.verdict.isin(_show)] if _show else _tbl.iloc[0:0]
        if _by != "(none)" and not _sel.empty:
            _vals = sorted(_sel[_by].unique())
            _pick = _select_all_when_options_change(_by, _vals, "flyview_groupvals")
            _sel = _sel[_sel[_by].isin(_pick)]

        st.dataframe(
            _sel.sort_values(["verdict", "id"]),
            width="stretch",
            hide_index=True,
            height=260,
        )

        if not _sel.empty:
            _ids = list(_sel.id)
            _per_page = 8
            _chosen = st.multiselect(
                "Draw these flies",
                _ids,
                default=_ids[: min(_per_page, len(_ids))],
                key="flyview_ids",
                help="Each fly gets its own row: kept in grey, trimmed in "
                "amber, dropped in red.",
            )
            b1, b2 = st.columns([1, 2])
            with b1:
                _bin = st.selectbox(
                    "Bin (minutes)", [1, 5, 15, 30, 60], index=3, key="flyview_bin"
                )
            with b2:
                _same_y = st.checkbox(
                    "Same y-scale for every fly",
                    value=True,
                    key="flyview_samey",
                    help="Comparable heights across flies. Uncheck to give a quiet "
                    "fly its own scale.",
                )

            if len(_chosen) > _per_page:
                _pages = (len(_chosen) + _per_page - 1) // _per_page
                _pg = st.number_input(
                    f"Page (of {_pages}, {_per_page} flies each)",
                    min_value=1,
                    max_value=_pages,
                    value=1,
                    step=1,
                    key="flyview_page",
                )
                _draw = _chosen[(int(_pg) - 1) * _per_page : int(_pg) * _per_page]
            else:
                _draw = _chosen

            if _draw:
                _live_ids = (
                    {str(v) for v in _live["id"].values} if _live is not None else set()
                )
                _verdict_of = dict(zip(_sel.id, _sel.verdict))
                _COLOUR = {"kept": "#3D3D3D", "trimmed": "#E08A00", "dropped": "#C62828"}
                _NOTE = {"kept": "", "trimmed": "  —  trimmed", "dropped": "  —  dropped"}

                _traces = []
                for _fid in _draw:
                    _src = _live if _fid in _live_ids else _dead
                    _one = _src.sel(id=_fid)["activity"].values.astype(float)
                    _n_use = (len(_one) // int(_bin)) * int(_bin)
                    _b = np.nan_to_num(_one[:_n_use]).reshape(-1, int(_bin)).sum(axis=1)
                    _x = (np.arange(_b.size) + 0.5) * int(_bin) / 1440.0
                    _traces.append((_fid, _x, _b, _verdict_of.get(_fid, "kept")))

                _ymax = (
                    max((float(np.nanmax(t[2])) for t in _traces if t[2].size), default=1.0)
                    * 1.05
                )
                _xmax = max((float(t[1][-1]) for t in _traces if t[1].size), default=1.0)

                # One row per fly. Overlaid, a dozen traces are just noise; stacked,
                # a fly that stops halfway through is obvious at a glance.
                _fig = make_subplots(
                    rows=len(_traces),
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=min(0.055, 1.1 / max(len(_traces), 1)),
                )
                for _r, (_fid, _x, _b, _verdict) in enumerate(_traces, start=1):
                    _col = _COLOUR.get(_verdict, "#3D3D3D")
                    _fig.add_trace(
                        go.Scatter(
                            x=_x,
                            y=_b,
                            mode="lines",
                            showlegend=False,
                            line=dict(width=0.8, color=_col),
                            fill="tozeroy",
                            fillcolor=_col,
                            hovertemplate="day %{x:.2f}<br>%{y:.0f} counts"
                            f"<extra>{_fid}</extra>",
                        ),
                        row=_r,
                        col=1,
                    )
                    # A rotated y-axis title is hard to read and collides with the
                    # ticks; label each panel horizontally along its top edge instead.
                    _fig.add_annotation(
                        text=f"{_fid}{_NOTE.get(_verdict, '')}",
                        xref="x domain",
                        yref=f"y{_r if _r > 1 else ''} domain",
                        x=0,
                        y=1.0,
                        xanchor="left",
                        yanchor="bottom",
                        showarrow=False,
                        font=dict(size=12, color=_col),
                    )
                    _fig.update_yaxes(
                        title_text="counts" if _r == 1 else None,
                        title_font=dict(size=12, color="#000000"),
                        title_standoff=4,
                        range=[0, _ymax] if _same_y else None,
                        showline=True,
                        linecolor="#000000",
                        linewidth=1,
                        mirror=True,
                        ticks="outside",
                        tickfont=dict(size=10, color="#000000"),
                        nticks=3,
                        gridcolor="#E6E6E6",
                        row=_r,
                        col=1,
                    )
                    _fig.update_xaxes(
                        range=[0, _xmax],
                        dtick=1,
                        showline=True,
                        linecolor="#000000",
                        linewidth=1,
                        mirror=True,
                        ticks="outside",
                        tickfont=dict(size=11, color="#000000"),
                        gridcolor="#E6E6E6",
                        title_text="day" if _r == len(_traces) else None,
                        title_font=dict(size=13, color="#000000"),
                        row=_r,
                        col=1,
                    )

                _fig.update_layout(
                    height=100 * len(_traces) + 110,
                    title=dict(
                        text=f"Activity per fly — counts per {_bin}-minute bin",
                        x=0.5,
                        xanchor="center",
                        font=dict(size=16, color="#000000"),
                    ),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    font=dict(color="#000000", size=12),
                    margin=dict(t=70, b=55, l=65, r=25),
                    bargap=0,
                )
                charts.plotly_chart(_fig, filename="per_fly_activity", width="stretch")

        st.download_button(
            "Download this fly table (CSV)",
            _tbl.to_csv(index=False).encode("utf-8"),
            file_name="fly_curation_table.csv",
            mime="text/csv",
            key="flyview_csv",
        )

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

                # ds_dd / ds_ld stay LOCAL: they are reported on below, then
                # dropped. They used to be cached in session_state, which meant
                # four files had to keep that copy in sync with the master — and
                # sleep_detection.py had to regenerate both on every run to stop
                # them going stale. The split PARAMETERS are recorded on the
                # master instead, so any consumer reproduces the same slice.
                # Mark master as split-applied (but keep all timepoints).
                # Canonical phase stays 'full' since the master spans both
                # epochs; split_applied=True records that LD/DD partitions
                # are available via session_state. See core/dataset_meta.py.
                from dataset_meta import stamp_phase

                stamp_phase(ds, PHASE_FULL, split_applied=True)
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
