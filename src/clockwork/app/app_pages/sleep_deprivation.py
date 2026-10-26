"""
Sleep Deprivation Analysis — compare baseline sleep to post-SD recovery sleep.

Requires sleep analysis to be completed first (Sleep and Activity page).
"""


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from clockwork.app import export_helpers as ex
from clockwork.app.analysis_detection import detect_analyses
from clockwork.app.ui import charts, facet_panels, status
from clockwork.app.ui import stats as stats_ui
from clockwork.app.ui.guards import require_dataset
from clockwork.core import dam_utilities, facets, group_stats, plotting
from clockwork.core import sleep_deprivation as sd_module

# ============================================================
# Section 1: Prerequisites
# ============================================================
ds = require_dataset()
analyses = detect_analyses(ds)

if not analyses.get("sleep", False):
    st.warning(
        "Sleep has not been detected yet. Run it from the top of the **Sleep** tab "
        "on **Activity & Sleep**, then come back."
    )
    st.stop()

# ============================================================
# Section 2: LD-phase experiment overview
# ============================================================
# SD is an LD-structured protocol (light/dark baseline + recovery days). Per the
# current phase API (Stage-2), we pass the WHOLE dataset (the master) and let each
# SD function select the LD phase internally via phase="LD" — we do NOT pre-slice to
# a dataset_LD / split_xarray_dataset copy (the legacy pattern the rest of the app has
# moved off). See core/sleep_deprivation.py: get_experiment_days and compute_sd_analysis
# both resolve_phase→LD and restrict to the LD boundary; get_day_sleep_traces limits to
# the LD days via get_experiment_days. A DD-only dataset has no LD epoch to analyse.
from clockwork.core.dataset_meta import PHASE_DD, dataset_phase

if dataset_phase(ds) == PHASE_DD:
    st.error("Dataset contains only DD data. Sleep deprivation analysis requires LD data.")
    st.stop()

n_days, day_bounds = sd_module.get_experiment_days(ds, phase="LD")
if n_days < 3:
    st.error(
        f"Only {n_days} complete LD day(s) detected. Need at least 3 days "
        "(baseline + SD + recovery)."
    )
    st.stop()

st.caption("Sleep deprivation runs on the **LD** phase (selected internally, whole-dataset in).")
st.info(f"**{n_days}** complete LD days detected in the dataset.")

# ============================================================
# Section 3: SD window configuration
# ============================================================
st.subheader("Define Sleep Deprivation Window")

if ds.attrs.get("sd_day_number") is not None:
    _sd_start = ds.attrs["sd_start_zt_minutes"]
    _sd_dur = ds.attrs["sd_duration_minutes"]
    _sd_end = _sd_start + _sd_dur
    st.info(
        f"Previous SD analysis recorded — "
        f"Day **{ds.attrs['sd_day_number']}**, "
        f"ZT{_sd_start // 60}:{_sd_start % 60:02d}–ZT{_sd_end // 60}:{_sd_end % 60:02d}, "
        f"**{ds.attrs['sd_bin_size_minutes']} min** bins"
    )

col_plot, col_manual = st.columns([3, 1])

# --- Manual inputs (right column) ---
with col_manual:
    st.markdown("**Manual Input**")
    sd_day = st.number_input(
        "SD Day (1-indexed)",
        min_value=2,
        max_value=n_days - 1,
        value=2,
        help="Day on which SD occurs. Must have at least 1 baseline day before and 1 recovery day after.",
    )
    sd_day_index = sd_day - 1  # convert to 0-indexed

    sd_start_hour = st.number_input("SD Start (ZT hour)", min_value=0, max_value=23, value=12)
    sd_start_min = st.number_input("SD Start (minute)", min_value=0, max_value=59, value=0)
    sd_start_zt = sd_start_hour * 60 + sd_start_min

    sd_dur_hours = st.number_input("SD Duration (hours)", min_value=0, max_value=23, value=6)
    sd_dur_mins = st.number_input("SD Duration (minutes)", min_value=0, max_value=59, value=0)
    sd_duration = sd_dur_hours * 60 + sd_dur_mins

    if sd_duration <= 0:
        st.warning("SD duration must be > 0.")

    sd_end_zt = sd_start_zt + sd_duration
    st.markdown(
        f"**SD window:** ZT{sd_start_hour}:{sd_start_min:02d} – "
        f"ZT{sd_end_zt // 60}:{sd_end_zt % 60:02d} on Day {sd_day}"
    )

    bin_size = st.slider("Bin size (min)", min_value=5, max_value=60, value=15, step=5)

# --- Interactive plot (left column) ---
with col_plot:
    st.markdown("**Sleep Trace Overview** (averaged across flies)")
    try:
        trace_df = sd_module.get_day_sleep_traces(ds, bin_size_minutes=bin_size)
    except Exception as e:
        st.error(f"Error computing sleep traces: {e}")
        st.stop()

    if trace_df.empty:
        st.warning("No sleep trace data available.")
    else:
        fig_sel = go.Figure()

        # Plot each day as a separate trace
        for day_idx in sorted(trace_df["day"].unique()):
            day_data = trace_df[trace_df["day"] == day_idx].sort_values("zt_bin_minute")
            zt_hours = dam_utilities.zt_bin_to_hours(day_data["zt_bin_minute"], bin_size)
            label = f"Day {day_idx + 1}"
            if day_idx < sd_day_index:
                label += " (Baseline)"
                color = "rgba(100,100,100,0.5)"
                dash = "dot"
            elif day_idx == sd_day_index:
                label += " (SD)"
                color = "red"
                dash = "dash"
            else:
                rec_num = day_idx - sd_day_index
                label += f" (Recovery {rec_num})"
                color = None
                dash = "solid"

            fig_sel.add_trace(
                go.Scatter(
                    x=zt_hours,
                    y=day_data["sleep_mean"],
                    mode="lines",
                    name=label,
                    line=dict(color=color, dash=dash) if color else dict(dash=dash),
                )
            )

        # Shade light/dark regions
        fig_sel.add_vrect(x0=0, x1=12, fillcolor="yellow", opacity=0.08, line_width=0)
        fig_sel.add_vrect(x0=12, x1=24, fillcolor="gray", opacity=0.12, line_width=0)

        # Highlight SD window
        sd_start_h = sd_start_zt / 60.0
        sd_end_h = sd_end_zt / 60.0
        fig_sel.add_vrect(
            x0=sd_start_h,
            x1=min(sd_end_h, 24),
            fillcolor="red",
            opacity=0.18,
            line_width=2,
            annotation_text="SD",
            annotation_position="top left",
        )

        fig_sel.update_layout(
            xaxis_title="ZT (hours)",
            yaxis_title="Sleep (fraction)",
            xaxis=dict(range=[0, 24], dtick=2),
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        charts.plotly_chart(fig_sel, width="stretch")

# ============================================================
# Section 4: Run analysis
# ============================================================
st.divider()

if sd_duration <= 0:
    st.stop()

if st.button("Run Sleep Deprivation Analysis", type="primary"):
    with st.spinner("Computing SD analysis..."):
        try:
            results = sd_module.compute_sd_analysis(
                ds,
                sd_start_zt_minutes=sd_start_zt,
                sd_duration_minutes=sd_duration,
                sd_day_index=sd_day_index,
                bin_size_minutes=bin_size,
                phase="LD",
            )
            st.session_state.sd_results = results

            # Store SD parameters in dataset attrs for reproducibility
            ds.attrs["sd_day_number"] = int(sd_day)
            ds.attrs["sd_start_zt_minutes"] = int(sd_start_zt)
            ds.attrs["sd_duration_minutes"] = int(sd_duration)
            ds.attrs["sd_bin_size_minutes"] = int(bin_size)
            st.session_state.dataset = ds
            # SD records its results in attrs, not data_vars, so the status grid
            # only learns about them if we re-detect here.
            status.refresh(ds)

            st.success(
                f"Analysis complete: {results['n_baseline_days']} baseline day(s), "
                f"{results['n_recovery_days']} recovery day(s)."
            )
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

# ============================================================
# Section 5: Results
# ============================================================
if "sd_results" not in st.session_state:
    st.info("Configure SD parameters above and click **Run** to see results.")
    st.stop()

results = st.session_state.sd_results
sd_params = results["sd_params"]
baseline_prof = results["baseline_profile"]
recovery_profs = results["recovery_profiles"]
diff_profs = results["difference_profiles"]
cum_diffs = results["cumulative_diff"]
phase_totals = results["phase_totals"]
rebound_pct = results["rebound_pct"]
_bin = sd_params["bin_size_minutes"]

st.subheader("Results")

# Lazy tabs: only the open view's figures are built on a rerun.
tab_timecourse, tab_bars, tab_export = st.tabs(
    ["ZT Time-Course Plots", "Bar Graphs", "Data Export"],
    key="sleep_deprivation_tab",
    on_change="rerun",
)

# --- Helper: add light/dark shading to a figure ---
def _shade_ld(fig):
    fig.add_vrect(x0=0, x1=12, fillcolor="yellow", opacity=0.07, line_width=0)
    fig.add_vrect(x0=12, x1=24, fillcolor="gray", opacity=0.10, line_width=0)

# --- Color palette for genotypes ---
_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

def _group_color(groups, grp):
    return _COLORS[groups.index(grp) % len(_COLORS)]

# ================================================================
# Panels. The time-course figures overlay one colour per group, so with panels on
# they draw once per panel (a genotype, say) with one colour per compared level
# (its temperatures) — the same layout as the daily patterns on Activity & Sleep.
# ================================================================
facet_spec, _ = facet_panels.facet_controls(ds)
_all_groups = sorted(baseline_prof["group"].unique())
_facet_rows = facet_panels.arrange_groups(ds, facet_spec, _all_groups)
if _facet_rows:
    _level_of = facet_panels.group_levels(ds, facet_spec)
    _level_colours = facets.facet_colours(
        facets.order_levels({_level_of[g] for _, gs in _facet_rows for g in gs})
    )


def _sets():
    """``[(panel title, groups, label_of, colour_of)]``: one entry with every group
    when panels are off, one per panel when they are on."""
    if not _facet_rows:
        return [("", _all_groups, str, lambda g: _group_color(_all_groups, g))]
    return [
        (title, gs, _level_of.get, lambda g: _level_colours[_level_of[g]])
        for title, gs in _facet_rows
    ]


def _titled(title, text):
    return f"{text} — {title}" if title else text


def _show(pairs, key, *, grid=True):
    """One figure full width, or a panel per row entry when panels are on."""
    if not _facet_rows:
        charts.plotly_chart(pairs[0][1], width="stretch")
        return
    facet_panels.render_panels(
        [(facets.Panel(key={}, title=t, series=[]), f) for t, f in pairs],
        facet_spec,
        ds,
        key=key,
        filename=key,
        grid=grid,
    )


def _baseline_recovery_fig(title, groups, label_of, colour_of):
    fig1 = go.Figure()
    _shade_ld(fig1)

    for grp in groups:
        color = colour_of(grp)
        name = label_of(grp)
        bl = baseline_prof[baseline_prof["group"] == grp].sort_values("zt_bin_minute")
        zt_h = dam_utilities.zt_bin_to_hours(bl["zt_bin_minute"], _bin)

        # Baseline (dashed)
        _lg_bl = f"{name} — Baseline"
        fig1.add_trace(
            go.Scatter(
                x=zt_h,
                y=bl["mean"],
                mode="lines",
                name=_lg_bl,
                legendgroup=_lg_bl,
                line=dict(color=color, dash="dash"),
            )
        )
        # SAME legendgroup as the mean so a legend click toggles both together.
        fig1.add_trace(
            go.Scatter(
                x=pd.concat([zt_h, zt_h[::-1]]),
                y=pd.concat([bl["mean"] + bl["sem"], (bl["mean"] - bl["sem"])[::-1]]),
                fill="toself",
                line=dict(color="rgba(0,0,0,0)"),
                fillcolor=color.replace(")", ",0.12)").replace("rgb", "rgba")
                if "rgb" in color
                else "rgba(100,100,200,0.12)",
                legendgroup=_lg_bl,
                showlegend=False,
            )
        )

        # Recovery days (solid, increasing opacity)
        for rec_num, rec_df in recovery_profs.items():
            rec = rec_df[rec_df["group"] == grp].sort_values("zt_bin_minute")
            if rec.empty:
                continue
            zt_h_r = dam_utilities.zt_bin_to_hours(rec["zt_bin_minute"], _bin)
            _lg_rec = f"{name} — Recovery Day {rec_num}"
            fig1.add_trace(
                go.Scatter(
                    x=zt_h_r,
                    y=rec["mean"],
                    mode="lines",
                    name=_lg_rec,
                    legendgroup=_lg_rec,
                    line=dict(color=color, dash="solid", width=2),
                )
            )
            # SAME legendgroup as the mean so a legend click toggles both together.
            fig1.add_trace(
                go.Scatter(
                    x=pd.concat([zt_h_r, zt_h_r[::-1]]),
                    y=pd.concat([rec["mean"] + rec["sem"], (rec["mean"] - rec["sem"])[::-1]]),
                    fill="toself",
                    line=dict(color="rgba(0,0,0,0)"),
                    fillcolor="rgba(100,100,200,0.10)",
                    legendgroup=_lg_rec,
                    showlegend=False,
                )
            )

    fig1.update_layout(
        xaxis_title="ZT (hours)",
        yaxis_title="Sleep (fraction of bin)",
        xaxis=dict(range=[0, 24], dtick=2),
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    if title:
        fig1.update_layout(title=_titled(title, "Baseline vs recovery"))
    return fig1


def _difference_fig(title, groups, label_of, colour_of):
    n_rec = len(diff_profs)
    fig2 = make_subplots(
        rows=1,
        cols=n_rec,
        subplot_titles=[f"Recovery Day {r}" for r in diff_profs],
        shared_yaxes=True,
    )
    _shade_ld(fig2)

    for col_idx, diff_df in enumerate(diff_profs.values(), start=1):
        for grp in groups:
            d = diff_df[diff_df["group"] == grp].sort_values("zt_bin_minute")
            if d.empty:
                continue
            zt_h = dam_utilities.zt_bin_to_hours(d["zt_bin_minute"], _bin)
            color = colour_of(grp)
            fig2.add_trace(
                go.Scatter(
                    x=zt_h,
                    y=d["mean"],
                    mode="lines",
                    name=label_of(grp) if col_idx == 1 else None,
                    showlegend=(col_idx == 1),
                    line=dict(color=color),
                ),
                row=1,
                col=col_idx,
            )
            fig2.add_trace(
                go.Scatter(
                    x=zt_h,
                    y=[0] * len(zt_h),
                    mode="lines",
                    line=dict(color="gray", dash="dot", width=1),
                    showlegend=False,
                ),
                row=1,
                col=col_idx,
            )

    fig2.update_xaxes(title_text="ZT (hours)", range=[0, 24], dtick=4)
    fig2.update_yaxes(title_text="Sleep Difference (fraction)", col=1)
    fig2.update_layout(height=400)
    if title:
        fig2.update_layout(title=_titled(title, "Recovery − baseline"))
    return fig2


def _cumulative_fig(title, groups, label_of, colour_of):
    fig3 = go.Figure()
    _shade_ld(fig3)

    for rec_num, cum_df in cum_diffs.items():
        for grp in groups:
            c = cum_df[cum_df["group"] == grp].sort_values("zt_bin_minute")
            if c.empty:
                continue
            zt_h = dam_utilities.zt_bin_to_hours(c["zt_bin_minute"], _bin)
            color = colour_of(grp)
            fig3.add_trace(
                go.Scatter(
                    x=zt_h,
                    y=c["mean"],
                    mode="lines",
                    name=f"{label_of(grp)} — Recovery Day {rec_num}",
                    line=dict(color=color),
                )
            )

    fig3.add_hline(y=0, line_dash="dot", line_color="gray")
    fig3.update_layout(
        xaxis_title="ZT (hours)",
        yaxis_title="Cumulative Sleep Difference (min)",
        xaxis=dict(range=[0, 24], dtick=2),
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    if title:
        fig3.update_layout(title=_titled(title, "Cumulative difference"))
    return fig3


def _shared(pairs):
    if _facet_rows and facet_spec.shared_y:
        plotting.apply_shared_y([f for _, f in pairs])
    return pairs


# Per-fly tables; a result computed before they existed has none, and gets no tests.
_empty = pd.DataFrame()
totals_per_fly = results.get("phase_totals_per_fly", _empty)
rebound_per_fly = results.get("rebound_per_fly", _empty)
cumulative_end = results.get("cumulative_end_per_fly", _empty)
_sd_family = stats_ui.family(group_stats.WELCH)
_PHASE_COLS = (("light_sleep_min", "Light Phase (ZT0-12)"), ("dark_sleep_min", "Dark Phase (ZT12-24)"))
_REB_PHASES = sd_module.PHASES


def _between(frame, col, groups, title, measure):
    """The groups of one panel compared on ``col`` of a per-fly frame."""
    if frame.empty:
        return None
    rows = frame[frame["group"].isin(groups)]
    present = [g for g in groups if g in set(rows["group"])]
    return group_stats.compare_frame(
        rows, col, "group", _sd_family, order=present, measure=measure, panel=title or None
    )


def _days(frame, col="day_label"):
    return list(dict.fromkeys(frame[col])) if not frame.empty else []


def _totals_stats(title, groups):
    """Between groups per (day, phase); then each group's baseline against each
    recovery day on the same flies, Holm across recovery days."""
    between = {
        (day, col): _between(
            totals_per_fly[totals_per_fly["day_label"] == day], col, groups, title,
            f"{day} — {name}",
        )
        for day in _days(totals_per_fly)
        for col, name in _PHASE_COLS
    }
    paired = []
    if not totals_per_fly.empty:
        base = totals_per_fly[totals_per_fly["day_label"] == sd_module.BASELINE_LABEL]
        for grp in groups:
            for col, name in _PHASE_COLS:
                b = base[base["group"] == grp].set_index("id")[col]
                tests = []
                for day in _days(totals_per_fly):
                    if day == sd_module.BASELINE_LABEL:
                        continue
                    rec = totals_per_fly[
                        (totals_per_fly["day_label"] == day) & (totals_per_fly["group"] == grp)
                    ].set_index("id")[col]
                    r = group_stats.compare_paired(
                        b, rec, _sd_family, labels=(sd_module.BASELINE_LABEL, day),
                        measure=f"{grp} — {name}",
                    )
                    if r is not None:
                        r.panel = title or None
                    tests.append(r)
                paired += group_stats.holm(tests, "recovery days")
    return between, [r for r in paired if r is not None]


def _rebound_stats(title, groups):
    return {
        (day, phase): _between(
            rebound_per_fly[
                (rebound_per_fly["recovery_day"] == day) & (rebound_per_fly["phase"] == phase)
            ],
            "rebound_pct", groups, title, f"Recovery Day {day} — {phase}",
        )
        for day in _days(rebound_per_fly, "recovery_day")
        for phase in _REB_PHASES
    }


def _cumulative_stats(title, groups):
    return [
        _between(
            cumulative_end[cumulative_end["recovery_day"] == day], "cumulative_diff_min",
            groups, title, f"Recovery Day {day} — cumulative difference at ZT24 (min)",
        )
        for day in _days(cumulative_end, "recovery_day")
    ]


# ================================================================
# Tab 1: ZT Time-Course Plots
# ================================================================
if tab_timecourse.open:
    with tab_timecourse:
        # --- 1a. Baseline vs Recovery line plot ---
        st.markdown("#### Baseline vs Recovery Sleep Profiles")
        _show(
            _shared([(t, _baseline_recovery_fig(t, *rest)) for t, *rest in _sets()]),
            "sd_baseline_recovery",
        )

        # --- 1b. Difference plot (recovery − baseline) ---
        st.markdown("#### Sleep Difference (Recovery − Baseline)")
        # Each is a row of recovery-day subplots, so it cannot go into a combined
        # grid, and apply_shared_y leaves multi-axis figures alone.
        _show(
            [(t, _difference_fig(t, *rest)) for t, *rest in _sets()],
            "sd_difference",
            grid=False,
        )

        # --- 1c. Cumulative sleep difference ---
        st.markdown("#### Cumulative Sleep Difference (minutes)")
        _show(
            _shared([(t, _cumulative_fig(t, *rest)) for t, *rest in _sets()]),
            "sd_cumulative",
        )
        _cum_stats = [r for t, gs, *_ in _sets() for r in _cumulative_stats(t, gs)]
        if any(r is not None for r in _cum_stats):
            st.dataframe(group_stats.stats_table(_cum_stats), width="stretch", hide_index=True)
            stats_ui.notice(_cum_stats)


def _grouped_bars(title, groups, label_of, days, bars, stats, yaxis_title, height):
    """Bars clustered by day: one category per (day, group), one trace per phase.

    ``bars`` is ``[(trace name, colour, {(day, group): value})]``; ``stats`` maps
    ``(day, trace index)`` to the between-group result drawn over that day.
    """
    cats, x_of_cat = [], {}
    for day in days:
        for grp in groups:
            if any((day, grp) in values for _, _, values in bars):
                x_of_cat[(day, grp)] = len(cats)
                cats.append(f"{label_of(grp)}<br>{day}")
    fig = go.Figure()
    for name, colour, values in bars:
        fig.add_trace(
            go.Bar(
                name=name,
                x=cats,
                y=[values.get(key, np.nan) for key in x_of_cat],
                marker_color=colour,
            )
        )
    plotting.apply_category_ticks(fig, cats)
    fig.update_layout(
        barmode="group",
        yaxis_title=yaxis_title,
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    # Two phases share each category: barmode="group" puts them at -0.2 and +0.2.
    offsets = np.linspace(-0.2, 0.2, len(bars)) if len(bars) > 1 else [0.0]
    every = [v for _, _, values in bars for v in values.values() if np.isfinite(v)]
    span = (max(every + [0.0]) - min(every + [0.0])) or 1.0
    for day in days:
        heights = [values.get((day, g), np.nan) for _, _, values in bars for g in groups]
        top = max([v for v in heights if np.isfinite(v)] + [0.0])
        for i in range(len(bars)):
            result = stats.get((day, i))
            if result is None:
                continue
            x_of = {g: x_of_cat[(day, g)] + offsets[i] for g in groups if (day, g) in x_of_cat}
            reached = plotting.add_group_stats(
                fig, result, x_of, n_slots=len(cats) * len(bars), y_base=top, span=span
            )
            top = reached if reached is not None else top
    if title:
        fig.update_layout(title=_titled(title, yaxis_title))
    return fig


def _totals_fig(title, groups, label_of):
    between, _ = _totals_stats(title, groups)
    days = _days(phase_totals)
    bars = [
        (
            name,
            "gold" if i == 0 else "navy",
            {
                (row["day_label"], row["group"]): row[col]
                for _, row in phase_totals[phase_totals["group"].isin(groups)].iterrows()
            },
        )
        for i, (col, name) in enumerate(_PHASE_COLS)
    ]
    stats = {(day, i): between.get((day, col)) for day in days for i, (col, _) in enumerate(_PHASE_COLS)}
    return _grouped_bars(title, groups, label_of, days, bars, stats, "Total Sleep (minutes)", 500)


def _rebound_fig(title, groups, label_of):
    reb_stats = _rebound_stats(title, groups)
    rec_days = _days(rebound_pct, "recovery_day")
    sub = rebound_pct[rebound_pct["group"].isin(groups)]
    bars = [
        (
            phase,
            "gold" if "Light" in phase else "navy",
            {
                (f"Recovery Day {row['recovery_day']}", row["group"]): row["rebound_pct"]
                for _, row in sub[sub["phase"] == phase].iterrows()
            },
        )
        for phase in _REB_PHASES
    ]
    stats = {
        (f"Recovery Day {day}", i): reb_stats.get((day, phase))
        for day in rec_days
        for i, phase in enumerate(_REB_PHASES)
    }
    fig = _grouped_bars(
        title, groups, label_of, [f"Recovery Day {d}" for d in rec_days], bars, stats,
        "Sleep Rebound (%)", 450,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    return fig


# ================================================================
# Tab 2: Bar Graphs
# ================================================================
if tab_bars.open:
    with tab_bars:
        st.markdown("#### Sleep Totals by Phase (Light / Dark)")

        if phase_totals.empty:
            st.warning("No phase total data available.")
        else:
            _bar_groups = set(phase_totals["group"].unique())
            if _facet_rows:
                _bar_sets = [(t, [g for g in gs if g in _bar_groups], lab) for t, gs, lab, _ in _sets()]
            else:
                _bar_sets = [("", sorted(_bar_groups), str)]
            _show(
                _shared([(t, _totals_fig(t, gs, lab)) for t, gs, lab in _bar_sets if gs]),
                "sd_totals",
            )
            _tot = [_totals_stats(t, gs) for t, gs, _ in _bar_sets if gs]
            stats_ui.notice([r for between, _ in _tot for r in between.values()])
            _paired = [r for _, paired in _tot for r in paired]
            if _paired:
                st.markdown("##### Baseline vs recovery, same flies")
                st.dataframe(group_stats.stats_table(_paired), width="stretch", hide_index=True)
                stats_ui.notice(_paired)

        # --- Rebound % bar chart ---
        st.markdown("#### Sleep Rebound (%)")
        if rebound_pct.empty:
            st.warning("No rebound data available.")
        else:
            if _facet_rows:
                _reb_sets = [(t, gs, lab) for t, gs, lab, _ in _sets()]
            else:
                _reb_sets = [("", list(rebound_pct["group"].unique()), str)]
            _show(
                _shared([(t, _rebound_fig(t, gs, lab)) for t, gs, lab in _reb_sets]),
                "sd_rebound",
            )
            stats_ui.notice([r for t, gs, _ in _reb_sets for r in _rebound_stats(t, gs).values()])

# ================================================================
# Tab 3: Data Export
# ================================================================
if tab_export.open:
    with tab_export:
        st.markdown("#### Download Analysis Data")

        # Baseline profile
        st.download_button(
            "Download Baseline ZT Profile",
            ex.excel_bytes([("baseline_profile", baseline_prof)]),
            "sd_baseline_profile.xlsx",
            ex.XLSX_MIME,
            key="dl_bl",
        )

        # Recovery profiles
        for rec_num, rec_df in recovery_profs.items():
            st.download_button(
                f"Download Recovery Day {rec_num} ZT Profile",
                ex.excel_bytes([(f"recovery_day{rec_num}_profile", rec_df)]),
                f"sd_recovery_day{rec_num}_profile.xlsx",
                ex.XLSX_MIME,
                key=f"dl_rec_{rec_num}",
            )

        # Difference profiles
        for rec_num, diff_df in diff_profs.items():
            st.download_button(
                f"Download Difference Day {rec_num}",
                ex.excel_bytes([(f"difference_day{rec_num}", diff_df)]),
                f"sd_difference_day{rec_num}.xlsx",
                ex.XLSX_MIME,
                key=f"dl_diff_{rec_num}",
            )

        # Cumulative difference
        for rec_num, cum_df in cum_diffs.items():
            _end = (
                cumulative_end[cumulative_end["recovery_day"] == rec_num]
                if not cumulative_end.empty
                else cumulative_end
            )
            _cum_day_stats = [
                _between(
                    _end, "cumulative_diff_min", gs, t,
                    f"Recovery Day {rec_num} — cumulative difference at ZT24 (min)",
                )
                for t, gs, *_ in _sets()
            ]
            st.download_button(
                f"Download Cumulative Diff Day {rec_num}",
                ex.excel_bytes(
                    [
                        (f"cumulative_diff_day{rec_num}", cum_df),
                        ("per_fly_end_of_day", _end),
                        stats_ui.sheet(_cum_day_stats),
                    ]
                ),
                f"sd_cumulative_diff_day{rec_num}.xlsx",
                ex.XLSX_MIME,
                key=f"dl_cum_{rec_num}",
            )

        # Phase totals
        if not phase_totals.empty:
            _tot_all = [_totals_stats(t, gs) for t, gs, *_ in _sets()]
            st.download_button(
                "Download Phase Totals",
                ex.excel_bytes(
                    [
                        ("phase_totals", phase_totals),
                        ("per_fly", totals_per_fly),
                        stats_ui.sheet(
                            [r for between, paired in _tot_all for r in [*between.values(), *paired]]
                        ),
                    ]
                ),
                "sd_phase_totals.xlsx",
                ex.XLSX_MIME,
                key="dl_phase",
            )

        # Rebound %
        if not rebound_pct.empty:
            st.download_button(
                "Download Rebound %",
                ex.excel_bytes(
                    [
                        ("rebound_pct", rebound_pct),
                        ("per_fly", rebound_per_fly),
                        stats_ui.sheet(
                            [r for t, gs, *_ in _sets() for r in _rebound_stats(t, gs).values()]
                        ),
                    ]
                ),
                "sd_rebound_pct.xlsx",
                ex.XLSX_MIME,
                key="dl_reb",
            )
