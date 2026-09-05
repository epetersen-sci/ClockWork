"""
Sleep Deprivation Analysis — compare baseline sleep to post-SD recovery sleep.

Requires sleep analysis to be completed first (Sleep and Activity page).
"""

import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Add core/ and app/ to import path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORE_DIR = os.path.join(PROJECT_ROOT, "core")
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [CORE_DIR, APP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import dam_utilities
import sleep_deprivation as sd_module
from analysis_detection import detect_analyses

st.header("Sleep Deprivation Analysis")

# ============================================================
# Section 1: Prerequisites
# ============================================================
if st.session_state.get("dataset") is None:
    st.warning("No dataset loaded. Go to **Data Loading** first.")
    st.stop()

ds = st.session_state.dataset
analyses = detect_analyses(ds)

if not analyses.get("sleep", False):
    st.warning(
        "Sleep analysis has not been run. Go to **Sleep and Activity** and run it first."
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
from dataset_meta import PHASE_DD, dataset_phase

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
        st.plotly_chart(fig_sel, width="stretch")

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

tab_timecourse, tab_bars, tab_export = st.tabs(
    ["ZT Time-Course Plots", "Bar Graphs", "Data Export"]
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
# Tab 1: ZT Time-Course Plots
# ================================================================
with tab_timecourse:
    groups = sorted(baseline_prof["group"].unique())

    # --- 1a. Baseline vs Recovery line plot ---
    st.markdown("#### Baseline vs Recovery Sleep Profiles")
    fig1 = go.Figure()
    _shade_ld(fig1)

    for grp in groups:
        color = _group_color(groups, grp)
        bl = baseline_prof[baseline_prof["group"] == grp].sort_values("zt_bin_minute")
        zt_h = dam_utilities.zt_bin_to_hours(bl["zt_bin_minute"], _bin)

        # Baseline (dashed)
        _lg_bl = f"{grp} — Baseline"
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
            _lg_rec = f"{grp} — Recovery Day {rec_num}"
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
    st.plotly_chart(fig1, width="stretch")

    # --- 1b. Difference plot (recovery − baseline) ---
    st.markdown("#### Sleep Difference (Recovery − Baseline)")
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
            color = _group_color(groups, grp)
            fig2.add_trace(
                go.Scatter(
                    x=zt_h,
                    y=d["mean"],
                    mode="lines",
                    name=grp if col_idx == 1 else None,
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
    st.plotly_chart(fig2, width="stretch")

    # --- 1c. Cumulative sleep difference ---
    st.markdown("#### Cumulative Sleep Difference (minutes)")
    fig3 = go.Figure()
    _shade_ld(fig3)

    for rec_num, cum_df in cum_diffs.items():
        for grp in groups:
            c = cum_df[cum_df["group"] == grp].sort_values("zt_bin_minute")
            if c.empty:
                continue
            zt_h = dam_utilities.zt_bin_to_hours(c["zt_bin_minute"], _bin)
            color = _group_color(groups, grp)
            fig3.add_trace(
                go.Scatter(
                    x=zt_h,
                    y=c["mean"],
                    mode="lines",
                    name=f"{grp} — Recovery Day {rec_num}",
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
    st.plotly_chart(fig3, width="stretch")


# ================================================================
# Tab 2: Bar Graphs
# ================================================================
with tab_bars:
    st.markdown("#### Sleep Totals by Phase (Light / Dark)")

    if phase_totals.empty:
        st.warning("No phase total data available.")
    else:
        groups_bar = sorted(phase_totals["group"].unique())
        day_labels = phase_totals["day_label"].unique()

        fig_bar = go.Figure()
        bar_x = []
        light_vals = []
        dark_vals = []

        for grp in groups_bar:
            grp_data = phase_totals[phase_totals["group"] == grp]
            for _, row in grp_data.iterrows():
                bar_x.append(f"{grp}<br>{row['day_label']}")
                light_vals.append(row["light_sleep_min"])
                dark_vals.append(row["dark_sleep_min"])

        fig_bar.add_trace(
            go.Bar(
                name="Light Phase (ZT0-12)",
                x=bar_x,
                y=light_vals,
                marker_color="gold",
            )
        )
        fig_bar.add_trace(
            go.Bar(
                name="Dark Phase (ZT12-24)",
                x=bar_x,
                y=dark_vals,
                marker_color="navy",
            )
        )
        fig_bar.update_layout(
            barmode="group",
            yaxis_title="Total Sleep (minutes)",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_bar, width="stretch")

    # --- Rebound % bar chart ---
    st.markdown("#### Sleep Rebound (%)")
    if rebound_pct.empty:
        st.warning("No rebound data available.")
    else:
        fig_reb = go.Figure()
        for phase in rebound_pct["phase"].unique():
            ph_data = rebound_pct[rebound_pct["phase"] == phase]
            x_labels = [
                f"{row['group']}<br>Recovery Day {row['recovery_day']}"
                for _, row in ph_data.iterrows()
            ]
            fig_reb.add_trace(
                go.Bar(
                    name=phase,
                    x=x_labels,
                    y=ph_data["rebound_pct"],
                    marker_color="gold" if "Light" in phase else "navy",
                )
            )
        fig_reb.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_reb.update_layout(
            barmode="group",
            yaxis_title="Sleep Rebound (%)",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_reb, width="stretch")


# ================================================================
# Tab 3: Data Export
# ================================================================
with tab_export:
    st.markdown("#### Download Analysis Data")

    # Baseline profile
    csv_bl = baseline_prof.to_csv(index=False)
    st.download_button(
        "Download Baseline ZT Profile", csv_bl, "sd_baseline_profile.csv", "text/csv", key="dl_bl"
    )

    # Recovery profiles
    for rec_num, rec_df in recovery_profs.items():
        csv_rec = rec_df.to_csv(index=False)
        st.download_button(
            f"Download Recovery Day {rec_num} ZT Profile",
            csv_rec,
            f"sd_recovery_day{rec_num}_profile.csv",
            "text/csv",
            key=f"dl_rec_{rec_num}",
        )

    # Difference profiles
    for rec_num, diff_df in diff_profs.items():
        csv_diff = diff_df.to_csv(index=False)
        st.download_button(
            f"Download Difference Day {rec_num}",
            csv_diff,
            f"sd_difference_day{rec_num}.csv",
            "text/csv",
            key=f"dl_diff_{rec_num}",
        )

    # Cumulative difference
    for rec_num, cum_df in cum_diffs.items():
        csv_cum = cum_df.to_csv(index=False)
        st.download_button(
            f"Download Cumulative Diff Day {rec_num}",
            csv_cum,
            f"sd_cumulative_diff_day{rec_num}.csv",
            "text/csv",
            key=f"dl_cum_{rec_num}",
        )

    # Phase totals
    if not phase_totals.empty:
        csv_phase = phase_totals.to_csv(index=False)
        st.download_button(
            "Download Phase Totals", csv_phase, "sd_phase_totals.csv", "text/csv", key="dl_phase"
        )

    # Rebound %
    if not rebound_pct.empty:
        csv_reb = rebound_pct.to_csv(index=False)
        st.download_button(
            "Download Rebound %", csv_reb, "sd_rebound_pct.csv", "text/csv", key="dl_reb"
        )
