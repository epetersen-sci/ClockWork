"""
Phase Shift Analysis — measure each fly's phase shift after a light pulse.

Requires a ``pulse_time`` column in the metadata (a ZT hour such as ZT15) plus
``first_DD_day``, which anchors that ZT to the last entrained day before DD release.
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Add core/ and app/ to import path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORE_DIR = os.path.join(PROJECT_ROOT, "core")
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [CORE_DIR, APP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import dam_utilities
import phase_shift as ps_module
import plotting
from calibrations import (
    DEFAULT_PHASE_SHIFT_FILTER_HOURS,
    DEFAULT_PHASE_SHIFT_MARKER_SEARCH_HALF_WIDTH_HOURS,
    DEFAULT_PHASE_SHIFT_MIN_POST_DAYS,
    DEFAULT_PHASE_SHIFT_MIN_PRE_DAYS,
    DEFAULT_PHASE_SHIFT_ONSET_SMOOTH_MINUTES,
    DEFAULT_PHASE_SHIFT_ONSET_THRESHOLD_FRAC,
    DEFAULT_PHASE_SHIFT_PEAK_DISTANCE_HOURS,
    DEFAULT_PHASE_SHIFT_PEAK_PROMINENCE_FRAC,
    DEFAULT_PHASE_SHIFT_TRANSIENT_SKIP_DAYS,
)

st.header("Phase Shift Analysis")
st.caption(
    "Measures how far each fly's rhythm shifted after its light pulse, by comparing "
    "the rhythm's daily phase before and after the pulse."
)

# ============================================================
# Section 1: Prerequisites
# ============================================================
if st.session_state.get("dataset") is None:
    st.warning("No dataset loaded. Go to **Data Loading** first.")
    st.stop()

ds = st.session_state.dataset

if "pulse_zt_hour" not in ds.coords:
    st.error(
        "This dataset has no **pulse_time** column. Phase-shift analysis needs to know at "
        "what circadian time the light pulse was given.\n\n"
        "Add a `pulse_time` column to your metadata file — a ZT hour such as `ZT15` — and "
        "optionally `pulse_duration_min`, then reload the dataset on the **Data Loading** "
        "page. See `metadata_template.csv` for the format. Leave the cell blank for any "
        "group that received no pulse."
    )
    st.stop()

if ds.attrs.get("time_is_relative_minutes", 0) != 1:
    st.error(
        "Phase-shift analysis needs the relative-minute time axis produced by the standard "
        "loading path. Reload the dataset on the **Data Loading** page."
    )
    st.stop()

# Pulse position per fly, in minutes from that fly's recording start.
try:
    ds_pulse = ds if "pulse_minute" in ds.coords else dam_utilities.add_pulse_metadata(ds)
except Exception as exc:  # surfacing the reason beats a blank page
    st.error(f"Could not derive the pulse position from the metadata: {exc}")
    st.stop()

pulse_minutes = np.asarray(ds_pulse["pulse_minute"].values, dtype=float)
n_pulsed = int(np.isfinite(pulse_minutes).sum())
n_total = pulse_minutes.size

if n_pulsed == 0:
    st.error(
        "Every fly's `pulse_time` is blank, so there is no pulse to measure a shift "
        "around. Fill in the pulse time for at least one group and reload."
    )
    st.stop()

minutes_axis = np.asarray(ds_pulse["time"].values, dtype=float)
n_days = int(np.floor((minutes_axis[-1] + 1) / 1440.0))

col_a, col_b, col_c = st.columns(3)
col_a.metric("Flies with a pulse", f"{n_pulsed} / {n_total}")
col_b.metric("Complete days", n_days)
_pulse_days = sorted({int(m // 1440) for m in pulse_minutes if np.isfinite(m)})
col_c.metric("Pulse on day", ", ".join(str(d) for d in _pulse_days))

if n_pulsed < n_total:
    st.info(
        f"{n_total - n_pulsed} fly/flies have no pulse time (unpulsed controls). They are "
        "reported with status `no_pulse` rather than dropped."
    )

_has_boundary = "split_minute" in ds_pulse.coords or "first_DD_day" in ds_pulse.coords
if _has_boundary:
    st.caption(
        "Each regression is kept inside a single LD/DD epoch, using the LD-DD boundary from "
        "`first_DD_day`. A fit spanning the boundary would average an entrained slope with a "
        "free-running one."
    )
else:
    st.caption(
        "No `first_DD_day` in the metadata, so all days are treated as one epoch. If this "
        "experiment released the flies into DD, add that column so the fits respect the "
        "boundary."
    )

if ds.attrs.get("phase_shift_method") is not None:
    st.info(
        f"Previous phase-shift analysis recorded — method **{ds.attrs['phase_shift_method']}**."
    )

# ============================================================
# Section 2: Reference — what the shifted phase is compared against
# ============================================================
st.subheader("Reference")

_pre_pulse_days = min(int(m // 1440) for m in pulse_minutes if np.isfinite(m))

REFERENCE_LABELS = {
    "Unpulsed control group (per day)": "control",
    "Each fly's own pre-pulse rhythm": "self",
}
reference_label = st.radio(
    "Compare the post-pulse phase against what?",
    list(REFERENCE_LABELS.keys()),
    index=0 if _pre_pulse_days < int(DEFAULT_PHASE_SHIFT_MIN_PRE_DAYS) else 1,
    horizontal=True,
    help="A phase shift is a difference from where the rhythm would have been. That "
    "reference can come from an unpulsed control cohort, or from extrapolating each "
    "fly's own pre-pulse trend.",
)
reference = REFERENCE_LABELS[reference_label]

if reference == "control":
    st.caption(
        "This is the lab's `peakphaseplot.m` comparison: average each group's activity, "
        "smooth it, take one peak per day, and report the pulsed group's peak time minus "
        "the unpulsed control's on the same day. Validated against the lab's own SCAMP "
        "exports — from day 1 on, the two agree to a median of 2 min (worst 12 min). "
        "Results are per group per day, not per fly."
    )
    if _pre_pulse_days < int(DEFAULT_PHASE_SHIFT_MIN_PRE_DAYS):
        st.info(
            f"This protocol leaves only **{_pre_pulse_days}** full day(s) before the pulse, "
            "which is too few to fit each fly's own pre-pulse rhythm — so the control-group "
            "reference is the appropriate choice here and is preselected."
        )
else:
    st.caption(
        "Per fly: fit the daily phase before the pulse, extrapolate it across the pulse, and "
        "measure how far the post-pulse rhythm sits from that line. Gives one shift per fly "
        "(so a distribution per group), but needs several pre-pulse days."
    )
    if _pre_pulse_days < int(DEFAULT_PHASE_SHIFT_MIN_PRE_DAYS):
        st.warning(
            f"Only **{_pre_pulse_days}** full day(s) precede the pulse in this dataset, below "
            f"the {int(DEFAULT_PHASE_SHIFT_MIN_PRE_DAYS)}-day minimum, so most or all flies "
            "will abstain with `insufficient_pre`. Use the control-group reference instead, or "
            "lower the minimum below (knowing the fit gets noisier)."
        )

st.subheader("Method")

if reference == "control":
    # peakphaseplot.m is a peak-matching method; there is no onset variant of it to
    # be faithful to, so don't offer one here.
    method = "peak"
    st.caption("Peak matching, as in `peakphaseplot.m`.")

    _labels, _cols = ps_module._group_labels(ds_pulse, ("genotype", "condition"))
    _groups = sorted(set(_labels))
    # Default to whatever looks like the unpulsed arm, so the common case is one click.
    _guess = next(
        (g for g in _groups if any(t in g.lower() for t in ("nolp", "no_lp", "control"))),
        _groups[0],
    )
    control_group = st.selectbox(
        "Unpulsed control group (the phase reference)",
        _groups,
        index=_groups.index(_guess),
        help="Every other group's daily peak time is reported relative to this one.",
    )
else:
    METHOD_LABELS = {
        "Peak matching (default)": "peak",
        "Onset regression (Aschoff / Daan-Pittendrigh)": "onset",
    }
    method_label = st.radio(
        "How should each day's phase be measured?",
        list(METHOD_LABELS.keys()),
        index=0,
        horizontal=True,
    )
    method = METHOD_LABELS[method_label]
    control_group = None

    if method == "peak":
        st.caption(
            "Peak matching: low-pass filters the activity trace and takes each day's peak — "
            "the lab's `peakphaseplot.m` approach (Levine et al. 2002), run per fly with the "
            "peaks matched automatically instead of by clicking. On the validation cohort this "
            "recovers a known shift to about 1 min median error."
        )
    else:
        st.warning(
            "Onset regression takes the start of each day's active phase. It agrees with peak "
            "matching in the typical case (~3 min median error on the validation cohort) but "
            "has a much worse tail — about 1 in 10 flies off by over 2.5 h — because the onset "
            "of a sparse 1-minute activity trace is a far less well-defined feature than a "
            "peak. Its calibrations are provisional. Check the actogram before trusting these "
            "numbers."
        )

with st.expander("Detection and fitting parameters"):
    st.caption(
        "Defaults come from `core/calibrations.py`, which documents where each value came "
        "from and which ones are provisional."
    )
    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown("**Fitting**")
        min_pre_days = st.number_input(
            "Minimum days before pulse",
            min_value=2,
            max_value=30,
            value=int(DEFAULT_PHASE_SHIFT_MIN_PRE_DAYS),
            help="Fewer usable days than this and the fly abstains instead of reporting a number.",
        )
        min_post_days = st.number_input(
            "Minimum days after pulse",
            min_value=2,
            max_value=30,
            value=int(DEFAULT_PHASE_SHIFT_MIN_POST_DAYS),
        )
        transient_skip_days = st.number_input(
            "Skip transient days after pulse",
            min_value=0,
            max_value=10,
            value=int(DEFAULT_PHASE_SHIFT_TRANSIENT_SKIP_DAYS),
            help="Excludes the first N days after the pulse from the fit, letting transient "
            "cycles pass before the new steady state is read. 0 = off.",
        )
        search_half_width_hours = st.number_input(
            "Day-to-day tracking window (± hours)",
            min_value=0.5,
            max_value=12.0,
            value=float(DEFAULT_PHASE_SHIFT_MARKER_SEARCH_HALF_WIDTH_HOURS),
            step=0.5,
            help="How far from the previous day's marker to look for the next one. Does not "
            "limit how large a shift can be measured — the post-pulse days are tracked "
            "independently of the pre-pulse days.",
        )
        override_ref = st.checkbox(
            "Override reference day",
            value=False,
            help="By default the two fitted lines are compared at the pulse day (the standard "
            "convention). Change only if you want the shift read at a different day.",
        )
        reference_day_index = None
        if override_ref:
            reference_day_index = st.number_input(
                "Reference day index", min_value=0, max_value=max(0, n_days - 1), value=0
            )
    with pc2:
        if method == "peak":
            st.markdown("**Peak detection**")
            filter_hours = st.number_input(
                "Low-pass filter (hours)",
                min_value=0.0,
                max_value=24.0,
                value=float(DEFAULT_PHASE_SHIFT_FILTER_HOURS),
                step=1.0,
                help="peakphaseplot.m's own default is 12 h. Raise it if false peaks appear.",
            )
            peak_prominence_frac = st.number_input(
                "Peak prominence (fraction of daily range)",
                min_value=0.0,
                max_value=1.0,
                value=float(DEFAULT_PHASE_SHIFT_PEAK_PROMINENCE_FRAC),
                step=0.01,
            )
            peak_distance_hours = st.number_input(
                "Minimum peak separation (hours)",
                min_value=1.0,
                max_value=24.0,
                value=float(DEFAULT_PHASE_SHIFT_PEAK_DISTANCE_HOURS),
                step=1.0,
            )
            onset_threshold_frac = DEFAULT_PHASE_SHIFT_ONSET_THRESHOLD_FRAC
            onset_smooth_minutes = DEFAULT_PHASE_SHIFT_ONSET_SMOOTH_MINUTES
        else:
            st.markdown("**Onset detection**")
            onset_threshold_frac = st.number_input(
                "Onset threshold (× mean activity)",
                min_value=0.05,
                max_value=6.0,
                value=float(DEFAULT_PHASE_SHIFT_ONSET_THRESHOLD_FRAC),
                step=0.05,
                help="Relative to the fly's own mean smoothed activity, so it transfers across "
                "flies and recorders. Provisional.",
            )
            onset_smooth_minutes = st.number_input(
                "Smoothing window (minutes)",
                min_value=5.0,
                max_value=360.0,
                value=float(DEFAULT_PHASE_SHIFT_ONSET_SMOOTH_MINUTES),
                step=15.0,
                help="Width of the rolling mean the onset is read from. Provisional.",
            )
            filter_hours = DEFAULT_PHASE_SHIFT_FILTER_HOURS
            peak_prominence_frac = DEFAULT_PHASE_SHIFT_PEAK_PROMINENCE_FRAC
            peak_distance_hours = DEFAULT_PHASE_SHIFT_PEAK_DISTANCE_HOURS

if reference == "control":
    params = dict(
        control_group=control_group,
        group_by=("genotype", "condition"),
        filter_hours=float(filter_hours),
        peak_prominence_frac=float(peak_prominence_frac),
        peak_distance_hours=float(peak_distance_hours),
        search_half_width_hours=float(search_half_width_hours),
    )
else:
    params = dict(
        method=method,
        reference_day_index=reference_day_index,
        min_pre_days=int(min_pre_days),
        min_post_days=int(min_post_days),
        transient_skip_days=int(transient_skip_days),
        search_half_width_hours=float(search_half_width_hours),
        filter_hours=float(filter_hours),
        peak_prominence_frac=float(peak_prominence_frac),
        peak_distance_hours=float(peak_distance_hours),
        onset_threshold_frac=float(onset_threshold_frac),
        onset_smooth_minutes=float(onset_smooth_minutes),
    )

# ============================================================
# Section 3: Run (control-referenced group comparison)
# ============================================================
if reference == "control":
    if st.button("Run Group Phase Comparison", type="primary"):
        with st.spinner("Comparing each group's daily peak time to the control..."):
            try:
                gres = ps_module.compute_group_phase_difference(ds_pulse, **params)
                st.session_state.phase_shift_group_results = gres
                ds.attrs["phase_shift_method"] = "group_peak_vs_control"
                ds.attrs["phase_shift_control_group"] = str(control_group)
                ds.attrs["phase_shift_filter_hours"] = float(filter_hours)
                st.session_state.dataset = ds
                st.success(f"Done: {gres['per_day']['group'].nunique()} groups compared.")
                st.rerun()
            except Exception as exc:
                st.error(f"Error: {exc}")
                st.stop()

    if "phase_shift_group_results" not in st.session_state:
        st.info("Pick the control group above and click **Run** to compare.")
        st.stop()

    gres = st.session_state.phase_shift_group_results
    per_day = gres["per_day"]

    st.subheader("Results")
    st.caption(
        f"Phase difference = each group's daily peak time minus **{gres['control_group']}**'s. "
        "Positive = later (delayed) than the control."
    )

    tab_plot, tab_table = st.tabs(["Phase Difference by Day", "Table & Export"])

    with tab_plot:
        fig = go.Figure()
        others = [g for g in sorted(per_day["group"].unique()) if g != gres["control_group"]]
        palette = [
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
        for i, grp in enumerate(others):
            sub = per_day[per_day["group"] == grp].sort_values("day_index")
            fig.add_trace(
                go.Scatter(
                    x=sub["day_index"],
                    y=sub["phase_difference_hours"],
                    mode="lines+markers",
                    name=grp,
                    line=dict(color=palette[i % len(palette)]),
                )
            )
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        _pulse_day = _pulse_days[0] if _pulse_days else None
        if _pulse_day is not None:
            fig.add_vline(
                x=_pulse_day,
                line_dash="dash",
                line_color="goldenrod",
                annotation_text="pulse",
            )
        fig.update_layout(
            title=f"Phase difference vs {gres['control_group']}",
            xaxis_title="day",
            yaxis_title="phase difference (hours)",
            height=460,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Day 0 is flagged `filter_edge`: the smoothing filter pads the start of the "
            "record, so the first day's peak rests partly on synthetic padding and its "
            "position is not well determined. It is a pre-pulse baseline day and carries no "
            "shift information — read the trend from day 1 on."
        )

    with tab_table:
        st.dataframe(per_day, use_container_width=True)
        st.download_button(
            "Download per-day phase differences (CSV)",
            per_day.to_csv(index=False).encode("utf-8"),
            file_name="phase_difference_vs_control_per_day.csv",
            mime="text/csv",
        )
        st.markdown("**Parameters used**")
        st.json({k: (list(v) if isinstance(v, tuple) else v) for k, v in gres["params"].items()})
    st.stop()

# ============================================================
# Section 3: Preview one fly before committing to the cohort
# ============================================================
st.subheader("Preview")
st.caption(
    "Check one fly's markers and fits before running the whole cohort — if the markers are "
    "landing in the wrong place, the parameters need adjusting, not the cohort re-run."
)

fly_ids = [str(i) for i in ds_pulse["id"].values]
pulsed_ids = [f for f, m in zip(fly_ids, pulse_minutes) if np.isfinite(m)]
preview_id = st.selectbox("Fly", pulsed_ids, index=0)

if st.checkbox("Show preview actogram", value=True):
    with st.spinner("Computing preview..."):
        try:
            one = ds_pulse.sel(id=[preview_id])
            res1 = ps_module.compute_phase_shift_analysis(one, **params)
            row1 = res1["per_fly"].iloc[0]
            minutes = np.asarray(one["time"].values, dtype=float)
            values = np.asarray(one["activity"].transpose("time", "id").values[:, 0], dtype=float)
            shift_txt = (
                f"{row1['phase_shift_hours']:+.2f} h"
                if row1["status"] == "ok"
                else f"no value ({row1['status']})"
            )
            fig1 = plotting.phase_shift_actogram(
                minutes,
                values,
                day_markers=res1["day_markers"].get(preview_id),
                pre_fit=res1["fits"].get(preview_id, {}).get("pre"),
                post_fit=res1["fits"].get(preview_id, {}).get("post"),
                pulse_minute=row1.get("pulse_minute"),
                pulse_duration_minutes=(
                    float(one["pulse_duration_minutes"].values[0])
                    if "pulse_duration_minutes" in one.coords
                    else None
                ),
                reference_day_index=(
                    int(row1["reference_day_index"])
                    if np.isfinite(row1.get("reference_day_index", np.nan))
                    else None
                ),
                title=f"{preview_id} — {method} method, shift {shift_txt}",
            )
            st.plotly_chart(fig1, use_container_width=True)
            if row1["status"] == "ok":
                m1, m2, m3 = st.columns(3)
                m1.metric("Phase shift", f"{row1['phase_shift_hours']:+.2f} h")
                m2.metric("Period before", f"{row1['pre_period_hours']:.2f} h")
                m3.metric("Period after", f"{row1['post_period_hours']:.2f} h")
            else:
                st.warning(f"No phase shift for this fly: **{row1['status']}**")
        except Exception as exc:
            st.error(f"Preview failed: {exc}")

# ============================================================
# Section 4: Run
# ============================================================
if st.button("Run Phase Shift Analysis", type="primary"):
    with st.spinner(f"Measuring phase shifts for {n_total} flies..."):
        try:
            results = ps_module.compute_phase_shift_analysis(ds_pulse, **params)
            st.session_state.phase_shift_results = results

            # Record the parameters (scalars only) so the run is reproducible and the
            # sidebar can report the analysis as done. Results themselves stay in
            # session state — same convention as the Sleep Deprivation page.
            for key, value in results["params"].items():
                if value is None:
                    continue
                ds.attrs[f"phase_shift_{key}"] = int(value) if isinstance(value, bool) else value
            ds.attrs["phase_shift_method"] = results["method"]
            st.session_state.dataset = ds

            n_ok = int((results["per_fly"]["status"] == "ok").sum())
            st.success(f"Done: {n_ok} of {len(results['per_fly'])} flies produced a phase shift.")
            st.rerun()
        except Exception as exc:
            st.error(f"Error: {exc}")
            st.stop()

# ============================================================
# Section 5: Results
# ============================================================
if "phase_shift_results" not in st.session_state:
    st.info("Set the method above and click **Run** to analyse the whole cohort.")
    st.stop()

results = st.session_state.phase_shift_results
per_fly = results["per_fly"]

st.subheader("Results")

tab_summary, tab_actogram, tab_export = st.tabs(["Summary", "Per-Fly Actogram", "Data Export"])

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

with tab_summary:
    ok = per_fly[per_fly["status"] == "ok"]
    status_counts = per_fly["status"].value_counts()

    if len(ok) == 0:
        st.warning("No fly produced a usable phase shift.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Flies with a shift", f"{len(ok)} / {len(per_fly)}")
        c2.metric("Median shift", f"{ok['phase_shift_hours'].median():+.2f} h")
        c3.metric("Median period after", f"{ok['post_period_hours'].median():.2f} h")

        groups = sorted(ok["group"].dropna().unique())
        fig = go.Figure()
        for i, grp in enumerate(groups):
            vals = ok.loc[ok["group"] == grp, "phase_shift_hours"]
            fig.add_trace(
                go.Box(
                    y=vals,
                    name=str(grp),
                    boxpoints="all",
                    jitter=0.4,
                    pointpos=0,
                    marker=dict(color=_COLORS[i % len(_COLORS)]),
                )
            )
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(
            title="Phase shift by group (positive = delay, negative = advance)",
            yaxis_title="phase shift (hours)",
            showlegend=False,
            height=460,
        )
        st.plotly_chart(fig, use_container_width=True)

    if len(status_counts) > 1 or "ok" not in status_counts:
        st.markdown("**Why some flies have no value**")
        _explain = {
            "ok": "phase shift measured",
            "no_pulse": "no pulse_time in the metadata (unpulsed control)",
            "pulse_outside_record": "the pulse time falls outside this fly's recording",
            "insufficient_pre": "too few usable days before the pulse",
            "insufficient_post": "too few usable days after the pulse",
            "implausible_period": "the fitted period was not circadian — the daily markers "
            "did not track a consistent rhythm",
        }
        for status, count in status_counts.items():
            st.markdown(f"- `{status}` — {count} fly/flies: {_explain.get(status, '')}")

    st.dataframe(per_fly, use_container_width=True)

with tab_actogram:
    st.caption(
        "The dashed line continues the pre-pulse rhythm across the pulse. The gap between it "
        "and the post-pulse line is the reported shift, so the number can be checked by eye."
    )
    view_ids = per_fly["fly_id"].tolist()
    view_id = st.selectbox("Fly", view_ids, index=0, key="actogram_fly")
    row = per_fly.set_index("fly_id").loc[view_id]

    if view_id not in results["day_markers"]:
        st.warning(f"No markers were detected for this fly (status `{row['status']}`).")
    else:
        one = ds_pulse.sel(id=[view_id])
        minutes = np.asarray(one["time"].values, dtype=float)
        values = np.asarray(one["activity"].transpose("time", "id").values[:, 0], dtype=float)
        shift_txt = (
            f"{row['phase_shift_hours']:+.2f} h"
            if row["status"] == "ok"
            else f"no value ({row['status']})"
        )
        fig = plotting.phase_shift_actogram(
            minutes,
            values,
            day_markers=results["day_markers"].get(view_id),
            pre_fit=results["fits"].get(view_id, {}).get("pre"),
            post_fit=results["fits"].get(view_id, {}).get("post"),
            pulse_minute=row.get("pulse_minute"),
            pulse_duration_minutes=(
                float(one["pulse_duration_minutes"].values[0])
                if "pulse_duration_minutes" in one.coords
                else None
            ),
            reference_day_index=(
                int(row["reference_day_index"])
                if np.isfinite(row.get("reference_day_index", np.nan))
                else None
            ),
            title=f"{view_id} ({row['group']}) — {results['method']} method, shift {shift_txt}",
        )
        st.plotly_chart(fig, use_container_width=True)

with tab_export:
    st.download_button(
        "Download per-fly phase shifts (CSV)",
        per_fly.to_csv(index=False).encode("utf-8"),
        file_name=f"phase_shift_{results['method']}_per_fly.csv",
        mime="text/csv",
    )

    marker_rows = [
        {"fly_id": fly, "day_index": day, "marker_minute": minute}
        for fly, markers in results["day_markers"].items()
        for day, minute in sorted(markers.items())
    ]
    if marker_rows:
        markers_df = pd.DataFrame(marker_rows)
        st.download_button(
            "Download daily phase markers (CSV)",
            markers_df.to_csv(index=False).encode("utf-8"),
            file_name=f"phase_shift_{results['method']}_daily_markers.csv",
            mime="text/csv",
            help="The per-day marker times the fits were built from — useful for checking "
            "a suspicious result.",
        )

    st.markdown("**Parameters used**")
    st.json(results["params"])
