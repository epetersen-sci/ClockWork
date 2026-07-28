"""
Period Analysis Page - CWT, Lomb-Scargle, and Autocorrelation analyses.
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
import export_helpers as ex
import periodograms
import plotting as plotting_mod
from analysis_detection import detect_analyses
from calibrations import AC_RI_SCAMP_REFERENCE, DEFAULT_CWT_METHOD
from preprocessing import preprocess_activity
from preprocessing_widgets import render_preprocess_expander
from rhythmicity_classification import (
    DEFAULT_AC_RI_THRESHOLD,
    DEFAULT_CWT_RIDGE_THRESHOLD,
    DEFAULT_LS_POWER_THRESHOLD,
    classify_all,
    compare_thresholds,
    cwt_threshold_for,
    per_fly_classification_df,
    summarize_rhythmicity,
)

st.header("Period Analysis")

if st.session_state.get("dataset") is None:
    st.warning("No dataset loaded. Go to **Data Loading** first.")
    st.stop()

ds = st.session_state.dataset
analyses = detect_analyses(ds)

# ------------------------------------------------------------------ #
# Plain-language method descriptions (mechanism / main use / major
# drawback), shown under each method's heading so a user reads what a
# method does BEFORE running it. Verified against the period-method
# literature. The Periodograms page renders the group-averaged spectra;
# these explanations live here, at the point of use.
# ------------------------------------------------------------------ #
METHOD_DESCRIPTIONS = {
    "cwt": (
        "**Continuous Wavelet Transform** compares the signal against a wave-packet "
        "stretched to many scales, producing a time-by-period map that shows how the "
        "dominant period changes over the recording. Its strength is capturing "
        "non-stationary rhythms — period drift, or rhythmicity that appears and "
        "disappears — that single-number methods average away. Its costs are a "
        "time-vs-frequency resolution tradeoff and edge artifacts near the start and "
        "end of the record."
    ),
    "ls": (
        "**Lomb-Scargle** fits a sine wave at each candidate period and scores how "
        "well it fits, building a periodogram whose tallest peak is the estimated "
        "period; a Baluev false-alarm probability flags whether that peak could be "
        "noise. Its strength is working directly on unevenly sampled or gappy data "
        "without interpolation. Its main limitation is assuming a roughly sinusoidal "
        "rhythm, so strongly non-sinusoidal activity can be mis-scored."
    ),
    "ac": (
        "**Autocorrelation** slides the activity signal against a time-shifted copy "
        "of itself; a rhythmic fly shows evenly spaced correlation peaks whose "
        "spacing gives the period, and the peak height (rhythmicity index) measures "
        "rhythm strength. It is robust, intuitive, and comes with a simple "
        "significance bound. Its drawback is coarser period precision and weaker "
        "performance on faint or drifting rhythms."
    ),
    "mesa": (
        "**MESA (Maximum Entropy Spectral Analysis)** fits a small predictive "
        "(autoregressive) model to the evenly sampled activity and computes the "
        "spectrum from that model, giving very sharp, well-resolved peaks even from "
        "short records. Its strength is high period precision and separating closely "
        "spaced periodicities. Its limitations are dependence on the chosen model "
        "order, a requirement for evenly sampled gap-free data, and no built-in test "
        "of whether a rhythm is real — so it is paired with autocorrelation or "
        "Lomb-Scargle for the rhythmic/arrhythmic call."
    ),
}

# ============================================================
# Data phase selection (LD / DD)
# ============================================================
import numpy as np

from dataset_meta import PHASE_DD, PHASE_LD, dataset_phase, is_split_applied

_ds_phase = dataset_phase(ds)
_has_split_datasets = (
    st.session_state.get("dataset_DD") is not None
    and st.session_state.get("dataset_LD") is not None
)
has_dd = "first_DD_day" in ds.coords

if _ds_phase in (PHASE_LD, PHASE_DD):
    # The loaded file is itself a partition — use it directly. No phase
    # picker needed: the choice was made when the file was saved.
    phase_selection = _ds_phase
    period_ds = ds
    if _ds_phase == PHASE_DD:
        st.success(
            f"Using the loaded **DD** dataset — "
            f"{len(period_ds['id'])} flies, {len(period_ds['time'])} timepoints."
        )
    else:
        st.warning(
            f"Using the loaded **LD** dataset — period estimates will reflect "
            f"the imposed light cycle, not the endogenous circadian period. "
            f"{len(period_ds['id'])} flies, {len(period_ds['time'])} timepoints."
        )
elif _has_split_datasets:
    # Pre-split datasets exist — let user choose which to use
    phase_choice = st.radio(
        "Data phase for period analysis",
        ["DD (recommended)", "LD"],
        index=0,
        horizontal=True,
        help="DD (constant darkness) is the standard for circadian period estimation "
        "(free-running rhythm). LD periods reflect the imposed light cycle.",
        key="period_phase",
    )
    phase_selection = "DD" if "DD" in phase_choice else "LD"
    if phase_selection == "DD":
        period_ds = st.session_state.dataset_DD
        st.success(
            f"Using **DD (constant darkness)** data — "
            f"{len(period_ds['id'])} flies, {len(period_ds['time'])} timepoints."
        )
    else:
        period_ds = st.session_state.dataset_LD
        st.warning(
            f"Using **LD (light-dark)** data — period estimates will reflect "
            f"the imposed light cycle, not the endogenous circadian period. "
            f"{len(period_ds['id'])} flies, {len(period_ds['time'])} timepoints."
        )
elif has_dd:
    # Dataset has an LD/DD transition. Whether the split was APPLIED must be read
    # from ROUND-TRIP-SAFE dataset state — `is_split_applied(ds)` inspects the
    # split_applied/split_phase attrs (both survive the NetCDF round-trip; C3) —
    # NOT the session-state dataset_LD/DD caches (`_has_split_datasets`), which are
    # derivative and ABSENT on a fresh .nc load. Keying off the caches made a
    # reloaded split-applied dataset falsely report "split not applied". The
    # analysis runs on-the-fly from the whole master via select_phase (below) in
    # BOTH cases, so only the message differs.
    if is_split_applied(ds):
        st.success(
            "LD/DD split applied — analysing the selected phase on-the-fly from the loaded dataset."
        )
    else:
        st.warning(
            "LD/DD split has **not** been applied on the Preprocessing page. "
            "Period analysis will split the data on-the-fly, but applying the split "
            "in Preprocessing first is recommended (for gap detection and consistency)."
        )
    phase_choice = st.radio(
        "Data phase for period analysis",
        ["DD (recommended)", "LD"],
        index=0,
        horizontal=True,
        help="DD (constant darkness) is the standard for circadian period estimation. "
        "LD periods reflect the imposed light cycle.",
        key="period_phase",
    )
    phase_selection = "DD" if "DD" in phase_choice else "LD"
    period_ds = ds  # analyses will split on-the-fly via _select_phase
else:
    st.info("No LD/DD transition found — using full dataset for period analysis.")
    phase_selection = "full"
    period_ds = ds

# ------------------------------------------------------------------ #
# Stage-2 phase API: build the ANALYSIS SOURCE from the WHOLE dataset.
# ------------------------------------------------------------------ #
# The analyses run on a per-fly NaN-masked VIEW of the whole dataset (the one core
# selector), NOT a pre-sliced/re-zeroed object. Feeding a re-zeroed slice and then
# letting the analysis re-mask it dropped the first phase days (Stage-1 foot-gun).
# Selecting the phase BEFORE preprocessing also means detrending sees only the
# in-phase region (correct baseline) — verified bit-identical to preprocessing the
# legacy slice. `period_ds` is kept only for reading existing results / display.
if _ds_phase in (PHASE_LD, PHASE_DD):
    # Loaded file is itself the single phase — analyse it as-is. Drop the boundary
    # so the selector treats it as already-selected (no re-derive / re-mask).
    _src = ds.drop_vars([c for c in ("first_DD_day", "split_minute") if c in ds.coords])
    _analysis_src, _ = dam_utilities.select_phase(_src, "auto")
    _period_phase = "auto"
elif phase_selection in ("DD", "LD"):
    _whole = st.session_state.dataset
    _analysis_src, _ = dam_utilities.select_phase(_whole, phase_selection)
    _period_phase = phase_selection
else:  # "full" — no LD/DD transition; analyse the whole recording
    _analysis_src, _ = dam_utilities.select_phase(ds, "auto")
    _period_phase = "auto"

# C2: the recording-duration summary is shown AFTER the DD-days floor filter
# below, so it describes the flies actually RETAINED (not the pre-filter master).

# Common period range
st.subheader("Period Range")
st.caption(
    "This range drives BOTH the period search and the classification window "
    "(they track together). Rejecting arrhythmic red-noise ramps is the "
    "metric's job (CWT `global_rednoise`), not the window's — so the range can "
    "be widened for long-period lines without inflating false positives. A "
    "short record cannot resolve the top of the band; CWT warns and analyses "
    "only the resolvable sub-band."
)
col1, col2 = st.columns(2)
with col1:
    min_period = st.number_input(
        "Min period (hours)",
        min_value=1.0,
        max_value=48.0,
        value=float(periodograms.DEFAULT_CWT_MIN_PERIOD),
        step=1.0,
    )
with col2:
    max_period = st.number_input(
        "Max period (hours)",
        min_value=1.0,
        max_value=72.0,
        value=float(periodograms.DEFAULT_CWT_MAX_PERIOD),
        step=1.0,
    )

if min_period >= max_period:
    st.error("Min period must be less than max period.")
    st.stop()

# C3: explain the knob at point of use — the default is circadian-focused; a known
# long-period line (e.g. ~43 h) reads ARRHYTHMIC at the default ceiling by design.
st.caption(
    f"Default {periodograms.DEFAULT_CWT_MIN_PERIOD:g}–{periodograms.DEFAULT_CWT_MAX_PERIOD:g} h "
    "(circadian). **Widen the max** for known long-period lines — a ~43 h rhythm reads "
    "arrhythmic at a 36 h ceiling; `global_rednoise` stays range-robust when widened. "
    "This range drives both the CWT search and the classify window (they track together)."
)

min_days_floor = st.number_input(
    "Minimum DD days for retention (recommended ≥ 4)",
    min_value=0.0,
    max_value=30.0,
    value=float(periodograms.DEFAULT_MIN_DD_DAYS_FLOOR),
    step=0.5,
    help="Flies whose longest analysable DD block is below this many days are EXCLUDED "
    "from period analysis (their period is not computed and they do not enter the "
    "period graphs/aggregates). This is your choice (§2c: each fly contributes its "
    "valid DD ≥ the floor). ~4 DD days is a reasonable floor — below it the period "
    "estimate has converged less toward the longest-record result. Set to 0 to "
    "retain every fly (no exclusion).",
    key="min_days_floor_shared",
)

# Gap-bridge ceiling — ONE user knob for the CWT/AC block extractor. Gaps ≤ this
# are bridged (worker-local linear interpolation of the interior NaN cells, NEVER
# written to the .nc — §2a); gaps larger are a hard break → the longest clean
# segment is analysed. LS does NOT use this (it is gap-native — see below).
max_bridge_gap = st.number_input(
    "Max gap to bridge (minutes)",
    min_value=0.0,
    max_value=120.0,
    value=float(periodograms.DEFAULT_MAX_BRIDGE_GAP_MINUTES),
    step=5.0,
    help="Interior gaps whose missing span is ≤ this many minutes are BRIDGED by "
    "worker-local linear interpolation of the interior NaN cells (this filled "
    "value feeds CWT/AC only — it is NEVER written to the .nc; stored truth "
    "stays NaN, §2a). Gaps LARGER than this are a hard break → the longest "
    "clean segment is analysed (so a real multi-day outage still severs the "
    "record). **0 = bridging off.** Validated ≤ 60 min on the control cohort; "
    "60–120 is user-adjustable but beyond the validated range. Applies to CWT "
    "and autocorrelation; Lomb-Scargle is gap-native and ignores this.",
    key="max_bridge_gap",
)


# B1 (2026-06-30): the floor is now a FILTER, not a flag. It is passed to the period
# kernels as min_num_days=min_days_floor (below), so a fly whose longest analysable DD
# block is under the floor returns NO period and is therefore absent from the downstream
# period graphs/aggregations (§3 functional exclusion). min_days_floor=0 → no exclusion.
def _dd_record_days(ds):
    """Per-fly valid-data span in days (robust, aligned to id). Used to report which
    flies the floor excludes; the kernel enforces the floor on the longest analysable
    block (equal to this span for a gap-free record)."""
    out = {}
    for _fid in ds["id"].values:
        _v = ds["activity"].sel(id=_fid).dropna("time")
        if _v.size >= 2:
            _t = _v["time"].values
            if np.issubdtype(_t.dtype, np.datetime64):
                _span = (np.datetime64(_t[-1]) - np.datetime64(_t[0])) / np.timedelta64(1, "D")
            else:
                _span = (float(_t[-1]) - float(_t[0])) / 1440.0
        else:
            _span = 0.0
        out[str(_fid)] = float(_span)
    return out


_fly_record_days = _dd_record_days(period_ds)
_floor_fids = (
    [f for f, d in _fly_record_days.items() if d < min_days_floor] if min_days_floor > 0 else []
)
if _floor_fids:
    st.info(
        f"**{len(_floor_fids)} of {len(_fly_record_days)} flies** have < "
        f"**{min_days_floor:g}** DD days and are **EXCLUDED** from period analysis "
        f"(no period computed; absent from the period graphs/aggregates). Lower the "
        f"floor to retain them."
    )

# C2: descriptive summary of the flies RETAINED after the floor filter (sequenced
# AFTER the B1 floor so it describes what actually ENTERS period analysis, not the
# pre-filter master). Excludes the sub-floor flies above.
_retained_days = {
    f: d for f, d in _fly_record_days.items() if (min_days_floor <= 0) or (d >= min_days_floor)
}
if _retained_days:
    _rd_arr = np.array(list(_retained_days.values()), dtype=float)
    _phase_word = {
        "DD": "DD (constant darkness)",
        "LD": "LD (light-dark)",
        "full": "full-recording",
    }.get(phase_selection, str(phase_selection))
    st.markdown(
        f"**{len(_retained_days)} flies** enter period analysis "
        f"({_phase_word} data): average record length "
        f"**{_rd_arr.mean():.1f} d**, range "
        f"**{_rd_arr.min():.1f}–{_rd_arr.max():.1f} d**."
        + (
            f" ({len(_floor_fids)} excluded below the {min_days_floor:g}-day floor.)"
            if _floor_fids
            else ""
        )
    )

# Shared preprocessing controls — one expander, one config per method.
# Defaults match the pre-refactor production behavior (LS/CWT no-op,
# AC 4-h Butterworth + linear detrend).
preprocess_configs = render_preprocess_expander("page3", expanded=False)

st.divider()


# ============================================================
# Helper: store analysis results on phase dataset and master
# ============================================================
def _merge_analysis_outputs(master, result_ds):
    """Attach analysis outputs from ``result_ds`` onto ``master`` without
    overwriting `master`'s `activity` / `time` / other shared variables.

    The CWT/LS/AC pipelines run on a *preprocessed* copy of the input
    (`preprocess_activity` may detrend/normalize/etc.) and return a
    merged dataset whose `activity` is the preprocessed version. If we
    blindly assigned `result_ds` back to the master we'd contaminate
    downstream visualization (e.g. `summary_bars` would sum detrended
    activity, producing negative totals for DD recordings — see issue
    1D in the period-analysis cleanup audit). This helper merges only
    the analysis-output variables and per-fly classification flags so
    raw activity stays untouched on master.
    """
    # Per-fly scalar data_vars (period, amplitude, FAP, RI, etc.)
    _scalar_vars = [v for v in result_ds.data_vars if result_ds[v].dims == ("id",)]
    # Per-fly array data_vars (cwt_powerseries, cwt_ridge_periods, etc.)
    # — anything keyed on id plus an analysis-specific axis.
    _array_vars = [
        v
        for v in result_ds.data_vars
        if v != "activity"
        and "id" in result_ds[v].dims
        and result_ds[v].dims != ("id",)
        and v.startswith(("cwt_", "ls_", "ac_", "mesa_"))
    ]
    # Per-fly classification coords (ac_rhythmic, ls_rhythmic, etc.)
    _scalar_coords = [
        c
        for c in result_ds.coords
        if c not in ("id", "time") and c in result_ds and result_ds[c].dims == ("id",)
    ]
    _to_drop = [v for v in (_scalar_vars + _array_vars + _scalar_coords) if v in master]
    if _to_drop:
        master = master.drop_vars(_to_drop, errors="ignore")
    if _scalar_vars or _array_vars:
        master = master.merge(result_ds[_scalar_vars + _array_vars])
    if _scalar_coords:
        master = master.assign_coords({c: result_ds[c] for c in _scalar_coords})
    # Copy analysis attrs (CWT/LS/AC/MESA + classification thresholds + paths)
    for k, v in result_ds.attrs.items():
        if k.startswith(("cwt_", "ls_", "ac_", "mesa_")):
            master.attrs[k] = v
    return master


def _store_period_results(result_ds):
    """Store analysis results on the phase-specific dataset and merge
    per-fly outputs onto the master dataset.

    Single-phase loads (e.g. DD-only NetCDFs) and split workflows take
    the same merge path so master's `activity` is never replaced with
    the preprocessed (detrended) version that the analysis pipelines
    work on internally. See ``_merge_analysis_outputs`` docstring."""
    # Stage-2: result_ds is the WHOLE-dataset masked-view + per-fly outputs (the
    # analysis no longer runs on a re-zeroed slice). Per-fly results are phase-
    # independent (id,)/(id, analysis-axis) vars, so MERGE them onto the existing
    # sliced phase dataset rather than replacing it — keeps dataset_DD/LD's sliced
    # time series intact for unmigrated downstream pages (transitional; retires
    # with the dataset_LD/DD sweep). _merge_analysis_outputs never touches activity.
    if _has_split_datasets or phase_selection in ("DD", "LD"):
        _name = "dataset_DD" if phase_selection == "DD" else "dataset_LD"
        _tgt = st.session_state.get(_name)
        if _tgt is not None:
            st.session_state[_name] = _merge_analysis_outputs(_tgt, result_ds)

    master = st.session_state.dataset
    master = _merge_analysis_outputs(master, result_ds)
    st.session_state.dataset = master
    st.session_state.analyses = detect_analyses(master)


# ============================================================
# CWT Analysis
# ============================================================
st.subheader("Continuous Wavelet Transform (CWT)")

st.markdown(METHOD_DESCRIPTIONS["cwt"])

st.caption(
    "Default `global_rednoise`: a range-robust local-prominence strength — "
    "the COI-excluded global-spectrum peak divided by the AR(1) red-noise floor "
    "at the peak period. Rejects arrhythmic band-edge ramps on a wide search."
)

# Check for existing CWT on the selected phase dataset
_cwt_exists = "cwt_period" in period_ds.data_vars
if _cwt_exists:
    _cwt_method = period_ds.attrs.get("cwt_method", "?")
    _cwt_phase = period_ds.attrs.get("cwt_phase", "?")
    st.info(
        f"CWT analysis already completed — "
        f"**{_cwt_phase}** phase, "
        f"**{period_ds.attrs.get('cwt_wavelet', '?')}** wavelet, "
        f"**{period_ds.attrs.get('cwt_min_period', '?')}–{period_ds.attrs.get('cwt_max_period', '?')}h** range, "
        f"method=**{_cwt_method}**"
    )
    rerun_cwt = st.checkbox("Re-run CWT with different parameters", value=False, key="rerun_cwt")
else:
    rerun_cwt = True

if rerun_cwt:
    with st.expander("Advanced — testing only", expanded=False):
        cwt_resolution_voices = st.select_slider(
            "CWT resolution (voices per octave)",
            options=[8, 10, 12, 16, 32, 64, 128, 256, 512, 1024],
            value=periodograms.DEFAULT_CWT_VOICES_PER_OCTAVE,
            format_func=lambda x: f"{x} voices/octave",
            help="Density of the logarithmic period grid (scales per doubling of "
            "period), spanning the Min/Max period range above. Higher = finer "
            "period readout but proportionally slower. Default "
            f"{periodograms.DEFAULT_CWT_VOICES_PER_OCTAVE} voices/octave "
            "(~0.5 h period steps near 24 h) is comfortably fine without the "
            "over-resolution of 512+; verified resolution-stable (CWT calls "
            "unchanged) vs the prior 10 v/oct.",
            key="cwt_resolution_voices",
        )

        _cwt_method_options = [
            "ar1",
            "global_rednoise",
            "global",
            "global_baseline",
            "global_neighbor",
            "ridge",
        ]
        # Default selection comes from the single calibration source
        # (core/calibrations.py: DEFAULT_CWT_METHOD), never a hardcoded index.
        _cwt_method_default_idx = (
            _cwt_method_options.index(DEFAULT_CWT_METHOD)
            if DEFAULT_CWT_METHOD in _cwt_method_options
            else 0
        )
        cwt_method = st.selectbox(
            "Reduction method",
            options=_cwt_method_options,
            index=_cwt_method_default_idx,
            help="`global_rednoise` (default): RANGE-ROBUST, length-stable "
            "local-prominence strength — peak / AR(1) red-noise expected "
            "power AT the peak period. Rejects arrhythmic band-edge ramps "
            "while searching WIDE (16-36 h); the strength form of `ar1` "
            "without its length-sensitive threshold. "
            "`ar1` (former default): Torrence & Compo 1998 AR(1) red-noise "
            "significance with continuous SNR score (length-sensitive). "
            "`global`: length-stable COI-excluded peak / MEAN band power — "
            "over-calls on a wide band (band-edge artifact). "
            "`global_baseline` / `global_neighbor`: other local-prominence "
            "candidates (comparison only; no calibrated cutoff). "
            "`ridge`: legacy Pythomics ridge tracker (median ridge period, "
            "valid-fraction rhythmicity).",
            key="cwt_method_select",
        )

    # ----- Group-averaged scalogram (Phase 3) -----
    cwt_avg_cols = st.columns(2)
    with cwt_avg_cols[0]:
        compute_group_averages = st.checkbox(
            "Compute group-averaged scalograms (saved to disk)",
            value=False,
            help="Folds per-group 2D averaging into the CWT pass and writes "
            "one PNG + matching CSV per group to "
            "'<working_dir>/Averaged Scalograms/'. Memory-bounded: "
            "per-fly 2D matrices are accumulated as running sums and "
            "discarded immediately. PNG (not interactive plotly) is "
            "used to bypass Streamlit's browser-payload ceiling.",
            key="cwt_compute_group_averages",
        )
    with cwt_avg_cols[1]:
        avg_filter_nonrhythmic = st.checkbox(
            "Filter non-rhythmic flies (AC RI gate) from average",
            value=True,
            help="Restricts each group's average to AC-rhythmic flies. "
            "Off-rhythm power spectra are dominated by noise and "
            "would smear the group mean. Disable when the biology "
            "of interest is loss of rhythmicity.",
            key="cwt_avg_filter_nonrhythmic",
            disabled=not compute_group_averages,
        )

    # Pre-flight: when filter is requested but ac_rhythmic is missing,
    # block the run. CWT can take many minutes; the user should have a
    # chance to cancel and run AC classification first rather than
    # discover after the fact that the filter no-opped.
    _ac_classified = "ac_rhythmic" in period_ds.coords
    _need_confirm = compute_group_averages and avg_filter_nonrhythmic and not _ac_classified
    if _need_confirm:
        st.warning(
            "**AC rhythmic classification has not been run.** "
            "Group averages would include every fly regardless of "
            "rhythmicity. Cancel and run AC classification (below), "
            "or check the box to proceed anyway."
        )
        proceed_anyway = st.checkbox(
            "Proceed without AC rhythmic filtering",
            value=False,
            key="cwt_avg_proceed_unfiltered",
        )
    else:
        proceed_anyway = True

    if st.button(
        "Run CWT Analysis", key="run_cwt", disabled=(_need_confirm and not proceed_anyway)
    ):
        cwt_progress = st.progress(0, text="Starting CWT...")
        with st.spinner("Running CWT analysis... This may take several minutes."):
            try:

                def _cwt_cb(completed, total):
                    cwt_progress.progress(completed / total, text=f"CWT: fly {completed}/{total}")

                _cwt_phase_arg = _period_phase
                # Apply shared preprocessing pipeline before CWT (on the phase view)
                ds_pp = preprocess_activity(_analysis_src, preprocess_configs["CWT"])
                # Resolve output folder for averaged scalograms.
                _avg_out_dir = None
                _phase_label = None
                if compute_group_averages:
                    # Save next to the source monitor data (survives a .nc reload
                    # via ds.attrs['source_data_dir']) — never Streamlit's launch dir.
                    _wd = dam_utilities.resolve_export_dir(ds, st.session_state.get("working_dir"))
                    _avg_out_dir = os.path.join(_wd, "Averaged Scalograms")
                    # Phase label encodes which slice we're CWT'ing so that
                    # DD and LD reruns don't silently overwrite each other.
                    if _has_split_datasets:
                        _phase_label = str(phase_selection)
                    else:
                        _phase_label = "full"
                result = periodograms.wavelet_analysis(
                    ds_pp,
                    min_period=min_period,
                    max_period=max_period,
                    cwt_method=cwt_method,
                    phase=_cwt_phase_arg,
                    resolution=1 / cwt_resolution_voices,
                    min_num_days=min_days_floor,  # B1: floor is a FILTER — exclude sub-floor flies
                    max_bridge_gap_minutes=max_bridge_gap,  # ONE gap-bridge ceiling (§2a-transient)
                    progress_callback=_cwt_cb,
                    compute_group_averages=bool(compute_group_averages),
                    group_coord="group",
                    filter_nonrhythmic_for_average=bool(avg_filter_nonrhythmic),
                    average_output_dir=_avg_out_dir,
                    phase_label=_phase_label,
                )
                period_ds = result
                _store_period_results(result)
                if compute_group_averages and _avg_out_dir:
                    st.success(
                        f"CWT analysis complete. Averaged scalograms saved "
                        f"to `{_avg_out_dir}` (one PNG + CSV per group)."
                    )
                else:
                    st.success("CWT analysis complete!")
                st.rerun()
            except Exception as e:
                st.error(f"CWT Error: {e}")
            finally:
                cwt_progress.empty()

st.divider()

# ============================================================
# Lomb-Scargle Analysis
# ============================================================
st.subheader("Lomb-Scargle Periodogram")

st.markdown(METHOD_DESCRIPTIONS["ls"])

st.caption(
    "Generalized Lomb-Scargle (Zechmeister-Kürster) with floating-mean fit "
    "per trial frequency. Baluev FAP on standard-normalised power. "
    "Oversampling=8 (Rethomics `ofac`). Z-scoring is configurable above and "
    "recommended for cross-recording amplitude comparability."
)

_ls_exists = "ls_period" in period_ds.data_vars
if _ls_exists:
    _ls_phase = period_ds.attrs.get("ls_phase", "?")
    _ls_oversampling = period_ds.attrs.get("ls_oversampling", "?")
    _ls_fap = period_ds.attrs.get("ls_fap_method", "?")
    st.info(
        f"Lomb-Scargle analysis already completed — "
        f"**{_ls_phase}** phase, "
        f"**{period_ds.attrs.get('ls_min_period', '?')}–{period_ds.attrs.get('ls_max_period', '?')}h** range, "
        f"oversampling=**{_ls_oversampling}**, FAP=**{_ls_fap}**"
    )
    rerun_ls = st.checkbox(
        "Re-run Lomb-Scargle with different parameters", value=False, key="rerun_ls"
    )
else:
    rerun_ls = True

if rerun_ls:
    with st.expander("Advanced LS knobs", expanded=False):
        ls_oversampling = st.number_input(
            "Oversampling (samples_per_peak)",
            min_value=1,
            max_value=64,
            value=8,
            step=1,
            key="ls_oversampling",
            help="Rethomics default is 8. Astropy default is 5.",
        )
        ls_fap_method = st.selectbox(
            "FAP method",
            options=["baluev", "naive", "bootstrap"],
            index=0,
            key="ls_fap_method",
            help="astropy false_alarm_probability method for "
            "normalization='standard'. `baluev` (Baluev 2008, default), "
            "`naive` (Šidák-corrected Beta tail), or `bootstrap` "
            "(slow but gold-standard).",
        )

    if st.button("Run Lomb-Scargle Analysis", key="run_ls"):
        ls_progress = st.progress(0, text="Starting Lomb-Scargle...")
        with st.spinner("Running Lomb-Scargle analysis... This may take several minutes."):
            try:

                def _ls_cb(completed, total):
                    ls_progress.progress(
                        completed / total, text=f"Lomb-Scargle: fly {completed}/{total}"
                    )

                _ls_phase_arg = _period_phase
                # Apply shared preprocessing pipeline before LS
                ds_pp = preprocess_activity(_analysis_src, preprocess_configs["LS"])
                result = periodograms.lomb_scargle_analysis(
                    ds_pp,
                    min_period=min_period,
                    max_period=max_period,
                    oversampling=ls_oversampling,
                    fap_method=ls_fap_method,
                    phase=_ls_phase_arg,
                    min_num_days=min_days_floor,  # B1: floor is a FILTER — exclude sub-floor flies
                    progress_callback=_ls_cb,
                )
                period_ds = result
                _store_period_results(result)
                st.success("Lomb-Scargle analysis complete!")
                st.rerun()
            except Exception as e:
                st.error(f"Lomb-Scargle Error: {e}")
            finally:
                ls_progress.empty()

st.divider()

# ============================================================
# Autocorrelation Analysis
# ============================================================
st.subheader("Autocorrelation")

st.markdown(METHOD_DESCRIPTIONS["ac"])

st.caption(
    "SCAMP-style: 4-h Butterworth low-pass + linear detrend + 2nd-day peak window "
    "[40,56]h, RI = peak autocorrelation, RS = RI / (1.965/√N) (matches "
    "`autoco.m` / `rindex_raw.m`)."
)

_ac_exists = "ac_period" in period_ds.data_vars
if _ac_exists:
    _ac_phase = period_ds.attrs.get("ac_phase", "?")
    _ac_method = period_ds.attrs.get("ac_method", "?")
    _ac_lopass = period_ds.attrs.get("prep_lopass_hours", "?")
    _ac_peak = period_ds.attrs.get("ac_peak", "?")
    st.info(
        f"Autocorrelation analysis already completed — "
        f"**{_ac_phase}** phase, "
        f"**{period_ds.attrs.get('ac_min_period', '?')}–{period_ds.attrs.get('ac_max_period', '?')}h** range, "
        f"method=**{_ac_method}**, low-pass=**{_ac_lopass}h**, peak=**day {_ac_peak}**"
    )
    rerun_ac = st.checkbox(
        "Re-run autocorrelation with different parameters", value=False, key="rerun_ac"
    )
else:
    rerun_ac = True

if rerun_ac:
    with st.expander("Advanced AC knobs", expanded=False):
        ac_peak = st.number_input(
            "Peak day (1=first AC peak, 2=second-day peak)",
            min_value=1,
            max_value=5,
            value=2,
            step=1,
            key="ac_peak",
            help="SCAMP `acplot.m` searches the 2nd-day peak (lag window [(k-1)·24+min, (k-1)·24+max] h, k=2 → [40,56]h).",
        )

    if st.button("Run Autocorrelation Analysis", key="run_ac"):
        ac_progress = st.progress(0, text="Starting autocorrelation...")
        with st.spinner("Running autocorrelation analysis... This may take several minutes."):
            try:

                def _ac_cb(completed, total):
                    ac_progress.progress(
                        completed / total, text=f"Autocorrelation: fly {completed}/{total}"
                    )

                _ac_phase_arg = _period_phase
                # Apply shared preprocessing pipeline before AC. Default
                # config matches SCAMP: 4-h Butterworth low-pass + linear detrend.
                ds_pp = preprocess_activity(_analysis_src, preprocess_configs["AC"])
                result = periodograms.autocorrelation_analysis(
                    ds_pp,
                    min_period=min_period,
                    max_period=max_period,
                    ac_peak=ac_peak,
                    phase=_ac_phase_arg,
                    min_num_days=min_days_floor,  # B1: floor is a FILTER — exclude sub-floor flies
                    max_bridge_gap_minutes=max_bridge_gap,  # ONE gap-bridge ceiling (§2a-transient)
                    progress_callback=_ac_cb,
                )
                period_ds = result
                _store_period_results(result)
                st.success("Autocorrelation analysis complete!")
                st.rerun()
            except Exception as e:
                st.error(f"Autocorrelation Error: {e}")
            finally:
                ac_progress.empty()

# ============================================================
# MESA (Maximum Entropy / Burg AR) Analysis
# ============================================================
st.divider()
st.subheader("MESA — Maximum Entropy Spectral Analysis")
st.markdown(METHOD_DESCRIPTIONS["mesa"])
st.caption(
    "Burg autoregressive spectral estimate (SCAMP pairs AC + MESA). Sharp period "
    "resolution from short records — but a period ESTIMATOR only (no significance "
    "test; rhythmic/arrhythmic gating stays with AC/LS). Activity is binned before "
    "the fit and uses the same detrend + low-pass preprocessing as autocorrelation."
)

_mesa_exists = "mesa_period" in period_ds.data_vars
if _mesa_exists:
    st.info(
        f"MESA analysis already completed — "
        f"**{period_ds.attrs.get('mesa_phase', '?')}** phase, "
        f"**{period_ds.attrs.get('mesa_min_period', '?')}–{period_ds.attrs.get('mesa_max_period', '?')}h** range, "
        f"bin=**{period_ds.attrs.get('mesa_bin_minutes', '?')} min**, "
        f"order=**{period_ds.attrs.get('mesa_order', '?')}**"
    )
    rerun_mesa = st.checkbox("Re-run MESA with different parameters", value=False, key="rerun_mesa")
else:
    rerun_mesa = True

if rerun_mesa:
    with st.expander("Advanced MESA knobs", expanded=False):
        mesa_bin = st.number_input(
            "Bin size (minutes)",
            min_value=5,
            max_value=60,
            value=int(periodograms.DEFAULT_MESA_BIN_MINUTES),
            step=5,
            key="mesa_bin",
            help="Activity is binned to this before the Burg fit. AR spectral estimation "
            "needs a modest number of samples per cycle; ~30 min is the chronobiology "
            "convention (Dowse & Ringo).",
        )
        mesa_order_choice = st.selectbox(
            "AR model order",
            ["N/3 (Dowse default)", "N/4", "FPE auto", "Fixed"],
            index=0,
            key="mesa_order_choice",
            help="Order controls resolution: too low merges peaks, too high invents "
            "spurious ones. N/3 is Dowse's circadian 'safe maximum'.",
        )
        mesa_fixed_order = None
        if mesa_order_choice == "Fixed":
            mesa_fixed_order = int(
                st.number_input(
                    "Fixed order",
                    min_value=2,
                    max_value=150,
                    value=24,
                    step=1,
                    key="mesa_fixed_order",
                )
            )

    if st.button("Run MESA Analysis", key="run_mesa"):
        _order_map = {
            "N/3 (Dowse default)": None,
            "N/4": "n_over_4",
            "FPE auto": "fpe",
            "Fixed": mesa_fixed_order,
        }
        mesa_progress = st.progress(0, text="Starting MESA...")
        with st.spinner("Running MESA analysis..."):
            try:

                def _mesa_cb(completed, total):
                    mesa_progress.progress(completed / total, text=f"MESA: fly {completed}/{total}")

                # MESA reuses AC's preprocessing (linear detrend + low-pass) — MESA
                # benefits from mean/trend removal; mean is also removed per-fly in
                # the worker.
                ds_pp = preprocess_activity(_analysis_src, preprocess_configs["AC"])
                result = periodograms.mesa_analysis(
                    ds_pp,
                    min_period=min_period,
                    max_period=max_period,
                    bin_minutes=mesa_bin,
                    order=_order_map[mesa_order_choice],
                    phase=_period_phase,
                    min_num_days=min_days_floor,  # B1: floor is a FILTER
                    max_bridge_gap_minutes=max_bridge_gap,  # ONE gap-bridge ceiling (§2a-transient)
                    progress_callback=_mesa_cb,
                )
                period_ds = result
                _store_period_results(result)
                st.success("MESA analysis complete!")
                st.rerun()
            except Exception as e:
                st.error(f"MESA Error: {e}")
            finally:
                mesa_progress.empty()

# ============================================================
# Summary of period results
# ============================================================
st.divider()
st.subheader("Period Analysis Summary")

# Use the phase-specific dataset for the summary (it has the results)
import pandas as pd

# Re-read period_ds in case analyses were run above
if _has_split_datasets:
    period_ds = (
        st.session_state.dataset_DD if phase_selection == "DD" else st.session_state.dataset_LD
    )
else:
    period_ds = st.session_state.dataset


@st.cache_data(show_spinner=False)
def _build_period_summary_df(_fp, _ds):
    """Build the per-fly Period Analysis summary table. Cached so a page
    rerun (no analysis-state change) doesn't redo the per-fly `.sel()`
    loop. The fingerprint key is invalidated whenever the analysis attrs
    or rhythmic flags change."""
    rows = []
    for fly_id in _ds["id"].values:
        row = {"ID": fly_id}
        if "group" in _ds.coords:
            row["Group"] = str(_ds["group"].sel(id=fly_id).values)
        if "cwt_period" in _ds.data_vars:
            row["CWT Period (h)"] = float(_ds["cwt_period"].sel(id=fly_id).values)
        if "ls_period" in _ds.data_vars:
            row["LS Period (h)"] = float(_ds["ls_period"].sel(id=fly_id).values)
        if "ac_period" in _ds.data_vars:
            row["AC Period (h)"] = float(_ds["ac_period"].sel(id=fly_id).values)
        if "mesa_period" in _ds.data_vars:
            row["MESA Period (h)"] = float(_ds["mesa_period"].sel(id=fly_id).values)
        # Per-algorithm STRENGTH metrics — the values the Interactive threshold
        # explorer plots (so this table is a superset of the explorer: no separate
        # export needed). AC RI = ac_power; LS power/FAP; CWT rhythmicity; MESA SNR.
        if "ac_power" in _ds.data_vars:
            row["AC RI (strength)"] = float(_ds["ac_power"].sel(id=fly_id).values)
        if "ls_power" in _ds.data_vars:
            row["LS Power (strength)"] = float(_ds["ls_power"].sel(id=fly_id).values)
        if "ls_fap" in _ds.data_vars:
            row["LS FAP"] = float(_ds["ls_fap"].sel(id=fly_id).values)
        if "cwt_rhythmicity" in _ds.data_vars:
            row["CWT Rhythmicity (strength)"] = float(_ds["cwt_rhythmicity"].sel(id=fly_id).values)
        if "mesa_power" in _ds.data_vars:
            row["MESA SNR (peak/median)"] = float(_ds["mesa_power"].sel(id=fly_id).values)
        # Per-algorithm rhythmic flags. AC is the canonical filter; LS/CWT
        # appear here for diagnostic comparison only and never gate downstream.
        if "ac_rhythmic" in _ds.coords:
            row["AC Rhythmic"] = bool(_ds["ac_rhythmic"].sel(id=fly_id).values)
        if "ls_rhythmic" in _ds.coords:
            row["LS Rhythmic (diagnostic)"] = bool(_ds["ls_rhythmic"].sel(id=fly_id).values)
        if "cwt_rhythmic" in _ds.coords:
            row["CWT Rhythmic (diagnostic)"] = bool(_ds["cwt_rhythmic"].sel(id=fly_id).values)
        rows.append(row)
    _df = pd.DataFrame(rows)
    # Alphabetical (Group then ID) for a predictable, GraphPad-friendly export.
    _sort_keys = [c for c in ("Group", "ID") if c in _df.columns]
    return _df.sort_values(_sort_keys).reset_index(drop=True) if _sort_keys else _df


from dataset_meta import dataset_fingerprint as _ds_fingerprint

summary_df = _build_period_summary_df(_ds_fingerprint(period_ds), period_ds)
summary_data = summary_df.to_dict("records")

if summary_data and len(summary_data[0]) > 2:
    st.dataframe(summary_df, width="stretch", height=300)
    st.caption(
        "This table includes every per-fly value shown in the Interactive threshold "
        "explorer below (period + strength + rhythmic call per algorithm), so exporting "
        "it captures the explorer data too."
    )
    ex.save_df_button(
        "Save Period Summary to working folder",
        summary_df,
        period_ds,
        "period_summary.csv",
        key="download_period_csv",
    )
else:
    st.info("No period analysis results to display. Run an analysis above.")

# ============================================================
# Rhythmicity Classification (per-algorithm: LS / AC / CWT)
# ============================================================
# Use period_ds for rhythmicity (results are on the phase dataset)
_has_ls = "ls_fap" in period_ds
_has_ac = "ac_power" in period_ds
_has_cwt = "cwt_rhythmicity" in period_ds
# MESA appears in the explorer only (it is a period method with no significance
# test — not part of the Rhythmicity Classification / classify_all section below).
# Its rhythmic call BORROWS the AC RI (MESA has no metric of its own), so the tab
# needs both the MESA period AND autocorrelation to render.
_has_mesa = ("mesa_period" in period_ds) and _has_ac

if _has_ls or _has_ac or _has_cwt or _has_mesa:
    st.divider()

    # --- Interactive threshold explorer (Part 2b) --------------------------
    # C2: shown ABOVE the Classification section (visual-before-cutoff) — see the
    # per-fly distribution first, then set the persistent cutoffs below.
    # Three parallel per-algorithm widget sets (AC / LS / CWT). Each set: a PERIOD
    # panel (left, rhythmic flies only) + a STRENGTH panel (right, all analysed
    # flies) coupled to a threshold slider. The slider is a PURE DISPLAY FILTER over
    # precomputed per-fly values (read from the stored (id) data_vars) — it
    # re-derives the rhythmic call live (strength > cutoff AND period in window),
    # never re-runs analysis. Arrhythmic flies LEAVE the period panel (filtered, not
    # zeroed); NaN-strength flies (failed analysis) appear in neither. Slider
    # defaults come from the single calibration source (core/calibrations.py).
    _expl = [
        (a, lbl)
        for a, lbl, has in (
            ("ac", "Autocorrelation", _has_ac),
            ("ls", "Lomb-Scargle", _has_ls),
            ("cwt", "CWT", _has_cwt),
            ("mesa", "MESA", _has_mesa),
        )
        if has
    ]
    if _expl:
        st.subheader("Interactive threshold explorer")
        st.caption(
            "Drag a threshold to see exactly which flies it calls rhythmic. Points "
            "are precomputed per-fly values; the slider only filters the display "
            "(arrhythmic flies leave the period panel — filtered, not zeroed; "
            "failed-analysis NaN flies appear in neither). Defaults come from "
            "`core/calibrations.py`. To PERSIST a cutoff, enter it in the "
            "classification controls below and re-run."
        )
        _expl_thresholds = {}  # {algo: live slider value}, collected for the export
        for _tab, (_algo, _lbl) in zip(st.tabs([l for _, l in _expl]), _expl):
            with _tab:
                _ref = None
                _min = 0.0
                if _algo in ("ac", "mesa"):
                    # AC RI slider — MESA borrows the SAME metric/threshold (it has
                    # no significance test of its own), so both share this config.
                    _default, _max, _step, _fmt = float(DEFAULT_AC_RI_THRESHOLD), 1.0, 0.01, "%.3f"
                    _ref = float(AC_RI_SCAMP_REFERENCE)
                    # The AC RI sliding scale is floored at the SCAMP historical
                    # reference (0.195) — the established convention is the lowest
                    # cutoff the tool will offer; period-shift groups are separated
                    # by sliding UP from here, never below it.
                    _min = _ref
                    if _algo == "ac":
                        st.caption(
                            "AC has two distinct numbers: the live per-fly "
                            "classification cutoff (slider, default 0.3) and the SCAMP "
                            "historical reference (0.195, dotted line). The group-average "
                            "check (mean RI < 0.3) is a separate eval criterion, not this slider."
                        )
                    else:  # mesa borrows AC's RI as its rhythmicity
                        st.caption(
                            "MESA has **no significance test**, so its rhythmic call borrows "
                            "the **Autocorrelation RI**: the strength panel and threshold here "
                            "are AC's RI, and the left panel shows the **MESA period** of the "
                            "AC-rhythmic flies. Floored at the SCAMP 0.195 reference. (MESA's "
                            "own peak/median SNR is still in the summary table above.)"
                        )
                elif _algo == "ls":
                    _smax = (
                        float(np.nanmax(period_ds["ls_power"].values))
                        if "ls_power" in period_ds
                        else 0.05
                    )
                    _default, _max, _step, _fmt = (
                        float(DEFAULT_LS_POWER_THRESHOLD),
                        max(0.05, round(_smax, 3)),
                        0.001,
                        "%.4f",
                    )
                elif _algo == "cwt":
                    _smax = (
                        float(np.nanmax(period_ds["cwt_rhythmicity"].values))
                        if "cwt_rhythmicity" in period_ds
                        else 2.0
                    )
                    _default = float(
                        cwt_threshold_for(period_ds.attrs.get("cwt_method", DEFAULT_CWT_METHOD))
                    )
                    _max = max(2.0, round(_smax, 1))
                    # Step scales with range so the slider stays usable across
                    # ar1/global (~1-4) and global_rednoise (~3-70+ strength).
                    _step = 0.05 if _max <= 5.0 else (0.1 if _max <= 20.0 else 0.5)
                    _fmt = "%.2f"
                _default = min(max(_default, _min), float(_max))
                _thr = st.slider(
                    f"{_lbl} threshold",
                    min_value=float(_min),
                    max_value=float(_max),
                    value=_default,
                    step=_step,
                    format=_fmt,
                    key=f"expl_thr_{_algo}",
                )
                _expl_thresholds[_algo] = _thr
                try:
                    _fig, _summ = plotting_mod.threshold_coupled_figure(
                        period_ds,
                        _algo,
                        _thr,
                        period_window=(min_period, max_period),
                        scamp_ref=_ref,
                    )
                    st.plotly_chart(_fig, width="stretch", key=f"expl_fig_{_algo}")
                    _t = _summ.get("_total", {})
                    if _t:
                        st.caption(
                            f"{_t.get('n_rhythmic', 0)}/{_t.get('n_analyzed', 0)} flies "
                            f"rhythmic at cutoff {_thr:g} (period "
                            f"{min_period:g}-{max_period:g} h). NaN-strength flies "
                            f"(failed analysis) are excluded from both panels."
                        )
                except Exception as _e:
                    st.error(f"Threshold explorer error ({_algo}): {_e}")

        # --- One-click export: one CSV per RUN analysis, at its live threshold ---
        # Each file is the per-fly table behind that tab (ID, Group, period,
        # strength, Rhythmic-at-cutoff), built via the SAME masks as the plot
        # (plotting.threshold_explorer_table). An analysis that was not run yields
        # an empty table and no file (save_multi_df_button skips it).
        _explorer_exports = []
        for _algo, _lbl in _expl:
            _thr_cur = _expl_thresholds.get(_algo)
            if _thr_cur is None:
                continue
            _tbl = plotting_mod.threshold_explorer_table(
                period_ds, _algo, float(_thr_cur), period_window=(min_period, max_period)
            )
            if not _tbl.empty:
                _explorer_exports.append((f"threshold_explorer_{_algo.upper()}.csv", _tbl))
        ex.save_multi_df_button(
            "Export explorer data (one CSV per analysis)",
            _explorer_exports,
            period_ds,
            key="export_explorer_all",
            help="Writes one CSV per analysis that has been run (AC / LS / CWT / "
            "MESA) into the working folder's Graph Exports/, each at its "
            "current slider cutoff. Un-run analyses are skipped.",
        )

    st.divider()
    st.subheader("Rhythmicity Classification")
    st.markdown(
        "**Autocorrelation** is the canonical classifier; its `ac_rhythmic` "
        "flag is the universal filter for group statistics. **Lomb-Scargle** "
        "and **CWT** classifiers are diagnostic — their flags are reported "
        "alongside AC's but never filter group statistics."
    )
    st.caption(
        "Thresholds are sensitive to recording duration. Shorter recordings "
        "(<7 days DD) may require relaxing cutoffs or enabling the dynamic "
        "2/√N CI for AC."
    )

    # --- Autocorrelation (canonical, always runs) ---
    st.markdown("**Autocorrelation (canonical)**")
    if not _has_ac:
        st.info("Autocorrelation not run — go back and run AC analysis first.")
        ac_ri_thr = DEFAULT_AC_RI_THRESHOLD
        ac_dynamic_ci = False
        run_ac = False
    else:
        run_ac = True  # AC is structural — always classified when available
        ac_cols = st.columns([1, 1, 2])
        with ac_cols[0]:
            ac_ri_thr = st.number_input(
                "RI threshold",
                min_value=float(AC_RI_SCAMP_REFERENCE),
                max_value=1.0,
                value=float(DEFAULT_AC_RI_THRESHOLD),
                step=0.01,
                help="Rhythmic iff RI = peak autocorrelation > threshold. "
                "Floored at the SCAMP historical reference (0.195); "
                "default 0.3 is a stricter practical cutoff. See "
                "`ac_rhythm_strength` for the normalized statistical "
                "(Levine 2002 95% CI) interpretation.",
                key="rc_ac_ri",
            )
        with ac_cols[1]:
            ac_dynamic_ci = st.checkbox(
                "Use dynamic 2/√N CI",
                value=False,
                help="If enabled, the effective per-fly threshold is "
                "max(RI threshold, 2/√N) — Levine's 95% CI.",
                key="rc_ac_dyn",
            )
        with ac_cols[2]:
            st.caption(
                "AC always runs — its flag drives every downstream filter. "
                "The `ac_rhythm_strength` variable (RI/CI) is retained for "
                "display only; the gating metric is raw RI (`ac_power`)."
            )

    # --- Diagnostic classifiers (LS / CWT) ---
    with st.expander(
        "Additional classifiers (diagnostic — not used for filtering)", expanded=False
    ):
        st.caption(
            "LS and CWT classifications are reported per fly for method "
            "comparison but never participate in group filtering. Leave "
            "these unchecked unless you specifically want their flags."
        )
        d1, d2 = st.columns(2)

        # --- Lomb-Scargle ---
        with d1:
            st.markdown("**Lomb-Scargle**")
            if not _has_ls:
                st.info("LS not run.")
                ls_power_thr = DEFAULT_LS_POWER_THRESHOLD
                run_ls = False
            else:
                run_ls = st.checkbox("Classify LS", value=False, key="rc_run_ls")
                ls_power_thr = st.number_input(
                    "Power threshold (strength)",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(DEFAULT_LS_POWER_THRESHOLD),
                    step=0.001,
                    format="%.4f",
                    help="Rhythmic iff LS power (standard-normalized peak, R^2 "
                    "strength = fraction of variance the best period explains) "
                    "> threshold AND peak period in window. Power is LS's "
                    "continuous strength index (its RI/CWT-strength equivalent). "
                    "Default ~0.006 is a soft, movable line (like AC's 0.3 RI), "
                    "data-grounded from the rhythmic-vs-arrhythmic separation. "
                    "FAP (Baluev) is reported separately but does NOT gate: it "
                    "fires on weak periodicity in long records. (FAP refs: "
                    "Horne & Baliunas 1986 / Refinetti 2007, not Pfeiffenberger.)",
                    key="rc_ls_power",
                )

        # --- CWT ---
        with d2:
            st.markdown("**CWT**")
            if not _has_cwt:
                st.info("CWT not run.")
                cwt_thr = DEFAULT_CWT_RIDGE_THRESHOLD
                run_cwt = False
            else:
                run_cwt = st.checkbox("Classify CWT", value=False, key="rc_run_cwt")
                cwt_thr = st.number_input(
                    "Rhythmicity threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(DEFAULT_CWT_RIDGE_THRESHOLD),
                    step=0.05,
                    help="⚠ No published rhythmic/arrhythmic cutoff exists for "
                    "CWT-based metrics in fly DAM data (Leise 2011, 2013, "
                    "2015). This threshold is a lab-choice default — use "
                    "with caution and report the exact value used.",
                    key="rc_cwt_thr",
                )

    st.caption(
        f"Classification period window tracks the analysis range above "
        f"({min_period:.1f}–{max_period:.1f} h). Edit the analysis range to "
        f"change the classification window."
    )

    if st.button("Run Classification", key="run_rhythmicity", type="primary"):
        try:
            period_ds = classify_all(
                period_ds,
                ls_power_threshold=ls_power_thr,
                ac_ri_threshold=ac_ri_thr,
                cwt_rhythmicity_threshold=cwt_thr,
                period_window=(min_period, max_period),
                ac_use_dynamic_ci=bool(ac_dynamic_ci) if _has_ac else False,
                run_ls=run_ls and _has_ls,
                run_ac=run_ac and _has_ac,
                run_cwt=run_cwt and _has_cwt,
            )
            _store_period_results(period_ds)
            _pf = per_fly_classification_df(period_ds)
            # §4: flag (never filter) each fly's DD record length vs the recommended
            # floor so under-floor flies are visible per fly, not dropped.
            _rd = _dd_record_days(period_ds)
            _pf["record_days"] = [round(_rd.get(str(f), float("nan")), 2) for f in _pf["fly_id"]]
            _pf["below_floor"] = [
                bool(_rd.get(str(f), 0.0) < min_days_floor) for f in _pf["fly_id"]
            ]
            st.session_state["rhythmicity_per_fly_df"] = _pf
            st.session_state["rhythmicity_summary_df"] = summarize_rhythmicity(period_ds)
            st.success(
                "Classification complete. See the **Visualization** page "
                "for period and rhythmicity-metric distributions."
            )
        except Exception as e:
            st.error(f"Rhythmicity classification error: {e}")
            st.exception(e)

    per_fly_df = st.session_state.get("rhythmicity_per_fly_df")
    summary_df = st.session_state.get("rhythmicity_summary_df")

    if summary_df is not None and not summary_df.empty:
        st.markdown("#### Group summary")
        st.dataframe(summary_df, width="stretch", height=260)
        ex.save_df_button(
            "Save group summary to working folder",
            summary_df,
            period_ds,
            "rhythmicity_group_summary.csv",
            key="dl_rhyth_group_summary",
        )

    if per_fly_df is not None and not per_fly_df.empty:
        with st.expander("Per-fly classification table"):
            st.dataframe(per_fly_df, width="stretch", height=300)
            ex.save_df_button(
                "Save per-fly classification to working folder",
                per_fly_df,
                period_ds,
                "rhythmicity_per_fly.csv",
                key="dl_rhyth_per_fly",
            )

    # Period & rhythmicity-metric distributions and the group-averaged
    # CWT periodogram are rendered on the **Visualization** page (the
    # natural workflow split: configure & run here, explore there).
    st.info(
        "Period and rhythmicity-metric plots are on the **Visualization** "
        "page. Use the AC rhythmic filter in that page's sidebar to gate "
        "group statistics. Averaged CWT scalograms are produced as PNG "
        "files (saved to disk) when CWT is run with the corresponding "
        "checkbox enabled — see the CWT controls above."
    )

    # Threshold sensitivity analysis
    with st.expander("Threshold sensitivity analysis"):
        st.markdown(
            "Sweep a single algorithm's threshold to see how the rhythmic/"
            "arrhythmic call changes. Useful for checking robustness of your "
            "conclusions to cutoff choice."
        )
        sens_algo = st.selectbox(
            "Algorithm",
            options=[
                a for a in ("ls", "ac", "cwt") if {"ls": _has_ls, "ac": _has_ac, "cwt": _has_cwt}[a]
            ],
            format_func=lambda a: {
                "ls": "Lomb-Scargle (power)",
                "ac": "Autocorrelation (RI)",
                "cwt": "CWT (rhythmicity)",
            }.get(a, a),
            key="sens_algo",
        )
        default_range = {
            "ls": (0.001, 0.05),  # LS sweeps POWER now (higher = rhythmic), not FAP
            "ac": (0.05, 0.5),
            "cwt": (0.05, 0.8),
        }[sens_algo]
        thresh_min = st.number_input(
            "Min threshold", 0.0, 1.0, float(default_range[0]), 0.005, format="%.4f", key="sens_min"
        )
        thresh_max = st.number_input(
            "Max threshold", 0.0, 1.0, float(default_range[1]), 0.005, format="%.4f", key="sens_max"
        )
        thresh_steps = st.slider("Number of steps", 3, 30, 10, key="sens_steps")

        if st.button("Run Sensitivity Analysis", key="run_sens"):
            try:
                thresholds = list(np.linspace(thresh_min, thresh_max, thresh_steps))
                st.session_state["_period_sens_df"] = compare_thresholds(
                    period_ds, algorithm=sens_algo, thresholds=thresholds
                )
            except Exception as e:
                st.error(f"Sensitivity analysis error: {e}")
                st.exception(e)
        _sens_df = st.session_state.get("_period_sens_df")
        if _sens_df is not None and not _sens_df.empty:
            st.dataframe(_sens_df, width="stretch")
            ex.save_df_button(
                "Save Sensitivity to working folder",
                _sens_df,
                period_ds,
                "rhythmicity_sensitivity.csv",
                key="dl_rhyth_sens",
            )
