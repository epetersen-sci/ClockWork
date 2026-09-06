"""
Period Analysis Page - CWT, Lomb-Scargle, and Autocorrelation analyses.
"""

import os

import numpy as np
import streamlit as st

import dam_utilities
import export_helpers
import periodograms
from analysis_detection import detect_analyses
from calibrations import DEFAULT_CWT_METHOD
from preprocessing import preprocess_activity
from preprocessing_widgets import render_preprocess_expander
from ui.guards import require_dataset
from ui.period_context import (
    dd_record_days,
    remember_min_days_floor,
    render_period_range,
    render_phase_picker,
    store_period_results,
)

ds = require_dataset()
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
# Data phase and period range
# ============================================================
# Both live in ui/period_context so the Rhythmicity page renders the SAME
# controls under the same widget keys. The range is not just a search range:
# it is also the classification window, so the two pages must not disagree.
phase_selection, period_ds, _analysis_src, _period_phase = render_phase_picker(ds)

# C2: the recording-duration summary is shown AFTER the DD-days floor filter
# below, so it describes the flies actually RETAINED (not the pre-filter master).
min_period, max_period = render_period_range()

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
remember_min_days_floor(min_days_floor)

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

_fly_record_days = dd_record_days(period_ds)
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


# One tab per method. The shared controls above stay OUTSIDE the tabs: they
# must execute before any runner reads them, and hiding the period range
# behind a tab would make it look method-specific when it governs all four.
# Tab bodies all execute every rerun, which is fine here because every
# expensive call sits behind its own Run button.
tab_cwt, tab_ls, tab_ac, tab_mesa = st.tabs(
    ["CWT", "Lomb-Scargle", "Autocorrelation", "MESA"]
)

with tab_cwt:
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
                        # Keyed off the phase actually selected — it used to ask
                        # whether the dataset_DD/LD caches existed, which was a
                        # proxy for the same thing and stopped being one when
                        # those caches were removed.
                        _phase_label = (
                            str(phase_selection)
                            if phase_selection in ("DD", "LD")
                            else "full"
                        )
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
                        phase_label=_phase_label,
                    )
                    # wavelet_analysis returns the group averages as ARRAYS and
                    # writes nothing; deciding where files go is the caller's
                    # job, which is why it no longer has to import plotting.
                    result, _group_averages = result
                    period_ds = result
                    store_period_results(result, phase_selection)
                    _saved = []
                    if _group_averages and _avg_out_dir:
                        with st.spinner("Saving averaged scalograms…"):
                            _saved = export_helpers.save_group_average_scalograms(
                                _group_averages, _avg_out_dir, ds=result
                            )
                    if _saved:
                        st.success(
                            f"CWT analysis complete. {len(_saved)} averaged scalogram(s) "
                            f"saved to `{_avg_out_dir}` (one PNG + CSV per group)."
                        )
                    else:
                        st.success("CWT analysis complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"CWT Error: {e}")
                finally:
                    cwt_progress.empty()


with tab_ls:
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
                    store_period_results(result, phase_selection)
                    st.success("Lomb-Scargle analysis complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Lomb-Scargle Error: {e}")
                finally:
                    ls_progress.empty()


with tab_ac:
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
                    store_period_results(result, phase_selection)
                    st.success("Autocorrelation analysis complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Autocorrelation Error: {e}")
                finally:
                    ac_progress.empty()

with tab_mesa:
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
                    store_period_results(result, phase_selection)
                    st.success("MESA analysis complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"MESA Error: {e}")
                finally:
                    mesa_progress.empty()


st.divider()
st.info(
    "Per-fly results, the interactive threshold explorer and the "
    "rhythmic/arrhythmic classification are on the **Rhythmicity** page."
)
