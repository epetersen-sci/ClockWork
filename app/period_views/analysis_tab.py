"""Analysis tab: pick the period methods, run them, one set of advanced options.

The four methods used to be four tabs, each with its own "already completed —
re-run?" checkbox and its own Run button. They are four checkboxes and one button
now: which methods to run is the decision, and it is made once.
"""

import os

import streamlit as st

import dam_utilities
import export_helpers
import periodograms
from calibrations import DEFAULT_CWT_METHOD
from preprocessing import preprocess_activity
from preprocessing_widgets import render_preprocess_controls
from ui.period_context import store_period_results

#: (key, label, the per-fly output that says it has been run)
METHODS = [
    ("cwt", "CWT", "cwt_period"),
    ("ls", "Lomb-Scargle", "ls_period"),
    ("ac", "Autocorrelation", "ac_period"),
    ("mesa", "MESA", "mesa_period"),
]
LABELS = {key: label for key, label, _ in METHODS}

# Plain-language method descriptions (mechanism / main use / major drawback),
# verified against the period-method literature.
METHOD_DESCRIPTIONS = {
    "cwt": (
        "**Continuous Wavelet Transform** compares the signal against a wave-packet "
        "stretched to many scales, producing a time-by-period map that shows how the "
        "dominant period changes over the recording. Its strength is capturing "
        "non-stationary rhythms — period drift, or rhythmicity that appears and "
        "disappears — that single-number methods average away. Its costs are a "
        "time-vs-frequency resolution tradeoff and edge artifacts near the start and "
        "end of the record. Default `global_rednoise` strength: the COI-excluded "
        "global-spectrum peak over the AR(1) red-noise floor at the peak period."
    ),
    "ls": (
        "**Lomb-Scargle** fits a sine wave at each candidate period and scores how "
        "well it fits, building a periodogram whose tallest peak is the estimated "
        "period; a Baluev false-alarm probability flags whether that peak could be "
        "noise. Its strength is working directly on unevenly sampled or gappy data "
        "without interpolation. Its main limitation is assuming a roughly sinusoidal "
        "rhythm, so strongly non-sinusoidal activity can be mis-scored. Generalized "
        "(Zechmeister-Kürster) with a floating-mean fit per trial frequency."
    ),
    "ac": (
        "**Autocorrelation** slides the activity signal against a time-shifted copy "
        "of itself; a rhythmic fly shows evenly spaced correlation peaks whose "
        "spacing gives the period, and the peak height (rhythmicity index) measures "
        "rhythm strength. It is robust, intuitive, and comes with a simple "
        "significance bound. Its drawback is coarser period precision and weaker "
        "performance on faint or drifting rhythms. SCAMP-style: 4-h Butterworth "
        "low-pass, linear detrend, 2nd-day peak window."
    ),
    "mesa": (
        "**MESA (Maximum Entropy Spectral Analysis)** fits a small predictive "
        "(autoregressive) model to the evenly sampled activity and computes the "
        "spectrum from that model, giving very sharp, well-resolved peaks even from "
        "short records. Its strength is high period precision and separating closely "
        "spaced periodicities. Its limitations are dependence on the chosen model "
        "order, a requirement for evenly sampled gap-free data, and no built-in test "
        "of whether a rhythm is real — so its rhythmic call borrows autocorrelation's."
    ),
}

_NOTES_KEY = "_period_run_notes"


def _status(period_ds, key):
    """One line saying what is stored for method ``key``, or None if nothing is."""
    a = period_ds.attrs
    rng = f"{a.get(f'{key}_min_period', '?')}–{a.get(f'{key}_max_period', '?')} h"
    phase = a.get(f"{key}_phase", "?")
    extra = {
        "cwt": f"method {a.get('cwt_method', '?')}, {a.get('cwt_wavelet', '?')} wavelet",
        "ls": f"oversampling {a.get('ls_oversampling', '?')}, FAP {a.get('ls_fap_method', '?')}",
        "ac": f"low-pass {a.get('prep_lopass_hours', '?')} h, peak day {a.get('ac_peak', '?')}",
        "mesa": f"{a.get('mesa_bin_minutes', '?')}-min bins, order {a.get('mesa_order', '?')}",
    }[key]
    return f"{phase}, {rng}, {extra}"


def _advanced_options():
    """Every method's options, in one expander. Returns them as a dict."""
    opts = {}
    with st.expander("Advanced options", expanded=False):
        t_pre, t_cwt, t_ls, t_ac, t_mesa = st.tabs(
            ["Preprocessing", "CWT", "Lomb-Scargle", "Autocorrelation", "MESA"]
        )
        with t_pre:
            opts["preprocess"] = render_preprocess_controls("page3")
        with t_cwt:
            opts["cwt_voices"] = st.select_slider(
                "CWT resolution (voices per octave)",
                options=[8, 10, 12, 16, 32, 64, 128, 256, 512, 1024],
                value=periodograms.DEFAULT_CWT_VOICES_PER_OCTAVE,
                format_func=lambda x: f"{x} voices/octave",
                help="Density of the logarithmic period grid (scales per doubling of "
                "period), spanning the period range. Higher = finer period readout but "
                "proportionally slower. Default "
                f"{periodograms.DEFAULT_CWT_VOICES_PER_OCTAVE} voices/octave (~0.5 h "
                "period steps near 24 h).",
                key="cwt_resolution_voices",
                persist_state="session",
            )
            _methods = ["ar1", "global_rednoise", "global", "global_baseline", "global_neighbor", "ridge"]
            opts["cwt_method"] = st.selectbox(
                "Reduction method",
                options=_methods,
                index=_methods.index(DEFAULT_CWT_METHOD) if DEFAULT_CWT_METHOD in _methods else 0,
                help="`global_rednoise` (default): range-robust, length-stable "
                "local-prominence strength — peak / AR(1) red-noise expected power at "
                "the peak period. `ar1`: Torrence & Compo 1998 red-noise significance "
                "(length-sensitive). `global`: peak / mean band power — over-calls on a "
                "wide band. `global_baseline` / `global_neighbor`: comparison only. "
                "`ridge`: legacy ridge tracker.",
                key="cwt_method_select",
                persist_state="session",
            )
            c1, c2 = st.columns(2)
            with c1:
                opts["cwt_group_averages"] = st.checkbox(
                    "Compute group-averaged scalograms (saved to disk)",
                    value=False,
                    help="Writes one PNG + CSV per group to "
                    "'<working_dir>/Averaged Scalograms/'.",
                    key="cwt_compute_group_averages",
                    persist_state="session",
                )
            with c2:
                opts["cwt_avg_filter"] = st.checkbox(
                    "Only AC-rhythmic flies in the average",
                    value=True,
                    help="Off-rhythm power spectra are mostly noise and smear the group "
                    "mean. Disable when loss of rhythmicity is the point.",
                    key="cwt_avg_filter_nonrhythmic",
                    persist_state="session",
                    disabled=not opts["cwt_group_averages"],
                )
        with t_ls:
            opts["ls_oversampling"] = st.number_input(
                "Oversampling (samples_per_peak)",
                min_value=1,
                max_value=64,
                value=8,
                step=1,
                key="ls_oversampling",
                persist_state="session",
                help="Rethomics default is 8. Astropy default is 5.",
            )
            opts["ls_fap_method"] = st.selectbox(
                "FAP method",
                options=["baluev", "naive", "bootstrap"],
                index=0,
                key="ls_fap_method",
                persist_state="session",
                help="`baluev` (Baluev 2008, default), `naive` (Šidák-corrected Beta "
                "tail), or `bootstrap` (slow but gold-standard).",
            )
        with t_ac:
            opts["ac_peak"] = st.number_input(
                "Peak day (1 = first AC peak, 2 = second-day peak)",
                min_value=1,
                max_value=5,
                value=2,
                step=1,
                key="ac_peak",
                persist_state="session",
                help="SCAMP `acplot.m` searches the 2nd-day peak ([40, 56] h for k = 2).",
            )
        with t_mesa:
            opts["mesa_bin"] = st.number_input(
                "Bin size (minutes)",
                min_value=5,
                max_value=60,
                value=int(periodograms.DEFAULT_MESA_BIN_MINUTES),
                step=5,
                key="mesa_bin",
                persist_state="session",
                help="Activity is binned to this before the Burg fit; ~30 min is the "
                "chronobiology convention (Dowse & Ringo).",
            )
            _choice = st.selectbox(
                "AR model order",
                ["N/3 (Dowse default)", "N/4", "FPE auto", "Fixed"],
                index=0,
                key="mesa_order_choice",
                persist_state="session",
                help="Too low merges peaks, too high invents spurious ones. N/3 is "
                "Dowse's circadian 'safe maximum'.",
            )
            fixed = None
            if _choice == "Fixed":
                fixed = int(
                    st.number_input(
                        "Fixed order", min_value=2, max_value=150, value=24, step=1,
                        key="mesa_fixed_order", persist_state="session",
                    )
                )
            opts["mesa_order"] = {
                "N/3 (Dowse default)": None,
                "N/4": "n_over_4",
                "FPE auto": "fpe",
                "Fixed": fixed,
            }[_choice]
    return opts


def _run_one(key, ctx, opts, progress):
    """Run method ``key`` and store its results. Returns a success note."""
    def _cb(done, total):
        progress.progress(done / total, text=f"{LABELS[key]}: fly {done}/{total}")

    pre = opts["preprocess"]
    common = dict(
        min_period=ctx.min_period,
        max_period=ctx.max_period,
        phase=ctx.phase_arg,
        min_num_days=ctx.min_days_floor,  # the floor is a FILTER — sub-floor flies get no period
        progress_callback=_cb,
    )
    if key == "cwt":
        out_dir = label = None
        if opts["cwt_group_averages"]:
            # Beside the source monitor data (survives a .nc reload), never the launch dir.
            wd = dam_utilities.resolve_export_dir(ctx.ds, st.session_state.get("working_dir"))
            out_dir = os.path.join(wd, "Averaged Scalograms")
            # Encodes the slice, so DD and LD reruns don't overwrite each other.
            label = ctx.phase_selection if ctx.phase_selection in ("DD", "LD") else "full"
        result, averages = periodograms.wavelet_analysis(
            preprocess_activity(ctx.analysis_src, pre["CWT"]),
            cwt_method=opts["cwt_method"],
            resolution=1 / opts["cwt_voices"],
            max_bridge_gap_minutes=ctx.max_bridge_gap,
            compute_group_averages=bool(opts["cwt_group_averages"]),
            group_coord="group",
            filter_nonrhythmic_for_average=bool(opts["cwt_avg_filter"]),
            phase_label=label,
            **common,
        )
        store_period_results(result, ctx.phase_selection)
        saved = []
        if averages and out_dir:
            saved = export_helpers.save_group_average_scalograms(averages, out_dir, ds=result)
        if saved:
            return f"CWT done; {len(saved)} averaged scalogram(s) saved to `{out_dir}`."
        return "CWT done."
    if key == "ls":
        result = periodograms.lomb_scargle_analysis(
            preprocess_activity(ctx.analysis_src, pre["LS"]),
            oversampling=opts["ls_oversampling"],
            fap_method=opts["ls_fap_method"],
            **common,
        )
    elif key == "ac":
        result = periodograms.autocorrelation_analysis(
            preprocess_activity(ctx.analysis_src, pre["AC"]),
            ac_peak=opts["ac_peak"],
            max_bridge_gap_minutes=ctx.max_bridge_gap,
            **common,
        )
    else:  # mesa — reuses AC's detrend + low-pass preprocessing
        result = periodograms.mesa_analysis(
            preprocess_activity(ctx.analysis_src, pre["AC"]),
            bin_minutes=opts["mesa_bin"],
            order=opts["mesa_order"],
            max_bridge_gap_minutes=ctx.max_bridge_gap,
            **common,
        )
    store_period_results(result, ctx.phase_selection)
    return f"{LABELS[key]} done."


def render(ctx):
    for kind, msg in st.session_state.get(_NOTES_KEY, []):
        (st.success if kind == "ok" else st.error)(msg)

    with st.expander("About the four methods"):
        for key, _, _ in METHODS:
            st.markdown(METHOD_DESCRIPTIONS[key])

    st.markdown("**Methods to run**")
    chosen = []
    for col, (key, label, var) in zip(st.columns(len(METHODS)), METHODS):
        done = var in ctx.period_ds.data_vars
        with col:
            if st.checkbox(
                label,
                value=not done,
                key=f"run_method_{key}",
                persist_state="session",
                help=("Stored: " + _status(ctx.period_ds, key)) if done else "Not run yet.",
            ):
                chosen.append(key)
            st.caption(":material/check_circle: Stored" if done else "Not run yet")

    opts = _advanced_options()

    # Averaging CWT scalograms over AC-rhythmic flies needs the AC classification;
    # without it the filter would quietly include every fly. CWT can take many
    # minutes, so say so before the run rather than after.
    need_confirm = (
        "cwt" in chosen
        and opts["cwt_group_averages"]
        and opts["cwt_avg_filter"]
        and "ac_rhythmic" not in ctx.period_ds.coords
    )
    proceed = True
    if need_confirm:
        st.warning(
            "**Autocorrelation has not been classified**, so the averaged scalograms "
            "would include every fly. Classify on the **Rhythmicity cutoff** tab first, "
            "or proceed anyway."
        )
        proceed = st.checkbox("Proceed without AC rhythmic filtering", key="cwt_avg_proceed_unfiltered")

    if st.button(
        "Run analysis",
        type="primary",
        key="run_period_analysis",
        disabled=not chosen or not proceed,
        help="Runs each ticked method in turn on the phase and period range above.",
    ):
        notes = []
        for key in chosen:
            label = LABELS[key]
            bar = st.progress(0.0, text=f"Starting {label}…")
            try:
                notes.append(("ok", _run_one(key, ctx, opts, bar)))
            except Exception as e:
                notes.append(("error", f"{label} failed: {e}"))
            finally:
                bar.empty()
        st.session_state[_NOTES_KEY] = notes
        st.rerun()
