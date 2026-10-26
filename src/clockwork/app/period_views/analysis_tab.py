"""Analysis tab: pick the period methods, run them, one set of advanced options.

The four methods used to be four tabs, each with its own "already completed —
re-run?" checkbox and its own Run button. They are four checkboxes and one button
now: which methods to run is the decision, and it is made once.
"""

import os

import streamlit as st

from clockwork import pipeline
from clockwork.app import export_helpers
from clockwork.app.preprocessing_widgets import render_preprocess_controls
from clockwork.app.ui.changed import flag_if_changed
from clockwork.app.ui.period_context import store_master
from clockwork.core import dam_utilities, periodograms
from clockwork.core.calibrations import DEFAULT_CWT_METHOD

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
        # ac_prep_*: the preprocessing is recorded per method now. The unprefixed
        # prep_* this used to read was never on the master, so it always said "?".
        "ac": f"low-pass {a.get('ac_prep_lopass_hours', '?')} h, peak day {a.get('ac_peak', '?')}",
        "mesa": f"{a.get('mesa_bin_minutes', '?')}-min bins, order {a.get('mesa_order', '?')}",
    }[key]
    return f"{phase}, {rng}, {extra}"


#: Readable names for the settings :func:`_changed_settings` reports.
_SETTING_LABELS = {
    "period_range_hours": "period range (h)",
    "min_dd_days": "min DD days",
    "max_bridge_gap_minutes": "max gap bridged (min)",
    "voices_per_octave": "voices/octave",
    "reduction": "reduction",
    "group_scalograms": "group scalograms",
    "scalogram_flies": "scalogram flies",
    "oversampling": "oversampling",
    "fap": "FAP method",
    "peak_day": "peak day",
    "bin_minutes": "bin (min)",
    "order": "AR order",
    "rhythmic_threshold": "rhythmic cutoff",
    "rhythmic_window_hours": "classification window (h)",
    "smooth_sigma_minutes": "smoothing σ (min)",
    "lowpass_hours": "low-pass (h)",
    "detrend": "detrend",
    "normalize": "normalize",
    "rolling_window_hours": "rolling window (h)",
}


def _fmt_setting(value):
    if isinstance(value, (list, tuple)):
        return "–".join(f"{v:g}" if isinstance(v, float) else str(v) for v in value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _changed_settings(period_ds, key):
    """The settings method ``key`` was last run with that differ from the defaults,
    as ``[(label, value)]`` — or None when nothing is stored for it.

    Read back through ``PeriodConfig.from_attrs``, the same reader the Export
    settings button uses, so this lists exactly what an exported config would.
    """
    try:
        cfg = pipeline.PeriodConfig.from_attrs(period_ds.attrs, period_ds.coords)
        method = cfg.method(key) if cfg is not None else None
    except (KeyError, ValueError):  # not run, or attrs from an older version
        return None
    if method is None:
        return None
    out = []
    for name, value in cfg.effective(key).items():
        if name == "phase":  # shown on its own
            continue
        if name == "max_bridge_gap_minutes" and not method.BRIDGES_GAPS:
            continue
        if value != type(cfg).model_fields[name].default:
            out.append((_SETTING_LABELS[name], _fmt_setting(value)))
    for name, value in method.overrides().items():
        if name in ("phase", "period_range_hours", "min_dd_days", "max_bridge_gap_minutes"):
            continue  # resolved above, against the period-level default
        if name == "classify":  # whether the cutoff tab has run yet, not a setting
            continue
        if name == "preprocessing":
            out += [
                (f"preprocessing {_SETTING_LABELS.get(k, k)}", _fmt_setting(v))
                for k, v in value.items()
            ]
            continue
        out.append((_SETTING_LABELS.get(name, name), _fmt_setting(value)))
    return out


def _stored_summary(period_ds, key):
    """Caption text for a stored method: its phase, then either its non-default
    settings or a plain statement that every one was left at its default."""
    phase = period_ds.attrs.get(f"{key}_phase", "?")
    changed = _changed_settings(period_ds, key)
    if changed is None:
        return f"Phase {phase}."
    if not changed:
        return f"Phase {phase} · all settings at defaults."
    return f"Phase {phase} · changed from default: " + "; ".join(
        f"{label} **{value}**" for label, value in changed
    )


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
            flag_if_changed(opts["cwt_voices"], periodograms.DEFAULT_CWT_VOICES_PER_OCTAVE)
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
            flag_if_changed(opts["cwt_method"], DEFAULT_CWT_METHOD)
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
                flag_if_changed(opts["cwt_group_averages"], False)
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
                flag_if_changed(opts["cwt_avg_filter"], True)
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
            flag_if_changed(opts["ls_oversampling"], 8)
            opts["ls_fap_method"] = st.selectbox(
                "FAP method",
                options=["baluev", "naive", "bootstrap"],
                index=0,
                key="ls_fap_method",
                persist_state="session",
                help="`baluev` (Baluev 2008, default), `naive` (Šidák-corrected Beta "
                "tail), or `bootstrap` (slow but gold-standard).",
            )
            flag_if_changed(opts["ls_fap_method"], "baluev")
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
            flag_if_changed(opts["ac_peak"], 2)
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
            flag_if_changed(opts["mesa_bin"], int(periodograms.DEFAULT_MESA_BIN_MINUTES))
            _choice = st.selectbox(
                "AR model order",
                ["N/3 (Dowse default)", "N/4", "FPE auto", "Fixed"],
                index=0,
                key="mesa_order_choice",
                persist_state="session",
                help="Too low merges peaks, too high invents spurious ones. N/3 is "
                "Dowse's circadian 'safe maximum'.",
            )
            flag_if_changed(_choice, "N/3 (Dowse default)")
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


def _config(key, ctx, opts):
    """The page's controls for method ``key``, as the pipeline's PeriodConfig.

    The same object a config file would produce, so what the page runs and what
    the Export settings button later writes down cannot drift apart.
    """
    pre = opts["preprocess"]
    if key == "cwt":
        method = {
            "voices_per_octave": opts["cwt_voices"],
            "reduction": opts["cwt_method"],
            "preprocessing": pipeline.Preprocessing.from_core(pre["CWT"]),
            "group_scalograms": bool(opts["cwt_group_averages"]),
            "scalogram_flies": "rhythmic" if opts["cwt_avg_filter"] else "all",
        }
    elif key == "ls":
        method = {
            "oversampling": opts["ls_oversampling"],
            "fap": opts["ls_fap_method"],
            "preprocessing": pipeline.Preprocessing.from_core(pre["LS"]),
        }
    elif key == "ac":
        method = {
            "peak_day": opts["ac_peak"],
            "preprocessing": pipeline.Preprocessing.from_core(pre["AC"]),
        }
    else:  # mesa — reuses AC's detrend + low-pass preprocessing
        method = {
            "bin_minutes": opts["mesa_bin"],
            "order": "n_over_3" if opts["mesa_order"] is None else opts["mesa_order"],
            "preprocessing": pipeline.Preprocessing.from_core(pre["AC"]),
        }
    return pipeline.PeriodConfig(
        phase=ctx.phase_selection,
        period_range_hours=(ctx.min_period, ctx.max_period),
        # The floor is a FILTER: sub-floor flies get no period.
        min_dd_days=ctx.min_days_floor,
        max_bridge_gap_minutes=ctx.max_bridge_gap,
        methods={_METHOD_FOR_KEY[key]: method},
    )


#: The page's method keys -> the pipeline's.
_METHOD_FOR_KEY = {"cwt": "cwt", "ls": "lomb_scargle", "ac": "autocorrelation", "mesa": "mesa"}


def _run_one(key, ctx, opts, progress):
    """Run method ``key`` and store its results. Returns a success note."""
    def _cb(done, total):
        progress.progress(done / total, text=f"{LABELS[key]}: fly {done}/{total}")

    out_dir = None
    if key == "cwt" and opts["cwt_group_averages"]:
        # Beside the source monitor data (survives a .nc reload), never the launch dir.
        wd = dam_utilities.resolve_export_dir(ctx.ds, st.session_state.get("working_dir"))
        out_dir = os.path.join(wd, "Averaged Scalograms")

    # The MASTER, as every write must be (ARCHITECTURE rule 11). The averages'
    # file names carry the phase run, so DD and LD reruns don't overwrite each other.
    master, averages = pipeline.run_period_method(
        st.session_state.dataset,
        _config(key, ctx, opts),
        _METHOD_FOR_KEY[key],
        progress=_cb,
    )
    store_master(master)
    if key == "cwt":
        saved = []
        if averages and out_dir:
            saved = export_helpers.save_group_average_scalograms(averages, out_dir, ds=master)
        if saved:
            return f"CWT done; {len(saved)} averaged scalogram(s) saved to `{out_dir}`."
        return "CWT done."
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
            if done:
                st.caption(
                    ":material/check_circle: Stored. " + _stored_summary(ctx.period_ds, key)
                )
            else:
                st.caption("Not run yet")

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
