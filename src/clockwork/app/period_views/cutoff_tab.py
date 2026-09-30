"""Rhythmicity cutoff tab: see where a threshold falls, then apply it.

The threshold explorer and the classification used to be two separate controls —
a slider that only filtered the display, and a number box further down that had
to be typed to match it before "Run classification" wrote anything. There is one
control now. The slider IS the cutoff, and Classify applies it.

Autocorrelation is the canonical classifier: its ``ac_rhythmic`` flag is the one
downstream group statistics filter on. The Lomb-Scargle and CWT flags are
diagnostic. MESA has no significance test of its own, so its rhythmic call is
autocorrelation's, and classifying "MESA" sets the AC flag.
"""

import numpy as np
import streamlit as st

from clockwork.app import export_helpers as ex
from clockwork.app.ui import charts
from clockwork.app.ui.period_context import store_period_results
from clockwork.core import plotting
from clockwork.core.calibrations import AC_RI_SCAMP_REFERENCE, DEFAULT_CWT_METHOD
from clockwork.core.rhythmicity_classification import (
    DEFAULT_AC_RI_THRESHOLD,
    DEFAULT_LS_POWER_THRESHOLD,
    classify_all,
    cwt_threshold_for,
)

_NOTE_KEY = "_classify_note"

#: The stored classification each algorithm writes: (flag coord, threshold attr).
_STORED = {
    "ac": ("ac_rhythmic", "ac_ri_threshold"),
    "mesa": ("ac_rhythmic", "ac_ri_threshold"),
    "ls": ("ls_rhythmic", "ls_power_threshold"),
    "cwt": ("cwt_rhythmic", "cwt_rhythmicity_threshold"),
}


def available(period_ds):
    """``[(algo, label), ...]`` for the analyses that have results to classify."""
    has_ac = "ac_power" in period_ds
    return [
        (a, lbl)
        for a, lbl, ok in (
            ("ac", "Autocorrelation", has_ac),
            ("ls", "Lomb-Scargle", "ls_fap" in period_ds),
            ("cwt", "CWT", "cwt_rhythmicity" in period_ds),
            # MESA's call borrows the AC RI, so it needs autocorrelation too.
            ("mesa", "MESA", "mesa_period" in period_ds and has_ac),
        )
        if ok
    ]


def slider_config(period_ds, algo):
    """``(min, max, default, step, format, reference line)`` for ``algo``'s slider."""
    if algo in ("ac", "mesa"):
        # Floored at the SCAMP historical reference (0.195): the established
        # convention is the lowest cutoff offered.
        ref = float(AC_RI_SCAMP_REFERENCE)
        return ref, 1.0, float(DEFAULT_AC_RI_THRESHOLD), 0.01, "%.3f", ref
    if algo == "ls":
        smax = float(np.nanmax(period_ds["ls_power"].values)) if "ls_power" in period_ds else 0.05
        return 0.0, max(0.05, round(smax, 3)), float(DEFAULT_LS_POWER_THRESHOLD), 0.001, "%.4f", None
    smax = float(np.nanmax(period_ds["cwt_rhythmicity"].values)) if "cwt_rhythmicity" in period_ds else 2.0
    mx = max(2.0, round(smax, 1))
    # Step scales with range: ar1/global run ~1-4, global_rednoise ~3-70+.
    step = 0.05 if mx <= 5.0 else (0.1 if mx <= 20.0 else 0.5)
    default = float(cwt_threshold_for(period_ds.attrs.get("cwt_method", DEFAULT_CWT_METHOD)))
    return 0.0, mx, default, step, "%.2f", None


def _threshold(period_ds, algo):
    """The live cutoff for ``algo``: the slider's value, kept inside this dataset's range."""
    lo, hi, default, *_ = slider_config(period_ds, algo)
    key = f"expl_thr_{algo}"
    if key in st.session_state:
        # A remembered cutoff from another dataset can fall outside this one's
        # range; Streamlit rejects a slider value outside its bounds.
        st.session_state[key] = min(max(float(st.session_state[key]), lo), hi)
    return st.session_state.get(key, min(max(default, lo), hi))


def _stored_line(period_ds, algo):
    flag, attr = _STORED[algo]
    if flag not in period_ds.coords:
        return "Not classified yet."
    n = int(np.asarray(period_ds[flag].values, dtype=bool).sum())
    thr = period_ds.attrs.get(attr)
    at = f" at **{float(thr):g}**" if isinstance(thr, (int, float, np.number)) else ""
    return f"Currently classified{at}: {n} of {period_ds.sizes['id']} flies rhythmic."


def render(ctx):
    period_ds = ctx.period_ds
    algos = available(period_ds)
    if not algos:
        st.info("Nothing to classify yet — run a period analysis on the **Analysis** tab.")
        return

    note = st.session_state.get(_NOTE_KEY)
    if note:
        st.success(note)

    labels = dict(algos)
    algo = st.selectbox(
        "Algorithm",
        [a for a, _ in algos],
        format_func=labels.get,
        key="cutoff_algo",
        persist_state="session",
        help="Autocorrelation is the canonical classifier: its flag is what group "
        "statistics filter on. Lomb-Scargle and CWT flags are diagnostic. MESA has no "
        "significance test, so it borrows the autocorrelation RI and sets the AC flag.",
    ) or algos[0][0]

    lo, hi, default, step, fmt, ref = slider_config(period_ds, algo)
    key = f"expl_thr_{algo}"
    thr = _threshold(period_ds, algo)
    # Seeded through the key, not value=: _threshold may already have written it,
    # and a widget given both logs a warning.
    st.session_state.setdefault(key, thr)
    thr = st.slider(
        f"{labels[algo]} threshold",
        min_value=float(lo),
        max_value=float(hi),
        step=step,
        format=fmt,
        key=key,
        persist_state="session",
    )
    if algo == "mesa":
        st.caption(
            "MESA's strength panel and threshold are the **autocorrelation RI**; the "
            "period panel shows the MESA period of the AC-rhythmic flies."
        )
    elif algo == "ac":
        st.caption(
            "Dotted line: the SCAMP historical reference (0.195), the lowest cutoff offered."
        )

    try:
        fig, summ = plotting.threshold_coupled_figure(
            period_ds,
            algo,
            thr,
            period_window=(ctx.min_period, ctx.max_period),
            scamp_ref=ref,
        )
        charts.plotly_chart(fig, width="stretch", key=f"expl_fig_{algo}")
        t = summ.get("_total", {})
        if t:
            st.caption(
                f"{t.get('n_rhythmic', 0)}/{t.get('n_analyzed', 0)} flies rhythmic at "
                f"{thr:g} (period {ctx.min_period:g}–{ctx.max_period:g} h). Flies whose "
                "analysis failed are in neither panel."
            )
    except Exception as e:
        st.error(f"Threshold explorer error ({algo}): {e}")

    st.caption(_stored_line(period_ds, algo))
    if st.button("Classify", type="primary", key="classify_rhythmicity"):
        target = "ac" if algo == "mesa" else algo
        try:
            out = classify_all(
                period_ds,
                ls_power_threshold=thr if target == "ls" else DEFAULT_LS_POWER_THRESHOLD,
                ac_ri_threshold=thr if target == "ac" else DEFAULT_AC_RI_THRESHOLD,
                cwt_rhythmicity_threshold=thr if target == "cwt" else None,
                period_window=(ctx.min_period, ctx.max_period),
                run_ls=target == "ls",
                run_ac=target == "ac",
                run_cwt=target == "cwt",
            )
            store_period_results(out, ctx.phase_selection)
            n = int(np.asarray(out[_STORED[target][0]].values, dtype=bool).sum())
            st.session_state[_NOTE_KEY] = (
                f"Classified by {labels[algo]} at {thr:g}: {n} of {out.sizes['id']} flies "
                "rhythmic. The **Period length** and **Rhythmicity** tabs now use it."
            )
            st.rerun()
        except Exception as e:
            st.error(f"Classification error: {e}")

    # One CSV per analysis, each at its current cutoff — the per-fly table behind
    # the figure (ID, group, period, strength, rhythmic at cutoff).
    exports = []
    for a, _ in algos:
        # The drawn slider's own value for this algorithm (its key cannot be
        # written once the widget exists); the others' remembered cutoffs.
        cut = thr if a == algo else _threshold(period_ds, a)
        tbl = plotting.threshold_explorer_table(
            period_ds, a, float(cut), period_window=(ctx.min_period, ctx.max_period)
        )
        if not tbl.empty:
            exports.append((f"threshold_explorer_{a.upper()}.csv", tbl))
    ex.save_multi_df_button(
        "Export explorer data (one CSV per analysis)",
        exports,
        period_ds,
        key="export_explorer_all",
        help="One CSV per analysis that has been run, each at its current cutoff.",
    )
