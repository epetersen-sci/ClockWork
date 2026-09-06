"""
Rhythmicity — per-fly period results, threshold exploration, and the
rhythmic/arrhythmic classification.

The explore half of what used to be one 1361-line Period analysis page. That
page's own comments already described this split ("configure & run here, explore
there") and pointed three times at a "Visualization" page that was never in the
tree; this is the destination those pointers wanted.

Autocorrelation is the CANONICAL classifier — its call is what gates downstream
group filtering. The Lomb-Scargle and CWT classifiers are diagnostic.

The phase picker and period range are rendered by ui/period_context under the
same widget keys the Period analysis page uses, because the period range is also
the classification window. Change it on either page and both follow.
"""

import numpy as np
import pandas as pd
import streamlit as st

import export_helpers as ex
import plotting as plotting_mod
from calibrations import AC_RI_SCAMP_REFERENCE, DEFAULT_CWT_METHOD
from dataset_meta import dataset_fingerprint
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
from ui.guards import require_dataset
from ui.period_context import (
    dd_record_days,
    min_days_floor,
    render_period_range,
    render_phase_picker,
    store_period_results,
)

ds = require_dataset()

phase_selection, period_ds, _analysis_src, _period_phase = render_phase_picker(ds)
min_period, max_period = render_period_range(show_caption=False)

st.divider()

st.subheader("Per-fly period summary")

# Re-read period_ds in case analyses were run above. Always the master now: the
# pre-sliced dataset_DD / dataset_LD caches this used to prefer are gone, and the
# per-fly period outputs this table reads are phase-independent ``(id,)`` vars
# merged onto the master anyway, so the slice never added anything here.
period_ds = st.session_state.dataset

@st.cache_data(show_spinner=False)
def _build_period_summary_df(fp, _ds):
    """Build the per-fly Period Analysis summary table. Cached so a page
    rerun (no analysis-state change) doesn't redo the per-fly `.sel()`
    loop. The fingerprint key is invalidated whenever the analysis attrs
    or rhythmic flags change.

    ``fp`` carries no leading underscore, and must not grow one. Streamlit's
    underscore rule is syntactic — it drops ANY leading-underscore parameter
    from the cache key — and these are the only two parameters, so as ``_fp``
    the key was EMPTY: the table was built once per session and then returned
    unchanged for every later dataset, which is exactly the invalidation this
    docstring promises. ``_ds`` keeps its underscore because a Dataset is what
    the fingerprint exists to stand in for.
    """
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


summary_df = _build_period_summary_df(dataset_fingerprint(period_ds), period_ds)
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
            store_period_results(period_ds, phase_selection)
            _pf = per_fly_classification_df(period_ds)
            # §4: flag (never filter) each fly's DD record length vs the recommended
            # floor so under-floor flies are visible per fly, not dropped.
            _rd = dd_record_days(period_ds)
            _pf["record_days"] = [round(_rd.get(str(f), float("nan")), 2) for f in _pf["fly_id"]]
            _pf["below_floor"] = [
                bool(_rd.get(str(f), 0.0) < min_days_floor()) for f in _pf["fly_id"]
            ]
            st.session_state["rhythmicity_per_fly_df"] = _pf
            st.session_state["rhythmicity_summary_df"] = summarize_rhythmicity(period_ds)
            st.success(
                "Classification complete. Group-averaged spectra are on the "
                "**Periodograms** page."
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

    st.info(
        "Group-averaged spectra for every method you have run are on the "
        "**Periodograms** page. Averaged CWT scalograms are written to disk as "
        "PNG files when CWT is run with the corresponding checkbox enabled — "
        "see the CWT controls on **Period analysis**."
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
