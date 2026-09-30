"""Rhythmicity tab: how many flies are rhythmic, and how strongly, by condition."""

import numpy as np
import streamlit as st

import export_helpers as ex
from period_views import by_condition
from rhythmicity_classification import (
    compare_thresholds,
    per_fly_classification_df,
    summarize_rhythmicity,
)
from ui.period_context import dd_record_days

#: (algo, label, strength column in the per-fly table, axis title)
_STRENGTHS = (
    ("ac", "Autocorrelation", "ac_metric", "Rhythmicity index (RI)"),
    ("ls", "Lomb-Scargle", "ls_metric", "LS power"),
    ("cwt", "CWT", "cwt_metric", "CWT rhythmicity"),
)


def render(ctx):
    period_ds = ctx.period_ds
    per_fly = per_fly_classification_df(period_ds)
    classified = [c for c in ("ac_rhythmic", "ls_rhythmic", "cwt_rhythmic") if c in period_ds.coords]
    strengths = [s for s in _STRENGTHS if s[2] in per_fly.columns and per_fly[s[2]].notna().any()]
    if not strengths:
        st.info("No rhythm strengths yet — run a period analysis on the **Analysis** tab.")
        return

    # --- Strength by condition --------------------------------------------
    algo = st.segmented_control(
        "Strength from",
        [s[0] for s in strengths],
        format_func={s[0]: s[1] for s in strengths}.get,
        default=strengths[0][0],
        key="rhythm_strength_violin_algo",
        persist_state="session",
    ) or strengths[0][0]
    _, label, col, y_title = next(s for s in strengths if s[0] == algo)
    by_condition.render(
        ctx,
        per_fly,
        col,
        title=f"Rhythm strength ({label})",
        y_title=y_title,
        key=f"strength_violin_{algo}",
        y_from_zero=True,
    )

    # --- Classification summary -------------------------------------------
    st.markdown("#### Rhythmic flies by group")
    if not classified:
        st.info("Not classified yet — set a cutoff on the **Rhythmicity cutoff** tab.")
    else:
        summary = summarize_rhythmicity(period_ds)
        st.dataframe(summary, width="stretch", height=260)
        ex.save_df_button(
            "Save group summary to working folder",
            summary,
            period_ds,
            "rhythmicity_group_summary.csv",
            key="dl_rhyth_group_summary",
        )
        with st.expander("Per-fly classification table"):
            table = per_fly.copy()
            # Flag (never filter) each fly's DD record length against the floor, so
            # short records are visible per fly rather than silently dropped.
            days = dd_record_days(period_ds)
            table["record_days"] = [round(days.get(str(f), float("nan")), 2) for f in table["fly_id"]]
            table["below_floor"] = [
                bool(days.get(str(f), 0.0) < ctx.min_days_floor) for f in table["fly_id"]
            ]
            st.dataframe(table, width="stretch", height=300)
            ex.save_df_button(
                "Save per-fly classification to working folder",
                table,
                period_ds,
                "rhythmicity_per_fly.csv",
                key="dl_rhyth_per_fly",
            )

    # --- Sensitivity --------------------------------------------------------
    with st.expander("Threshold sensitivity analysis"):
        st.markdown(
            "Sweep one algorithm's threshold to see how the rhythmic/arrhythmic call "
            "changes — a check on how much a conclusion depends on the cutoff."
        )
        sens_algo = st.selectbox(
            "Algorithm",
            options=[s[0] for s in strengths],
            format_func={"ls": "Lomb-Scargle (power)", "ac": "Autocorrelation (RI)", "cwt": "CWT (rhythmicity)"}.get,
            key="sens_algo",
        )
        lo_d, hi_d = {"ls": (0.001, 0.05), "ac": (0.05, 0.5), "cwt": (0.05, 0.8)}[sens_algo]
        c1, c2, c3 = st.columns(3)
        with c1:
            lo = st.number_input("Min threshold", 0.0, 1.0, float(lo_d), 0.005, format="%.4f", key="sens_min")
        with c2:
            hi = st.number_input("Max threshold", 0.0, 1.0, float(hi_d), 0.005, format="%.4f", key="sens_max")
        with c3:
            steps = st.slider("Number of steps", 3, 30, 10, key="sens_steps")
        if st.button("Run sensitivity analysis", key="run_sens"):
            try:
                st.session_state["_period_sens_df"] = compare_thresholds(
                    period_ds, algorithm=sens_algo, thresholds=list(np.linspace(lo, hi, steps))
                )
            except Exception as e:
                st.error(f"Sensitivity analysis error: {e}")
        sens = st.session_state.get("_period_sens_df")
        if sens is not None and not sens.empty:
            st.dataframe(sens, width="stretch")
            ex.save_df_button(
                "Save sensitivity to working folder",
                sens,
                period_ds,
                "rhythmicity_sensitivity.csv",
                key="dl_rhyth_sens",
            )
