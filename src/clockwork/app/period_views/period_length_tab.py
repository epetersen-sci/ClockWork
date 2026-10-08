"""Period length tab: every fly's period, as a table and by condition."""

import pandas as pd
import streamlit as st

from clockwork.app import export_helpers as ex
from clockwork.app.period_views import by_condition
from clockwork.core.dataset_meta import dataset_fingerprint
from clockwork.core.rhythmicity_classification import per_fly_classification_df

_ALGOS = (("ac", "Autocorrelation"), ("ls", "Lomb-Scargle"), ("cwt", "CWT"))


@st.cache_data(show_spinner=False, max_entries=8)
def _build_period_summary_df(fp, _ds):
    """The per-fly period summary table.

    ``fp`` carries no leading underscore, and must not grow one: Streamlit drops
    ANY leading-underscore parameter from the cache key, and as ``_fp`` the key
    was empty — the table was built once per session and returned unchanged for
    every later dataset. ``_ds`` keeps its underscore because a Dataset is what
    the fingerprint stands in for.
    """
    cols = {"ID": list(_ds["id"].values)}
    if "group" in _ds.coords:
        cols["Group"] = [str(g) for g in _ds["group"].values]
    for col, var in (
        ("CWT Period (h)", "cwt_period"),
        ("LS Period (h)", "ls_period"),
        ("AC Period (h)", "ac_period"),
        ("MESA Period (h)", "mesa_period"),
        # The strength each algorithm's cutoff is applied to.
        ("AC RI (strength)", "ac_power"),
        ("LS Power (strength)", "ls_power"),
        ("LS FAP", "ls_fap"),
        ("CWT Rhythmicity (strength)", "cwt_rhythmicity"),
        ("MESA SNR (peak/median)", "mesa_power"),
    ):
        if var in _ds.data_vars:
            cols[col] = [float(v) for v in _ds[var].values]
    # AC is the canonical filter; LS/CWT flags are diagnostic only.
    for col, coord in (
        ("AC Rhythmic", "ac_rhythmic"),
        ("LS Rhythmic (diagnostic)", "ls_rhythmic"),
        ("CWT Rhythmic (diagnostic)", "cwt_rhythmic"),
    ):
        if coord in _ds.coords:
            cols[col] = [bool(v) for v in _ds[coord].values]
    df = pd.DataFrame(cols)
    # Alphabetical (Group then ID) for a predictable, GraphPad-friendly export.
    keys = [c for c in ("Group", "ID") if c in df.columns]
    return df.sort_values(keys).reset_index(drop=True) if keys else df


def render(ctx):
    period_ds = ctx.period_ds
    per_fly = per_fly_classification_df(period_ds)
    algos = [
        (a, lbl)
        for a, lbl in _ALGOS
        if f"{a}_period" in per_fly.columns and per_fly[f"{a}_period"].notna().any()
    ]
    if not algos:
        st.info("No periods yet — run a period analysis on the **Analysis** tab.")
        return

    c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
    with c1:
        algo = st.segmented_control(
            "Period from",
            [a for a, _ in algos],
            format_func=dict(algos).get,
            default=algos[0][0],
            key="rhythm_period_violin_algo",
            persist_state="session",
        ) or algos[0][0]
    flag = f"{algo}_rhythmic"
    with c2:
        rhythmic_only = st.toggle(
            "Rhythmic flies only",
            value=True,
            key="rhythm_period_violin_rhythmic",
            persist_state="session",
            disabled=flag not in per_fly.columns,
            help="An arrhythmic fly's period is just its periodogram's tallest peak. "
            "Needs a classification (Rhythmicity cutoff tab).",
        )
    shown = per_fly[per_fly[flag].astype(bool)] if rhythmic_only and flag in per_fly else per_fly
    by_condition.render(
        ctx,
        shown,
        f"{algo}_period",
        title=f"Period ({dict(algos)[algo]})",
        y_title="Period (h)",
        key=f"period_violin_{algo}",
        y_from_zero=False,
    )

    st.markdown("#### Per-fly periods")
    summary = _build_period_summary_df(dataset_fingerprint(period_ds), period_ds)
    st.dataframe(summary, width="stretch", height=300)
    ex.save_df_button(
        "Save period summary to working folder",
        summary,
        period_ds,
        "period_summary.csv",
        key="download_period_csv",
    )
