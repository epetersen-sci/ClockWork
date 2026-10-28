"""Scalograms tab: the CWT's power over period and time, per group and per fly.

Built after the analysis rather than during it, so which flies are averaged can
follow a rhythmic call made after the CWT ran — autocorrelation's, typically,
which is only classified on the Rhythmicity cutoff tab.
"""

import os

import numpy as np
import streamlit as st

from clockwork.app import export_helpers
from clockwork.app.ui import charts
from clockwork.core import dam_utilities, periodograms, plotting
from clockwork.pipeline import scalograms

_RESULT_KEY = "_scalogram_averages"


def _master():
    return st.session_state.dataset


def _group_averages(ds, groups):
    st.markdown("#### Group averages")
    flags = {f: label for f, label in scalograms.RHYTHMIC_FLAGS.items() if f in ds.coords}
    options = [*flags, None]
    flag = st.radio(
        "Flies averaged",
        options,
        format_func=lambda f: "All flies" if f is None else flags[f],
        horizontal=True,
        key="scalogram_flag",
        help="A rhythmic call is made on the Rhythmicity cutoff tab; until then only "
        "'All flies' is offered.",
    )
    if not flags:
        st.caption("No rhythmic call has been made yet, so every fly is averaged.")
    picked = st.multiselect(
        "Groups",
        groups,
        default=groups,
        key="scalogram_groups",
    )

    if st.button("Compute scalograms", type="primary", disabled=not picked, key="scalogram_run"):
        bar = st.progress(0.0, text="Recomputing the CWT for the chosen flies…")
        try:
            averages, counts = scalograms.compute_group_average_scalograms(
                ds,
                groups=picked,
                flag=flag,
                progress=lambda done, total: bar.progress(
                    done / total, text=f"Recomputing the CWT: fly {done}/{total}"
                ),
            )
        except ValueError as exc:
            st.error(str(exc))
            return
        finally:
            bar.empty()
        out_dir = os.path.join(
            dam_utilities.resolve_export_dir(ds, st.session_state.get("working_dir")),
            "Averaged Scalograms",
        )
        saved = export_helpers.save_group_average_scalograms(averages, out_dir, ds=ds)
        st.session_state[_RESULT_KEY] = {
            "flag": flag,
            "saved": saved,
            "counts": counts,
            "out_dir": out_dir,
        }

    result = st.session_state.get(_RESULT_KEY)
    if not result:
        return
    who = "all flies" if result["flag"] is None else flags.get(result["flag"], result["flag"])
    st.caption(
        f"Averaged over {who}. Saved as PNG + CSV to `{result['out_dir']}`."
    )
    cols = st.columns(2)
    for i, item in enumerate(result["saved"]):
        n, total = result["counts"].get(item["group"], (item["n"], item["n"]))
        with cols[i % 2]:
            if os.path.exists(item["png"]):
                st.image(item["png"], caption=f"{item['group']} — {n} of {total} flies")


def _single_fly(ds, groups):
    st.markdown("#### One fly")
    labels = np.asarray(ds["group"].values).astype(str)
    c1, c2 = st.columns(2)
    with c1:
        group = st.selectbox("Group", groups, key="scalogram_fly_group")
    ids = [str(i) for i, g in zip(ds["id"].values, labels) if g == group]
    with c2:
        fly = st.selectbox("Fly", ids, key="scalogram_fly_id")
    if not fly:
        return
    try:
        data = periodograms.compute_single_fly_scalogram(ds, fly)
    except ValueError as exc:
        st.info(str(exc))
        return
    charts.plotly_chart(
        plotting.single_fly_scalogram_plotly(data, fly_id=fly),
        filename=f"scalogram_{fly}",
        width="stretch",
    )


def render(ctx):
    ds = _master()
    if "cwt_period" not in ds.data_vars:
        st.info("Run the CWT on the **Analysis** tab first.")
        return
    groups = sorted({str(g) for g in ds["group"].values})
    if ctx.selected_groups:
        groups = [g for g in groups if g in set(map(str, ctx.selected_groups))] or groups
    st.caption(
        "The CWT is recomputed for the chosen flies with the settings the CWT run "
        f"recorded (phase {ds.attrs.get('cwt_phase', '?')}, "
        f"{ds.attrs.get('cwt_min_period', '?')}–{ds.attrs.get('cwt_max_period', '?')} h). "
        "It takes about as long per fly as the CWT run did."
    )
    _group_averages(ds, groups)
    st.divider()
    _single_fly(ds, groups)
