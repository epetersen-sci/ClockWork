"""
Actograms — one double-plotted group-mean actogram per genotype / sex / condition.

Follows SCAMP's ``actogram2.m``: 30-minute bins, the group averaged before plotting,
each row showing its own day then the next, one amplitude scale for the whole panel,
day 1 at the top and ticks every 6 hours. The conventions, and the one place this
deliberately departs from SCAMP, are documented in ``core/actograms.py``.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import actograms as act_module
import dam_utilities
import phase_shift as ps_module
from dataset_meta import PHASE_DD, PHASE_LD, dataset_phase
from ui import charts
from ui.guards import require_dataset

st.caption(
    "One double-plotted actogram per group, averaged over the group's flies — the "
    "layout SCAMP's `actogram2.m` draws."
)

# ============================================================
# Section 1: Prerequisites
# ============================================================
ds = require_dataset()

if ds.attrs.get("time_is_relative_minutes", 0) != 1:
    st.error(
        "Actograms need the relative-minute time axis produced by the standard "
        "loading path. Reload the dataset on the **Data → Import** page."
    )
    st.stop()

_minutes = np.asarray(ds["time"].values, dtype=float)
_n_days = int(np.floor((_minutes[-1] + 1) / 1440.0))
if _n_days < 1:
    st.error("The recording is shorter than one full day — there is nothing to stack.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Flies", int(ds.sizes["id"]))
c2.metric("Complete days", _n_days)
c3.metric("Recording", f"{_minutes[-1] / 1440.0:.1f} days")

# ============================================================
# Section 2: Grouping
# ============================================================
st.subheader("Groups")

# Via group_defining_coords rather than a blocklist of the coords analyses attach.
# It answers "which coords came from the metadata" from provenance (a per-id coord
# that is ALSO an attr key — see its docstring), so a coord added by a later
# analysis cannot appear here and there is no list to keep in step with core/. It is
# the same question Groups & subsets asks, so both pages offer the same factors.
_coord_opts = dam_utilities.group_defining_coords(ds)
if not _coord_opts:
    st.error("This dataset has no categorical metadata coordinates to group by.")
    st.stop()

_default = [c for c in ("genotype", "condition", "sex", "block") if c in _coord_opts]
group_by = st.multiselect(
    "One actogram per",
    _coord_opts,
    default=_default or _coord_opts[:1],
    help="A group is one combination of these, and each group gets its own actogram.",
)
if not group_by:
    st.error("Pick at least one column.")
    st.stop()
group_by = tuple(group_by)

_labels, _cols = ps_module.group_labels(ds, group_by)
_groups = sorted(set(_labels))
_counts = {g: int((_labels == g).sum()) for g in _groups}

# Name each panel by the apparatus its flies sat in, without grouping on it. A group
# holding more than one box shows all of them rather than silently picking one.
_box_col = next(
    (c for c in ("flybox", "Monitor") if c in ds.coords and c not in group_by), None
)
_box_vals = np.asarray(ds[_box_col].values).astype(str) if _box_col else None


def _box_of(grp):
    if _box_vals is None:
        return ""
    return "+".join(sorted(set(_box_vals[_labels == grp].tolist())))


def _panel_name(grp):
    box = _box_of(grp)
    return f"{grp} · {box}" if box else str(grp)


with st.expander("Detection and layout parameters"):
    p1, p2 = st.columns(2)
    with p1:
        bin_minutes = st.selectbox(
            "Bin width (minutes)",
            [1, 5, 10, 15, 30, 60],
            index=4,
            help="SCAMP bins activity into 30-minute bins (`s30`) before plotting.",
        )
        reps = st.number_input(
            "Days plotted per row",
            min_value=1,
            max_value=3,
            value=int(act_module.DEFAULT_REPS),
            help="2 is the double-plotted convention: each row shows its own day and "
            "the next, so a drifting rhythm reads as a diagonal.",
        )
    with p2:
        bar_colour = st.color_picker("Bar colour", "#0000CD")
        panel_height = st.number_input(
            "Panel height (px)",
            min_value=200,
            max_value=1400,
            value=max(260, 60 * _n_days),
            step=20,
        )
    # LD highlighting needs to know where each group was released into DD. On an
    # LD-only or DD-only partition the answer is the whole panel or none of it —
    # split_minute is stale there, because split_xarray_dataset re-bases time to 0.
    _phase = dataset_phase(ds)
    _has_phase_meta = "split_minute" in ds.coords or "first_DD_day" in ds.coords
    _can_mark_ld = _phase in (PHASE_LD, PHASE_DD) or _has_phase_meta
    h1, h2 = st.columns(2)
    with h1:
        mark_ld = st.checkbox(
            "Colour LD days",
            value=False,
            disabled=not _can_mark_ld,
            key="actogram_mark_ld",
            help=(
                "Draw the bins recorded before the release into DD in their own "
                "colour, so the entrained part of the record is visible at a glance."
                if _can_mark_ld
                else "This dataset has no LD/DD boundary (no `first_DD_day` metadata)."
            ),
        )
    with h2:
        ld_colour = st.color_picker("LD colour", "#FF8C00", disabled=not mark_ld)

selected = st.multiselect(
    "Groups to draw",
    _groups,
    default=_groups,
    format_func=lambda g: f"{_panel_name(g)}  (n={_counts[g]})",
)
if not selected:
    st.info("Pick at least one group to draw.")
    st.stop()

# ============================================================
# Section 3: Draw
# ============================================================
try:
    res = act_module.compute_group_actograms(ds, group_by, bin_minutes=int(bin_minutes))
except Exception as exc:
    st.error(f"Could not build the actograms: {exc}")
    st.stop()

bins_per_day = res["_params"]["bins_per_day"]
bin_hours = int(bin_minutes) / 60.0
_span_hours = 24 * int(reps)

st.subheader("Results")
st.caption(
    f"{int(bin_minutes)}-minute bins, {int(reps)} day(s) per row, day 1 at the top. "
    "Every row in a panel shares one amplitude scale, so heights are comparable "
    "within an actogram but not between two of them."
)

# Per-group LD→DD boundary, resolved once. On an LD-only / DD-only partition the
# per-fly split_minute is meaningless (time was re-based to 0 by the split), so the
# dataset's own phase answers instead: everything, or nothing.
_splits, _split_disagree = ({}, set())
if mark_ld:
    if _phase == PHASE_LD:
        _splits = dict.fromkeys(_groups, np.inf)
    elif _phase == PHASE_DD:
        _splits = dict.fromkeys(_groups)
    else:
        _splits, _split_disagree = act_module.group_split_minutes(ds, group_by)
    _no_boundary = [g for g in selected if _splits.get(g) is None]
    if _no_boundary and _phase != PHASE_DD:
        st.warning(
            "No LD/DD boundary for: "
            + ", ".join(_panel_name(g) for g in _no_boundary)
            + ". Those panels are drawn in the bar colour throughout."
        )
    _disagree_here = [g for g in selected if g in _split_disagree]
    if _disagree_here:
        st.info(
            "Flies enter DD on different days within: "
            + ", ".join(_panel_name(g) for g in _disagree_here)
            + ". The highlight stops at the earliest fly's release, so a bin is only "
            "coloured while every fly in the group was still entrained."
        )

for grp in selected:
    matrix = res[grp]["matrix"]
    rows = act_module.actogram_rows(matrix, reps=int(reps))
    vmin, shift = act_module.actogram_scale(matrix)
    n_days = matrix.shape[0]

    # Bin centres in hours across the whole double-plotted row.
    x = (np.arange(rows.shape[1]) + 0.5) * bin_hours

    # Per-BIN LD flags, not per row: with reps=2 a row can straddle the release.
    ld_mask = (
        act_module.ld_bin_mask(
            n_days, bins_per_day, int(reps), int(bin_minutes), _splits.get(grp)
        )
        if mark_ld
        else np.zeros(rows.shape, dtype=bool)
    )

    fig = go.Figure()
    for i in range(n_days):
        voff = -i * shift
        _row_ld = ld_mask[i]
        fig.add_trace(
            go.Bar(
                x=x,
                y=rows[i] - vmin,
                base=voff,
                width=bin_hours,
                marker=dict(
                    color=np.where(_row_ld, ld_colour, bar_colour).tolist()
                    if mark_ld
                    else bar_colour,
                    line=dict(width=0),
                ),
                customdata=np.where(_row_ld, "LD", "DD"),
                showlegend=False,
                hovertemplate=(
                    f"day {i + 1}<br>%{{x:.1f}} h<br>%{{y:.2f}} counts/fly"
                    + ("<br>%{customdata}" if mark_ld else "")
                    + "<extra></extra>"
                ),
            )
        )

    if mark_ld and ld_mask.any():
        # Legend-only stand-ins: the real bars carry per-bin colours, which plotly
        # cannot summarise in a legend entry on its own.
        for _name, _col in (("LD", ld_colour), ("DD", bar_colour)):
            fig.add_trace(
                go.Bar(x=[None], y=[None], name=_name, marker=dict(color=_col), showlegend=True)
            )

    finite = matrix[np.isfinite(matrix)]
    _top = (float(finite.max()) - vmin) if finite.size else 1.0
    fig.update_layout(
        title=f"{_panel_name(grp)}   (n={res[grp]['n_flies']})",
        bargap=0,
        height=int(panel_height),
        margin=dict(t=60, b=50, l=70, r=20),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(
        title_text="time (h)",
        range=[0, _span_hours],
        tickmode="array",
        tickvals=list(range(0, _span_hours + 1, 6)),
        ticktext=[str(h % 24) for h in range(0, _span_hours + 1, 6)],
        showgrid=True,
        gridcolor="rgba(128,128,128,0.35)",
    )
    fig.update_yaxes(
        title_text="day",
        tickmode="array",
        tickvals=[-i * shift for i in range(n_days)],
        ticktext=[str(i + 1) for i in range(n_days)],
        range=[-(n_days - 1) * shift - 0.1 * shift, _top * 1.05],
        showgrid=True,
        gridcolor="rgba(128,128,128,0.35)",
    )
    # Named rather than left to the title, so the file says what kind of figure it
    # is; ui.charts adds the experiment prefix and the router offers the lot as PNGs.
    charts.plotly_chart(fig, filename=f"actogram_{_panel_name(grp)}", width="stretch")

st.caption(
    "Bars are the group's mean activity per fly in each bin. A bin measured in no fly "
    "of the group is left blank rather than drawn as zero — zero would claim the flies "
    "were still. The last row's second half is empty because no day follows it."
)
if mark_ld:
    st.caption(
        "Coloured bins were recorded before the group's release into DD. The label is "
        "per bin, not per row, so the row that straddles the release is part-coloured; "
        "the bin containing the transition itself is left in the bar colour, since it "
        "belongs to both epochs."
    )
