"""
Sleep (SCAMP style) — per-day profiles, averaged profiles, and the sleep-feature bars.

Every metric is computed by ``core/scamp_sleep``, which reproduces SCAMP's
``sleepcalc3.m``. Days are numbered within their epoch: LD day 1..n from the
recording start, DD day 1..n from ``first_DD_day``, so a selection reads the way a
free-run is described.

This is deliberately NOT the same sleep as **Sleep analysis**. That page uses
``core/sleep_analysis``, whose definition bridges short gaps and treats an
unmeasured minute as unknown. SCAMP does neither: a bout is an unbroken run of
zero counts, and a missing reading is scored as zero, hence as sleep. The point of
this page is a number that can be put beside a SCAMP run, so the two definitions
stay separate rather than one being made to approximate the other.
"""

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import dam_utilities
import export_helpers
import phase_shift as ps_module
import scamp_sleep as ss
from dataset_meta import dataset_fingerprint
from ui import charts, days
from ui.guards import require_dataset

# SCAMP's own figure colours: dark grey / orange / green, then extras.
PALETTE = ["#3D3D3D", "#FF8C00", "#00A000", "#8B008B", "#00868B", "#B22222", "#556B2F"]
# The figures are drawn on white so they match SCAMP and print cleanly, but the app
# itself may be in dark mode — plotly would then inherit the theme's near-white text
# and put it on the white panel, invisibly. Every colour is therefore stated here
# rather than inherited.
INK = "#000000"
WHITE = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(color=INK, size=16),
    title_font=dict(color=INK, size=21),
    legend=dict(font=dict(color=INK), bgcolor="rgba(255,255,255,0.85)",
                bordercolor=INK, borderwidth=1),
)
AXIS = dict(showline=True, linecolor=INK, mirror=True, ticks="outside",
            showgrid=False, zeroline=False, color=INK,
            title_font=dict(color=INK, size=16), tickfont=dict(color=INK, size=14))

# `layout="wide"` still caps the main container; lift it so the wide panels get
# the room they need on screen. Export width is set per figure, independently.
st.markdown(
    "<style>.block-container{max-width:98rem !important;padding-top:2.2rem;}</style>",
    unsafe_allow_html=True,
)

# Streamlit discards a widget's state once the widget stops being rendered, so
# switching view would wipe the other view's settings. Re-assigning each key marks
# it as script-owned and keeps it across the switch.
#
# This has to be an ALLOWLIST of the stateful widgets, never a prefix match on
# everything: a button refuses to be created at all if its key already carries a
# value (StreamlitValueAssignmentNotAllowedError), and that error is raised when
# st.button runs, not when the value is assigned — so it cannot be caught here.
# Buttons need no persisting anyway: a button's value is only True on the run
# right after the click. The one button key on this page, kept out deliberately,
# is scamp_avg_xlsx.
_PERSIST_KEYS = {
    "scamp_view",         # radio: which view
    "scamp_facets",       # multiselect: figures to draw
    "scamp_show_tests",   # checkbox: significance tests
    "perday_which",       # multiselect: per-day facets
    "perday_cols",        # selectbox: panel columns
    "avg_epoch",          # radio: LD / DD
}
for _k in _PERSIST_KEYS & set(st.session_state.keys()):
    st.session_state[_k] = st.session_state[_k]
# The day tick-boxes, by the same rule and for the same reason.
days.persist_checkbox_keys("perday_cb_", "avg_LD_cb_", "avg_DD_cb_")

st.caption(
    "Sleep and activity in the SCAMP layout. Metrics match `sleepcalc3.m` — "
    "sleep is 5+ minutes of zero counts, a bout counts in the bin it starts in."
)

# ============================================================
# Section 1: Prerequisites
# ============================================================
ds = require_dataset()

if ds.attrs.get("time_is_relative_minutes", 0) != 1:
    st.error(
        "SCAMP sleep metrics need the relative-minute time axis produced by the "
        "standard loading path. Reload the dataset on the **Data → Import** page."
    )
    st.stop()

minutes = np.asarray(ds["time"].values, dtype=float)
n_days_total = int(np.floor((minutes[-1] + 1) / 1440.0))
if n_days_total < 1:
    st.error("The recording is shorter than one full day.")
    st.stop()

# Day numbering, the tick-box widget and the label/filename spellings all come from
# ui.days, which the Sleep bouts page uses too — see its docstring for why there is
# one copy rather than one per page.
DAYS, dd_day = days.day_table(ds, warn=st.warning)

m1, m2, m3 = st.columns(3)
m1.metric("Flies", int(ds.sizes["id"]))
m2.metric("LD days", int((DAYS.epoch == "LD").sum()))
m3.metric("DD days", int((DAYS.epoch == "DD").sum()))
if dd_day is None:
    st.caption("No `first_DD_day` in the metadata — every day is treated as LD.")

# A group subset on Groups & subsets replaces `dataset` with a narrowed copy and
# keeps the original in `dataset_full`. That is invisible here otherwise, and it is
# the usual reason a group holds fewer flies than the metadata implies.
_full = st.session_state.get("dataset_full")
if _full is not None and int(_full.sizes.get("id", 0)) > int(ds.sizes["id"]):
    st.warning(
        f"A group filter is active: **{int(ds.sizes['id'])} of "
        f"{int(_full.sizes['id'])}** flies are loaded, so every n below counts only "
        "the filtered set. Reset it on the **Data → Groups & subsets** page to use "
        "all flies."
    )

# ============================================================
# Section 2: Grouping
# ============================================================
st.subheader("Groups")

# Via group_defining_coords, so this page offers exactly the factors Groups &
# subsets does — see its docstring for why provenance beats a blocklist.
_opts = dam_utilities.group_defining_coords(ds)
if not _opts:
    st.error("This dataset has no categorical metadata coordinates to group by.")
    st.stop()

gc1, gc2 = st.columns(2)
with gc1:
    series_col = st.selectbox(
        "One line/bar per",
        _opts,
        index=_opts.index("genotype") if "genotype" in _opts else 0,
        help="The factor compared inside each figure — genotype in the SCAMP figures.",
    )
with gc2:
    _facet_default = [c for c in ("sex", "condition", "block") if c in _opts]
    facet_cols = st.multiselect(
        "Separate figures for each",
        [c for c in _opts if c != series_col],
        default=_facet_default,
        help="Add `sex` to get males and females on separate sets of graphs.",
    )

group_by = tuple([series_col] + list(facet_cols))
labels, _cols = ps_module.group_labels(ds, group_by)
series_vals = np.asarray(ds[series_col].values).astype(str)
facet_vals = (
    np.array(["_".join(v) for v in zip(*[np.asarray(ds[c].values).astype(str)
                                         for c in facet_cols])])
    if facet_cols
    else np.array([""] * ds.sizes["id"])
)

# Which apparatus each figure's flies sat in. A figure split by condition alone does
# not say which box produced it, and that is the first thing to check when two panels
# disagree — so it is named in the picker AND in every title. Skipped when flybox is
# already one of the grouping columns, since the label or legend then says it anyway.
_BOX_COL = next((c for c in ("flybox", "Monitor") if c in ds.coords), None)
_BOX_VALS = np.asarray(ds[_BOX_COL].values).astype(str) if _BOX_COL else None
_NAME_BOX = _BOX_COL is not None and _BOX_COL not in group_by


def _boxes_of(facet):
    """``'cake'`` / ``'bun+tart'`` — the box(es) behind one figure, or ``''``."""
    if not _NAME_BOX:
        return ""
    member = facet_vals == facet
    return "+".join(sorted(set(_BOX_VALS[member].tolist())))


def _facet_label(facet):
    """How a figure is named in the picker and in its own title.

    With no figure split there is no facet value to lead with, so the boxes stand
    alone rather than trailing a redundant "all flies".
    """
    boxes = _boxes_of(facet)
    if not facet:
        return boxes or "all flies"
    return f"{facet} · {boxes}" if boxes else facet


_all_facets = sorted(set(facet_vals.tolist()))
chosen_facets = st.multiselect(
    "Figures to draw", _all_facets, default=_all_facets[:1],
    key="scamp_facets",
    format_func=_facet_label,
    help="Each figure costs about a second to build and send. Add them as you need them.",
) if facet_cols else [""]
if len(chosen_facets) > 3:
    st.warning(
        f"{len(chosen_facets)} figures selected. Each one is rebuilt on every click, "
        "so the page will feel sluggish — narrow this down while you are adjusting "
        "settings, then widen it once to export."
    )
if not chosen_facets:
    st.info("Pick at least one figure to draw.")
    st.stop()

colour_of = {v: PALETTE[i % len(PALETTE)]
             for i, v in enumerate(sorted(set(series_vals.tolist())))}

# ============================================================
# Section 3: Sleep scoring (once per dataset)
# ============================================================


@st.cache_data(show_spinner="Scoring sleep (SCAMP definition)…")
def _score_sleep(fp, _ds):
    """``(counts, sleep, bout_len)`` — the SCAMP scoring, which runs per fly.

    ``fp`` carries the dataset identity and takes NO leading underscore; ``_ds``
    does, so streamlit does not try to hash the Dataset. Getting either wrong is
    backlog item 14 — see ``dataset_meta.dataset_fingerprint``.

    The count matrix comes back with the masks rather than being rebuilt outside,
    because every caller needs all three and the transpose is itself a copy of the
    whole record.
    """
    counts = np.asarray(_ds["activity"].transpose("time", "id").values, dtype=float)
    sleep, bout_len = ss.per_minute(counts)
    return counts, sleep, bout_len


counts, sleep, bout_len = _score_sleep(dataset_fingerprint(ds), ds)

_measured = np.isfinite(counts).mean()
if _measured < 1.0:
    st.info(
        f"{(1 - _measured) * 100:.2f}% of minutes have no reading. Following SCAMP "
        "these are scored as zero counts, so they count toward sleep."
    )


def composition_note(in_facet, svals):
    """Say which apparatus each line averaged over.

    A group pools every fly matching its grouping columns, so unless flybox is one
    of them a line can silently combine two boxes. Stating it means the question
    "is this one box or several?" is answered on the figure instead of guessed at.
    """
    if _BOX_VALS is None:
        return ""
    parts = []
    for sv in svals:
        member = in_facet & (series_vals == sv)
        box_counts = {}
        for b in _BOX_VALS[member]:
            box_counts[b] = box_counts.get(b, 0) + 1
        detail = ", ".join(f"{b} {n}" for b, n in sorted(box_counts.items()))
        parts.append(f"**{sv}** n={int(member.sum())} ({detail})")
    return " · ".join(parts)


def _legend_name(value, n):
    """'Mito-gfp (n=28)'. n belongs on the figure, not only in the stats sheet."""
    return f"{value} (n={n})"


def _day_window(absolute_day):
    return slice(absolute_day * 1440, (absolute_day + 1) * 1440)


view = st.radio("View", ["Per-day profiles", "Averaged days + sleep features"],
                horizontal=True, key="scamp_view")

# ============================================================
# Section 4: Per-day profiles — one panel per day
# ============================================================
if view == "Per-day profiles":
    st.caption(
        "One panel per day, mean ± SEM across the flies in each group — SCAMP's "
        "'All days' figures."
    )
    which = st.radio("Show", ["Activity", "Sleep"], horizontal=True, key="perday_which")
    metric = "amean" if which == "Activity" else "s30"
    ylabel = ss.PROFILE_LABELS[metric]
    st.markdown("**Days**")
    sel_days, day_choice = days.day_checkboxes(DAYS, "perday")
    if not day_choice:
        st.info("Tick at least one day.")
        st.stop()
    ncols = st.slider("Columns", 1, 4, 3, key="perday_cols")

    x = (np.arange(48) + 0.5) * 0.5  # 30-min bin centres, hours
    for facet in chosen_facets:
        in_facet = facet_vals == facet
        nrows = int(np.ceil(len(sel_days) / ncols))
        fig = make_subplots(
            rows=nrows, cols=ncols,
            subplot_titles=[r.label for _, r in sel_days.iterrows()],
            horizontal_spacing=0.07, vertical_spacing=0.12,
        )
        for k, (_, row) in enumerate(sel_days.iterrows()):
            r, c = k // ncols + 1, k % ncols + 1
            win = _day_window(int(row.absolute))
            prof = ss.profiles(counts, sleep, win)[metric]
            for sv in sorted(set(series_vals[in_facet].tolist())):
                member = in_facet & (series_vals == sv)
                mean, sem, _ = ss.mean_sem(prof[:, member], axis=1)
                fig.add_trace(
                    go.Scatter(
                        x=x, y=mean, name=_legend_name(sv, int(member.sum())),
                        legendgroup=sv, showlegend=(k == 0),
                        mode="lines+markers", marker=dict(size=4),
                        line=dict(color=colour_of[sv], width=1.5),
                        error_y=dict(type="data", array=sem, visible=True, thickness=1,
                                     width=2, color=colour_of[sv]),
                    ),
                    row=r, col=c,
                )
            fig.update_xaxes(title_text="ZT/CT" if r == nrows else "", row=r, col=c)
            fig.update_yaxes(title_text=ylabel if c == 1 else "", row=r, col=c)
        # This grid is exported much wider than the other figures, so the shared
        # font sizes come out looking small against it — scale them up here.
        DAY_TICK, DAY_AXIS_TITLE, DAY_SUBPLOT, DAY_TITLE = 18, 21, 24, 30
        fig.update_xaxes(tickvals=[0, 6, 12, 18, 24], range=[0, 24], **AXIS)
        fig.update_yaxes(**AXIS)
        fig.update_xaxes(tickfont=dict(size=DAY_TICK, color=INK),
                         title_font=dict(size=DAY_AXIS_TITLE, color=INK))
        fig.update_yaxes(tickfont=dict(size=DAY_TICK, color=INK),
                         title_font=dict(size=DAY_AXIS_TITLE, color=INK))
        _days_txt = days.days_label(sel_days)
        _facet_txt = _facet_label(facet) if (facet or _boxes_of(facet)) else ""
        title = (f"{which} — {_days_txt} — {_facet_txt}" if _facet_txt
                 else f"{which} — {_days_txt}")
        # Filename built from the same pieces, with ranges spelled 'to' (see
        # ui.days.compress_runs) so a different day pick can never reuse this name.
        _file_txt = f"sleep perday {which} {days.days_label(sel_days, dash='to')} {facet}"
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center", y=0.985, yanchor="top",
                       font=dict(size=DAY_TITLE, color=INK)),
            height=max(340, 290 * nrows) + 60,
            # An explicit width is what the PNG export uses; without it kaleido
            # falls back to 700 px and the day panels come out cramped.
            width=620 * ncols + 280,
            margin=dict(t=130, b=75, l=100, r=30), **WHITE,
        )
        fig.update_layout(legend=dict(font=dict(size=DAY_TICK, color=INK)),
                          title_font=dict(size=DAY_TITLE, color=INK))
        # Subplot titles are annotations and do not inherit layout.font.
        for ann in fig.layout.annotations:
            ann.font.color = INK
            ann.font.size = DAY_SUBPLOT
        charts.plotly_chart(fig, filename=_file_txt.strip(), width="stretch")
        _note = composition_note(in_facet, sorted(set(series_vals[in_facet].tolist())))
        if _note:
            st.caption("Flies averaged — " + _note)

# ============================================================
# Section 5: Averaged days + sleep-feature bars
# ============================================================
if view == "Averaged days + sleep features":
    st.caption(
        "Select days within one epoch; every metric is averaged over those days per "
        "fly, then across flies. Bars are 24 h and the two halves of the day."
    )
    ec1, ec2 = st.columns([1, 3])
    with ec1:
        epochs = sorted(set(DAYS.epoch))
        epoch = st.radio("Epoch", epochs, horizontal=True, key="avg_epoch")
    pool = DAYS[DAYS.epoch == epoch]
    with ec2:
        st.markdown(f"**{epoch} days to average**")
        # Day 1 of an epoch is the transition day, so it starts unticked.
        sel, picked = days.day_checkboxes(
            pool, f"avg_{epoch}", default=list(pool.label)[1:] or list(pool.label)
        )
    if not picked:
        st.info("Tick at least one day.")
        st.stop()
    abs_days = [int(d) for d in sel.absolute]
    day_label = days.day_summary(sel, epoch)
    # Same days, spelled for a filename (ranges as 'to' — see ui.days).
    day_slug = days.day_summary(sel, epoch, dash="to")

    # SCAMP names the halves LP/DP in LD and sLP/sDP in DD (subjective).
    lp, dp = ("LP", "DP") if epoch == "LD" else ("sLP", "sDP")
    BINS = [("24", slice(0, 1440)), (lp, slice(0, 720)), (dp, slice(720, 1440))]
    xaxis_name = "ZT" if epoch == "LD" else "CT"

    # --- per-fly values, averaged over the chosen days -----------------------------
    per_fly = {}  # (metric, bin) -> array over flies
    for bname, bslice in BINS:
        stack = []
        for d in abs_days:
            off = d * 1440
            win = slice(off + bslice.start, off + bslice.stop)
            stack.append(ss.bin_metrics(counts, sleep, bout_len, win))
        for m in ss.METRIC_LABELS:
            # A fly with no bout in every selected day gives an all-NaN column; the
            # mean of that is legitimately NaN, so the warning is noise.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                per_fly[(m, bname)] = np.nanmean(
                    np.vstack([s[m] for s in stack]), axis=0
                )

    prof_stack = {m: [] for m in ss.PROFILE_LABELS}
    for d in abs_days:
        p = ss.profiles(counts, sleep, _day_window(d))
        for m in prof_stack:
            prof_stack[m].append(p[m])
    # A fly not recorded on ANY selected day gives an all-NaN column — its mean is
    # legitimately NaN (mean_sem then leaves it out of the group), so the warning
    # numpy raises for that slice is noise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        prof_mean = {m: np.nanmean(np.stack(v), axis=0) for m, v in prof_stack.items()}

    x = (np.arange(48) + 0.5) * 0.5
    summary_rows, perfly_rows, test_specs = [], [], []

    for facet in chosen_facets:
        in_facet = facet_vals == facet
        svals = sorted(set(series_vals[in_facet].tolist()))
        _facet_txt = _facet_label(facet) if (facet or _boxes_of(facet)) else ""
        tag = f" — {_facet_txt}" if _facet_txt else ""
        st.markdown(f"#### {day_label}{tag}")
        _note = composition_note(in_facet, svals)
        if _note:
            st.caption("Flies averaged — " + _note)

        # Activity and sleep share one wide 1x2 figure — two panels that are read
        # together and export as one image.
        prof_keys = list(ss.PROFILE_LABELS)
        prof_fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[ss.PROFILE_LABELS[k] for k in prof_keys],
            horizontal_spacing=0.10,
        )
        for pi, m in enumerate(prof_keys):
            for sv in svals:
                member = in_facet & (series_vals == sv)
                mean, sem, _ = ss.mean_sem(prof_mean[m][:, member], axis=1)
                prof_fig.add_trace(
                    go.Scatter(
                        x=x, y=mean, name=_legend_name(sv, int(member.sum())),
                        legendgroup=sv, showlegend=(pi == 0),
                        mode="lines+markers", marker=dict(size=4),
                        line=dict(color=colour_of[sv], width=1.5),
                        error_y=dict(type="data", array=sem, visible=True, thickness=1,
                                     width=2, color=colour_of[sv]),
                    ),
                    row=1, col=pi + 1,
                )
            prof_fig.update_xaxes(title_text=xaxis_name, row=1, col=pi + 1)
            prof_fig.update_yaxes(title_text=ss.PROFILE_LABELS[m], row=1, col=pi + 1)
        prof_fig.update_xaxes(range=[0, 24], tickvals=[0, 6, 12, 18, 24], **AXIS)
        prof_fig.update_yaxes(**AXIS)
        prof_fig.update_layout(
            title=dict(text=f"Activity and sleep — {day_label}{tag}", x=0.5,
                       xanchor="center", y=0.975, yanchor="top"),
            height=470, width=1650, margin=dict(t=100, b=65, l=95, r=40), **WHITE,
        )
        for ann in prof_fig.layout.annotations:
            ann.font.color = INK
            ann.font.size = 18
        # Named after the DAYS, not just the epoch: exporting DD 2-4 and then DD
        # 6-10 would otherwise write the same sleep_profiles_DD.png twice.
        charts.plotly_chart(
            prof_fig, filename=f"sleep_profiles_{day_slug}{tag}", width="stretch"
        )

        # The six sleep features live in one 2x3 subplot figure, so they read as a
        # single panel and export as a single image.
        metric_keys = list(ss.METRIC_LABELS)
        bars_fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[ss.METRIC_LABELS[k] for k in metric_keys],
            horizontal_spacing=0.11, vertical_spacing=0.17,
        )
        for bi, m in enumerate(metric_keys):
            brow, bcol = bi // 3 + 1, bi % 3 + 1
            for sv in svals:
                member = in_facet & (series_vals == sv)
                means, sems = [], []
                for bname, _ in BINS:
                    vals = per_fly[(m, bname)][member]
                    mean, sem, n = ss.mean_sem(vals[:, None], axis=0)
                    means.append(float(mean[0]))
                    sems.append(float(sem[0]))
                    summary_rows.append({
                        "figure": facet or "all", "metric": m,
                        "label": ss.METRIC_LABELS[m], "bin": bname, series_col: sv,
                        "mean": float(mean[0]), "sem": float(sem[0]), "n": int(n[0])})
                    for fly, value in zip(np.asarray(ds["id"].values)[member], vals):
                        perfly_rows.append({
                            "figure": facet or "all", "metric": m, "bin": bname,
                            series_col: sv, "id": str(fly), "value": float(value)})
                bars_fig.add_trace(
                    go.Bar(
                        x=list(range(len(BINS))), y=means,
                        name=_legend_name(sv, int(member.sum())), legendgroup=sv,
                        showlegend=(bi == 0),
                        marker=dict(color=colour_of[sv],
                                    line=dict(color="black", width=0.6)),
                        error_y=dict(type="data", array=sems, visible=True,
                                     color="black", thickness=1, width=4),
                    ),
                    row=brow, col=bcol,
                )

            # Tests are deferred: they cost about a fifth of the whole page and
            # are only read in the preview table or the workbook.
            test_specs.append((facet or "all", m, tuple(svals), in_facet))

        # Fixed numeric ticks: every bin keeps its slot even with no data in it.
        bars_fig.update_xaxes(
            tickmode="array", tickvals=list(range(len(BINS))),
            ticktext=[b for b, _ in BINS], range=[-0.6, len(BINS) - 0.4], **AXIS)
        bars_fig.update_yaxes(**AXIS)
        bars_fig.update_layout(
            title=dict(text=f"Sleep features — {day_label}{tag}",
                       x=0.5, xanchor="center", y=0.975, yanchor="top"),
            barmode="group", bargap=0.25, bargroupgap=0.05,
            height=740, width=1750, margin=dict(t=110, b=60, l=85, r=40), **WHITE,
        )
        for ann in bars_fig.layout.annotations:
            ann.font.color = INK
            ann.font.size = 18
        charts.plotly_chart(
            bars_fig, filename=f"sleep_features_{day_slug}{tag}", width="stretch"
        )

    def compute_tests():
        """Run every pairwise comparison. Called only on demand — see test_specs."""
        rows = []
        for fig_name, m, svals_t, in_facet_t in test_specs:
            for bname, _ in BINS:
                groups = {sv: per_fly[(m, bname)][in_facet_t & (series_vals == sv)]
                          for sv in svals_t}
                for row in ss.pairwise_tests(groups):
                    row.update({"figure": fig_name, "metric": m, "bin": bname,
                                "epoch": epoch, "days": day_label})
                    rows.append(row)
        return rows

    # ------------------------------------------------------------------ exports
    # The figures are offered by the router's own button, after the page body, like
    # every other page's. This is the one export that button cannot cover.
    st.markdown("**Export**")
    _notes = pd.DataFrame([
        {"field": "epoch", "value": epoch},
        {"field": "days averaged", "value": day_label},
        {"field": "sleep definition",
         "value": f"{ss.SLEEP_THRESHOLD_MIN}+ min of zero counts "
                  "(SCAMP fly_sleepthresh, no gap bridging)"},
        {"field": "missing minutes",
         "value": "scored as zero counts, as SCAMP does — they count toward sleep"},
        {"field": "tests",
         "value": "Welch t-test and Mann-Whitney U per pair, each Holm-corrected "
                  "within metric x bin x figure"},
    ])
    # A callable, not a list: the tests sheet is the expensive one, and building it
    # to populate an argument would run it on every rerun whether or not anyone
    # clicks — which is what test_specs exists to avoid.
    export_helpers.save_excel_button(
        "Save stats workbook (.xlsx)",
        lambda: [
            ("summary", pd.DataFrame(summary_rows)),
            ("per_fly", pd.DataFrame(perfly_rows)),
            ("tests", pd.DataFrame(compute_tests())),
            ("notes", _notes),
        ],
        ds,
        f"sleep_scamp_stats_{day_slug}.xlsx",
        key="scamp_avg_xlsx",
        help="The summary, the per-fly values behind it, and every pairwise test.",
    )

    if st.checkbox("Show significance tests", key="scamp_show_tests",
                   help="Adds a few seconds; the workbook always includes them."):
        st.dataframe(pd.DataFrame(compute_tests()), width="stretch")
