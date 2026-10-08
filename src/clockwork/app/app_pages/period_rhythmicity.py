"""
Period & rhythmicity — run the period analyses, set the rhythmic cutoff, and read
the results, on one page.

This was three pages (Period analysis, Rhythmicity, Periodograms) that had to
agree on the phase and the period range — the range is also the classification
window — and did so through shadow session keys. They are tabs now under one set
of controls, so the agreement is structural: every tab gets the same
``PeriodContext``.

Tabs:
- **Analysis** — tick the methods, Run analysis; advanced options for all four.
- **Rhythmicity cutoff** — the threshold explorer for one algorithm, and Classify.
- **Period length** — per-fly periods, by condition and as a table.
- **Rhythmicity** — rhythm strength by condition, the rhythmic counts, sensitivity.
- **Periodograms** — group-averaged spectra.
"""

import numpy as np
import streamlit as st

from clockwork.app.period_views import (
    PeriodContext,
    analysis_tab,
    cutoff_tab,
    period_length_tab,
    periodograms_tab,
    rhythmicity_tab,
)
from clockwork.app.ui import facet_panels
from clockwork.app.ui.filters import DISPLAY_GROUPS_KEY, group_filter_sidebar
from clockwork.app.ui.guards import require_dataset
from clockwork.app.ui.period_context import dd_record_days, render_period_row, render_phase_picker

ds = require_dataset()

phase_selection, period_ds, analysis_src, phase_arg = render_phase_picker(ds)
min_period, max_period, min_days_floor, max_bridge_gap = render_period_row()

# Which flies the DD-days floor keeps — the flies that actually enter the analysis.
_days = dd_record_days(period_ds)
_excluded = [f for f, d in _days.items() if d < min_days_floor] if min_days_floor > 0 else []
_kept = np.array([d for f, d in _days.items() if f not in set(_excluded)], dtype=float)
if len(_kept):
    _phase_word = {"DD": "DD", "LD": "LD", "full": "full-recording"}.get(
        phase_selection, str(phase_selection)
    )
    st.caption(
        f"**{len(_kept)} flies** enter period analysis ({_phase_word} data): record "
        f"length {_kept.mean():.1f} d on average, {_kept.min():.1f}–{_kept.max():.1f} d."
        + (
            f" **{len(_excluded)}** excluded below the {min_days_floor:g}-day floor."
            if _excluded
            else ""
        )
    )

# Sidebar: the app-wide group filter (the Periodograms tab reads it) and the panel
# layout (every figure tab reads it).
_, _, selected_groups, _ = group_filter_sidebar(ds, key=DISPLAY_GROUPS_KEY)
facet_spec, facet_table = facet_panels.facet_controls(period_ds)

ctx = PeriodContext(
    ds=ds,
    period_ds=period_ds,
    analysis_src=analysis_src,
    phase_selection=phase_selection,
    phase_arg=phase_arg,
    min_period=min_period,
    max_period=max_period,
    min_days_floor=min_days_floor,
    max_bridge_gap=max_bridge_gap,
    facet_spec=facet_spec,
    facet_table=facet_table,
    selected_groups=list(selected_groups),
)

# Lazy tabs: only the open one runs, and the key keeps it open across the rerun
# that Run analysis and Classify cause.
_TABS = [
    ("Analysis", analysis_tab),
    ("Rhythmicity cutoff", cutoff_tab),
    ("Period length", period_length_tab),
    ("Rhythmicity", rhythmicity_tab),
    ("Periodograms", periodograms_tab),
]
for tab, (_, view) in zip(
    st.tabs([name for name, _ in _TABS], key="period_rhythm_tab", on_change="rerun"), _TABS
):
    if tab.open:
        with tab:
            view.render(ctx)
