"""One per-fly value by condition: a violin per group, or per facet panel.

Shared by the Period length and Rhythmicity tabs, which draw the same kind of
figure for different numbers (a period; a rhythm strength).
"""

import streamlit as st

from clockwork.app.ui import charts, facet_panels
from clockwork.app.ui import stats as stats_ui
from clockwork.core import facets, group_stats, plotting


def render(ctx, per_fly, value_col, *, title, y_title, key, y_from_zero, family=group_stats.WELCH):
    """Draw ``per_fly[value_col]`` (one row per fly, ``fly_id`` and ``group``) with
    its group statistics. Returns the statistics shown (a list, possibly empty)."""
    if per_fly[value_col].notna().sum() == 0:
        st.info("No flies with a value for this selection.")
        return []
    family = stats_ui.family(family)
    if ctx.facet_spec.active:
        panels = facets.resolve_panels(ctx.facet_table, ctx.facet_spec)
        results = group_stats.per_panel(
            per_fly, value_col, panels, family, id_col="fly_id", measure=title
        )
        pairs = plotting.faceted_violins(
            per_fly,
            value_col,
            panels,
            facets.facet_colours(facets.layout_levels(panels)),
            id_col="fly_id",
            title=title,
            y_title=y_title,
            x_title=ctx.facet_spec.compare_by,
            shared_y=ctx.facet_spec.shared_y,
            y_from_zero=y_from_zero,
            stats=results,
        )
        facet_panels.render_panels(pairs, ctx.facet_spec, ctx.period_ds, key=key, filename=key)
        stats_ui.notice(results)
        return [r for r in results if r is not None]
    df = per_fly.copy()
    df["group"] = df["group"].astype(str) if "group" in df else "All flies"
    result = group_stats.compare_frame(df, value_col, "group", family, measure=title)
    fig, _ = plotting.group_violins(
        df, value_col, group_col="group", title=title, y_title=y_title, stats=result
    )
    if not y_from_zero:
        fig.update_yaxes(rangemode="normal")
    charts.plotly_chart(fig, width="stretch", theme=None, key=key)
    stats_ui.notice([result])
    return [result] if result is not None else []
