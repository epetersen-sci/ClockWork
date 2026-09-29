"""One per-fly value by condition: a violin per group, or per facet panel.

Shared by the Period length and Rhythmicity tabs, which draw the same kind of
figure for different numbers (a period; a rhythm strength).
"""

import streamlit as st

import facets
import plotting
from ui import charts, facet_panels


def render(ctx, per_fly, value_col, *, title, y_title, key, y_from_zero):
    """Draw ``per_fly[value_col]`` (one row per fly, ``fly_id`` and ``group``)."""
    if per_fly[value_col].notna().sum() == 0:
        st.info("No flies with a value for this selection.")
        return
    if ctx.facet_spec.active:
        panels = facets.resolve_panels(ctx.facet_table, ctx.facet_spec)
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
        )
        facet_panels.render_panels(pairs, ctx.facet_spec, ctx.period_ds, key=key, filename=key)
        return
    df = per_fly.copy()
    df["group"] = df["group"].astype(str) if "group" in df else "All flies"
    fig, _ = plotting.group_violins(df, value_col, group_col="group", title=title, y_title=y_title)
    if not y_from_zero:
        fig.update_yaxes(rangemode="normal")
    charts.plotly_chart(fig, width="stretch", theme=None, key=key)
