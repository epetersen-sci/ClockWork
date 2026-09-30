"""The "Compare within panels" sidebar, and the grid the panels are drawn in.

The layout itself — which flies go in which panel, the level order, the colours
— is ``core/facets.py``, so a command-line run can produce the same figures from
a saved :class:`facets.FacetSpec`. This module only turns sidebar widgets into
a spec and lays the resulting figures out on the page.

Every control carries ``persist_state="session"`` under one app-wide key prefix,
like the display group filter in ``ui/filters.py``: choosing "one panel per
genotype, temperatures within it" on Sleep & activity keeps it chosen on
Periodograms and Rhythmicity. The default is OFF — one figure, one trace per
group — so nothing changes on a page until someone asks for panels.
"""

import re

import streamlit as st

from clockwork.app import export_helpers
from clockwork.app.ui import charts
from clockwork.core import facets, plotting

#: The shared key prefix behind every facet control.
KEY = "facet"

_OFF = "Off (one graph)"
_NO_REF = "None"


def _keep_valid(key, options, *, multi=False):
    """Drop a remembered selection the current dataset cannot offer.

    The selection is session-wide, so it can outlive the dataset it was made on;
    a remembered ``sex`` on a dataset without one must fall back to the default
    rather than error.
    """
    if key not in st.session_state:
        return
    val = st.session_state[key]
    if multi:
        kept = [v for v in (val or []) if v in options]
        if kept != list(val or []):
            st.session_state[key] = kept
    elif val not in options:
        del st.session_state[key]


def _slug(text):
    return "_".join(p for p in re.split(r"[^A-Za-z0-9]+", str(text)) if p) or "all"


def varying_factors(table):
    """Factor columns that actually vary in this dataset (one value splits nothing)."""
    return [c for c in table.columns if c != "group" and table[c].nunique() > 1]


def facet_controls(ds, *, container=None):
    """Render the sidebar block and return ``(spec, table)``.

    ``table`` is :func:`facets.fly_factor_table` for ``ds``; pages pass it straight
    to :func:`facets.resolve_panels`. An inactive spec means "draw as before".
    """
    box = container if container is not None else st.sidebar
    table = facets.fly_factor_table(ds)
    factors = varying_factors(table)
    if not factors:
        return facets.FacetSpec(), table

    from clockwork.core import dam_utilities

    back = {v: k for k, v in dam_utilities.METADATA_COORD_RENAMES.items()}
    grouped_by = [back.get(c, c) for c in dam_utilities.get_group_columns(ds)]

    box.subheader("Compare within panels")
    compare_opts = [_OFF] + factors
    _keep_valid(f"{KEY}_compare", compare_opts)
    compare = box.selectbox(
        "Compare",
        compare_opts,
        key=f"{KEY}_compare",
        persist_state="session",
        help="The factor compared inside each graph — one line (or one violin) per "
        "value. The graphs are then split by the factors below: compare "
        "temperature, one panel per genotype.",
    )
    if compare == _OFF:
        if len([c for c in grouped_by if c in factors]) >= 2:
            box.caption(
                f"Groups combine {', '.join(grouped_by)}. Pick one to compare it "
                "within a panel per value of the others."
            )
        return facets.FacetSpec(), table

    panel_opts = [c for c in factors if c != compare]
    # Keyed by the compared factor, so each choice remembers its own panels and a
    # panel factor can never be the one being compared.
    panel_key = f"{KEY}_panel_by__{compare}"
    _keep_valid(panel_key, panel_opts, multi=True)
    panel_by = box.multiselect(
        "One panel per",
        panel_opts,
        default=[c for c in grouped_by if c in panel_opts],
        key=panel_key,
        persist_state="session",
        help="One graph for each combination of these — genotype and sex gives a "
        "Mutant · F and a Mutant · M graph.",
    ) or []

    reference = None
    if panel_by:
        ref_col = panel_by[0]
        ref_vals = facets.order_levels(table[ref_col])
        ref_opts = [_NO_REF] + ref_vals
        ref_key = f"{KEY}_reference__{ref_col}"
        _keep_valid(ref_key, ref_opts)
        ref_val = box.selectbox(
            f"Grey reference ({ref_col})",
            ref_opts,
            key=ref_key,
            persist_state="session",
            help="Drawn in grey behind every other panel, matched on the remaining "
            "panel factors — typically the control genotype. Violins show it at "
            "every level; line graphs only in a panel with a single level, where "
            "one grey line can still be told apart.",
        )
        if ref_val != _NO_REF:
            reference = {ref_col: ref_val}

    shared_y = box.toggle(
        "Same y-axis on every panel",
        value=True,
        key=f"{KEY}_shared_y",
        persist_state="session",
    )
    ncols = box.segmented_control(
        "Panels per row",
        [1, 2, 3, 4],
        default=3,
        key=f"{KEY}_ncols",
        persist_state="session",
    ) or 3

    spec = facets.FacetSpec(
        compare_by=compare,
        panel_by=tuple(panel_by),
        reference=reference,
        shared_y=bool(shared_y),
        ncols=int(ncols),
    )
    export_helpers.save_csv_button(
        "Save layout (.json)",
        spec.to_json(),
        ds,
        "figure_layout.json",
        key=f"{KEY}_save_spec",
        help="The panel layout as a file, to reproduce these figures from the "
        "command line.",
    )
    return spec, table


def render_panels(panel_figs, spec, ds, *, key, filename, grid=True):
    """Lay ``[(panel, fig), ...]`` out ``spec.ncols`` to a row, plus a button that
    saves the whole set as one image (``grid=False`` leaves it out, for figures
    that are themselves subplot grids and cannot be copied into one).

    Each panel goes through ``charts.plotly_chart``, so the page's "Save N figures"
    button writes every panel too.
    """
    if not panel_figs:
        st.info("No flies to show for this layout.")
        return
    ncols = max(1, min(spec.ncols, len(panel_figs)))
    for start in range(0, len(panel_figs), ncols):
        row = panel_figs[start : start + ncols]
        cols = st.columns(ncols)
        for col, (panel, fig) in zip(cols, row):
            if ncols > 1:
                # A side legend eats half a narrow column; under the plot it doesn't.
                fig.update_layout(
                    legend=dict(orientation="h", yanchor="top", y=-0.22, x=0),
                    margin=dict(l=50, r=10, t=50, b=40),
                )
            with col:
                charts.plotly_chart(
                    fig,
                    width="stretch",
                    theme=None,
                    key=f"{key}_{start}_{panel.title}",
                    filename=f"{filename}_{_slug(panel.title)}",
                )

    if not grid:
        return
    title = panel_figs[0][1].layout.title.text or filename
    base = title.split(" — ")[0]
    export_helpers.save_figures_png_button(
        "Save all panels as one image",
        lambda: [
            (
                f"{filename}_grid.png",
                plotting.combine_panels(
                    panel_figs, ncols=spec.ncols, title=base, shared_y=spec.shared_y
                ),
            )
        ],
        ds,
        key=f"{key}_grid_png",
        help="One PNG with every panel on a grid, in the working folder.",
    )


def arrange_groups(ds, spec, groups):
    """Panel rows for a page that draws one figure per group, or None to keep the
    page's own order. See :func:`facets.arrange_groups`."""
    if not spec.active:
        return None
    return facets.arrange_groups(facets.fly_factor_table(ds), spec, groups)


def group_levels(ds, spec):
    """``{group: its compared level}`` — the label a group takes inside a panel."""
    table = facets.fly_factor_table(ds)
    return table.groupby("group")[spec.compare_by].first().to_dict()


def render_group_figures(groups, draws, keys, spec, rows):
    """Draw per-group figures, arranged by panel when ``rows`` is given.

    ``draws`` is a list of ``group -> figure or None`` callables, one per kind of
    figure the page shows for each group; ``keys`` the matching element-key
    formats (``"prof_{g}"``).

    Without ``rows`` (panels off, or a grouping the layout cannot place) this is
    the page's own layout: each group in turn, its figures side by side. With
    them: a heading per panel, then for each kind a row of that panel's groups in
    level order — a genotype's temperatures next to each other — on one y-axis
    across every panel when the layout asks for it.
    """
    if rows is None:
        for g in groups:
            figs = [d(g) for d in draws]
            cols = st.columns(len(draws)) if len(draws) > 1 else [st.container()]
            for col, fig, k in zip(cols, figs, keys):
                if fig is not None:
                    with col:
                        charts.plotly_chart(fig, width="stretch", key=k.format(g=g))
        return

    figs = [{g: d(g) for _, gs in rows for g in gs} for d in draws]
    if spec.shared_y:
        for by_group in figs:
            plotting.apply_shared_y(list(by_group.values()))
    ncols = max(1, spec.ncols)
    for title, gs in rows:
        if title:
            st.markdown(f"##### {title}")
        for by_group, k in zip(figs, keys):
            drawn = [(g, by_group[g]) for g in gs if by_group[g] is not None]
            for start in range(0, len(drawn), ncols):
                for col, (g, fig) in zip(st.columns(ncols), drawn[start : start + ncols]):
                    with col:
                        charts.plotly_chart(fig, width="stretch", key=k.format(g=g))
