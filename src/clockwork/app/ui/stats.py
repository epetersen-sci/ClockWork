"""The figure statistics in the app: the notice line under a tested figure, and
the ``Stats`` sheet of its workbook.

The tests themselves are :mod:`clockwork.core.group_stats`; the marks on a figure
are :func:`clockwork.core.plotting.add_group_stats`.
"""

import streamlit as st

from clockwork.core import group_stats as gs


def family(recommended):
    """The test family a measure recommending ``recommended`` uses."""
    return gs.resolve_family(recommended)


def notice(results):
    """The one-line notice under a tested figure (nothing when no test ran)."""
    text = gs.combined_notice(results)
    if text:
        st.caption(text)


def sheet(results):
    """``("Stats", table)`` for a workbook's sheet list; the table is empty (and
    the sheet skipped) when no test ran."""
    return ("Stats", gs.stats_table(results))
