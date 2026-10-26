"""The figure statistics in the app: the sidebar setting, the notice line under a
tested figure, and the ``Stats`` sheet of its workbook.

The tests themselves are :mod:`clockwork.core.group_stats`; the marks on a figure
are :func:`clockwork.core.plotting.add_group_stats`.
"""

import streamlit as st

from clockwork.core import group_stats as gs

#: Session key of the sidebar's "Group statistics" setting.
SETTING_KEY = "group_stats_setting"


def setting_sidebar():
    """The app-wide "Group statistics" control, in the sidebar."""
    st.sidebar.selectbox(
        "Group statistics",
        list(gs.SETTINGS),
        format_func=gs.SETTINGS.get,
        key=SETTING_KEY,
        persist_state="session",
    )


def setting():
    return st.session_state.get(SETTING_KEY, gs.RECOMMENDED)


def family(recommended):
    """The test family a measure recommending ``recommended`` uses now."""
    return gs.resolve_family(recommended, setting())


def notice(results):
    """The one-line notice under a tested figure (nothing when no test ran)."""
    text = gs.combined_notice(results)
    if text:
        st.caption(text)


def sheet(results):
    """``("Stats", table)`` for a workbook's sheet list; the table is empty (and
    the sheet skipped) when no test ran."""
    return ("Stats", gs.stats_table(results))
