"""Mark a control whose value has been moved off its default.

The Period & rhythmicity options run to a few dozen settings, almost all left at
their defaults. A highlight on the few that are not is how someone returning to a
dataset — or about to re-run — sees what is unusual without reading every field.
"""

import streamlit as st


def is_changed(value, default):
    """``value != default``, tolerant of int/float spellings of one number."""
    try:
        return abs(float(value) - float(default)) > 1e-9
    except (TypeError, ValueError):
        return value != default


def flag_if_changed(value, default, fmt=str):
    """Under the widget just drawn, an orange note when ``value`` is not ``default``.

    Returns whether it was changed, so a caller can count them.
    """
    if not is_changed(value, default):
        return False
    st.caption(f":orange[:material/edit: Changed from default ({fmt(default)})]")
    return True
