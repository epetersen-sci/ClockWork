"""One place that decides how an analysis key is spelled for a human.

There were two maps. ``ClockWork.py`` carried seven entries and
``analysis_detection.format_status_summary`` carried nine, and the home page
laid out ``st.columns(len(analyses))`` over all nine keys — so ``phase_shift``
and ``sleep_states`` rendered as raw snake_case identifiers in two of the
columns. One map, used by both call sites, makes that class of drift impossible.
"""

import streamlit as st

DISPLAY_NAMES = {
    "preprocessing": "Curation",
    "sleep": "Sleep",
    "sleep_states": "Sleep states",
    "cwt": "CWT",
    "lomb_scargle": "Lomb-Scargle",
    "autocorrelation": "Autocorrelation",
    "hmm": "HMM",
    "sleep_deprivation": "Sleep depriv.",
    "phase_shift": "Phase shift",
}


def display_name(key):
    """Human-readable label for an analysis key, falling back to the key itself."""
    return DISPLAY_NAMES.get(key, key)


def render_status_grid(analyses, per_row=5):
    """Render the completed/not-run grid for :func:`detect_analyses` output.

    Wraps at ``per_row`` so nine analyses stay readable instead of being squeezed
    into nine columns across the full page width.
    """
    from analysis_detection import get_status_label

    items = list(analyses.items())
    for start in range(0, len(items), per_row):
        chunk = items[start : start + per_row]
        # Pad the final row so its metrics keep the same width as the rows above
        # instead of stretching to fill the page.
        cols = st.columns(per_row)
        for col, (key, completed) in zip(cols, chunk):
            col.metric(display_name(key), get_status_label(completed))


def refresh(ds):
    """Re-detect completed analyses and store them on the session.

    Pages that write their results into ``ds.attrs`` rather than ``data_vars``
    (sleep deprivation, phase shift) must call this after a run, or the home
    page's status grid stays stale until the user happens to visit a page that
    recomputes it.
    """
    from analysis_detection import detect_analyses

    st.session_state.analyses = detect_analyses(ds)
    return st.session_state.analyses
