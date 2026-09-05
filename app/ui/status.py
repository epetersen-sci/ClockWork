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


def render_status_grid(analyses):
    """Render the completed/not-run state for :func:`detect_analyses` output.

    Badges rather than ``st.metric``: the old home page did
    ``st.columns(len(analyses))``, and nine metric columns squeezed the value
    hard enough that "Completed" rendered as "Compl...". Badges are sized by
    their text, wrap on their own, and stay legible at any window width.
    """
    from analysis_detection import get_status_label

    done = [k for k, v in analyses.items() if v]
    todo = [k for k, v in analyses.items() if not v]

    if done:
        st.markdown(" ".join(f":green-badge[:material/check: {display_name(k)}]" for k in done))
    if todo:
        st.markdown(" ".join(f":gray-badge[{display_name(k)}]" for k in todo))
    if not analyses:
        st.caption(get_status_label(False))


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
