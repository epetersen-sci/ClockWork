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


def render_integrity_counters(ds):
    """Show the import-time data-quality counters a dataset carries, if any.

    ``core/dam_integrity.py`` runs a full load-time quality analysis, but its
    output only ever existed during a raw import — a reloaded ``.nc`` could say
    nothing about how clean the recording it came from was. The aggregate
    counters are stamped onto ``attrs`` at import (see
    ``MetadataProcessor.integrity_scalars``) and survive the round-trip, so this
    renders them wherever a dataset is summarized.

    Silent on datasets that predate the counters — their absence means "not
    recorded", which is different from "clean", so it must not claim clean.
    """
    n_bad = ds.attrs.get("integrity_n_status_bad")
    if n_bad is None:
        return False
    n_cos = int(ds.attrs.get("integrity_n_cosmetic_slots", 0) or 0)
    n_dl = int(ds.attrs.get("integrity_n_dataloss_slots", 0) or 0)
    n_mon = int(ds.attrs.get("integrity_n_monitors", 0) or 0)
    n_bad = int(n_bad)

    scope = f" across {n_mon} monitor(s)" if n_mon else ""
    if n_bad == 0 and n_dl == 0:
        st.caption(f"Data integrity at import: clean{scope} — no failed reads, no gaps.")
        return True

    detail = (
        f"**{n_bad}** non-status-1 rows handled (→NaN), "
        f"**{n_cos}** cosmetic slot(s) with 0 data lost, "
        f"**{n_dl}** DATA-LOSS slot(s)"
    )
    if n_dl:
        # Real holes: the user has to know, and the per-monitor detail that says
        # WHERE only exists during the original import.
        st.warning(
            f"Data integrity at import{scope}: {detail}. Per-monitor detail is "
            "only available during a raw import, not from a reloaded `.nc`."
        )
    else:
        st.caption(f"Data integrity at import{scope}: {detail}.")
    return True
