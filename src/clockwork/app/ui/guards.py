"""Prerequisite guards shared by every page that needs a loaded dataset.

Nine pages opened with a byte-identical "no dataset loaded" warning followed by
``st.stop()``. Nine copies is nine chances for the wording, the target page name,
or the stop-vs-return decision to drift — and it had already drifted: some pages
called ``st.stop()``, others fell through and rendered an empty body.
"""

import streamlit as st


def require_dataset():
    """Stop the page unless a dataset is loaded, and return it.

    Returns the dataset so the caller can write ``ds = require_dataset()``
    instead of a guard followed by a separate session-state read.
    """
    ds = st.session_state.get("dataset")
    if ds is None:
        st.warning("No dataset loaded. Go to **Data → Import** to get started.")
        st.stop()
    return ds


def require_analysis(ds, key, where):
    """Stop the page unless the analysis ``key`` has already been run.

    ``key`` is one of :func:`analysis_detection.detect_analyses`'s keys and
    ``where`` names the page that produces it, so the message tells the user
    where to go rather than only what is missing.
    """
    from clockwork.app.analysis_detection import detect_analyses

    if not detect_analyses(ds).get(key, False):
        st.warning(f"This page needs **{key}** results. Run it on **{where}** first.")
        st.stop()


def require_variable(ds, var, why):
    """Stop the page unless ``var`` is present on the dataset.

    ``why`` explains what produces it, e.g. "Curation computes the movement data
    that sleep analysis needs".
    """
    if var not in ds.data_vars:
        st.warning(why)
        st.stop()


#: Analyses that describe the flies they were run on, in the words the pages use.
_ANALYSIS_NAMES = (
    ("autocorrelation", "autocorrelation"),
    ("lomb_scargle", "Lomb-Scargle"),
    ("cwt", "CWT"),
    ("sleep", "sleep"),
    ("hmm", "HMM"),
    ("sleep_deprivation", "sleep deprivation"),
    ("phase_shift", "phase shift"),
)


def analyses_done(ds):
    """Names of the analyses whose results are on ``ds`` — what curating would drop."""
    from clockwork.app.analysis_detection import detect_analyses

    done = detect_analyses(ds)
    names = [name for key, name in _ANALYSIS_NAMES if done.get(key)]
    if "mesa_period" in ds.data_vars:
        names.insert(3, "MESA")
    return names


def warn_if_uncurated(ds, what):
    """Say, before ``what`` is run, that its results will not survive curation.

    Curation has to drop every analysis run before it (the flies it trims or
    removes are in those results), so running one first is wasted work.
    """
    if ds.attrs.get("curation_time_window_hours") is None:
        st.warning(
            f"Curation has not been run on this dataset. Run it on **Data → Curate & "
            f"split** first: curating afterwards removes {what} results, because they "
            "would include flies that died during the recording."
        )
