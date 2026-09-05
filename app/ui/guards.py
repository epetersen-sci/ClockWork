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
    from analysis_detection import detect_analyses

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
