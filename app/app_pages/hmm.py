"""HMM analysis — choose the model, then fit it.

Two tabs, in the order the work takes: cross-validate the number of states and
the emission model, then fit that model and read the states out. They were two
pages, one immediately above the other, for a single decision and its
consequence.

The bodies live in :mod:`hmm_selection_view` and :mod:`hmm_analysis_view` rather
than here, because between them they are the better part of a thousand lines and
a page whose job is to say "these two, in this order" should read like that.
"""

import streamlit as st

from hmm_analysis_view import render as render_analysis
from hmm_selection_view import render as render_selection

_tab_select, _tab_fit = st.tabs(["Model selection", "Analysis"])

with _tab_select:
    render_selection()

with _tab_fit:
    render_analysis()
