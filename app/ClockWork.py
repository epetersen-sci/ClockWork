"""
ClockWork — Drosophila Activity Monitor (DAM) analysis pipeline. Entry point.

Run from the repository root with:
    streamlit run app/ClockWork.py

This file is a ROUTER, not a page. It sets up the import path, seeds session
state, declares the navigation, and hands off to the selected page. The landing
content lives in ``app_pages/home.py`` like any other page.
"""

import os
import sys

import streamlit as st

# Add core/ (analysis modules) and app/ (ui package, analysis_detection, ...) to
# the path. This is the ONLY place it happens: st.navigation runs this file on
# every rerun before executing the selected page, so the path is always set by
# the time a page body runs. Pages therefore carry no bootstrap of their own and
# no longer each compute PROJECT_ROOT with a different number of dirname() calls.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_DIR = os.path.join(PROJECT_ROOT, "core")
APP_DIR = os.path.dirname(os.path.abspath(__file__))
for p in [CORE_DIR, APP_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Guard the actual app rendering behind __name__ == '__main__'.
#
# Why: the analysis pipeline uses multiprocessing.Pool, which on Windows
# (and on macOS Python 3.8+) defaults to the *spawn* start method. Each
# spawned worker re-runs this entry script via
# ``multiprocessing.spawn._fixup_main_from_path`` → ``runpy.run_path``,
# which sets ``__name__ == '__mp_main__'``. Under ``streamlit run``,
# runpy sets ``__name__ == '__main__'``. Wrapping all module-level
# Streamlit calls behind that guard means workers don't execute
# ``st.set_page_config`` / ``st.navigation`` / etc. without a
# ``ScriptRunContext`` (which would otherwise emit
# ``missing ScriptRunContext!`` warnings — one per worker spawn).
#
# ``page.run()`` is separately safe — it returns early when there is no
# ScriptRunContext — but ``st.navigation`` itself is not, so the guard stays.
#
# Imports and ``sys.path`` setup intentionally stay *outside* the guard:
# workers that pickle objects referencing streamlit types still need the
# module loaded, and any submodule import that resolves through
# ``CORE_DIR`` / ``APP_DIR`` needs the path entries.
if __name__ == "__main__":
    st.set_page_config(
        page_title="ClockWork",
        page_icon="🪰",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    from ui.state import init_session_state

    init_session_state()

    # Sections are the "folders" in the sidebar. st.navigation supports exactly
    # one level of them — a dict of section label -> pages — so this is as nested
    # as Streamlit goes. Order within each section follows the data-dependency
    # order the work actually takes, which is why HMM model selection precedes
    # HMM analysis (it chooses the parameters that page consumes) and why
    # Groups & subsets precedes Curate & split (subset before you curate, and
    # the curation heatmap draws its y-axis from the group coord).
    page = st.navigation(
        {
            "": [
                st.Page(
                    "app_pages/home.py",
                    title="Home",
                    icon=":material/home:",
                    default=True,
                ),
            ],
            "Data": [
                st.Page(
                    "app_pages/data_import.py",
                    title="Import",
                    icon=":material/upload_file:",
                ),
                st.Page(
                    "app_pages/data_groups.py",
                    title="Groups & subsets",
                    icon=":material/groups:",
                ),
                st.Page(
                    "app_pages/data_curate_split.py",
                    title="Curate & split",
                    icon=":material/content_cut:",
                ),
            ],
            "Circadian analysis": [
                st.Page(
                    "app_pages/period_analysis.py",
                    title="Period analysis",
                    icon=":material/schedule:",
                ),
                st.Page(
                    "app_pages/periodograms.py",
                    title="Periodograms",
                    icon=":material/graphic_eq:",
                ),
                st.Page(
                    "app_pages/rhythmicity.py",
                    title="Rhythmicity",
                    icon=":material/rule:",
                ),
                st.Page(
                    "app_pages/phase_shift.py",
                    title="Phase shift",
                    icon=":material/light_mode:",
                ),
            ],
            "Sleep & activity": [
                st.Page(
                    "app_pages/sleep_detection.py",
                    title="Sleep analysis",
                    icon=":material/bedtime:",
                ),
                st.Page(
                    "app_pages/sleep_activity.py",
                    title="Sleep & activity",
                    icon=":material/stacked_line_chart:",
                ),
                st.Page(
                    "app_pages/sleep_states.py",
                    title="Sleep states",
                    icon=":material/bar_chart:",
                ),
                st.Page(
                    "app_pages/sleep_deprivation.py",
                    title="Sleep deprivation",
                    icon=":material/alarm:",
                ),
                st.Page(
                    "app_pages/hmm_model_selection.py",
                    title="HMM model selection",
                    icon=":material/tune:",
                ),
                st.Page(
                    "app_pages/hmm_analysis.py",
                    title="HMM analysis",
                    icon=":material/psychology:",
                ),
            ],
            "Export": [
                st.Page(
                    "app_pages/export_data.py",
                    title="Save & export",
                    icon=":material/save:",
                ),
                st.Page(
                    "app_pages/export_scamp.py",
                    title="SCAMP export",
                    icon=":material/share:",
                ),
            ],
        },
        # 16 pages. Without this the menu collapses to ten with a "View 6 more"
        # button, which hides a whole section behind a click.
        expanded=True,
    )

    # The router owns the page title, so pages carry no st.header of their own.
    st.title(page.title, icon=page.icon)
    page.run()
