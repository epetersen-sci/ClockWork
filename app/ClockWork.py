"""
ClockWork — Drosophila Activity Monitor (DAM) analysis pipeline. Main entry point.

Run from the repository root with:
    streamlit run app/ClockWork.py
"""

import os
import sys

import streamlit as st

# Add core/ (analysis modules) and app/ (analysis_detection etc.) to the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_DIR = os.path.join(PROJECT_ROOT, "core")
APP_DIR = os.path.dirname(os.path.abspath(__file__))
for p in [CORE_DIR, APP_DIR]:
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
# ``st.set_page_config`` / ``st.title`` / etc. without a
# ``ScriptRunContext`` (which would otherwise emit
# ``missing ScriptRunContext!`` warnings — one per worker spawn).
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

    # Initialize session state defaults
    if "dataset" not in st.session_state:
        st.session_state.dataset = None
    if "dataset_full" not in st.session_state:
        # Unfiltered copy stashed at load time so the Data Loading page can
        # restore the full set after a group-subset filter is applied.
        st.session_state.dataset_full = None
    if "dataset_path" not in st.session_state:
        st.session_state.dataset_path = None
    if "analyses" not in st.session_state:
        st.session_state.analyses = {}
    if "working_dir" not in st.session_state:
        st.session_state.working_dir = None

    st.title("🪰 ClockWork")
    st.markdown("**Drosophila Activity Monitor (DAM) analysis pipeline**")

    st.markdown("""
### Getting started

Load your data on **Data Loading**, then work down the sidebar. Every module adds
its results to one shared dataset that you can save and reload at any point.

**Modules**

- **Data Loading** — Import raw Trikinetics `MonitorXXX.txt` files with a metadata
  table, or reload a saved `.nc` dataset. You pick which metadata columns define
  your comparison groups here.
- **Preprocessing** — Curate dead flies and split the recording into light–dark (LD)
  and constant-darkness (DD) phases.
- **Period Analysis** — Per-fly free-running circadian period with rhythmicity
  classification: autocorrelation, Lomb–Scargle, CWT, and MESA.
- **Periodograms** — Group-averaged period views — power spectra, autocorrelograms,
  and CWT scalograms — across the methods above.
- **Sleep & Activity** — Run the 5-minute-rule sleep analysis, then daily activity
  and sleep profiles (ZT-binned) with group comparisons and CSV export.
- **HMM Analysis** — Hidden Markov Model sleep/wake state classification (Wiggin,
  Harbison, and improved presets; Gaussian / Poisson / ZIP emissions).
- **HMM Model Selection** — Cross-validate the number of states and emission model
  before running the full HMM.
- **Sleep Deprivation** — Compare baseline sleep to post-deprivation recovery
  (rebound) sleep.
- **Phase Shift** — Measure how far each fly's rhythm shifted after a light pulse
  (needs a `pulse_time` ZT column in your metadata).
- **SCAMP Export** — Export curated, LD/DD-split data into the SCAMP MATLAB toolbox
  format.
- **Export** — Download per-fly CSV summaries and save the full dataset (all
  results) to a NetCDF `.nc` file.
""")

    # Show current dataset status if one is loaded
    if st.session_state.dataset is not None:
        ds = st.session_state.dataset
        st.success(f"Dataset loaded: {len(ds['id'])} flies, {len(ds['time'])} timepoints")

        from analysis_detection import detect_analyses, get_status_label

        analyses = detect_analyses(ds)
        st.session_state.analyses = analyses

        cols = st.columns(3)
        with cols[0]:
            st.metric("Flies", len(ds["id"]))
        with cols[1]:
            st.metric("Timepoints", len(ds["time"]))
        with cols[2]:
            groups = ds["group"].values if "group" in ds.coords else []
            st.metric("Groups", len(set(groups)))

        st.subheader("Analysis Status")
        display_names = {
            "preprocessing": "Preprocessing",
            "sleep": "Sleep",
            "cwt": "CWT",
            "lomb_scargle": "Lomb-Scargle",
            "autocorrelation": "Autocorrelation",
            "hmm": "HMM",
            "sleep_deprivation": "Sleep Depriv.",
        }
        status_cols = st.columns(len(analyses))
        for col, (key, completed) in zip(status_cols, analyses.items()):
            name = display_names.get(key, key)
            status = get_status_label(completed)
            col.metric(name, status)
    else:
        st.info("No dataset loaded. Go to **Data Loading** to get started.")
