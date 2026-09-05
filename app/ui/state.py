"""Session-state defaults and cache invalidation.

These lived in the Data Loading page, which meant the *import* page had to carry
a hardcoded registry of every other page's session keys — HMM, sleep deprivation,
rhythmicity, the preprocessing heatmap. Adding a cache to any page meant
remembering to edit a page you were not working on, and one entry
(``param_sweep_results``) already outlived the page that created it.

The registry belongs next to the session contract itself, not inside whichever
page happens to invalidate it first.
"""

import streamlit as st

# Keys the entry point seeds before any page runs, so no page has to defend
# against their absence on a cold start.
SESSION_DEFAULTS = {
    "dataset": None,
    # Unfiltered copy stashed at load time so the group-subset filter can be
    # undone without re-reading from disk.
    "dataset_full": None,
    "dataset_path": None,
    "dataset_DD": None,
    "dataset_LD": None,
    "analyses": {},
    "working_dir": None,
}

# Every key whose contents are derived from the *current* set of flies. All of
# these are keyed off fly id, so any change to that set makes them stale.
DERIVED_CACHE_KEYS = (
    # Phase splits
    "dataset_DD",
    "dataset_LD",
    # Sleep / waveform / rebound
    "wf_df",
    "ip_df",
    "rb_df",
    "bout_timing_df",
    "activity_zt",
    # Period / rhythmicity caches
    "sleep_cwt_ds",
    "ultra_ls_ds",
    "circ_df",
    "rhythmicity_df",
    "rhythmicity_summary",
    "rhythmicity_per_fly_df",
    "rhythmicity_summary_df",
    "_period_sens_df",
    # HMM
    "hmm_results",
    "hmm_config",
    "hmm_by_phase",
    "hmm_view_phase",
    "_last_hmm_preset",
    "_hmm_occ_df",
    "_hmm_tc_df",
    "_hmm_zt_df",
    "cv_fold_df",
    "cv_summary_df",
    # Sleep deprivation
    "sd_results",
    "sd_config",
    # Phase shift
    "phase_shift_results",
    "phase_shift_group_results",
    # Preprocessing UI bits
    "show_preprocessing_heatmap",
    "_heatmap_var_last",
    "curated_dead_data",
)

# Dynamically named keys are swept by prefix, so renaming or removing a group
# does not leave stragglers behind.
DERIVED_CACHE_PREFIXES = ("ultra_ls_ds_",)

# Keys cleared only on a fresh load, never by a group filter.
_DATASET_KEYS = (
    "dataset",
    "dataset_full",
    "dataset_path",
    "analyses",
    "_raw_metadata",
    "_raw_data",
)


def init_session_state():
    """Seed :data:`SESSION_DEFAULTS`. Called once per rerun from the entry point,
    before ``page.run()``, so every page can assume the keys exist."""
    for key, default in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def invalidate_derived_caches():
    """Drop every cache derived from the current set of flies.

    Called both when a fresh dataset is loaded (via :func:`clear_dataset_state`)
    and when a group-subset filter is applied or reset.
    """
    for key in DERIVED_CACHE_KEYS:
        st.session_state.pop(key, None)
    for key in [
        k
        for k in list(st.session_state.keys())
        if isinstance(k, str) and k.startswith(DERIVED_CACHE_PREFIXES)
    ]:
        del st.session_state[key]


def clear_dataset_state():
    """Reset all dataset-related session state to defaults."""
    invalidate_derived_caches()
    for key in _DATASET_KEYS:
        st.session_state.pop(key, None)
    for key, default in SESSION_DEFAULTS.items():
        st.session_state[key] = default
