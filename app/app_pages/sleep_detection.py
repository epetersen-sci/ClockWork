"""
Sleep analysis — apply the 5-minute immobility rule and classify each bout.

This is a PIPELINE STEP, not a plot: it writes ``sleep``, the per-state sleep
masks and the per-bout table back onto the master dataset, and the Sleep &
activity, Sleep deprivation, HMM and Export pages all read those results.

It used to live at the top of Sleep & activity, wrapped in ``@st.fragment`` with
a session-state message relay, purely so that it would execute before that
page's sidebar group filter rebound ``ds`` — running it from below the filter
would have silently dropped deselected flies from the MASTER dataset. Alone on
its own page there is no filter to race, so the FRAGMENT is gone and the block
is plain top-to-bottom code.

The message relay stays, for its own separate reason: the run ends in
``st.rerun()`` to refresh this page's "already completed" state, and that rerun
discards everything already drawn — so the outcome is stashed in
``_sleep_run_message`` and reported at the top of the next run instead.
"""

import streamlit as st

import dam_utilities
import sleep_analysis
from analysis_detection import detect_analyses
from dataset_meta import PHASE_DD, PHASE_LD, dataset_phase, is_split_applied
from ui.guards import require_dataset

# Outcome of the last successful run, stashed just before the rerun at the
# bottom — that rerun discards whatever was already drawn, so a plain
# st.success() at the point of the run would never be seen.
_last_msg = st.session_state.pop("_sleep_run_message", None)
if _last_msg:
    st.success(_last_msg)

ds = require_dataset()
analyses = detect_analyses(ds)

if "moving" not in ds.data_vars:
    st.warning(
        "Curate dead animals on the **Data → Curate & split** page first — "
        "curation computes the movement data that sleep analysis needs."
    )
    st.stop()
# Phase selection for sleep analysis. Read canonical phase metadata
# from the dataset itself (core/dataset_meta.py) so a DD-only or
# LD-only NetCDF doesn't trigger the "split not applied" prompt.
_ds_phase = dataset_phase(ds)
_has_split_datasets = (
    st.session_state.get("dataset_DD") is not None
    and st.session_state.get("dataset_LD") is not None
)
_has_dd_coord = "first_DD_day" in ds.coords

# Phase API (Stage-2): feed the WHOLE dataset and pass an explicit phase to
# core sleep_analysis (dam_utilities.select_phase derives the epoch per fly).
# NEVER feed a pre-sliced/re-zeroed object — the selector would re-mask it.
# `_sleep_ds` is the dataset handed to sleep_analysis; `_sleep_phase_arg` is
# the phase argument ("LD"/"DD"/"both"); `_sleep_phase` is the display label.
if _ds_phase in (PHASE_LD, PHASE_DD):
    # The loaded file is itself a single-phase partition; analyse it as-is
    # (no further masking — the file already IS the phase).
    _sleep_ds = ds
    _sleep_phase = _ds_phase
    # Pass the phase itself, NOT "both". This looks like it would re-mask an
    # already-sliced dataset — the Stage-1 foot-gun — but select_phase's B4
    # guard catches exactly this case: asked for the phase a dataset is already
    # stamped with, it returns it as-is, with no re-derive and no re-mask. And
    # it is strictly better than "both", because if the file is stamped LD and
    # something asks for DD the guard RAISES instead of silently handing back
    # the wrong epoch. Do not "fix" this back to "both".
    _sleep_phase_arg = _sleep_phase
    st.info(
        f"Using the loaded **{_sleep_phase}** dataset for sleep analysis "
        f"({len(_sleep_ds['id'])} flies, {len(_sleep_ds['time'])} timepoints)."
    )
elif _has_split_datasets:
    sleep_phase = st.radio(
        "Data phase for sleep analysis",
        ["LD (recommended)", "DD"],
        index=0,
        horizontal=True,
        help="Sleep analysis is typically performed on LD data where "
        "the light-dark cycle drives consolidated sleep/wake patterns. "
        "DD is an explicit request for constant-darkness sleep.",
        key="sleep_phase_radio",
    )
    _sleep_phase = "LD" if "LD" in sleep_phase else "DD"
    # Feed the whole (unsplit) master + explicit phase — not dataset_LD/DD.
    _sleep_ds = ds
    _sleep_phase_arg = _sleep_phase
    st.info(
        f"Computing **{_sleep_phase}** sleep from the full dataset "
        f"({len(ds['id'])} flies, {len(ds['time'])} timepoints)."
    )
elif _has_dd_coord and not is_split_applied(ds):
    st.warning(
        "LD/DD split has not been applied yet — it lives on the "
        "**Data → Curate & split** page. Sleep analysis will run on the **full unsplit dataset** (LD+DD). "
        "For best results, apply the split first so sleep analysis can use LD "
        "data only."
    )
    _sleep_ds = ds
    _sleep_phase = "unsplit"
    _sleep_phase_arg = "both"
else:
    _sleep_ds = ds
    _sleep_phase = "full"
    _sleep_phase_arg = "both"

# Check if sleep analysis was already done (on the target dataset)
if "sleep" in _sleep_ds.data_vars:
    _sleep_sec = _sleep_ds.attrs.get("sleep_threshold_seconds")
    _short_max = _sleep_ds.attrs.get("sleep_short_max_min")
    _inter_max = _sleep_ds.attrs.get("sleep_inter_max_min")
    if _sleep_sec is not None:
        _state_info = ""
        if _short_max is not None and _inter_max is not None:
            _state_info = (
                f", state thresholds: short <{int(_short_max)} min, "
                f"intermediate <{int(_inter_max)} min"
            )
        st.info(
            f"Sleep analysis already completed on this dataset — "
            f"**{int(_sleep_sec)}s** threshold ({int(_sleep_sec) // 60} min)"
            f"{_state_info}"
        )
    else:
        st.info("Sleep analysis already completed (parameters not recorded).")
    rerun_sleep = st.checkbox("Re-run sleep analysis with different parameters", value=False)
    if not rerun_sleep:
        st.stop()
elif analyses["sleep"]:
    # Sleep exists on master but not on the phase dataset
    st.info(
        "Sleep analysis was previously run but not on this phase dataset. "
        "Run it below to compute sleep for the selected phase."
    )

sleep_threshold = st.number_input(
    "Sleep threshold (seconds) - minimum immobility duration to classify as sleep",
    min_value=60,
    max_value=1800,
    value=300,
    step=60,
)

with st.expander("Sleep State Thresholds (Abhilash et al. 2026)"):
    short_max_min = st.number_input(
        "Short sleep upper bound (minutes)",
        min_value=5,
        max_value=120,
        value=30,
        step=5,
        help="Bouts 5–N min = short sleep (paper default: 30 min, DAM system).",
        key="short_max_min_input",
    )
    inter_max_min = st.number_input(
        "Intermediate sleep upper bound (minutes)",
        min_value=short_max_min + 1,
        max_value=360,
        value=max(60, short_max_min + 1),
        step=5,
        help="Bouts N–M min = intermediate sleep; >M min = long sleep (paper default: 60 min).",
        key="inter_max_min_input",
    )
    st.caption(
        "These thresholds are provisional and DAM-system specific. "
        "Adjust based on your experimental context and the paper's supplemental methods."
    )

if st.button("Run Sleep Analysis", key="run_sleep"):
    _run_msg = None
    with st.spinner(f"Running sleep analysis on {_sleep_phase} data..."):
        try:
            _sleep_ds = sleep_analysis.sleep_analysis(
                _sleep_ds,
                sleep_threshold_sec=sleep_threshold,
                short_max_min=short_max_min,
                inter_max_min=inter_max_min,
                phase=_sleep_phase_arg,
            )
            # _sleep_ds is now the WHOLE dataset with phase-masked sleep
            # (out-of-phase minutes are -1). The master always carries it.
            st.session_state.dataset = _sleep_ds
            # TRANSITIONAL (retires with the dataset_LD/DD sweep): unmigrated
            # downstream pages still read the pre-sliced dataset_LD/DD, so
            # regenerate the one for the phase just computed from the new sleep
            # result. Same slice params as the Curate & split step, so
            # activity/moving are identical and now carry the correct per-phase
            # sleep.
            if _has_split_datasets and _sleep_phase in ("LD", "DD"):
                _gap = int(_sleep_ds.attrs.get("gap_threshold_minutes", 60))
                if _sleep_phase == "LD":
                    _sliced = dam_utilities.split_xarray_dataset(
                        _sleep_ds, phase="LD", gap_threshold_minutes=_gap
                    )
                else:
                    _disc = bool(_sleep_ds.attrs.get("split_discard_first_dd_day", 0))
                    _sliced = dam_utilities.split_xarray_dataset(
                        _sleep_ds,
                        phase="DD",
                        discard_first_dd_day=_disc,
                        gap_threshold_minutes=_gap,
                    )
                # §2b: slicing upcasts the int8 sleep masks to float (NaN trim
                # padding). Restore int8 (padding/missing → -1) so the sliced
                # object keeps the efficient dtype the masks had on the master.
                for _sv in ("sleep", "sleep_short", "sleep_intermediate", "sleep_long"):
                    if _sv in _sliced.data_vars:
                        _sliced[_sv] = _sliced[_sv].fillna(-1).astype("int8")
                if _sleep_phase == "LD":
                    st.session_state.dataset_LD = _sliced
                else:
                    st.session_state.dataset_DD = _sliced
            st.session_state.analyses = detect_analyses(_sleep_ds)
            ds = _sleep_ds

            if "duration" in ds.data_vars:
                bout_df = (
                    ds["duration"].to_dataframe().reset_index().dropna(subset=["duration"])
                )
                n_bouts = len(bout_df)
                mean_dur = bout_df["duration"].mean()

                # Build per-state counts if sleep_state variable is present
                _state_msg = ""
                if "sleep_state" in ds.data_vars:
                    try:
                        state_counts = (
                            ds["sleep_state"]
                            .to_dataframe()
                            .reset_index()
                            .dropna()["sleep_state"]
                            .value_counts()
                        )
                        _state_parts = [f"{k}: {v}" for k, v in state_counts.items()]
                        _state_msg = f" | States — {', '.join(_state_parts)}"
                    except Exception:
                        pass

                _run_msg = (
                    f"Sleep analysis complete. "
                    f"Detected {n_bouts} sleep bouts (mean duration: {mean_dur:.1f} min)."
                    f"{_state_msg}"
                )
            else:
                _run_msg = "Sleep analysis complete."
        except Exception as e:
            st.error(f"Error during sleep analysis: {e}")

    # Outside the try on purpose: st.rerun raises RerunException, an Exception
    # subclass the handler above would swallow and report as a failure. The
    # rerun refreshes this page's own "already completed" state, and the message
    # rides across in session_state because the rerun discards what was drawn.
    if _run_msg:
        st.session_state["_sleep_run_message"] = _run_msg
        st.rerun()
