"""
HMM Sleep State Analysis Page - Configure and run Hidden Markov Model analysis.
"""

import copy
import os
import sys

import matplotlib
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add core/ (analysis modules) and app/ to the import path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORE_DIR = os.path.join(PROJECT_ROOT, "core")
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [CORE_DIR, APP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import dam_utilities
import export_helpers as ex
from analysis_detection import detect_analyses
from hmm_models import (
    STATE_NAMES_4,
    HMMConfig,
    get_advanced_hmm_summary,
    load_hmm_config_from_attrs,
    plot_group_state_timecourse,
    plot_hypnogram_heatmap,
    plot_state_occupancy_by_group,
    plot_zt_state_fractions,
    run_genotype_workflow,
)

st.header("HMM Sleep State Analysis")

if st.session_state.get("dataset") is None:
    st.warning("No dataset loaded. Go to **Data Loading** first.")
    st.stop()

master = st.session_state.dataset

# Active DISPLAY dataset. "Both (separate)" stores an independent HMM result per phase
# (st.session_state['hmm_by_phase'] = {'LD': {...}, 'DD': {...}}); a "View phase" radio
# in the results section (key 'hmm_view_phase') picks which one the plots/summary show.
# For every other phase choice the master carries the single (phase-selected) result.
_by_phase = st.session_state.get("hmm_by_phase")
if _by_phase:
    _view = st.session_state.get("hmm_view_phase") or sorted(_by_phase.keys())[0]
    ds = _by_phase.get(_view, {}).get("ds", master)
else:
    ds = master


def _merge_hmm(master_ds, res_ds):
    """Merge ONLY the HMM result vars (+ hmm_ attrs) from ``res_ds`` onto the clean
    master. ``res_ds`` is a select_phase() MASKED VIEW — its activity/moving/sleep are
    NaN-masked / float-upcast, so it must NOT replace the master; we take only
    hmm_state/hmm_sleep/hmm_confidence (full-axis, decoded in-phase, −1 out-of-phase)."""
    hmm_vars = [v for v in ("hmm_state", "hmm_sleep", "hmm_confidence") if v in res_ds.data_vars]
    out = master_ds.drop_vars([v for v in hmm_vars if v in master_ds.data_vars], errors="ignore")
    if hmm_vars:
        out = out.merge(res_ds[hmm_vars])
    for k, v in res_ds.attrs.items():
        if str(k).startswith("hmm_"):
            out.attrs[k] = v
    return out


analyses = detect_analyses(ds)

# ============================================================
# Configuration
# ============================================================
st.subheader("Model Configuration")

# --- Phase selection: which lighting paradigm to FIT the HMM on. ---
# Previously the page silently ran on dataset_LD if present, else the full LD+DD
# master — so with no split it mixed paradigms. Now the phase is explicit and the
# source is a select_phase() view of the master (out-of-phase minutes → NaN → −1),
# exactly like the Period Analysis page. LD vs DD transition structure may genuinely
# differ, so "Both (separate)" fits each independently for comparison.
_has_transition = ("first_DD_day" in master.coords) or ("split_minute" in master.coords)
if _has_transition:
    hmm_phase_choice = st.radio(
        "Data phase for HMM",
        ["LD", "DD", "Both (together)", "Both (separate)"],
        index=0,
        horizontal=True,
        key="hmm_phase_choice",
        help="LD (entrained; the standard sleep reference) or DD (constant darkness) "
        "fit ONLY that paradigm. Both (together) fits one model on the full "
        "LD+DD recording. Both (separate) fits LD and DD INDEPENDENTLY so you can "
        "compare transition structure between paradigms (switch which to view in "
        "the results below).",
    )
else:
    hmm_phase_choice = "Full"
    st.caption("No LD/DD transition in this dataset — fitting on the full recording.")

preset = st.radio(
    "Preset configuration",
    ["Improved (Recommended)", "Wiggin et al. 2020", "Harbison 2026"],
    horizontal=True,
    key="hmm_preset",
)

# Plain-language "which model, and why". The three presets are not interchangeable —
# each is built to interrogate a DIFFERENT axis of sleep/wake biology (which is also
# why they disagree on state occupancies). Surface that at the point of choice.
with st.expander("Which model should I pick? (what each one answers)", expanded=True):
    st.markdown(
        "Each preset is built to interrogate a **different axis of sleep/wake biology** — "
        "pick by what your experiment is about:\n\n"
        "- **Improved (Zero-Inflated Poisson) — *activity level / arousal*.** Models the raw "
        "activity **counts** directly, so it resolves not just whether the fly is awake but "
        "*how intensely* — separating quiet wakefulness from vigorous locomotion. Reach for it "
        "when the readout is the **amount** of movement (arousal, hyperactivity, drug response), "
        "not just its presence. Genuinely high-activity minutes are rare, so the top 'full wake' "
        "state is normally a small slice of time.\n\n"
        "- **Wiggin et al. 2020 — *sleep structure / depth*.** Reduces activity to "
        "moving / not-moving and defines sleep **depth** by how hard the fly is to rouse "
        "(transition stickiness), stepping through a graded deep→light→wake ladder. Reach for it "
        "when the question is sleep **structure** — depth, consolidation, fragmentation — "
        "independent of how active the fly is when awake. It ignores movement magnitude by "
        "design, so it cannot tell quiet from vigorous wake.\n\n"
        "- **Harbison 2026 — *differences between individuals*.** Fits a separate model to each "
        "fly on each day and normalizes to that day's own activity, putting flies on a common "
        "footing regardless of overall activity level. Reach for it when the signal is the "
        "**variation between individuals** (or between days) — e.g. mapping genetic differences "
        "across many lines, or day-to-day plasticity. Because it answers a different question, "
        "its sleep totals diverge from the standard definition — it is **not** a drop-in sleep "
        "scorer."
    )

# When the preset changes, sync the advanced option widgets to the preset's values.
# This runs before the widgets are rendered, so the widgets will show the correct defaults.
if st.session_state.get("_last_hmm_preset") != preset:
    if preset == "Wiggin et al. 2020":
        _sync_cfg = HMMConfig.wiggin()
    elif preset == "Harbison 2026":
        _sync_cfg = HMMConfig.harbison()
    else:
        _sync_cfg = HMMConfig.improved()
    st.session_state["hmm_n_states"] = _sync_cfg.n_states
    st.session_state["hmm_training_scope"] = _sync_cfg.training_scope
    st.session_state["hmm_emission"] = _sync_cfg.emission_model
    st.session_state["hmm_trans"] = _sync_cfg.transition_constraints
    st.session_state["hmm_restarts"] = _sync_cfg.n_restarts
    st.session_state["hmm_decoding"] = _sync_cfg.decoding_method
    st.session_state["_last_hmm_preset"] = preset

with st.expander("Advanced options"):
    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        n_states = st.selectbox("Number of states", [2, 3, 4, 5], key="hmm_n_states")
        training_scope = st.selectbox(
            "Training scope",
            ["per_genotype", "all_pooled", "per_fly", "per_fly_per_day"],
            key="hmm_training_scope",
            help="per_fly_per_day = one model per fly per day (Ghosh & Harbison 2026); "
            "the Harbison preset uses it. Expensive (many fits).",
        )
        emission_model = st.selectbox(
            "Emission model",
            ["zip", "gaussian", "binary"],
            key="hmm_emission",
        )
    with adv_col2:
        transition_constraints = st.selectbox(
            "Transition constraints",
            ["soft", "hard", "none"],
            key="hmm_trans",
        )
        n_restarts = st.slider("Training restarts", 5, 100, step=5, key="hmm_restarts")
        decoding_method = st.selectbox(
            "Decoding method",
            ["both", "viterbi", "posterior"],
            key="hmm_decoding",
        )

if analyses["hmm"]:
    # Show the parameters that were used for the existing run
    _saved_cfg = load_hmm_config_from_attrs(ds)
    if _saved_cfg is not None:
        st.info("HMM analysis already completed with the following parameters:")
        _cfg_col1, _cfg_col2, _cfg_col3 = st.columns(3)
        with _cfg_col1:
            st.markdown(f"**States:** {_saved_cfg.n_states}")
            st.markdown(f"**Emission:** {_saved_cfg.emission_model}")
        with _cfg_col2:
            st.markdown(f"**Training scope:** {_saved_cfg.training_scope}")
            st.markdown(f"**Transitions:** {_saved_cfg.transition_constraints}")
        with _cfg_col3:
            st.markdown(f"**Decoding:** {_saved_cfg.decoding_method}")
            st.markdown(f"**Restarts:** {_saved_cfg.n_restarts}")
    else:
        st.info("HMM analysis already completed (parameters not recorded).")
    rerun_hmm = st.checkbox("Re-run with different parameters", value=False, key="rerun_hmm")
else:
    rerun_hmm = True

if rerun_hmm and st.button("Run HMM Analysis", key="run_hmm"):
    hmm_progress = st.progress(0, text="Fitting HMM models...")
    with st.spinner("Running HMM analysis... This may take several minutes."):
        try:
            # Build config from preset (preserves non-widget params like n_iter, tol),
            # then apply the current widget values (which are synced to the preset unless
            # the user manually overrode them).
            if preset == "Wiggin et al. 2020":
                config = HMMConfig.wiggin()
            elif preset == "Harbison 2026":
                config = HMMConfig.harbison()
            else:
                config = HMMConfig.improved()

            config.n_states = n_states
            config.training_scope = training_scope
            config.emission_model = emission_model
            config.transition_constraints = transition_constraints
            config.n_restarts = n_restarts
            config.decoding_method = decoding_method

            # Progress now advances during the long TRAINING phase (per fitted
            # restart for per_genotype, per fly otherwise), not just the ~1 s
            # decode. Unit differs by scope, so keep the label generic.
            if config.training_scope == "per_genotype":
                _unit = "restarts fitted"
            elif config.training_scope in ("per_fly", "per_fly_per_day"):
                _unit = "flies fitted"
            else:
                _unit = "step"

            def _hmm_cb(completed, total):
                frac = completed / total if total else 0.0
                hmm_progress.progress(
                    frac, text=f"HMM training: {completed}/{total} {_unit} ({frac:.0%})"
                )

            # Which phase(s) to fit. select_phase() masks out-of-phase minutes to
            # NaN (the workflow drops them), so each fit is paradigm-pure.
            _PHASE_MAP = {"LD": "LD", "DD": "DD", "Both (together)": "both", "Full": "auto"}
            _run_phases = (
                ["LD", "DD"] if hmm_phase_choice == "Both (separate)" else [hmm_phase_choice]
            )
            _n_ph = len(_run_phases)
            n_expected = (
                len({str(v) for v in master["group"].values}) if "group" in master.coords else 1
            )

            per_phase = {}
            n_fitted_by_phase = {}
            for _pi, _ph in enumerate(_run_phases):
                _src, _phase_used = dam_utilities.select_phase(master, _PHASE_MAP.get(_ph, "auto"))
                _cfg = copy.deepcopy(config)

                def _cb(completed, total, _pi=_pi, _ph=_ph):
                    frac = (_pi + (completed / total if total else 0.0)) / _n_ph
                    hmm_progress.progress(
                        frac, text=f"HMM training [{_ph}]: {completed}/{total} {_unit}"
                    )

                # verbose=True → convergence info (restarts, LL) in the terminal
                _res_ds, _res = run_genotype_workflow(
                    _src, _cfg, verbose=True, progress_callback=_cb
                )
                _merged = _merge_hmm(master, _res_ds)  # clean master + this phase's hmm_state
                _merged.attrs["hmm_phase"] = _phase_used
                per_phase[_ph] = {"ds": _merged, "results": _res, "config": _cfg}
                n_fitted_by_phase[_ph] = len(_res) if _res else 0

            # Store: Both(separate) keeps both, tagged; else the single result on master.
            _active_ph = _run_phases[0]
            if hmm_phase_choice == "Both (separate)":
                st.session_state["hmm_by_phase"] = per_phase
                st.session_state["hmm_view_phase"] = _active_ph
            else:
                st.session_state.pop("hmm_by_phase", None)
                st.session_state.pop("hmm_view_phase", None)
            _active = per_phase[_active_ph]
            st.session_state.dataset = _active["ds"]
            st.session_state["hmm_results"] = _active["results"]
            st.session_state["hmm_config"] = _active["config"]
            st.session_state.analyses = detect_analyses(_active["ds"])

            _min_fitted = min(n_fitted_by_phase.values())
            _summary = ", ".join(f"{p}: {n}/{n_expected}" for p, n in n_fitted_by_phase.items())
            if _min_fitted == 0:
                st.error(
                    f"No groups fitted for at least one phase ({_summary}). Check the "
                    "terminal; try reducing **Number of states** or raising **restarts**."
                )
            elif _min_fitted < n_expected:
                st.warning(f"Some groups did not fit ({_summary}). Check the terminal.")
            else:
                st.success(f"HMM analysis complete! Groups fitted — {_summary}.")

            st.rerun()
        except Exception as e:
            st.error(f"HMM Error: {e}")
            st.exception(e)
        finally:
            hmm_progress.empty()

# ============================================================
# Results display (shown when HMM has been run)
# ============================================================
# Both (separate): let the user switch which phase's results the plots/summary show.
# Changing the radio re-points the active dataset/results to that phase and reruns.
_by_phase = st.session_state.get("hmm_by_phase")
if _by_phase:
    st.divider()
    _phases = sorted(_by_phase.keys())
    _cur = st.session_state.get("hmm_view_phase") or _phases[0]
    _view = st.radio(
        "Viewing phase (fitted separately)",
        _phases,
        index=_phases.index(_cur) if _cur in _phases else 0,
        horizontal=True,
        key="hmm_view_phase",
        help="LD and DD were fit independently. Pick which phase's model/decoding the "
        "summary and plots below reflect.",
    )
    _act = _by_phase.get(_view)
    if _act is not None:
        st.session_state.dataset = _act["ds"]
        st.session_state["hmm_results"] = _act["results"]
        st.session_state["hmm_config"] = _act["config"]

ds = st.session_state.dataset
analyses = detect_analyses(ds)

if not analyses["hmm"]:
    st.stop()

st.divider()

# Summary table
_phase_label = ds.attrs.get("hmm_phase")
st.subheader("State Occupancy Summary" + (f" — {_phase_label} phase" if _phase_label else ""))

results = st.session_state.get("hmm_results")

# Restore config: prefer session state, then dataset attrs, then infer from data
config = st.session_state.get("hmm_config")
if config is None:
    config = load_hmm_config_from_attrs(ds)
if config is None:
    config = HMMConfig()
    # Legacy fallback: infer n_states from the data itself
    if "hmm_state" in ds.data_vars:
        _max_state = int(ds["hmm_state"].where(ds["hmm_state"] != -1).max().item())
        config.n_states = _max_state + 1

if results is not None:
    try:
        summary_df = get_advanced_hmm_summary(ds, results, config)
        st.dataframe(summary_df, width="stretch", height=300)

        _fn = f"hmm_summary_{_phase_label}.csv" if _phase_label else "hmm_summary.csv"
        ex.save_df_button(
            "Save HMM Summary to working folder", summary_df, ds, _fn, key="dl_hmm_summary"
        )
    except Exception as e:
        st.warning(f"Could not compute advanced summary: {e}")

st.divider()

# ============================================================
# Hypnogram Heatmap
# ============================================================
st.subheader("Hypnogram Heatmap")

if "group" in ds.coords:
    group_options = sorted({str(v) for v in ds["group"].values})
    selected_group = st.selectbox(
        "Filter by group (optional)", ["All groups"] + group_options, key="hmm_heatmap_group"
    )
    group_filter = None if selected_group == "All groups" else selected_group
else:
    group_filter = None

n_states_display = config.n_states if config is not None else 4

if st.button("Generate Hypnogram Heatmap", key="gen_hypnogram"):
    with st.spinner("Generating heatmap..."):
        try:
            fig = plot_hypnogram_heatmap(ds, group_name=group_filter, n_states=n_states_display)
            if fig is not None:
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.warning("No data to display for the selected group.")
        except Exception as e:
            st.error(f"Heatmap error: {e}")

st.divider()

# ============================================================
# State Occupancy by Group
# ============================================================
st.subheader("State Occupancy by Group")

if st.button("Generate State Occupancy Plot", key="gen_occupancy"):
    with st.spinner("Generating state occupancy..."):
        try:
            fig, occ_df = plot_state_occupancy_by_group(
                ds, n_states=n_states_display, return_data=True
            )
            if fig is not None:
                st.pyplot(fig)
                plt.close(fig)
                st.session_state["_hmm_occ_df"] = occ_df
            else:
                st.warning("No state data found.")
        except Exception as e:
            st.error(f"State occupancy error: {e}")

_occ_df = st.session_state.get("_hmm_occ_df")
if _occ_df is not None and not _occ_df.empty:
    _fn = f"hmm_state_occupancy_{_phase_label}.csv" if _phase_label else "hmm_state_occupancy.csv"
    ex.save_df_button(
        "Save State Occupancy data to working folder", _occ_df, ds, _fn, key="dl_hmm_occupancy"
    )

st.divider()

# ============================================================
# Group time-course (overlaid group comparison, mean ± SEM)
# ============================================================
st.subheader("Group Time-Course (overlaid comparison)")
st.caption(
    "Group-averaged **metric-to-compare**: mean % time in one state (or total "
    "sleep = DS+LS) across ZT, with every group **overlaid on shared axes** (± SEM "
    "across flies) so genotype differences read at a glance — unlike the per-fly "
    "hypnogram heatmap or the per-group ZT subplots above."
)

_tc_metric_options = ["Sleep (DS+LS)"] + list(STATE_NAMES_4[:n_states_display])
tc_c1, tc_c2 = st.columns(2)
with tc_c1:
    tc_metric_choice = st.selectbox(
        "Metric",
        _tc_metric_options,
        index=0,
        key="tc_metric",
        help="Total sleep (DS+LS) or a single state's occupancy.",
    )
with tc_c2:
    tc_bin_size = st.selectbox(
        "ZT bin resolution (minutes)", [15, 30, 60], index=1, key="tc_bin_size"
    )

_tc_metric = (
    "sleep"
    if tc_metric_choice == "Sleep (DS+LS)"
    else list(STATE_NAMES_4[:n_states_display]).index(tc_metric_choice)
)

if st.button("Generate Group Time-Course", key="gen_group_timecourse"):
    with st.spinner("Computing group time-course..."):
        try:
            fig, tc_df = plot_group_state_timecourse(
                ds,
                n_states=n_states_display,
                metric=_tc_metric,
                bin_size_minutes=tc_bin_size,
                return_data=True,
            )
            if fig is not None:
                st.pyplot(fig)
                plt.close(fig)
                st.session_state["_hmm_tc_df"] = tc_df
            else:
                st.warning("No HMM state data found for the selected metric.")
        except Exception as e:
            st.error(f"Group time-course error: {e}")

_tc_df = st.session_state.get("_hmm_tc_df")
if _tc_df is not None and not _tc_df.empty:
    _fn = f"hmm_group_timecourse_{_phase_label}.csv" if _phase_label else "hmm_group_timecourse.csv"
    ex.save_df_button(
        "Save Group Time-Course data to working folder", _tc_df, ds, _fn, key="dl_hmm_timecourse"
    )

st.divider()

# ============================================================
# State Fractions by Time of Day (ZT)
# ============================================================
st.subheader("State Fractions by Time of Day")

zt_col1, zt_col2 = st.columns(2)
with zt_col1:
    zt_section_options = ["Full resolution", 4, 6, 8, 12, 24]
    zt_section_choice = st.selectbox(
        "Day sections",
        zt_section_options,
        index=1,
        key="zt_n_sections",
    )
with zt_col2:
    zt_bin_size = st.selectbox(
        "ZT bin resolution (minutes)",
        [15, 30, 60],
        index=1,
        key="zt_bin_size",
    )

n_zt_sections = None if zt_section_choice == "Full resolution" else int(zt_section_choice)

if st.button("Generate ZT State Fractions", key="gen_zt_fractions"):
    with st.spinner("Computing ZT state fractions..."):
        try:
            fig, zt_df = plot_zt_state_fractions(
                ds,
                n_states=n_states_display,
                bin_size_minutes=zt_bin_size,
                n_sections=n_zt_sections,
                return_data=True,
            )
            if fig is not None:
                st.pyplot(fig)
                plt.close(fig)
                st.session_state["_hmm_zt_df"] = zt_df
            else:
                st.warning("No HMM state data found.")
        except Exception as e:
            st.error(f"ZT fractions error: {e}")

_zt_df = st.session_state.get("_hmm_zt_df")
if _zt_df is not None and not _zt_df.empty:
    _fn = f"hmm_zt_fractions_{_phase_label}.csv" if _phase_label else "hmm_zt_fractions.csv"
    ex.save_df_button(
        "Save ZT State Fractions data to working folder", _zt_df, ds, _fn, key="dl_hmm_zt"
    )
