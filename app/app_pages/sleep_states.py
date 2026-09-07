"""
Sleep states — the Abhilash et al. 2026 figures, on this dataset.

Five tabs, one per figure of "Recognition of distinct sleep states in
Drosophila uncovers previously obscured homeostatic and circadian control of
sleep" (Current Biology 36, 968-978): normalised waveforms (Fig 1B), bout
initiation probability (Fig 2), rose plots and circadian gating (Fig 3),
scalograms and period-vs-amplitude (Fig 5), and ultradian rhythmicity
(Fig 6).

Read-only with respect to the dataset. Everything here derives from the
short/intermediate/long masks the **Sleep analysis** page writes, so run that
first — with the phase you want to look at here, since sleep detection is
per-epoch.

Figure 4 (homeostatic rebound) is deliberately absent: it needs a deprivation
experiment with matched undisturbed controls, which is a different experimental
design and already has its own **Sleep deprivation** page.

Provenance for every method used here — including where our defaults were
wrong and how they were checked — is in ``core/sleep_state_metrics.py`` and
``periodograms.sleep_cwt_analysis``.
"""

import numpy as np
import streamlit as st

import plotting
import sleep_state_metrics as ssm
from dam_utilities import select_phase
from dataset_meta import dataset_fingerprint, dataset_phase
from periodograms import sleep_cwt_analysis, ultradian_rhythmicity_chi_sq
from ui.filters import group_filter_sidebar
from ui.guards import require_dataset

ds = require_dataset()

# No st.title here — the router sets it from the st.Page title, as on every
# other page.
st.caption(
    "Abhilash, Evans & Shafer 2026, *Current Biology* 36:968-978 — "
    "short (5-30 min), intermediate (30-60 min) and long (>60 min) sleep."
)

# ---------------------------------------------------------------------------
# Cached helpers. First arg is the dataset fingerprint and is named `fp` with
# NO leading underscore: Streamlit's underscore rule is syntactic, so `_fp`
# would drop the fingerprint from the cache key and the caches would stop
# tracking which flies are in the dataset. `_ds` keeps its underscore because
# a Dataset is what the fingerprint exists to avoid hashing.
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _cached_profiles(fp, _ds, bin_size_min):
    return ssm.state_profiles(_ds, bin_size_min=bin_size_min)


@st.cache_data(show_spinner=False)
def _cached_waveforms(fp, _ds, bin_size_min):
    return ssm.compute_normalized_waveforms(_ds, bin_size_min=bin_size_min)


@st.cache_data(show_spinner=False)
def _cached_initiation(fp, _ds):
    return ssm.compute_initiation_probability(_ds)


@st.cache_data(show_spinner=False)
def _cached_circular(fp, _ds, bin_size_min, angle_doubling):
    return ssm.circular_state_stats(
        _ds, bin_size_min=bin_size_min, angle_doubling=angle_doubling
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

stamped = dataset_phase(ds)
phase_options = ["DD", "LD"]
default_phase = "DD" if stamped in (None, "full", "DD") else stamped
phase = st.sidebar.radio(
    "Phase",
    phase_options,
    index=phase_options.index(default_phase) if default_phase in phase_options else 0,
    help=(
        "The paper's circadian figures are under constant darkness, so DD is the "
        "default. A dataset already stamped with a single epoch can only be shown "
        "in that epoch."
    ),
)

bin_size_min = st.sidebar.select_slider(
    "Profile bin (minutes)",
    options=[10, 15, 20, 30, 60],
    value=30,
    help="30 minutes is the paper's choice for the rose plots and profiles.",
)

angle_doubling = st.sidebar.selectbox(
    "Angle doubling",
    ["auto", "never", "always"],
    help=(
        "The paper applies an angle-doubling transform to clearly bimodal "
        "profiles before computing circular statistics, judged by eye. 'auto' "
        "decides per fly and state from the shape of the profile."
    ),
)

try:
    phase_ds, phase_used = select_phase(ds, phase=phase)
except (ValueError, KeyError) as exc:
    st.error(f"Cannot show the {phase} epoch of this dataset: {exc}")
    st.stop()

group_values, all_groups, selected_groups, phase_ds = group_filter_sidebar(
    phase_ds, key="sleep_states_groups", subset=True
)

states_present = ssm.available_states(phase_ds)
if not states_present:
    st.warning(
        "This dataset has no sleep-state masks. Run **Sleep & activity → Sleep "
        f"analysis** on the {phase_used} epoch first — the states here come from "
        "the `sleep_short` / `sleep_intermediate` / `sleep_long` variables it writes."
    )
    st.stop()

fp = dataset_fingerprint(phase_ds)
st.caption(
    f"{phase_ds.sizes['id']} flies · {phase_used} epoch · "
    f"states: {', '.join(states_present)}"
)

tab_wave, tab_init, tab_rose, tab_scal, tab_ultra = st.tabs(
    [
        "Waveforms (Fig 1)",
        "Initiation (Fig 2)",
        "Rose & gating (Fig 3)",
        "Scalograms (Fig 5)",
        "Ultradian (Fig 6)",
    ]
)


def _group_frames(profiles):
    """Group-level stats plus the list of groups present, in a stable order."""
    stats = ssm.group_profiles(profiles)
    groups = list(dict.fromkeys(profiles["group"].tolist()))
    return stats, groups


# ---------------------------------------------------------------------------
# Fig 1 — normalised waveforms
# ---------------------------------------------------------------------------
with tab_wave:
    st.subheader("Normalised sleep-state waveforms")
    st.markdown(
        "Each state's mean profile is divided by its own peak, so shapes can be "
        "compared between states that differ several-fold in absolute amount. "
        "Flies are averaged **before** normalising, as the paper specifies."
    )
    waveforms = _cached_waveforms(fp, phase_ds, bin_size_min)
    if waveforms.empty:
        st.info("No waveforms could be computed for the current selection.")
    else:
        st.plotly_chart(
            plotting.normalized_waveform_overlay(
                waveforms, title=f"Normalised waveforms — {phase_used}"
            ),
            width="stretch",
        )
        st.info(
            "The paper's error band is between-RUN SEM across three independent "
            "experiments. This dataset is one run, so the band here is "
            "between-fly SEM within each group — a narrower claim.",
            icon=":material/info:",
        )
        st.download_button(
            "Download normalised waveforms (CSV)",
            waveforms.to_csv(index=False).encode(),
            file_name=f"sleep_state_waveforms_{phase_used}.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------------------
# Fig 2 — initiation probability
# ---------------------------------------------------------------------------
with tab_init:
    st.subheader("Probability of initiating a sleep bout")
    st.markdown(
        "For each fly and state, the bouts starting in each 1-hour window "
        "divided by that fly's total bouts of that state. The activity profile "
        "is overlaid in red — the paper's point is that long sleep is initiated "
        "in the hour *following* the day's largest bout of wakefulness."
    )
    if "start_time" not in phase_ds.data_vars:
        st.warning(
            "No per-bout table in this dataset, so initiation probability cannot "
            "be computed. Re-run **Sleep analysis**, which writes `start_time` "
            "and `sleep_state` alongside the masks."
        )
    else:
        init = _cached_initiation(fp, phase_ds)
        profiles = _cached_profiles(fp, phase_ds, bin_size_min)
        if init.empty:
            st.info("No sleep bouts were detected for the current selection.")
        else:
            init_stats = ssm.group_initiation_probability(init)
            prof_stats, groups = _group_frames(profiles)
            for group in groups:
                if len(groups) > 1:
                    st.markdown(f"**{group}**")
                st.plotly_chart(
                    plotting.initiation_probability_plot(
                        init_stats[init_stats["group"] == group],
                        activity_stats=prof_stats[
                            (prof_stats["group"] == group)
                            & (prof_stats["state"] == "activity")
                        ],
                        phase_label=phase_used,
                        title=None,
                    ),
                    width="stretch",
                    key=f"init_{group}",
                )
            st.download_button(
                "Download per-fly initiation probabilities (CSV)",
                init.to_csv(index=False).encode(),
                file_name=f"sleep_state_initiation_{phase_used}.csv",
                mime="text/csv",
            )

# ---------------------------------------------------------------------------
# Fig 3 — rose plots and circadian gating
# ---------------------------------------------------------------------------
with tab_rose:
    st.subheader("Temporal organisation of sleep states")
    st.markdown(
        "Rose plots are the daily profiles in polar coordinates: each wedge is "
        f"one {bin_size_min}-minute bin, averaged over days and flies, with "
        "activity overlaid. Each series is scaled to its own peak, because sleep "
        "(min/h) and activity (counts/h) share no unit."
    )
    profiles = _cached_profiles(fp, phase_ds, bin_size_min)
    prof_stats, groups = _group_frames(profiles)
    for group in groups:
        st.markdown(f"**{group}**")
        st.plotly_chart(
            plotting.rose_plot_with_activity(
                prof_stats[prof_stats["group"] == group],
                group=group,
                bin_size_min=bin_size_min,
                phase_label=phase_used,
            ),
            width="stretch",
            key=f"rose_{group}",
        )

    st.divider()
    st.subheader("Circadian gating")
    st.markdown(
        "Each fly's centre of mass gives a mean phase; the angular deviation "
        "about it stands in for gate width, the paper's proxy in the absence of "
        "an objective phase marker for a sleep state's onset. Inner rings are "
        "individual flies, the thick outer arcs are group means. Gates may "
        "overlap even though the states themselves cannot co-occur."
    )
    circular = _cached_circular(fp, phase_ds, bin_size_min, angle_doubling)
    if circular.empty:
        st.info("No circular statistics could be computed.")
    else:
        gates = ssm.group_gates(circular)
        for group in groups:
            g_stats = circular[circular["group"] == group]
            if g_stats.empty:
                continue
            if len(groups) > 1:
                st.markdown(f"**{group}**")
            st.plotly_chart(
                plotting.polar_gating_plot(
                    g_stats,
                    gates[gates["group"] == group],
                    phase_label=phase_used,
                ),
                width="stretch",
                key=f"gate_{group}",
            )
        n_doubled = int(circular["doubled"].sum())
        if n_doubled:
            st.caption(
                f"Angle doubling applied to {n_doubled} of {len(circular)} "
                "(fly, state) profiles judged bimodal."
            )
        st.dataframe(gates, width="stretch", hide_index=True)
        st.download_button(
            "Download per-fly circular statistics (CSV)",
            circular.to_csv(index=False).encode(),
            file_name=f"sleep_state_circular_{phase_used}.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------------------
# Fig 5 / 6 — the CWT. Expensive, so it runs on request and is kept in session
# state; the results feed both tabs.
# ---------------------------------------------------------------------------

CWT_KEY = "sleep_states_cwt"
CWT_META = "sleep_states_cwt_meta"


def _cwt_is_current(meta):
    return meta is not None and meta.get("fp") == fp


with tab_scal:
    st.subheader("Scalograms and period-vs-amplitude")
    st.markdown(
        "Continuous wavelet transforms of the 5-minute-binned state series, one "
        "surface per fly, each normalised to its own surface mean before the "
        "flies are averaged. The colour range is pinned to 0-1.5, as in the "
        "paper, so panels are comparable to each other and to the printed figure."
    )
    col_a, col_b = st.columns([1, 2])
    with col_a:
        p_min = st.number_input("Min period (h)", 0.5, 12.0, 1.0, 0.5)
        p_max = st.number_input("Max period (h)", 12.0, 48.0, 32.0, 1.0)
    with col_b:
        st.markdown(
            "The default 1-32 h span is the paper's Figure 5A axis. Running two "
            "narrow bands separately — as this code used to by default — "
            "normalises each by its own band mean, which makes the z values "
            "incomparable between bands and to the paper."
        )
        run_cwt = st.button("Run wavelet analysis", type="primary")

    if run_cwt:
        progress = st.progress(0.0, text="Running CWT…")

        def _tick(done, total):
            progress.progress(min(1.0, done / max(1, total)), text=f"CWT {done}/{total}")

        with st.spinner("Computing wavelet transforms…"):
            result = sleep_cwt_analysis(
                phase_ds,
                states=tuple(states_present),
                full_range=(float(p_min), float(p_max)),
                # phase_ds is already this epoch's masked view; re-requesting
                # the SAME phase is idempotent, and it must not be "both",
                # which the period-analysis phase guard rejects outright.
                phase=phase_used,
                progress_callback=_tick,
            )
        progress.empty()
        st.session_state[CWT_KEY] = result
        st.session_state[CWT_META] = {"fp": fp, "range": (p_min, p_max), "phase": phase_used}

    cwt = st.session_state.get(CWT_KEY)
    meta = st.session_state.get(CWT_META)
    if cwt is None or not len(getattr(cwt, "data_vars", {})):
        st.info("Press **Run wavelet analysis** to compute the scalograms.")
    else:
        if not _cwt_is_current(meta):
            st.warning(
                "These wavelet results were computed for a different dataset or "
                "selection. Re-run to refresh them.",
                icon=":material/warning:",
            )
        surfaces, axes, spectra, bands = {}, {}, {}, {}
        for state in states_present:
            key = f"sleep_cwt_{state}_full_avg_surface"
            if key not in cwt.data_vars:
                continue
            surfaces[state] = cwt[key].values
            axes[state] = cwt[f"sleep_cwt_{state}_full_period_axis"].values
            spectra[state] = cwt[f"sleep_cwt_{state}_full_fly_power"].values
            bands[state] = (2, 6) if state == "long" else (1, 4)

        if not surfaces:
            st.info("The wavelet run produced no surfaces for these states.")
        else:
            st.plotly_chart(
                plotting.sleep_state_scalogram(
                    surfaces, axes, bin_size_min=5, phase_label=phase_used
                ),
                width="stretch",
            )
            st.plotly_chart(
                plotting.period_amplitude_plot(spectra, axes, ultradian_band=bands),
                width="stretch",
            )

with tab_ultra:
    st.subheader("Ultradian rhythm amplitude and its circadian gating")
    st.markdown(
        "Mean normalised amplitude inside each state's ultradian band (1-4 h for "
        "short and intermediate sleep, 2-6 h for long sleep) over time, then a "
        "periodogram of that amplitude series. A ~24 h peak means the strength "
        "of the ultradian rhythm itself waxes and wanes with the circadian day."
    )
    cwt = st.session_state.get(CWT_KEY)
    if cwt is None or not len(getattr(cwt, "data_vars", {})):
        st.info("Run the wavelet analysis on the **Scalograms** tab first.")
    else:
        amp = {
            state: cwt[f"sleep_cwt_{state}_ultradian_amplitude"].values
            for state in states_present
            if f"sleep_cwt_{state}_ultradian_amplitude" in cwt.data_vars
        }
        if not amp:
            st.info("No ultradian amplitude series available.")
        else:
            st.plotly_chart(
                plotting.ultradian_amplitude_plot(amp),
                width="stretch",
            )
            st.divider()
            test = st.radio(
                "Rhythmicity test",
                ["Chi-squared periodogram (as published)", "Lomb-Scargle"],
                horizontal=True,
                help=(
                    "The paper uses a chi-squared periodogram, so that is the one "
                    "whose numbers can be checked against its Figure 6 and Table "
                    "S2. Lomb-Scargle is the better test on its merits — "
                    "calibrated false-alarm probabilities, tolerant of gaps — but "
                    "its values are not comparable to the published ones."
                ),
            )
            if test.startswith("Chi-squared"):
                chi = ultradian_rhythmicity_chi_sq(cwt, states=tuple(states_present))
                if not len(chi.data_vars):
                    st.info("No periodogram could be computed.")
                else:
                    st.plotly_chart(
                        plotting.chi_sq_periodogram_plot(chi, states=tuple(states_present)),
                        width="stretch",
                    )
                    rows = []
                    for state in states_present:
                        var = f"ultra_chisq_rhythmic_{state}"
                        if var not in chi.data_vars:
                            continue
                        rhythmic = chi[var].values
                        periods = chi[f"ultra_chisq_peak_period_{state}"].values
                        power = chi[f"ultra_chisq_peak_adjusted_{state}"].values
                        rows.append(
                            {
                                "state": state,
                                "% rhythmic": round(100.0 * float(np.mean(rhythmic)), 2),
                                "median period (h)": round(float(np.nanmedian(periods)), 2),
                                "median adj. power": round(float(np.nanmedian(power)), 1),
                            }
                        )
                    if rows:
                        st.markdown(
                            "Comparable to the paper's Table S2, which reports "
                            "93-100% rhythmic at 23.4-23.8 h for wild-type flies."
                        )
                        st.dataframe(rows, width="stretch", hide_index=True)
            else:
                from periodograms import ultradian_rhythmicity_ls

                ls = ultradian_rhythmicity_ls(cwt, states=tuple(states_present))
                if not len(ls.data_vars):
                    st.info("No Lomb-Scargle result could be computed.")
                else:
                    rows = []
                    for state in states_present:
                        var = f"ultra_ls_period_{state}"
                        if var not in ls.data_vars:
                            continue
                        rows.append(
                            {
                                "state": state,
                                "median period (h)": round(
                                    float(np.nanmedian(ls[var].values)), 2
                                ),
                                "% FAP < 0.05": round(
                                    100.0
                                    * float(
                                        np.nanmean(ls[f"ultra_ls_fap_{state}"].values < 0.05)
                                    ),
                                    2,
                                ),
                            }
                        )
                    st.dataframe(rows, width="stretch", hide_index=True)
