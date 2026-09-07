"""
Sleep states — the Abhilash et al. 2026 figures, on this dataset.

Five tabs, one per figure of "Recognition of distinct sleep states in
Drosophila uncovers previously obscured homeostatic and circadian control of
sleep" (Current Biology 36, 968-978): normalised waveforms (Fig 1B), daily
profiles and bout initiation probability (Fig 2), rose plots and circadian
gating (Fig 3), scalograms and period-vs-amplitude (Fig 5), and ultradian
rhythmicity (Fig 6).

Read-only with respect to the dataset. Everything here derives from the
short/intermediate/long masks the **Sleep analysis** page writes, so run that
first — with the phase you want to look at here, since sleep detection is
per-epoch.

Figure 4 (homeostatic rebound) is deliberately absent: it needs a deprivation
experiment with matched undisturbed controls, which is a different experimental
design and already has its own **Sleep deprivation** page.

Provenance for every method used here — including where our defaults were
wrong and how they were checked against the authors' own R code — is in
``core/sleep_state_metrics.py`` and ``periodograms.sleep_cwt_analysis``.

Tabs are DYNAMIC (``on_change="rerun"``) and every body is guarded by
``tab.open``. Streamlit renders all tab content by default, so without the
guard one visit to this page computed the waveforms, the initiation
probabilities, six rose rows and six gating rings whether or not anyone looked
at them — nineteen figures on 189 flies for the one the user was actually on.
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

# The paper's ultradian windows: "we considered period bands of 1 to 4 hours
# for short and intermediate sleep and 2 to 6 hours for long sleep... owing to
# our definition of long sleep being 60 minutes or longer, 1-h ultradian
# rhythms are impossible."
ULTRADIAN_BANDS = {
    "standard": (1, 4),
    "short": (1, 4),
    "intermediate": (1, 4),
    "long": (2, 6),
}
CWT_BIN_MIN = 5  # sleep_cwt_analysis bins to 5 minutes

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
#
# max_entries bounds them: the key includes the bin size and the group subset,
# so browsing combinations would otherwise accumulate 45k-row frames for the
# life of the session.
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_profiles(fp, _ds, bin_size_min):
    return ssm.state_profiles(_ds, bin_size_min=bin_size_min)


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_waveforms(fp, _ds, bin_size_min):
    return ssm.compute_normalized_waveforms(_ds, bin_size_min=bin_size_min)


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_initiation(fp, _ds):
    return ssm.compute_initiation_probability(_ds)


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_circular(fp, _ds, bin_size_min, angle_doubling):
    return ssm.circular_state_stats(
        _ds, bin_size_min=bin_size_min, angle_doubling=angle_doubling
    )


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_chi_sq(fp, _cwt, states):
    return ultradian_rhythmicity_chi_sq(_cwt, states=states)


# ---------------------------------------------------------------------------
# Sidebar and epoch selection. All of this is cheap and none of it depends on
# a figure, so it renders before any computation starts.
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

# on_change="rerun" makes these dynamic, so `.open` is True only for the
# selected tab. With the default "ignore" every body below would run on every
# rerun and `.open` would be None.
#
# The `key` is what makes the selection reachable from session state, which is
# the only way a headless AppTest can open a tab — its Tab object is read-only.
TAB_KEY = "sleep_states_tab"
TAB_LABELS = [
    "Waveforms (Fig 1)",
    "Initiation (Fig 2)",
    "Rose & gating (Fig 3)",
    "Scalograms (Fig 5)",
    "Ultradian (Fig 6)",
]
tab_wave, tab_init, tab_rose, tab_scal, tab_ultra = st.tabs(
    TAB_LABELS, on_change="rerun", key=TAB_KEY
)


def _groups_in(frame):
    """Group labels in a stable order."""
    return list(dict.fromkeys(frame["group"].tolist()))


# ---------------------------------------------------------------------------
# Fig 1 — normalised waveforms
# ---------------------------------------------------------------------------
if tab_wave.open:
    with tab_wave:
        st.subheader("Normalised sleep-state waveforms")
        st.markdown(
            "Each state's mean profile is divided by its own peak, so shapes can "
            "be compared between states that differ several-fold in absolute "
            "amount. Flies are averaged **before** normalising, as the paper "
            "specifies — the other order makes every fly's own peak 1.0 and "
            "flattens between-fly differences in profile shape."
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
                "The paper's error band is between-RUN SEM across three "
                "independent experiments. This dataset is one run, so the band "
                "here is between-fly SEM within each group — a narrower claim.",
                icon=":material/info:",
            )
            st.download_button(
                "Download normalised waveforms (CSV)",
                waveforms.to_csv(index=False).encode(),
                file_name=f"sleep_state_waveforms_{phase_used}.csv",
                mime="text/csv",
                icon=":material/download:",
            )

# ---------------------------------------------------------------------------
# Fig 2 — profiles (left column) and initiation probability (right column)
# ---------------------------------------------------------------------------
if tab_init.open:
    with tab_init:
        st.subheader("Daily profiles and bout initiation")
        st.markdown(
            "Left: the daily profile of each state with standard sleep behind it "
            "in grey. Right: for each fly and state, the bouts starting in each "
            "1-hour window divided by that fly's own total bouts of that state, "
            "with the activity profile overlaid in red. The paper's point is "
            "that long sleep is initiated in the hour *following* the day's "
            "largest bout of wakefulness."
        )
        profiles = _cached_profiles(fp, phase_ds, bin_size_min)
        prof_stats = ssm.group_profiles(profiles)
        groups = _groups_in(profiles)

        has_bouts = "start_time" in phase_ds.data_vars
        if not has_bouts:
            st.warning(
                "No per-bout table in this dataset, so initiation probability "
                "cannot be computed — the profiles below are still valid. "
                "Re-run **Sleep analysis**, which writes `start_time` and "
                "`sleep_state` alongside the masks."
            )
            init_stats = None
        else:
            init = _cached_initiation(fp, phase_ds)
            init_stats = ssm.group_initiation_probability(init) if not init.empty else None
            if init_stats is None:
                st.info("No sleep bouts were detected for the current selection.")

        for group in groups:
            if len(groups) > 1:
                st.markdown(f"**{group}**")
            g_prof = prof_stats[prof_stats["group"] == group]
            left, right = st.columns(2)
            with left:
                st.plotly_chart(
                    plotting.state_profile_plot(
                        g_prof, phase_label=phase_used, title=None
                    ),
                    width="stretch",
                    key=f"prof_{group}",
                )
            with right:
                if init_stats is None:
                    st.empty()
                else:
                    st.plotly_chart(
                        plotting.initiation_probability_plot(
                            init_stats[init_stats["group"] == group],
                            activity_stats=g_prof[g_prof["state"] == "activity"],
                            phase_label=phase_used,
                            title=None,
                        ),
                        width="stretch",
                        key=f"init_{group}",
                    )

        if init_stats is not None:
            st.download_button(
                "Download per-fly initiation probabilities (CSV)",
                init.to_csv(index=False).encode(),
                file_name=f"sleep_state_initiation_{phase_used}.csv",
                mime="text/csv",
                icon=":material/download:",
            )

# ---------------------------------------------------------------------------
# Fig 3 — rose plots and circadian gating
# ---------------------------------------------------------------------------
if tab_rose.open:
    with tab_rose:
        st.subheader("Temporal organisation of sleep states")
        st.markdown(
            "Rose plots are the daily profiles in polar coordinates: each wedge "
            f"is one {bin_size_min}-minute bin, averaged over days and flies, "
            "with activity overlaid. Each series is scaled to its own peak, "
            "because sleep (min/h) and activity (counts/h) share no unit."
        )
        profiles = _cached_profiles(fp, phase_ds, bin_size_min)
        prof_stats = ssm.group_profiles(profiles)
        groups = _groups_in(profiles)
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
            "about it stands in for gate width, the paper's proxy in the absence "
            "of an objective phase marker for a sleep state's onset. Inner rings "
            "are individual flies, the thick outer arcs are group means. Gates "
            "may overlap even though the states themselves cannot co-occur."
        )
        circular = _cached_circular(fp, phase_ds, bin_size_min, angle_doubling)
        if circular.empty:
            st.info("No circular statistics could be computed.")
        else:
            gates = ssm.group_gates(circular)
            for group in _groups_in(circular):
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
                icon=":material/download:",
            )

# ---------------------------------------------------------------------------
# Fig 5 / 6 — the CWT. Expensive, so it runs on request and its result lives in
# session state, which is also why it is not an st.fragment: both the
# Scalograms and the Ultradian tab read it, and a fragment rerun would refresh
# only its own tab.
# ---------------------------------------------------------------------------

CWT_KEY = "sleep_states_cwt"
CWT_META = "sleep_states_cwt_meta"


def _cwt_results():
    """(dataset, meta, is_current) for whatever the last wavelet run produced."""
    result = st.session_state.get(CWT_KEY)
    meta = st.session_state.get(CWT_META)
    if result is None or not len(getattr(result, "data_vars", {})):
        return None, None, False
    return result, meta, bool(meta and meta.get("fp") == fp)


def _stale_warning(is_current):
    if not is_current:
        st.warning(
            "These wavelet results were computed for a different dataset or "
            "selection. Re-run them on the **Scalograms** tab to refresh.",
            icon=":material/warning:",
        )


if tab_scal.open:
    with tab_scal:
        st.subheader("Scalograms and period-vs-amplitude")
        st.markdown(
            "Continuous wavelet transforms of the 5-minute-binned state series, "
            "one surface per fly, each normalised to its own surface mean before "
            "the flies are averaged. The colour range is pinned to 0-1.5, as in "
            "the paper, so panels are comparable to each other and to the "
            "printed figure."
        )
        with st.form("sleep_states_cwt_form"):
            col_a, col_b = st.columns([1, 2])
            with col_a:
                p_min = st.number_input("Min period (h)", 0.5, 12.0, 1.0, 0.5)
                p_max = st.number_input("Max period (h)", 12.0, 48.0, 32.0, 1.0)
            with col_b:
                st.markdown(
                    "The default 1-32 h span is the paper's Figure 5A axis. "
                    "Running two narrow bands separately — as this code used to "
                    "by default — normalises each by its own band mean, which "
                    "makes the z values incomparable between bands and to the "
                    "paper."
                )
                run_cwt = st.form_submit_button(
                    "Run wavelet analysis", type="primary", icon=":material/play_arrow:"
                )

        if run_cwt:
            progress = st.progress(0.0, text="Running CWT…")

            def _tick(done, total):
                progress.progress(
                    min(1.0, done / max(1, total)), text=f"CWT {done}/{total}"
                )

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
            st.session_state[CWT_META] = {
                "fp": fp,
                "range": (p_min, p_max),
                "phase": phase_used,
            }

        cwt, meta, is_current = _cwt_results()
        if cwt is None:
            st.info("Press **Run wavelet analysis** to compute the scalograms.")
        else:
            _stale_warning(is_current)
            surfaces, axes, spectra, bands = {}, {}, {}, {}
            for state in states_present:
                key = f"sleep_cwt_{state}_full_avg_surface"
                if key not in cwt.data_vars:
                    continue
                surfaces[state] = cwt[key].values
                axes[state] = cwt[f"sleep_cwt_{state}_full_period_axis"].values
                spectra[state] = cwt[f"sleep_cwt_{state}_full_fly_power"].values
                bands[state] = ULTRADIAN_BANDS.get(state, (1, 4))

            if not surfaces:
                st.info("The wavelet run produced no surfaces for these states.")
            else:
                st.plotly_chart(
                    plotting.sleep_state_scalogram(
                        surfaces,
                        axes,
                        bin_size_min=CWT_BIN_MIN,
                        phase_label=meta.get("phase", phase_used) if meta else phase_used,
                    ),
                    width="stretch",
                )
                st.plotly_chart(
                    plotting.period_amplitude_plot(
                        spectra, axes, ultradian_band=bands
                    ),
                    width="stretch",
                )

if tab_ultra.open:
    with tab_ultra:
        st.subheader("Ultradian rhythm amplitude and its circadian gating")
        st.markdown(
            "Mean normalised amplitude inside each state's ultradian band "
            "(1-4 h for short and intermediate sleep, 2-6 h for long sleep) over "
            "time, then a periodogram of that amplitude series. A ~24 h peak "
            "means the strength of the ultradian rhythm itself waxes and wanes "
            "with the circadian day."
        )
        cwt, meta, is_current = _cwt_results()
        if cwt is None:
            st.info("Run the wavelet analysis on the **Scalograms** tab first.")
        else:
            _stale_warning(is_current)
            amp = {
                state: cwt[f"sleep_cwt_{state}_ultradian_amplitude"].values
                for state in states_present
                if f"sleep_cwt_{state}_ultradian_amplitude" in cwt.data_vars
            }
            if not amp:
                st.info("No ultradian amplitude series available.")
            else:
                # Figure 6A/C/E puts the scalogram CROPPED to each state's
                # ultradian band directly above the amplitude trace, so the
                # trace can be read as the band average it is. Same surfaces as
                # the Scalograms tab, just sliced to the band's rows.
                cropped, cropped_axes = {}, {}
                for state in amp:
                    surf_key = f"sleep_cwt_{state}_full_avg_surface"
                    if surf_key not in cwt.data_vars:
                        continue
                    periods = cwt[f"sleep_cwt_{state}_full_period_axis"].values
                    lo, hi = ULTRADIAN_BANDS.get(state, (1, 4))
                    rows = (periods >= lo) & (periods <= hi)
                    if not rows.any():
                        continue
                    cropped[state] = cwt[surf_key].values[rows]
                    cropped_axes[state] = periods[rows]
                if cropped:
                    st.plotly_chart(
                        plotting.sleep_state_scalogram(
                            cropped,
                            cropped_axes,
                            bin_size_min=CWT_BIN_MIN,
                            phase_label=meta.get("phase", phase_used)
                            if meta
                            else phase_used,
                            title="Ultradian band only",
                        ),
                        width="stretch",
                    )
                st.plotly_chart(
                    plotting.ultradian_amplitude_plot(
                        amp,
                        bin_size_min=CWT_BIN_MIN,
                        phase_label=meta.get("phase", phase_used) if meta else phase_used,
                        bands={s: ULTRADIAN_BANDS.get(s, (1, 4)) for s in amp},
                    ),
                    width="stretch",
                )
                st.divider()
                test = st.segmented_control(
                    "Rhythmicity test",
                    ["Chi-squared (as published)", "Lomb-Scargle"],
                    default="Chi-squared (as published)",
                    help=(
                        "The paper uses a chi-squared periodogram, so that is the "
                        "one whose numbers can be checked against its Figure 6 "
                        "and Table S2. Lomb-Scargle is the better test on its "
                        "merits — calibrated false-alarm probabilities, tolerant "
                        "of gaps — but its values are not comparable to the "
                        "published ones."
                    ),
                )
                if test == "Lomb-Scargle":
                    from periodograms import ultradian_rhythmicity_ls

                    ls = ultradian_rhythmicity_ls(cwt, states=tuple(states_present))
                    rows = [
                        {
                            "state": state,
                            "median period (h)": round(
                                float(np.nanmedian(ls[f"ultra_ls_period_{state}"].values)), 2
                            ),
                            "% FAP < 0.05": round(
                                100.0
                                * float(
                                    np.nanmean(ls[f"ultra_ls_fap_{state}"].values < 0.05)
                                ),
                                2,
                            ),
                        }
                        for state in states_present
                        if f"ultra_ls_period_{state}" in ls.data_vars
                    ]
                    if rows:
                        st.dataframe(rows, width="stretch", hide_index=True)
                    else:
                        st.info("No Lomb-Scargle result could be computed.")
                else:
                    chi = _cached_chi_sq(fp, cwt, tuple(states_present))
                    if not len(chi.data_vars):
                        st.info("No periodogram could be computed.")
                    else:
                        st.plotly_chart(
                            plotting.chi_sq_periodogram_plot(
                                chi, states=tuple(states_present)
                            ),
                            width="stretch",
                        )
                        rows = []
                        for state in states_present:
                            var = f"ultra_chisq_rhythmic_{state}"
                            if var not in chi.data_vars:
                                continue
                            rows.append(
                                {
                                    "state": state,
                                    "% rhythmic": round(
                                        100.0 * float(np.mean(chi[var].values)), 2
                                    ),
                                    "median period (h)": round(
                                        float(
                                            np.nanmedian(
                                                chi[
                                                    f"ultra_chisq_peak_period_{state}"
                                                ].values
                                            )
                                        ),
                                        2,
                                    ),
                                    "median adj. power": round(
                                        float(
                                            np.nanmedian(
                                                chi[
                                                    f"ultra_chisq_peak_adjusted_{state}"
                                                ].values
                                            )
                                        ),
                                        1,
                                    ),
                                }
                            )
                        if rows:
                            st.markdown(
                                "Comparable to the paper's Table S2, which "
                                "reports 93-100% rhythmic at 23.4-23.8 h for "
                                "wild-type flies."
                            )
                            st.dataframe(rows, width="stretch", hide_index=True)
