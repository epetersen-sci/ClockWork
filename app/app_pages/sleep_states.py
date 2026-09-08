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

Every figure here is faceted BY GENOTYPE — one panel per group, per tab —
because the comparisons the paper makes (state against state, LD against DD)
are within a genotype, and overlaying groups on shared axes stops being
readable at more than two of them.

Tabs are DYNAMIC (``on_change="rerun"``) and every body is guarded by
``tab.open``. Streamlit renders all tab content by default, so without the
guard one visit to this page computed the waveforms, the initiation
probabilities, six rose rows and six gating rings whether or not anyone looked
at them — nineteen figures on 189 flies for the one the user was actually on.
Measured on that dataset, the landing tab went from ~14 s to 3.6 s. Faceting
the waveforms by genotype does not undo that: the frame is computed once per
epoch and sliced per group, so the extra cost is browser-side rendering only.

The tradeoff: switching tabs now costs a rerun, so a click made while the page
is already busy can be dropped and has to be repeated. That is inherent to
dynamic tabs (a segmented control behaves the same way), and on this page it
is the better bargain — the tabs that are expensive are expensive precisely
because nobody wants them computed unseen.
"""

import numpy as np
import pandas as pd
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
# The ONE reference to the source paper that the user sees. Everything on this
# page used to carry its own aside on how the output compared to the printed
# figures — which numbers matched, which conventions differed, where our
# defaults had been wrong. That is provenance for whoever maintains the code,
# not for whoever is reading their own data, and it belongs in the module
# docstrings (where it still is) rather than beside every figure.
st.caption(
    "Short (5-30 min), intermediate (30-60 min) and long (>60 min) sleep. "
    "The methods here follow Abhilash, Evans & Shafer 2026, *Current Biology* "
    "36:968-978 closely."
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
def _cached_chi_sq(fp, group, _cwt, states):
    """`group` is in the key on purpose.

    `_cwt` leads with an underscore, so it is not hashed — which was harmless
    while one pooled wavelet result existed per dataset, and wrong the moment
    the run became per genotype: without `group` the key is the same for every
    genotype and the first one's periodogram is served for all of them.
    """
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
    # Explicit key so the LD branch is reachable from a test; without one the
    # widget id is generated and nothing can select the non-default epoch.
    key="sleep_states_phase",
    help=(
        "DD is the default because the circadian measures here are cleanest "
        "without a light cycle driving them. A dataset already stamped with a "
        "single epoch can only be shown in that epoch."
    ),
)

bin_size_min = st.sidebar.select_slider(
    "Profile bin (minutes)",
    options=[10, 15, 20, 30, 60],
    value=30,
    help="Bin width for the profiles and rose plots. 30 minutes is a "
    "compromise between resolving temporal structure and smoothing out noise.",
)

angle_doubling = st.sidebar.selectbox(
    "Angle doubling",
    ["auto", "never", "always"],
    help=(
        "Circular statistics on a clearly bimodal profile need an "
        "angle-doubling transform first, or the mean phase lands between the "
        "two peaks rather than on either. 'auto' decides per fly and state "
        "from the shape of the profile."
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

states_declared = ssm.available_states(phase_ds)
if not states_declared:
    st.warning(
        "This dataset has no sleep-state masks. Run **Sleep & activity → Sleep "
        f"analysis** on the {phase_used} epoch first — the states here come from "
        "the `sleep_short` / `sleep_intermediate` / `sleep_long` variables it writes."
    )
    st.stop()

# Carrying the masks is not the same as having values in THIS epoch. Sleep
# analysis writes them over the whole time axis but marks every minute outside
# the epoch it ran on as missing, so asking for the other epoch gives four
# present-but-empty masks. Everything downstream then failed quietly and
# differently: empty profile panels, zero-radius rose wedges, a wavelet run
# that produced no surfaces — and initiation-probability bars that still had
# data, because the per-bout table is dimensioned on `sleep_bout_number` and
# phase selection never touches it. Catch it once, here, and say which epoch
# the masks are actually for.
states_present = ssm.states_with_data(phase_ds)
if not states_present:
    other = next(e for e in phase_options if e != phase_used)
    other_has = []
    try:
        other_ds, other_used = select_phase(ds, phase=other)
    except (ValueError, KeyError):
        other_used = None
    else:
        other_has = ssm.states_with_data(other_ds)
    hint = (
        f"Its masks do carry the **{other_used}** epoch — switch **Phase** to "
        f"{other_used} to see these figures, or re-run sleep analysis on "
        f"{phase_used}."
        if other_has
        else f"Re-run it with **Phase = {phase_used}**."
    )
    st.warning(
        f"Every sleep-state minute in the {phase_used} epoch of this dataset is "
        "missing, so nothing on this page can be computed. Sleep detection is "
        "per-epoch: **Sleep & activity → Sleep analysis** marks the minutes "
        f"outside the epoch it ran on as missing, and it was not run on "
        f"{phase_used}. " + hint,
        icon=":material/warning:",
    )
    st.stop()

fp = dataset_fingerprint(phase_ds)
# Says which epoch is on screen in words, not just as a label on an axis. The
# same line appears on the Sleep & activity page, which has the same choice.
_other_epoch = next((e for e in phase_options if e != phase_used), None)
if _other_epoch:
    st.caption(
        f"Data shown is from the **{phase_used}** dataset. To change to the "
        f"**{_other_epoch}** dataset, use the selector in the sidebar."
    )
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
    "Waveforms",
    "Initiation",
    "Rose & gating",
    "Scalograms",
    "Ultradian",
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
            "amount. Flies are averaged **before** normalising: the other order "
            "makes every fly's own peak 1.0 and flattens between-fly "
            "differences in profile shape."
        )
        # Figure 1B prints LD and DD beside each other, because its claim is
        # that the waveform SHAPES survive the loss of the light cycle. Showing
        # only the selected epoch loses that comparison, so both are drawn when
        # the dataset carries both. Same flies in each panel.
        fly_ids = phase_ds["id"].values
        epochs = {}
        for epoch in ("LD", "DD"):
            if epoch == phase_used:
                epochs[epoch] = phase_ds
                continue
            try:
                other, other_used = select_phase(ds, phase=epoch)
            except (ValueError, KeyError):
                continue  # this dataset holds only the one epoch
            try:
                other = other.sel(id=fly_ids)
            except KeyError:
                continue
            # Skip an epoch sleep analysis did not run on rather than binning
            # 189 flies x 4 all-missing masks to produce a panel that gets
            # filtered out below anyway.
            if not ssm.states_with_data(other):
                continue
            epochs[other_used] = other

        panels = [(name, _cached_waveforms(dataset_fingerprint(sub), sub, bin_size_min))
                  for name, sub in epochs.items()]
        panels = [(name, frame) for name, frame in panels if not frame.empty]

        if not panels:
            st.info("No waveforms could be computed for the current selection.")
        else:
            # ONE FIGURE PER GENOTYPE, the way every other tab on this page is
            # laid out. Overlaying the groups put four states x N groups on one
            # pair of axes, distinguished only by dash style: on this dataset
            # that is 24 solid-to-dashdot lines plus 24 SEM bands, and the
            # legend ran off the bottom of the panel. The states within one
            # genotype are the comparison Figure 1B actually makes; comparing
            # genotypes is what putting the panels in a column is for.
            #
            # Still one epoch per column inside each genotype's row, because
            # Figure 1B's claim is that the shapes survive the loss of the
            # light cycle, which needs LD and DD side by side.
            wave_groups = list(
                dict.fromkeys(
                    g for _, frame in panels for g in _groups_in(frame)
                )
            )
            for group in wave_groups:
                rows = [
                    (name, frame[frame["group"] == group])
                    for name, frame in panels
                ]
                rows = [(name, frame) for name, frame in rows if not frame.empty]
                if not rows:
                    continue
                for column, (name, frame) in zip(st.columns(len(rows)), rows):
                    with column:
                        st.plotly_chart(
                            plotting.normalized_waveform_overlay(
                                frame,
                                phase_label=name,
                                # Group and epoch both in the FIGURE title, so
                                # there is no second heading above it.
                                title=f"{group} — {name}",
                            ),
                            width="stretch",
                            key=f"waveform_{group}_{name}",
                        )
            st.info(
                "The band is between-fly SEM within each group. It describes "
                "the spread among these flies, not the reproducibility of the "
                "result across independent experiments.",
                icon=":material/info:",
            )
            # Every panel shown, tagged by epoch, so the CSV matches the figure
            # rather than only its left half.
            waveforms = pd.concat(
                [frame.assign(epoch=name) for name, frame in panels], ignore_index=True
            )
            st.download_button(
                "Download normalised waveforms (CSV)",
                waveforms.to_csv(index=False).encode(),
                file_name="sleep_state_waveforms_"
                + "_".join(name for name, _ in panels)
                + ".csv",
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
            "with the activity profile overlaid in red. Each fly's curve sums "
            "to 1, so a fly that slept little counts as much as one that slept "
            "a lot."
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

        # The group goes in each FIGURE's title rather than in a heading above
        # the pair. `title=None` used to suppress the figure title so the
        # heading could carry the group, but Plotly serialises a None title as
        # an empty title object, whose `text` is undefined — so both panels
        # printed the literal word "undefined" where the title belongs.
        for group in groups:
            g_prof = prof_stats[prof_stats["group"] == group]
            left, right = st.columns(2)
            with left:
                st.plotly_chart(
                    plotting.state_profile_plot(
                        g_prof,
                        phase_label=phase_used,
                        title=f"Daily profiles — {group}",
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
                            title=f"Bout initiation — {group}",
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
        # No st.markdown heading here: rose_plot_with_activity already prints
        # "Temporal organisation of sleep states — <group>" as the figure
        # title, so the heading repeated the group label directly above it.
        for group in groups:
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
            "about it stands in for gate width, there being no objective phase "
            "marker for the onset of a sleep state. Inner rings are individual "
            "flies, the thick outer arcs are group means. Gates may overlap "
            "even though the states themselves cannot co-occur."
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
                st.plotly_chart(
                    plotting.polar_gating_plot(
                        g_stats,
                        gates[gates["group"] == group],
                        phase_label=phase_used,
                        title=f"Circadian gating — {group}",
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
# The CWT, shared by the Scalograms and Ultradian tabs. Expensive, so it runs
# only on request, and it is not an st.fragment for that same sharing reason:
# a fragment rerun would refresh only its own tab.
# ---------------------------------------------------------------------------

# Session state holds the REQUEST (a small dict of fingerprint, genotype,
# period range and epoch), never the result. The result itself lives in
# `_cached_cwt` below, where Streamlit bounds it: one run of four states over a
# ten-day recording is ~19 MB of averaged surface, so keeping a result per
# genotype in session state would grow without limit as genotypes are browsed.
# The request is what says "the user has asked for this one" — the surfaces are
# far too expensive to compute on an unprompted rerun.
CWT_REQUEST = "sleep_states_cwt_request"


@st.cache_data(show_spinner=False, max_entries=3)
def _cached_cwt(fp, _ds, fly_ids, states, p_range, phase):
    """One genotype's wavelet run.

    ``fly_ids`` is the genotype's flies and NOT a detail: the transforms are
    averaged across whatever flies are passed, so running the whole dataset at
    once produced a single surface per state with every genotype averaged into
    it. That is not a comparison anyone asked for — it is a genotype-blind mean
    that a two-genotype experiment makes meaningless — and it was the only
    thing these two tabs could show.

    Per-genotype costs no more in total than the pooled run did: the work is
    one transform per fly either way, and each fly belongs to one genotype.
    ``max_entries=3`` bounds the memory a held result can take (~19 MB of
    averaged surface for four states over a ten-day recording).

    The progress bar is created INSIDE the function on purpose. A callback
    closing over a bar created by the caller raises CacheReplayClosureError on
    the first cache hit: Streamlit replays the element calls a cached function
    made, and it cannot replay into a layout block that no longer exists.
    """
    progress = st.progress(0.0, text="Running CWT…")

    def _tick(done, total):
        progress.progress(min(1.0, done / max(1, total)), text=f"CWT {done}/{total}")

    try:
        return sleep_cwt_analysis(
            _ds,
            states=tuple(states),
            fly_ids=list(fly_ids),
            full_range=tuple(p_range),
            # _ds is already this epoch's masked view; re-requesting the SAME
            # phase is idempotent, and it must not be "both", which the
            # period-analysis phase guard rejects outright.
            phase=phase,
            progress_callback=_tick,
        )
    finally:
        progress.empty()


def _cwt_request():
    """The standing wavelet request, or None if it is not for this selection."""
    req = st.session_state.get(CWT_REQUEST)
    if not req or req.get("fp") != fp:
        return None
    return req


def _cwt_for(req):
    """The result for a request — instant on a cache hit."""
    return _cached_cwt(
        req["fp"],
        phase_ds,
        req["fly_ids"],
        req["states"],
        req["range"],
        req["phase"],
    )


CWT_EMPTY_MESSAGE = (
    "The wavelet run finished but produced no surfaces. Every fly's state "
    "series was too short or too gappy to transform: the CWT needs one "
    "continuous run of at least 50 minutes (ten 5-minute bins) per fly, and "
    "missing minutes break a run. Check that sleep analysis was run on this "
    "epoch and that the flies in this selection have a usable stretch of it."
)

# Genotypes offered to the wavelet tabs, in dataset order.
cwt_groups = (
    list(dict.fromkeys(str(g) for g in phase_ds["group"].values))
    if "group" in phase_ds.coords
    else ["All Flies"]
)


def _fly_ids_for(group):
    """The flies of one genotype, as a tuple so it can key a cache."""
    if "group" not in phase_ds.coords:
        return tuple(str(i) for i in phase_ds["id"].values)
    return tuple(
        str(i)
        for i, g in zip(phase_ds["id"].values, phase_ds["group"].values)
        if str(g) == group
    )


if tab_scal.open:
    with tab_scal:
        st.subheader("Scalograms and period-vs-amplitude")
        st.markdown(
            "Continuous wavelet transforms of the 5-minute-binned state series, "
            "one surface per fly, each normalised to its own surface mean before "
            "the flies are averaged. The colour range is pinned to 0-1.5, so "
            "the panels are comparable to each other."
        )
        # ONE GENOTYPE PER RUN. The selector sits inside the form so changing it
        # does not silently start a run: nothing recomputes until Submit.
        with st.form("sleep_states_cwt_form"):
            col_a, col_b = st.columns([1, 1])
            with col_a:
                cwt_group = st.selectbox(
                    "Genotype",
                    cwt_groups,
                    help=(
                        "The transforms are averaged across the flies included, "
                        "so one genotype is run at a time — an average over "
                        "several genotypes at once would not describe any of "
                        "them. The last few runs are kept, so returning to a "
                        "genotype you have already run is immediate."
                    ),
                )
            with col_b:
                p_min = st.number_input("Min period (h)", 0.5, 12.0, 1.0, 0.5)
                p_max = st.number_input("Max period (h)", 12.0, 48.0, 32.0, 1.0)
            run_cwt = st.form_submit_button(
                "Run wavelet analysis", type="primary", icon=":material/play_arrow:"
            )

        if run_cwt:
            st.session_state[CWT_REQUEST] = {
                "fp": fp,
                "group": cwt_group,
                "fly_ids": _fly_ids_for(cwt_group),
                "states": tuple(states_present),
                "range": (float(p_min), float(p_max)),
                "phase": phase_used,
            }

        req = _cwt_request()
        if req is None:
            st.info("Press **Run wavelet analysis** to compute the scalograms.")
        elif not req["fly_ids"]:
            st.warning(
                f"No flies of **{req['group']}** are in the current group selection."
            )
        else:
            with st.spinner(f"Computing wavelet transforms for {req['group']}…"):
                cwt = _cwt_for(req)

            st.caption(
                f"{req['group']} · {len(req['fly_ids'])} flies · "
                f"{req['range'][0]:g}-{req['range'][1]:g} h · "
                f"{req['phase']} epoch"
            )
            if not len(cwt.data_vars):
                st.error(CWT_EMPTY_MESSAGE, icon=":material/error:")
            else:
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
                            phase_label=req["phase"],
                            title=f"Normalised average scalograms — {req['group']}",
                        ),
                        width="stretch",
                    )
                    st.plotly_chart(
                        plotting.period_amplitude_plot(
                            spectra,
                            axes,
                            ultradian_band=bands,
                            title=f"Period vs. amplitude — {req['group']}",
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
        # Reads the SAME request as the Scalograms tab, so the genotype chosen
        # there is the genotype shown here — and a cache hit means no recompute.
        req = _cwt_request()
        cwt = None
        if req and req["fly_ids"]:
            # Usually a cache hit and instant. It can miss — the cache holds
            # only the last few runs — and then this recomputes, so say so.
            with st.spinner(f"Loading wavelet results for {req['group']}…"):
                cwt = _cwt_for(req)
        if cwt is None:
            st.info("Run the wavelet analysis on the **Scalograms** tab first.")
        elif not len(cwt.data_vars):
            st.error(CWT_EMPTY_MESSAGE, icon=":material/error:")
        else:
            st.caption(
                f"{req['group']} · {len(req['fly_ids'])} flies · "
                f"{req['phase']} epoch"
            )
            amp = {
                state: cwt[f"sleep_cwt_{state}_ultradian_amplitude"].values
                for state in states_present
                if f"sleep_cwt_{state}_ultradian_amplitude" in cwt.data_vars
            }
            if not amp:
                st.info("No ultradian amplitude series available.")
            else:
                # The scalogram CROPPED to each state's ultradian band sits
                # directly above the amplitude trace, so the trace can be read
                # as the band average it is. Same surfaces as the Scalograms
                # tab, just sliced to the band's rows.
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
                            phase_label=req["phase"],
                            title=f"Ultradian band only — {req['group']}",
                        ),
                        width="stretch",
                    )
                st.plotly_chart(
                    plotting.ultradian_amplitude_plot(
                        amp,
                        bin_size_min=CWT_BIN_MIN,
                        phase_label=req["phase"],
                        bands={s: ULTRADIAN_BANDS.get(s, (1, 4)) for s in amp},
                        title=f"Ultradian amplitude over time — {req['group']}",
                    ),
                    width="stretch",
                )
                st.divider()
                # Lomb-Scargle leads and is the default: it gives calibrated
                # false-alarm probabilities and tolerates the gaps a real
                # recording has, neither of which the chi-squared periodogram
                # does. Chi-squared stays available because its numbers are the
                # ones comparable to previously published results.
                #
                # Key renamed alongside the option labels — a session that
                # hot-reloads this file would otherwise still hold
                # "Chi-squared (as published)", which is no longer an option.
                test = st.segmented_control(
                    "Rhythmicity test",
                    ["Lomb-Scargle", "Chi-squared"],
                    default="Lomb-Scargle",
                    key="sleep_states_rhythmicity_test",
                    help=(
                        "Lomb-Scargle is the better test on its merits: "
                        "calibrated false-alarm probabilities and tolerant of "
                        "gaps. Chi-squared reports a per-fly periodogram and "
                        "a % rhythmic, and its values are the ones comparable "
                        "to older published numbers."
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
                        st.markdown(
                            "Dominant period of each state's ultradian-amplitude "
                            "series, and the fraction of flies whose peak is "
                            "significant at a false-alarm probability below 0.05."
                        )
                        st.dataframe(rows, width="stretch", hide_index=True)
                        st.download_button(
                            "Download per-fly Lomb-Scargle results (CSV)",
                            pd.DataFrame(
                                {
                                    "id": ls["id"].values,
                                    **{
                                        f"{quantity}_{state}": ls[
                                            f"ultra_ls_{quantity}_{state}"
                                        ].values
                                        for state in states_present
                                        for quantity in ("period", "power", "fap")
                                        if f"ultra_ls_{quantity}_{state}" in ls.data_vars
                                    },
                                }
                            )
                            .to_csv(index=False)
                            .encode(),
                            file_name=(
                                f"ultradian_ls_{req['group']}_{req['phase']}.csv"
                            ),
                            mime="text/csv",
                            icon=":material/download:",
                        )
                    else:
                        st.info("No Lomb-Scargle result could be computed.")
                else:
                    chi = _cached_chi_sq(
                        fp, req["group"], cwt, tuple(states_present)
                    )
                    if not len(chi.data_vars):
                        st.info("No periodogram could be computed.")
                    else:
                        st.plotly_chart(
                            plotting.chi_sq_periodogram_plot(
                                chi,
                                states=tuple(states_present),
                                title=(
                                    "Chi-squared periodogram of ultradian "
                                    f"amplitude — {req['group']}"
                                ),
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
                            st.dataframe(rows, width="stretch", hide_index=True)
