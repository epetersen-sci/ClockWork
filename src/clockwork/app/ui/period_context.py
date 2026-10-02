"""The phase and period-range context for the Period & rhythmicity page.

The classification window is not its own control: it is ``min_period`` /
``max_period``, the same range the period search runs over. Those, the DD-days
floor and the gap-bridging ceiling are rendered here in one row
(:func:`render_period_row`), and the phase cascade that produces ``period_ds``
is resolved here too (:func:`render_phase_picker`). The page hands the result
to every tab, so the tabs cannot disagree about which epoch or which window a
number came from.
"""

import numpy as np
import streamlit as st

# Streamlit garbage-collects a keyed widget's session-state entry as soon as a
# run does not instantiate that widget — and on a page switch it does that
# BEFORE the new page's script runs. So a widget key alone does NOT carry a
# value away from the page and back: it re-seeds from its default. Each of these
# widgets is backed by a shadow key with no widget attached, which nothing
# collects, so the phase and range you set are still set when you return.
_PERSIST_PREFIX = "_persist_"


def _remembered(key, default):
    """Last value stored for ``key``, or ``default`` on first render."""
    return st.session_state.get(_PERSIST_PREFIX + key, default)


def _remember(key, value):
    """Stash ``value`` where the widget garbage collector cannot reach it."""
    st.session_state[_PERSIST_PREFIX + key] = value
    return value


def render_phase_picker(ds, *, quiet=False):
    """Resolve which phase to analyse and return
    ``(phase_selection, period_ds, analysis_src, phase_arg)``.

    ``quiet=True`` resolves without drawing anything — no radio, no status
    message — for a page that CONSUMES the choice rather than making it. The phase
    is picked on Period analysis; offering the same radio again downstream invites
    the two to be set apart and the second page to analyse something else without
    saying so. A quiet caller is expected to state which phase it used.

    - ``period_ds`` is for reading existing results and for display only.
    - ``analysis_src`` is the per-fly NaN-masked view of the *whole* dataset that
      the analyses actually consume. Feeding a pre-sliced, re-zeroed object and
      letting the analysis re-mask it dropped the first phase days, so the
      selection happens once, here, via the one core selector.
    - ``phase_arg`` is what to pass as ``phase=`` to the core analysis functions.
    """
    from clockwork.core import dam_utilities
    from clockwork.core.dataset_meta import PHASE_DD, PHASE_LD, dataset_phase, is_split_applied

    ds_phase = dataset_phase(ds)
    has_dd = "first_DD_day" in ds.coords

    if ds_phase in (PHASE_LD, PHASE_DD):
        # The loaded file is itself a partition — use it directly. No phase
        # picker needed: the choice was made when the file was saved.
        phase_selection = ds_phase
        period_ds = ds
        if ds_phase == PHASE_DD:
            st.success(
                f"Using the loaded **DD** dataset — "
                f"{len(period_ds['id'])} flies, {len(period_ds['time'])} timepoints."
            )
        else:
            st.warning(
                f"Using the loaded **LD** dataset — period estimates will reflect "
                f"the imposed light cycle, not the endogenous circadian period. "
                f"{len(period_ds['id'])} flies, {len(period_ds['time'])} timepoints."
            )
    # There used to be a branch here for "the pre-sliced dataset_DD/dataset_LD
    # caches exist", which handed the analyses a physical slice. It is gone with
    # the caches. The `has_dd` branch below already did the same job better — it
    # reads split state from attrs that survive a NetCDF round-trip, and lets the
    # analyses select the phase on-the-fly through the one core selector instead
    # of consuming a re-zeroed object.
    elif has_dd:
        # Whether the split was APPLIED must be read from ROUND-TRIP-SAFE dataset
        # state — is_split_applied() inspects the split_applied/split_phase attrs,
        # which survive the NetCDF round-trip — NOT the session-state dataset_LD/DD
        # caches, which are derivative and ABSENT on a fresh .nc load. Keying off
        # the caches made a reloaded split-applied dataset falsely report "split
        # not applied". The analysis runs on-the-fly in BOTH cases, so only the
        # message differs.
        if is_split_applied(ds):
            st.success(
                "LD/DD split applied — analysing the selected phase on-the-fly "
                "from the loaded dataset."
            )
        else:
            st.warning(
                "LD/DD split has **not** been applied on the **Data → Curate & split** page. "
                "Period analysis will split the data on-the-fly, but applying the split "
                "there first is recommended (for gap detection and consistency)."
            )
        _opts = ["DD (recommended)", "LD"]
        phase_choice = _remembered("period_phase", _opts[0]) if quiet else st.radio(
            "Data phase for period analysis",
            _opts,
            index=_opts.index(_remembered("period_phase", _opts[0])),
            horizontal=True,
            help="DD (constant darkness) is the standard for circadian period estimation. "
            "LD periods reflect the imposed light cycle.",
            key="period_phase",
        )
        _remember("period_phase", phase_choice)
        phase_selection = "DD" if "DD" in phase_choice else "LD"
        period_ds = ds  # analyses split on-the-fly via select_phase
    else:
        if not quiet:
            st.info("No LD/DD transition found — using full dataset for period analysis.")
        phase_selection = "full"
        period_ds = ds

    if ds_phase in (PHASE_LD, PHASE_DD):
        # Loaded file is itself the single phase — analyse it as-is. Drop the
        # boundary so the selector treats it as already-selected (no re-derive).
        src = ds.drop_vars([c for c in ("first_DD_day", "split_minute") if c in ds.coords])
        analysis_src, _ = dam_utilities.select_phase(src, "auto")
        phase_arg = "auto"
    elif phase_selection in ("DD", "LD"):
        analysis_src, _ = dam_utilities.select_phase(st.session_state.dataset, phase_selection)
        phase_arg = phase_selection
    else:  # "full" — no LD/DD transition; analyse the whole recording
        analysis_src, _ = dam_utilities.select_phase(ds, "auto")
        phase_arg = "auto"

    return phase_selection, period_ds, analysis_src, phase_arg


def render_period_row():
    """The four numbers every period analysis runs on, in ONE row:
    ``(min_period, max_period, min_dd_days, max_bridge_gap)``.

    Min/max period is the search range AND the classification window; the DD-days
    floor excludes short records; the gap ceiling is how much missing data CWT,
    AC and MESA bridge. Same widget keys as when they were separate rows, so a
    remembered value carries over.
    """
    from clockwork.core import periodograms

    st.subheader("Period range")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        min_period = st.number_input(
            "Min period (h)",
            min_value=1.0,
            max_value=48.0,
            value=_remembered("period_min_h", float(periodograms.DEFAULT_CWT_MIN_PERIOD)),
            step=1.0,
            key="period_min_h",
        )
        _remember("period_min_h", min_period)
    with c2:
        max_period = st.number_input(
            "Max period (h)",
            min_value=1.0,
            max_value=72.0,
            value=_remembered("period_max_h", float(periodograms.DEFAULT_CWT_MAX_PERIOD)),
            step=1.0,
            key="period_max_h",
            help="Widen for known long-period lines: a ~43 h rhythm reads arrhythmic at "
            "a 36 h ceiling. CWT `global_rednoise` stays range-robust when widened.",
        )
        _remember("period_max_h", max_period)
    with c3:
        floor = st.number_input(
            "Min DD days",
            min_value=0.0,
            max_value=30.0,
            value=_remembered("min_days_floor_shared", float(periodograms.DEFAULT_MIN_DD_DAYS_FLOOR)),
            step=0.5,
            key="min_days_floor_shared",
            help="Flies whose longest analysable DD block is shorter than this are "
            "EXCLUDED from period analysis (no period computed; absent from the period "
            "graphs). ~4 days is a reasonable floor. 0 keeps every fly.",
        )
        remember_min_days_floor(floor)
    with c4:
        max_gap = st.number_input(
            "Max gap to bridge (min)",
            min_value=0.0,
            max_value=120.0,
            value=_remembered("max_bridge_gap", float(periodograms.DEFAULT_MAX_BRIDGE_GAP_MINUTES)),
            step=5.0,
            key="max_bridge_gap",
            help="Interior gaps up to this long are bridged by linear interpolation for "
            "CWT, autocorrelation and MESA only (never written to the .nc). Longer gaps "
            "break the record and the longest clean segment is analysed. 0 = off. "
            "Validated to 60 min; Lomb-Scargle is gap-native and ignores this.",
        )
        _remember("max_bridge_gap", max_gap)
    st.caption(
        "The period range drives both the period search and the classification window."
    )

    if min_period >= max_period:
        st.error("Min period must be less than max period.")
        st.stop()
    return min_period, max_period, floor, max_gap


def remember_min_days_floor(value):
    """Record the DD-days floor past the widget's lifetime, so it survives a
    page switch (see ``_PERSIST_PREFIX``)."""
    return _remember("min_days_floor_shared", float(value))


def merge_analysis_outputs(master, result_ds):
    """Attach analysis outputs from ``result_ds`` onto ``master`` without replacing
    its raw ``activity``. The one implementation is
    :func:`clockwork.pipeline.merge_period_outputs`, which the CLI uses too."""
    from clockwork.pipeline import merge_period_outputs

    return merge_period_outputs(master, result_ds)


def store_master(master):
    """Make ``master`` the session's dataset after a pipeline step returned it."""
    from clockwork.app.analysis_detection import detect_analyses

    st.session_state.dataset = master
    st.session_state.analyses = detect_analyses(master)


def dd_record_days(ds):
    """Per-fly valid-data span in days (robust, aligned to id). Used to report which
    flies the floor excludes; the kernel enforces the floor on the longest analysable
    block (equal to this span for a gap-free record)."""
    # Every fly at once: its first and last non-NaN minute bound the span. (A
    # per-fly .sel().dropna() did the same one fly at a time, on every rerun.)
    t = ds["time"].values
    if len(t) == 0:
        return {str(fid): 0.0 for fid in ds["id"].values}
    valid = ~np.isnan(ds["activity"].transpose("time", "id").values)
    n_valid = valid.sum(axis=0)
    first = valid.argmax(axis=0)
    last = len(t) - 1 - valid[::-1].argmax(axis=0)
    if np.issubdtype(t.dtype, np.datetime64):
        spans = (t[last] - t[first]) / np.timedelta64(1, "D")
    else:
        spans = (t[last].astype(float) - t[first].astype(float)) / 1440.0
    spans = np.where(n_valid >= 2, spans, 0.0)
    return {str(fid): float(span) for fid, span in zip(ds["id"].values, spans)}
