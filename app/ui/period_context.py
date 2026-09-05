"""The phase and period-range context shared by Period analysis and Rhythmicity.

Splitting the old 1361-line Period Analysis page into a "run" half and an
"explore" half exposed a coupling that was previously invisible because both
halves lived in one script: the classification window is not its own control.
It is ``min_period`` / ``max_period`` from the top of the page, reused verbatim
when classifying (old ``3_Period_Analysis.py:1237-1241``). The phase cascade that
produces ``period_ds`` is shared the same way.

So both live here and both pages render them, sharing widget keys. Same defaults
and same values as before, so behaviour is unchanged — the Rhythmicity page just
now shows you which phase and which window it is classifying against, instead of
inheriting them from a scroll position.
"""

import streamlit as st


def render_phase_picker(ds):
    """Resolve which phase to analyse and return
    ``(phase_selection, period_ds, analysis_src, phase_arg)``.

    - ``period_ds`` is for reading existing results and for display only.
    - ``analysis_src`` is the per-fly NaN-masked view of the *whole* dataset that
      the analyses actually consume. Feeding a pre-sliced, re-zeroed object and
      letting the analysis re-mask it dropped the first phase days, so the
      selection happens once, here, via the one core selector.
    - ``phase_arg`` is what to pass as ``phase=`` to the core analysis functions.
    """
    import dam_utilities
    from dataset_meta import PHASE_DD, PHASE_LD, dataset_phase, is_split_applied

    ds_phase = dataset_phase(ds)
    has_split_datasets = (
        st.session_state.get("dataset_DD") is not None
        and st.session_state.get("dataset_LD") is not None
    )
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
    elif has_split_datasets:
        phase_choice = st.radio(
            "Data phase for period analysis",
            ["DD (recommended)", "LD"],
            index=0,
            horizontal=True,
            help="DD (constant darkness) is the standard for circadian period estimation "
            "(free-running rhythm). LD periods reflect the imposed light cycle.",
            key="period_phase",
        )
        phase_selection = "DD" if "DD" in phase_choice else "LD"
        if phase_selection == "DD":
            period_ds = st.session_state.dataset_DD
            st.success(
                f"Using **DD (constant darkness)** data — "
                f"{len(period_ds['id'])} flies, {len(period_ds['time'])} timepoints."
            )
        else:
            period_ds = st.session_state.dataset_LD
            st.warning(
                f"Using **LD (light-dark)** data — period estimates will reflect "
                f"the imposed light cycle, not the endogenous circadian period. "
                f"{len(period_ds['id'])} flies, {len(period_ds['time'])} timepoints."
            )
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
        phase_choice = st.radio(
            "Data phase for period analysis",
            ["DD (recommended)", "LD"],
            index=0,
            horizontal=True,
            help="DD (constant darkness) is the standard for circadian period estimation. "
            "LD periods reflect the imposed light cycle.",
            key="period_phase",
        )
        phase_selection = "DD" if "DD" in phase_choice else "LD"
        period_ds = ds  # analyses split on-the-fly via select_phase
    else:
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


def render_period_range(*, show_caption=True):
    """Render the min/max period inputs and return ``(min_period, max_period)``.

    Keyed so the value carries between Period analysis and Rhythmicity: the same
    range drives the search on one page and the classification window on the
    other, and they must not be allowed to disagree.
    """
    import periodograms

    st.subheader("Period range")
    if show_caption:
        st.caption(
            "This range drives BOTH the period search and the classification window "
            "(they track together). Rejecting arrhythmic red-noise ramps is the "
            "metric's job (CWT `global_rednoise`), not the window's — so the range can "
            "be widened for long-period lines without inflating false positives. A "
            "short record cannot resolve the top of the band; CWT warns and analyses "
            "only the resolvable sub-band."
        )
    col1, col2 = st.columns(2)
    with col1:
        min_period = st.number_input(
            "Min period (hours)",
            min_value=1.0,
            max_value=48.0,
            value=float(periodograms.DEFAULT_CWT_MIN_PERIOD),
            step=1.0,
            key="period_min_h",
        )
    with col2:
        max_period = st.number_input(
            "Max period (hours)",
            min_value=1.0,
            max_value=72.0,
            value=float(periodograms.DEFAULT_CWT_MAX_PERIOD),
            step=1.0,
            key="period_max_h",
        )

    if min_period >= max_period:
        st.error("Min period must be less than max period.")
        st.stop()

    return min_period, max_period
