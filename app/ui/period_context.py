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

import numpy as np
import streamlit as st

# Streamlit garbage-collects a keyed widget's session-state entry as soon as a
# run does not instantiate that widget — and on a page switch it does that
# BEFORE the new page's script runs. So a widget key alone does NOT carry a
# value from Period analysis to Rhythmicity: the second page re-seeds from its
# own default. These two pages must agree on the phase and the period range
# (the range IS the classification window), so each widget is backed by a
# shadow key with no widget attached to it, which nothing collects.
_PERSIST_PREFIX = "_persist_"


def _remembered(key, default):
    """Last value stored for ``key``, or ``default`` on first render."""
    return st.session_state.get(_PERSIST_PREFIX + key, default)


def _remember(key, value):
    """Stash ``value`` where the widget garbage collector cannot reach it."""
    st.session_state[_PERSIST_PREFIX + key] = value
    return value


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
        _opts = ["DD (recommended)", "LD"]
        phase_choice = st.radio(
            "Data phase for period analysis",
            _opts,
            index=_opts.index(_remembered("period_phase", _opts[0])),
            horizontal=True,
            help="DD (constant darkness) is the standard for circadian period estimation "
            "(free-running rhythm). LD periods reflect the imposed light cycle.",
            key="period_phase",
        )
        _remember("period_phase", phase_choice)
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
        _opts = ["DD (recommended)", "LD"]
        phase_choice = st.radio(
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
            value=_remembered("period_min_h", float(periodograms.DEFAULT_CWT_MIN_PERIOD)),
            step=1.0,
            key="period_min_h",
        )
        _remember("period_min_h", min_period)
    with col2:
        max_period = st.number_input(
            "Max period (hours)",
            min_value=1.0,
            max_value=72.0,
            value=_remembered("period_max_h", float(periodograms.DEFAULT_CWT_MAX_PERIOD)),
            step=1.0,
            key="period_max_h",
        )
        _remember("period_max_h", max_period)

    if min_period >= max_period:
        st.error("Min period must be less than max period.")
        st.stop()

    return min_period, max_period


def has_split_datasets():
    """True when the Curate & split page left pre-sliced phase caches behind."""
    return (
        st.session_state.get("dataset_DD") is not None
        and st.session_state.get("dataset_LD") is not None
    )


def remember_min_days_floor(value):
    """Record the DD-days floor so Rhythmicity can flag under-floor flies with
    the same number Period analysis filtered on. Called by the page that owns
    the widget; see :func:`min_days_floor` for the read side."""
    return _remember("min_days_floor_shared", float(value))


def min_days_floor():
    """The DD-days retention floor set on the Period analysis page.

    Read from the widget key rather than passed along, so the Rhythmicity page
    can flag under-floor flies with the same number the run page filtered on
    without the two pages having to hand it between them.
    """
    import periodograms

    return float(
        _remembered("min_days_floor_shared", float(periodograms.DEFAULT_MIN_DD_DAYS_FLOOR))
    )


def merge_analysis_outputs(master, result_ds):
    """Attach analysis outputs from ``result_ds`` onto ``master`` without
    overwriting `master`'s `activity` / `time` / other shared variables.

    The CWT/LS/AC pipelines run on a *preprocessed* copy of the input
    (`preprocess_activity` may detrend/normalize/etc.) and return a
    merged dataset whose `activity` is the preprocessed version. If we
    blindly assigned `result_ds` back to the master we'd contaminate
    downstream visualization (e.g. `summary_bars` would sum detrended
    activity, producing negative totals for DD recordings — see issue
    1D in the period-analysis cleanup audit). This helper merges only
    the analysis-output variables and per-fly classification flags so
    raw activity stays untouched on master.
    """
    # Per-fly scalar data_vars (period, amplitude, FAP, RI, etc.)
    _scalar_vars = [v for v in result_ds.data_vars if result_ds[v].dims == ("id",)]
    # Per-fly array data_vars (cwt_powerseries, cwt_ridge_periods, etc.)
    # — anything keyed on id plus an analysis-specific axis.
    _array_vars = [
        v
        for v in result_ds.data_vars
        if v != "activity"
        and "id" in result_ds[v].dims
        and result_ds[v].dims != ("id",)
        and v.startswith(("cwt_", "ls_", "ac_", "mesa_"))
    ]
    # Per-fly classification coords (ac_rhythmic, ls_rhythmic, etc.)
    _scalar_coords = [
        c
        for c in result_ds.coords
        if c not in ("id", "time") and c in result_ds and result_ds[c].dims == ("id",)
    ]
    _to_drop = [v for v in (_scalar_vars + _array_vars + _scalar_coords) if v in master]
    if _to_drop:
        master = master.drop_vars(_to_drop, errors="ignore")
    if _scalar_vars or _array_vars:
        master = master.merge(result_ds[_scalar_vars + _array_vars])
    if _scalar_coords:
        master = master.assign_coords({c: result_ds[c] for c in _scalar_coords})
    # Copy analysis attrs (CWT/LS/AC/MESA + classification thresholds + paths)
    for k, v in result_ds.attrs.items():
        if k.startswith(("cwt_", "ls_", "ac_", "mesa_")):
            master.attrs[k] = v
    return master


def store_period_results(result_ds, phase_selection):
    """Store analysis results on the phase-specific dataset and merge
    per-fly outputs onto the master dataset.

    Single-phase loads (e.g. DD-only NetCDFs) and split workflows take
    the same merge path so master's `activity` is never replaced with
    the preprocessed (detrended) version that the analysis pipelines
    work on internally. See ``_merge_analysis_outputs`` docstring."""
    from analysis_detection import detect_analyses

    # Stage-2: result_ds is the WHOLE-dataset masked-view + per-fly outputs (the
    # analysis no longer runs on a re-zeroed slice). Per-fly results are phase-
    # independent (id,)/(id, analysis-axis) vars, so MERGE them onto the existing
    # sliced phase dataset rather than replacing it — keeps dataset_DD/LD's sliced
    # time series intact for unmigrated downstream pages (transitional; retires
    # with the dataset_LD/DD sweep). _merge_analysis_outputs never touches activity.
    if has_split_datasets() or phase_selection in ("DD", "LD"):
        _name = "dataset_DD" if phase_selection == "DD" else "dataset_LD"
        _tgt = st.session_state.get(_name)
        if _tgt is not None:
            st.session_state[_name] = merge_analysis_outputs(_tgt, result_ds)

    master = st.session_state.dataset
    master = merge_analysis_outputs(master, result_ds)
    st.session_state.dataset = master
    st.session_state.analyses = detect_analyses(master)


def dd_record_days(ds):
    """Per-fly valid-data span in days (robust, aligned to id). Used to report which
    flies the floor excludes; the kernel enforces the floor on the longest analysable
    block (equal to this span for a gap-free record)."""
    out = {}
    for _fid in ds["id"].values:
        _v = ds["activity"].sel(id=_fid).dropna("time")
        if _v.size >= 2:
            _t = _v["time"].values
            if np.issubdtype(_t.dtype, np.datetime64):
                _span = (np.datetime64(_t[-1]) - np.datetime64(_t[0])) / np.timedelta64(1, "D")
            else:
                _span = (float(_t[-1]) - float(_t[0])) / 1440.0
        else:
            _span = 0.0
        out[str(_fid)] = float(_span)
    return out
