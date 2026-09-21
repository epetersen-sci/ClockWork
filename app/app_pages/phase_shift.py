"""
Phase Shift Analysis — measure each fly's phase shift after a light pulse.

Requires a ``pulse_time`` column in the metadata (a ZT hour such as ZT15) plus
``first_DD_day``, which anchors that ZT to the last entrained day before DD release.
"""


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import dam_utilities
import export_helpers
import phase_shift as ps_module
import plotting
from calibrations import (
    DEFAULT_PHASE_SHIFT_FILTER_HOURS,
    DEFAULT_PHASE_SHIFT_MARKER_SEARCH_HALF_WIDTH_HOURS,
    DEFAULT_PHASE_SHIFT_MIN_POST_DAYS,
    DEFAULT_PHASE_SHIFT_MIN_PRE_DAYS,
    DEFAULT_PHASE_SHIFT_ONSET_SMOOTH_MINUTES,
    DEFAULT_PHASE_SHIFT_ONSET_THRESHOLD_FRAC,
    DEFAULT_PHASE_SHIFT_PEAK_DISTANCE_HOURS,
    DEFAULT_PHASE_SHIFT_PEAK_PROMINENCE_FRAC,
    DEFAULT_PHASE_SHIFT_TRANSIENT_SKIP_DAYS,
)
from ui import charts, filters, status
from ui.guards import require_dataset

st.caption(
    "Measures how far each fly's rhythm shifted after its light pulse, by comparing "
    "the rhythm's daily phase before and after the pulse."
)

# ============================================================
# Section 1: Prerequisites
# ============================================================
ds = require_dataset()

if "pulse_zt_hour" not in ds.coords:
    st.error(
        "This dataset has no **pulse_time** column. Phase-shift analysis needs to know at "
        "what circadian time the light pulse was given.\n\n"
        "Add a `pulse_time` column to your metadata file — a ZT hour such as `ZT15` — and "
        "optionally `pulse_duration_min`, then reload the dataset on the **Data → Import** "
        "page. See `metadata_template.csv` for the format. Leave the cell blank for any "
        "group that received no pulse."
    )
    st.stop()

if ds.attrs.get("time_is_relative_minutes", 0) != 1:
    st.error(
        "Phase-shift analysis needs the relative-minute time axis produced by the standard "
        "loading path. Reload the dataset on the **Data → Import** page."
    )
    st.stop()

# Pulse position per fly, in minutes from that fly's recording start.
try:
    ds_pulse = ds if "pulse_minute" in ds.coords else dam_utilities.add_pulse_metadata(ds)
except Exception as exc:  # surfacing the reason beats a blank page
    st.error(f"Could not derive the pulse position from the metadata: {exc}")
    st.stop()

pulse_minutes = np.asarray(ds_pulse["pulse_minute"].values, dtype=float)
n_pulsed = int(np.isfinite(pulse_minutes).sum())
n_total = pulse_minutes.size

if n_pulsed == 0:
    st.error(
        "Every fly's `pulse_time` is blank, so there is no pulse to measure a shift "
        "around. Fill in the pulse time for at least one group and reload."
    )
    st.stop()

minutes_axis = np.asarray(ds_pulse["time"].values, dtype=float)
n_days = int(np.floor((minutes_axis[-1] + 1) / 1440.0))

col_a, col_b, col_c = st.columns(3)
col_a.metric("Flies with a pulse", f"{n_pulsed} / {n_total}")
col_b.metric("Complete days", n_days)
_pulse_days = sorted({int(m // 1440) for m in pulse_minutes if np.isfinite(m)})
col_c.metric("Pulse on day", ", ".join(str(d) for d in _pulse_days))

if n_pulsed < n_total:
    st.info(
        f"{n_total - n_pulsed} fly/flies have no pulse time (unpulsed controls). They are "
        "reported with status `no_pulse` rather than dropped."
    )

_has_boundary = "split_minute" in ds_pulse.coords or "first_DD_day" in ds_pulse.coords
if _has_boundary:
    st.caption(
        "Each regression is kept inside a single LD/DD epoch, using the LD-DD boundary from "
        "`first_DD_day`. A fit spanning the boundary would average an entrained slope with a "
        "free-running one."
    )
else:
    st.caption(
        "No `first_DD_day` in the metadata, so all days are treated as one epoch. If this "
        "experiment released the flies into DD, add that column so the fits respect the "
        "boundary."
    )

if ds.attrs.get("phase_shift_method") is not None:
    st.info(
        f"Previous phase-shift analysis recorded — method **{ds.attrs['phase_shift_method']}**."
    )

# ============================================================
# Setup and Results are separate tabs
# ============================================================
# The page had the pickers, the run button, the figures, the per-line pairing notes
# and the export tables all in one column, so reading a result meant scrolling past
# every control that produced it — and changing one control meant scrolling back
# down to see what it did. Streamlit runs both tab bodies on every rerun in document
# order, so the widgets below are still created before the results block reads them.
_tab_setup, _tab_results = st.tabs(["Setup", "Results"])

with _tab_setup:
    # ============================================================
    # Section 2: Reference — what the shifted phase is compared against
    # ============================================================
    st.subheader("Reference")

    _pre_pulse_days = min(int(m // 1440) for m in pulse_minutes if np.isfinite(m))

    REFERENCE_LABELS = {
        "Unpulsed control group (per day)": "control",
        "Each fly's own pre-pulse rhythm": "self",
    }
    reference_label = st.radio(
        "Compare the post-pulse phase against what?",
        list(REFERENCE_LABELS.keys()),
        # The control-referenced comparison is the lab's peakphaseplot.m workflow and
        # the one people come here for, so it leads regardless of how many pre-pulse
        # days the protocol left. The per-fly alternative needs several of them and
        # warns below when there are too few; that warning is the better place for the
        # distinction than a default that moves under you between experiments.
        index=0,
        horizontal=True,
        help="A phase shift is a difference from where the rhythm would have been. That "
        "reference can come from an unpulsed control cohort, or from extrapolating each "
        "fly's own pre-pulse trend.",
    )
    reference = REFERENCE_LABELS[reference_label]

    if reference == "control":
        st.caption(
            "This is the lab's `peakphaseplot.m` comparison: average each group's activity, "
            "smooth it, take one peak per day, and report the pulsed group's peak time minus "
            "the unpulsed control's on the same day. Validated against the lab's own SCAMP "
            "exports — from day 1 on, the two agree to a median of 2 min (worst 12 min). "
            "Results are per group per day, not per fly."
        )
        if _pre_pulse_days < int(DEFAULT_PHASE_SHIFT_MIN_PRE_DAYS):
            st.info(
                f"This protocol leaves only **{_pre_pulse_days}** full day(s) before the pulse, "
                "which is too few to fit each fly's own pre-pulse rhythm — so the control-group "
                "reference is the appropriate choice here and is preselected."
            )
    else:
        st.caption(
            "Per fly: fit the daily phase before the pulse, extrapolate it across the pulse, and "
            "measure how far the post-pulse rhythm sits from that line. Gives one shift per fly "
            "(so a distribution per group), but needs several pre-pulse days."
        )
        if _pre_pulse_days < int(DEFAULT_PHASE_SHIFT_MIN_PRE_DAYS):
            st.warning(
                f"Only **{_pre_pulse_days}** full day(s) precede the pulse in this dataset, below "
                f"the {int(DEFAULT_PHASE_SHIFT_MIN_PRE_DAYS)}-day minimum, so most or all flies "
                "will abstain with `insufficient_pre`. Use the control-group reference instead, or "
                "lower the minimum below (knowing the fit gets noisier)."
            )

    st.subheader("Method")

    if reference == "control":
        # peakphaseplot.m is a peak-matching method; there is no onset variant of it to
        # be faithful to, so don't offer one here.
        method = "peak"
        st.caption("Peak matching, as in `peakphaseplot.m`.")

        # Which metadata factors define a group. Splitting by sex as well as genotype and
        # condition is the usual reason to change this — the lab plots the sexes apart.
        #
        # Via group_defining_coords rather than a blocklist of the coords analyses attach:
        # it answers "which coords came from the metadata" from provenance, so nothing a
        # later analysis adds can appear here and there is no list to keep in step with
        # core/. The same question Groups & subsets, Actograms and Sleep (SCAMP) ask.
        _coord_opts = dam_utilities.group_defining_coords(ds_pulse)
        if not _coord_opts:
            st.error("This dataset has no categorical metadata coordinates to group by.")
            st.stop()
        # Start from the grouping the dataset already has — as COORD names, via
        # get_group_coord_names. attrs['group_columns'] records the metadata COLUMNS
        # ticked at import, and two of those live under different coord names
        # (pulse_time -> pulse_zt_hour), so filtering that list down to real coords
        # dropped them and left this page defaulting to genotype alone: a different
        # partition from the one the dataset is grouped by.
        #
        # Unlike the actograms page this cannot simply select the `group` coord: the
        # matched pairing needs a column that separates pulsed from unpulsed, which
        # means the grouping has to stay decomposable. A column that does that is
        # added when the import grouping did not include one — without it every
        # group is its own control and the pairing has nothing to do.
        #
        # Pulse duration leads the list because it is what now identifies a control
        # (a group whose pulse was 0 minutes long), and the unpulsed arm has to be a
        # group of its own before it can be found.
        _grp_default = [
            c for c in dam_utilities.get_group_coord_names(ds) if c in _coord_opts
        ]

        def _separates_something(col):
            """Whether ``col`` takes more than one value across these flies.

            A column that does not splits nothing, and adding it to the grouping
            only lengthens every label — ``A_LP`` becomes ``A_LP_nan``. That is
            exactly what happens with the ``pulse_duration_minutes`` that
            ``add_pulse_metadata`` invents, all-NaN, for a dataset whose metadata
            never carried one.
            """
            if col not in ds_pulse.coords:
                return False
            vals = {v for v in np.asarray(ds_pulse[col].values).astype(str) if v != "nan"}
            return len(vals) > 1

        _cond_like = next(
            (
                c
                for c in (
                    "pulse_duration_minutes",
                    "condition",
                    "treatment",
                    "pulse_zt_hour",
                )
                if c in _coord_opts and _separates_something(c)
            ),
            None,
        )
        if _cond_like and _cond_like not in _grp_default:
            _grp_default = _grp_default + [_cond_like]
        if not _grp_default:
            _grp_default = [c for c in ("genotype", "condition") if c in _coord_opts]
        group_by = st.multiselect(
            "Group-defining columns",
            _coord_opts,
            default=_grp_default or _coord_opts[:1],
            help="A group is one combination of these. Add `sex` to compare males and "
            "females separately rather than pooled.",
        )
        if not group_by:
            st.error("Pick at least one column to define the groups.")
            st.stop()
        group_by = tuple(group_by)

        _labels, _cols = ps_module.group_labels(ds_pulse, group_by)
        _groups = sorted(set(_labels))

        def _looks_unpulsed(text):
            return any(t in str(text).lower() for t in ("nolp", "no_lp", "no lp", "control"))

        def _is_numeric_coord(dset, col):
            """Whether ``col`` holds numbers — a duration, an intensity, a ZT hour.

            Asked of the VALUES rather than of the dtype: coords arrive as strings
            often enough (a metadata column with one blank cell comes in as object)
            that a dtype check would reject the very column the control test needs.
            """
            vals = pd.to_numeric(
                pd.Series(np.asarray(dset[col].values).astype(str)), errors="coerce"
            )
            return bool(vals.notna().any())

        # What a control has to SHARE with the group it references: everything that
        # defines a group except the pulse itself, since the pulse is what the two
        # arms differ in on purpose. With nothing left over there is only one
        # possible control, and matching it is the same as naming it — which is why
        # this also decides which pairing mode leads below.
        _match_cols = [c for c in _cols if c not in ps_module.PULSE_COORDS]
        _n_match_keys = (
            len(set(zip(*[np.asarray(ds_pulse[c].values).astype(str) for c in _match_cols])))
            if _match_cols
            else 1
        )

        # No apparatus column is singled out any more. This page used to hunt for
        # `flybox` (then `Monitor`) and thread it through every legend entry, figure
        # title, pairing note and exported column, on the reasoning that which box an
        # arm sat in bounds how much of a difference can be read as the pulse. True,
        # but it is one lab's collection habit wearing the clothes of an analysis
        # rule: it assumed one box per arm, it fired on any dataset that happened to
        # carry the column, and it put a factor nobody had chosen into the output of
        # every comparison. A box effect is a confound like any other — if it matters
        # here, `flybox` belongs in the grouping, where it is compared rather than
        # annotated.

        PAIRING_LABELS = {
            "Unpulsed control of the same genotype": "matched",
            "One control group for the whole experiment": "single",
        }
        pairing_label = st.radio(
            "Which control is each group compared against?",
            list(PAIRING_LABELS.keys()),
            index=0 if _n_match_keys > 1 else 1,
            horizontal=True,
            help="Genotypes differ in baseline phase, so referring every group to a "
            "single cohort folds that genotype difference into the reported shift.",
        )
        pairing = PAIRING_LABELS[pairing_label]

        if pairing == "matched":
            # Which group is the control is read off the PULSE: a group whose pulse
            # duration is 0 got no pulse, whatever the metadata calls it. The
            # previous version searched a `condition` column for the literal string
            # "noLP", which worked on this lab's spreadsheet and silently found
            # nothing on anyone else's — and took three widgets to configure a
            # question the data already answers. The control is matched on every
            # grouping column EXCEPT the pulse ones, so one unpulsed cohort serves
            # every dose its genotype received.
            # "No pulse" is the number zero, so it needs a column that holds
            # numbers. A dataset recorded before anyone wrote a pulse duration down
            # has only a categorical arm label ("noLP"), and for those the control
            # value is still named by hand — the old behaviour, kept for old data
            # rather than left in the way of new.
            _numeric_cols = [c for c in _cols if _is_numeric_coord(ds_pulse, c)]
            _dur_default = next(
                (
                    c
                    for c in ("pulse_duration_minutes", "pulse_duration", "duration")
                    if c in _numeric_cols
                ),
                _numeric_cols[0] if _numeric_cols else None,
            )

            if _dur_default is None:
                st.caption(
                    "None of the grouping columns hold numbers, so the unpulsed arm "
                    "cannot be recognised by its pulse being zero. Name it below. "
                    "Adding a `pulse_duration_min` column to the metadata — 0 for the "
                    "unpulsed flies — removes this step for good."
                )
                control_on = st.selectbox(
                    "Column that marks the unpulsed control",
                    _cols,
                    index=len(_cols) - 1,
                )
                _cond_vals = sorted(set(np.asarray(ds_pulse[control_on].values).astype(str)))
                control_value = st.selectbox(
                    "Value marking the control arm",
                    _cond_vals,
                    index=_cond_vals.index(
                        next((v for v in _cond_vals if _looks_unpulsed(v)), _cond_vals[0])
                    ),
                )
                # The arm label itself is what the two arms differ in, so it is the
                # one column a control must NOT share — the same rule the numeric
                # path applies to the pulse coords.
                _matched_on = [c for c in _match_cols if c != control_on]
                try:
                    control_group = ps_module.build_matched_control_map(
                        ds_pulse,
                        control_value,
                        group_by=group_by,
                        control_on=control_on,
                        match_on=_matched_on or [control_on],
                    )
                except ValueError as exc:
                    st.error(str(exc))
                    st.stop()
                _controls = sorted({g for g in _groups if control_group.get(g) == g})
            else:
                control_on = st.selectbox(
                    "Column that marks the unpulsed control",
                    _numeric_cols,
                    index=_numeric_cols.index(_dur_default),
                    help="A group whose value here is 0 is a control. Pulse duration "
                    "is the honest one — a fly that got no light got a 0-minute pulse.",
                )
                _matched_on = list(_match_cols)
                try:
                    control_group, _controls = ps_module.control_map_from_pulse(
                        ds_pulse, group_by, duration_coord=control_on
                    )
                except ValueError as exc:
                    st.error(str(exc))
                    st.stop()

            st.caption(
                "Each group is compared against the unpulsed group that shares its "
                + (
                    ", ".join(f"`{c}`" for c in _matched_on)
                    if _matched_on
                    else "grouping"
                )
                + " — so an `Hr38` light-pulse arm is measured against unpulsed "
                "`Hr38` rather than against another genotype's control."
            )

            if not _controls:
                _suggest = ps_module.grouping_suggestion(ds_pulse)
                st.warning(
                    "**No unpulsed control group found.** No group has "
                    f"`{control_on}` = "
                    + ("0" if _dur_default is not None else f"`{control_value}`")
                    + " for every one of its flies, so nothing can be measured "
                    "against anything and every result below will be blank."
                    + (
                        "\n\nTry grouping by "
                        + ", ".join(f"`{c}`" for c in _suggest)
                        + " — the unpulsed arm only becomes a group of its own once "
                        "the column that separates it is part of the grouping."
                        if _suggest
                        else ""
                    )
                    + "\n\nIf the unpulsed flies are recorded with a blank duration "
                    "rather than a 0, put a 0 in that column and reload."
                )

            # n is shown because it is the cheapest tell that a label covers two
            # sub-experiments: a group holding twice the flies you expect is pooling
            # them. Group, then what it is compared against, then the counts, then
            # everything else — the reading order of the question being asked.
            _n_by_group = {g: int((_labels == g).sum()) for g in _groups}
            _pairs = pd.DataFrame(
                [
                    {
                        "group": g,
                        "compared against": control_group.get(g) or "— none —",
                        "n": _n_by_group[g],
                        "n in control": _n_by_group.get(control_group.get(g), 0),
                    }
                    for g in _groups
                    if control_group.get(g) != g
                ]
            )
            _orphans = [g for g in _groups if control_group.get(g) is None]
            with st.expander(
                f"Pairings ({len(_pairs) - len(_orphans)} of {len(_pairs)} groups matched)",
                expanded=bool(_orphans),
            ):
                st.dataframe(_pairs, width="stretch", hide_index=True)
                if _controls:
                    st.caption(
                        "Controls: "
                        + ", ".join(f"`{c}`" for c in _controls)
                        + f" — the groups whose `{control_on}` is "
                        + ("0." if _dur_default is not None else f"`{control_value}`.")
                    )
            if _orphans:
                st.warning(
                    "No matching control for: "
                    + ", ".join(f"`{g}`" for g in _orphans)
                    + ". No unpulsed group shares their "
                    + (", ".join(_matched_on) if _matched_on else "grouping")
                    + ", so their phase difference is reported as blank rather than "
                    "measured against a different genotype."
                )
        else:
            # Default to whatever looks like the unpulsed arm, so the common case is one click.
            _guess = next((g for g in _groups if _looks_unpulsed(g)), _groups[0])
            control_group = st.selectbox(
                "Unpulsed control group (the phase reference)",
                _groups,
                index=_groups.index(_guess),
                help="Every other group's daily peak time is reported relative to this one.",
            )
            if _n_match_keys > 1:
                st.caption(
                    "All groups — including the other genotypes — are referred to this one "
                    "cohort, so a baseline phase difference between genotypes will appear in "
                    "the result alongside any real shift."
                )

        # Neither the day origin nor the direction of the subtraction is a choice
        # any more. The difference is reported control minus pulsed, so a delay
        # reads negative — the same sign the phase response curve uses. Two views
        # of one pulse disagreeing about which way is an advance is the kind of
        # thing nobody catches until a figure is in a manuscript.
        difference_sign = "control_minus_group"

        # Days are counted from the recording start, with one exception that is a
        # property of the dataset rather than a preference: flies released into DD
        # on different days of their own recording line up only when day 0 is the
        # first DD day in each. So it is detected, not asked about.
        _dd_days = dam_utilities.dd_onset_days(ds_pulse)
        _mixed_dd = len(_dd_days) > 1
        day_origin = "dd_onset" if _mixed_dd else "recording_start"
        if _mixed_dd:
            st.info(
                "Flies here are released into DD on different days of their recording "
                f"(day {', '.join(str(d) for d in _dd_days)}) — the signature of a "
                "combined dataset. Days are therefore counted from the **first DD "
                "day** rather than the recording start, so the runs are compared at "
                "the same free-running age; otherwise the control has free-run a day "
                "longer than the group it references and that drift lands in the "
                "difference. Under this origin the light pulse falls on day −1."
            )

        # Zero every group on the LAST DAY THE PULSE HAS NOT TOUCHED. The pulse is
        # given late on the last entrained day and only moves the NEXT day's peak
        # (the same reasoning that puts the pulse marker BETWEEN two days, below), so
        # that entrained day is the last clean one — and being adjacent to the pulse
        # it is the tightest possible reference for the offset the boxes already
        # carried. The switch itself now lives in Advanced Parameters.
        _baseline_default = (
            -1 if day_origin == "dd_onset" else (_pulse_days[0] if _pulse_days else 0)
        )

    else:
        METHOD_LABELS = {
            "Peak matching (default)": "peak",
            "Onset regression (Aschoff / Daan-Pittendrigh)": "onset",
        }
        method_label = st.radio(
            "How should each day's phase be measured?",
            list(METHOD_LABELS.keys()),
            index=0,
            horizontal=True,
        )
        method = METHOD_LABELS[method_label]
        control_group = None

        if method == "peak":
            st.caption(
                "Peak matching: low-pass filters the activity trace and takes each day's peak — "
                "the lab's `peakphaseplot.m` approach (Levine et al. 2002), run per fly with the "
                "peaks matched automatically instead of by clicking. On the validation cohort this "
                "recovers a known shift to about 1 min median error."
            )
        else:
            st.warning(
                "Onset regression takes the start of each day's active phase. It agrees with peak "
                "matching in the typical case (~3 min median error on the validation cohort) but "
                "has a much worse tail — about 1 in 10 flies off by over 2.5 h — because the onset "
                "of a sparse 1-minute activity trace is a far less well-defined feature than a "
                "peak. Its calibrations are provisional. Check the actogram before trusting these "
                "numbers."
            )

    with st.expander("Advanced Parameters"):
        st.caption(
            "Defaults come from `core/calibrations.py`, which documents where each value came "
            "from and which ones are provisional."
        )
        if reference == "control":
            # The starting offset lives here rather than on the main setup column:
            # it is on by default and the default is right, so it was three lines of
            # page between you and the thing you came to configure. It stays
            # reachable because turning it off is how you check that a difference
            # was not there before the pulse.
            st.markdown("**Starting offset**")
            rebase = st.checkbox(
                "Take out the starting offset (set one day to zero)",
                value=True,
                help="Subtracts each group's difference on the chosen day from all of "
                "its days. Without it, a cohort-to-cohort phase offset present before "
                "the pulse is carried through every later day.",
            )
            baseline_day = None
            if rebase:
                baseline_day = st.number_input(
                    "Day to set to zero",
                    min_value=-int(n_days),
                    max_value=int(n_days),
                    value=int(_baseline_default),
                    step=1,
                    help="Defaults to the last entrained day — the day of the pulse, "
                    "whose own peak still precedes it — so every group starts from "
                    "zero immediately before the pulse and the curve shows only what "
                    "the pulse did. Pick an earlier day to average out a noisy one, "
                    "but avoid the first recorded day: its peak sits on the filter "
                    "edge.",
                )
            st.divider()

        pc1, pc2 = st.columns(2)
        with pc1:
            st.markdown("**Fitting**")
            min_pre_days = st.number_input(
                "Minimum days before pulse",
                min_value=2,
                max_value=30,
                value=int(DEFAULT_PHASE_SHIFT_MIN_PRE_DAYS),
                help="Fewer usable days than this and the fly abstains instead of reporting a number.",
            )
            min_post_days = st.number_input(
                "Minimum days after pulse",
                min_value=2,
                max_value=30,
                value=int(DEFAULT_PHASE_SHIFT_MIN_POST_DAYS),
            )
            transient_skip_days = st.number_input(
                "Skip transient days after pulse",
                min_value=0,
                max_value=10,
                value=int(DEFAULT_PHASE_SHIFT_TRANSIENT_SKIP_DAYS),
                help="Excludes the first N days after the pulse from the fit, letting transient "
                "cycles pass before the new steady state is read. 0 = off.",
            )
            search_half_width_hours = st.number_input(
                "Day-to-day tracking window (± hours)",
                min_value=0.5,
                max_value=12.0,
                value=float(DEFAULT_PHASE_SHIFT_MARKER_SEARCH_HALF_WIDTH_HOURS),
                step=0.5,
                help="How far from the previous day's marker to look for the next one. Does not "
                "limit how large a shift can be measured — the post-pulse days are tracked "
                "independently of the pre-pulse days.",
            )
            override_ref = st.checkbox(
                "Override reference day",
                value=False,
                help="By default the two fitted lines are compared at the pulse day (the standard "
                "convention). Change only if you want the shift read at a different day.",
            )
            reference_day_index = None
            if override_ref:
                reference_day_index = st.number_input(
                    "Reference day index", min_value=0, max_value=max(0, n_days - 1), value=0
                )
        with pc2:
            if method == "peak":
                st.markdown("**Peak detection**")
                filter_hours = st.number_input(
                    "Low-pass filter (hours)",
                    min_value=0.0,
                    max_value=24.0,
                    value=float(DEFAULT_PHASE_SHIFT_FILTER_HOURS),
                    step=1.0,
                    help="peakphaseplot.m's own default is 12 h. Raise it if false peaks appear.",
                )
                peak_prominence_frac = st.number_input(
                    "Peak prominence (fraction of daily range)",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(DEFAULT_PHASE_SHIFT_PEAK_PROMINENCE_FRAC),
                    step=0.01,
                )
                peak_distance_hours = st.number_input(
                    "Minimum peak separation (hours)",
                    min_value=1.0,
                    max_value=24.0,
                    value=float(DEFAULT_PHASE_SHIFT_PEAK_DISTANCE_HOURS),
                    step=1.0,
                )
                onset_threshold_frac = DEFAULT_PHASE_SHIFT_ONSET_THRESHOLD_FRAC
                onset_smooth_minutes = DEFAULT_PHASE_SHIFT_ONSET_SMOOTH_MINUTES
            else:
                st.markdown("**Onset detection**")
                onset_threshold_frac = st.number_input(
                    "Onset threshold (× mean activity)",
                    min_value=0.05,
                    max_value=6.0,
                    value=float(DEFAULT_PHASE_SHIFT_ONSET_THRESHOLD_FRAC),
                    step=0.05,
                    help="Relative to the fly's own mean smoothed activity, so it transfers across "
                    "flies and recorders. Provisional.",
                )
                onset_smooth_minutes = st.number_input(
                    "Smoothing window (minutes)",
                    min_value=5.0,
                    max_value=360.0,
                    value=float(DEFAULT_PHASE_SHIFT_ONSET_SMOOTH_MINUTES),
                    step=15.0,
                    help="Width of the rolling mean the onset is read from. Provisional.",
                )
                filter_hours = DEFAULT_PHASE_SHIFT_FILTER_HOURS
                peak_prominence_frac = DEFAULT_PHASE_SHIFT_PEAK_PROMINENCE_FRAC
                peak_distance_hours = DEFAULT_PHASE_SHIFT_PEAK_DISTANCE_HOURS

    if reference == "control":
        params = dict(
            control_group=control_group,
            group_by=group_by,
            day_origin=day_origin,
            baseline_day=baseline_day,
            difference_sign=difference_sign,
            filter_hours=float(filter_hours),
            peak_prominence_frac=float(peak_prominence_frac),
            peak_distance_hours=float(peak_distance_hours),
            search_half_width_hours=float(search_half_width_hours),
        )
    else:
        params = dict(
            method=method,
            reference_day_index=reference_day_index,
            min_pre_days=int(min_pre_days),
            min_post_days=int(min_post_days),
            transient_skip_days=int(transient_skip_days),
            search_half_width_hours=float(search_half_width_hours),
            filter_hours=float(filter_hours),
            peak_prominence_frac=float(peak_prominence_frac),
            peak_distance_hours=float(peak_distance_hours),
            onset_threshold_frac=float(onset_threshold_frac),
            onset_smooth_minutes=float(onset_smooth_minutes),
        )


    # The run button lives at the FOOT OF SETUP, not on Results. Clicking it
    # reruns the script, and a rerun resets the tab selection to the first tab —
    # so a button on the Results tab bounced you back to Setup every time, and you
    # had to click back to see what it produced. Below the controls that feed it is
    # also simply where it belongs.
    if reference == "control" and st.button(
        "Run Group Phase Comparison", type="primary"
    ):
        with st.spinner("Comparing each group's daily peak time to the control..."):
            try:
                gres = ps_module.compute_group_phase_difference(ds_pulse, **params)
                st.session_state.phase_shift_group_results = gres
                ds.attrs["phase_shift_method"] = "group_peak_vs_control"
                # attrs must survive a netCDF round-trip, so a per-group mapping is
                # recorded as text rather than as a dict.
                ds.attrs["phase_shift_control_group"] = (
                    "; ".join(
                        f"{g}->{c}"
                        for g, c in sorted(gres["control_map"].items())
                        if c is not None
                    )
                    if isinstance(control_group, dict)
                    else str(control_group)
                )
                ds.attrs["phase_shift_filter_hours"] = float(filter_hours)
            except Exception as exc:
                st.error(f"Error: {exc}")
                st.stop()

        # The phase response is computed by the SAME click, because it answers the
        # question the experiment was run to answer and nobody should have to find
        # a second button to get it. It is a second pass over the data rather than a
        # rearrangement of the first: the per-day comparison detects one peak on
        # each group's MEAN trace, and a response per fly needs a peak detected in
        # each fly, which is both slower and noisier. Hence the progress bar.
        pres = None
        _bar = st.progress(0.0, text="Measuring each fly's own phase…")
        try:
            pres = ps_module.compute_phase_response(
                ds_pulse,
                group_by=group_by,
                # The pairing resolved above, handed over rather than worked out
                # again — so the curve and the per-day comparison cannot end up
                # measuring against different controls, and so a dataset whose
                # control arm is a name rather than a zero still gets a curve.
                control_map=(
                    control_group
                    if isinstance(control_group, dict)
                    else dict.fromkeys(_groups, control_group)
                ),
                progress_callback=lambda f: _bar.progress(
                    min(1.0, f), text="Measuring each fly's own phase…"
                ),
                filter_hours=float(filter_hours),
                peak_prominence_frac=float(peak_prominence_frac),
                peak_distance_hours=float(peak_distance_hours),
                search_half_width_hours=float(search_half_width_hours),
            )
        except Exception as exc:
            # A failure here must not cost the per-day comparison that already
            # succeeded — the two are independent answers and one is on screen.
            st.warning(
                f"The per-day comparison is ready, but no phase response could be "
                f"computed: {exc}"
            )
        finally:
            _bar.empty()
        st.session_state.phase_shift_response = pres

        st.session_state.dataset = ds
        status.refresh(ds)
        st.success(
            f"Done: {gres['per_day']['group'].nunique()} groups compared"
            + (
                f", {len(pres['per_fly'])} flies measured for the phase response."
                if pres is not None and not pres["per_fly"].empty
                else "."
            )
        )
        st.rerun()

with _tab_results:
    # ============================================================
    # Section 3: Run (control-referenced group comparison)
    # ============================================================
    if reference == "control":
        if "phase_shift_group_results" not in st.session_state:
            st.info("Pick the control group above and click **Run** to compare.")
            st.stop()

        gres = st.session_state.phase_shift_group_results
        per_day = gres["per_day"]
        control_map = gres.get("control_map") or {}
        single_control = gres["control_group"] if isinstance(gres["control_group"], str) else None

        # Two questions, two tabs, in the order they are asked. A pulse experiment
        # is run to produce a phase response curve; how the difference settles over
        # the days after the pulse is the diagnostic you look at when the curve says
        # something surprising. The page used to offer only the second.
        _tab_prc, _tab_over_time = st.tabs(
            ["Phase Response Curves", "Phase over Time"]
        )

        with _tab_prc:
            _pres = st.session_state.get("phase_shift_response")
            if not _pres:
                st.info(
                    "No phase response computed yet — re-run the comparison on the "
                    "**Setup** tab and it is built alongside the per-day result."
                )
            elif _pres["per_fly"].empty:
                st.warning(
                    "Not one fly got a phase response. Either no group has an unpulsed "
                    "control, or no peak could be detected on the response days."
                )
                if len(_pres["dropped"]):
                    st.dataframe(_pres["dropped"], width="stretch", hide_index=True)
            else:
                _rp = _pres["params"]
                _pf = _pres["per_fly"]
                _base = _rp["baseline_days"]
                st.caption(
                    "One number per fly: the mean over days "
                    + ", ".join(f"+{d}" for d in _rp["response_days"])
                    + " after the pulse of **its own control group's mean peak time "
                    "minus its own**"
                    + (
                        f", less the same difference on day {_base[0]:+d} as a baseline"
                        if _base
                        else ""
                    )
                    + ". **An advance reads positive, a delay negative.**"
                )

                # Controls are the reference, not a result: every other group's
                # response is measured against their mean, so drawing them would be
                # drawing zero against itself. They are computed all the same —
                # their spread is the noise floor the real responses are read over,
                # and it is in the export.
                _ctrls = set(_pres.get("controls") or [])
                _treated = _pf[~_pf["group"].isin(_ctrls)].copy()
                _drop = _pres["dropped"]
                _drop_treated = (
                    _drop[~_drop["group"].isin(_ctrls)] if len(_drop) else _drop
                )

                if _treated.empty:
                    st.warning(
                        "Every group here is an unpulsed control, so there is no "
                        "response to plot."
                    )
                else:
                    _gb_prc = list(_rp["group_by"])
                    _zt_coord = "pulse_zt_hour"

                    # ---------------------------------------- the curve
                    st.markdown("#### Phase response curve")
                    _zts = sorted({float(z) for z in _treated["zt"] if np.isfinite(z)})
                    MIN_ZT_FOR_A_CURVE = 3
                    # The pulse time is the x axis, so it is not also a line; every
                    # other grouping factor separates the lines.
                    _series_prc = [c for c in _gb_prc if c != _zt_coord]
                    if len(_zts) < MIN_ZT_FOR_A_CURVE:
                        st.info(
                            f"This experiment only contains {len(_zts)} timepoint"
                            f"{'' if len(_zts) == 1 else 's'}. At least "
                            f"{MIN_ZT_FOR_A_CURVE} timepoints are required to create a "
                            "phase response curve. Please refer to per-fly responses below."
                        )
                    else:
                        _summary = ps_module.summarize_phase_response(
                            _treated, by=("zt", *_series_prc)
                        )
                        charts.plotly_chart(
                            plotting.phase_response_curve(
                                _summary, series_cols=_series_prc
                            ),
                            filename="phase_response_curve",
                            width="stretch",
                        )
                        st.caption(
                            "Points are group means of the per-fly responses; bars are "
                            "±1 SEM over the flies. No bootstrap is needed here — the "
                            "response is measured once per fly, so the flies are the "
                            "replicates. (The group-level comparison on the next tab "
                            "has no per-fly values to average, which is why that one "
                            "does need resampling.)"
                        )

                    # ---------------------------------------- the violins
                    st.markdown("#### Phase response per fly")
                    # Intensity is drawn side by side WITHIN a column rather than
                    # getting columns of its own, so two intensities of the same dose
                    # can be read against each other without hunting along the axis.
                    _split = (
                        "pulse_intensity"
                        if "pulse_intensity" in _treated.columns
                        and _treated["pulse_intensity"].nunique(dropna=True) > 1
                        else None
                    )
                    # Keep the import order, but plot the NUMERIC zt column rather
                    # than the coord: the coord arrives as text, and as text ZT9
                    # sorts after ZT15.
                    _major = []
                    for _c in _gb_prc:
                        if _c == _zt_coord:
                            _major.append("zt")
                        elif _c != _split:
                            _major.append(_c)

                    _show_pts = st.checkbox(
                        "Show individual flies",
                        value=True,
                        key="ps_prc_points",
                        help="Each dot is one fly's own phase response. The violin is "
                        "their distribution and its line is the mean — the same "
                        "number the curve above plots.",
                    )
                    # Named the way Import named them. Two of these columns are
                    # stored under a different coord name than the one that was
                    # ticked (pulse_time -> pulse_zt_hour), and a figure that
                    # labels itself from the coords renames the user's own
                    # columns back at them.
                    _back = filters.column_for_coord(ds)
                    _fig_v, _drawn_v = plotting.phase_response_violins(
                        _treated,
                        major_cols=_major,
                        split_col=_split,
                        show_points=_show_pts,
                        x_title=" x ".join(
                            _back.get(_zt_coord if c == "zt" else c, c).replace("_", " ")
                            for c in _major
                        ),
                        split_title=(
                            _back.get(_split, _split).replace("_", " ") if _split else None
                        ),
                    )
                    charts.plotly_chart(
                        _fig_v, filename="phase_response_per_fly", width="stretch"
                    )

                    # What was left out, said plainly. A fly whose phase cannot be
                    # determined is dropped rather than counted as "no shift" — the
                    # §2a rule, and the difference between the two is a mean pulled
                    # toward zero by every arrhythmic fly in the cohort.
                    _n_drawn = len(_drawn_v)
                    _n_pulsed = (
                        int(_drop_treated["n_total"].sum()) if len(_drop_treated) else 0
                    )
                    _n_out = max(0, _n_pulsed - _n_drawn)
                    if _n_out:
                        st.caption(
                            f"**{_n_drawn} of {_n_pulsed}** pulsed flies are drawn; "
                            f"{_n_out} could not be measured — no matched control, or "
                            "no detectable peak on the response days — and are left "
                            "out rather than counted as no shift."
                        )
                        with st.expander("Flies left out, by group"):
                            st.dataframe(
                                _drop_treated, width="stretch", hide_index=True
                            )
                    else:
                        st.caption(f"All **{_n_drawn}** pulsed flies are drawn.")

                    _orphaned = [
                        g
                        for g, c in (_pres.get("control_map") or {}).items()
                        if c is None
                    ]
                    if _orphaned:
                        st.warning(
                            "No unpulsed control matched: "
                            + ", ".join(f"`{g}`" for g in sorted(_orphaned))
                            + ". Their flies have nothing to be measured against, so "
                            "they are absent from both figures."
                        )

                    def _prc_sheets():
                        """Built on the click, not on every rerun — the summary is a
                        groupby over every fly."""
                        return [
                            ("per_fly", _treated),
                            (
                                "summary",
                                ps_module.summarize_phase_response(
                                    _treated, by=("zt", *_series_prc)
                                ),
                            ),
                            ("controls_per_fly", _pf[_pf["group"].isin(_ctrls)]),
                            ("left_out", _drop_treated),
                            (
                                "params",
                                pd.DataFrame(
                                    [{"parameter": k, "value": str(v)} for k, v in _rp.items()]
                                ),
                            ),
                        ]

                    export_helpers.save_excel_button(
                        "Save the phase response (.xlsx)",
                        _prc_sheets,
                        ds,
                        "phase_response",
                        key="ps_prc_xlsx",
                        help="Every fly's own response, the group means behind the "
                        "curve, the control flies the responses are measured against, "
                        "and what was left out.",
                    )

        with _tab_over_time:

            st.subheader("Results")
            if single_control:
                st.caption(
                    "Phase difference = each group's daily peak time against "
                    f"**{single_control}**'s on the same day."
                )
            else:
                st.caption(
                    "Phase difference = each group's daily peak time minus **its own matched "
                    "control**'s on the same day (the `control_group` column names it)."
                )
            st.caption(
                "Reported as **control − pulsed**, so a delay reads negative and an "
                "advance positive — the same sign convention as the phase response "
                "curve on the other tab."
            )

            tab_plot, tab_table = st.tabs(["Phase Difference by Day", "Table & Export"])

            with tab_plot:
                _params = gres["params"]
                _origin = _params.get("day_origin", "recording_start")
                _gvals = gres.get("group_values") or {}
                _gcols = list(_params.get("group_by", []))

                # A group is its own reference (difference 0 by construction) — nothing to plot.
                others = [
                    g
                    for g in sorted(per_day["group"].unique())
                    if control_map.get(g) not in (g, None)
                ]

                def _include_boxes(label, values, key, *, per_row=6):
                    """A row of tick boxes, all on by default. Returns the ticked values."""
                    state_key = f"_ps_inc_{key}"
                    saved = st.session_state.setdefault(state_key, {})
                    for v in values:
                        saved.setdefault(v, True)
                    st.markdown(f"**{label}**")
                    for start in range(0, len(values), per_row):
                        chunk = values[start : start + per_row]
                        cols = st.columns(per_row)
                        for col, v in zip(cols, chunk):
                            saved[v] = col.checkbox(str(v), value=saved[v], key=f"ps_inc_{key}_{v}")
                    return [v for v in values if saved[v]]

                # Every grouping column gets its own row of tick boxes, plus the box the
                # flies sat in. Everything starts ticked, so the default view is unchanged.
                _PLURAL = {
                    "genotype": "Genotypes",
                    "condition": "Conditions",
                    "sex": "Sexes",
                    "flybox": "Flyboxes",
                    "block": "Blocks",
                }

                def _pretty(col):
                    return _PLURAL.get(col, str(col).replace("_", " ").capitalize())

                # One figure per light-pulse condition, one line per genotype — the lab's layout.
                # Whichever column carries the series becomes the legend; the rest split figures.
                pc1, pc2 = st.columns(2)
                with pc1:
                    _series_default = (
                        "genotype" if "genotype" in _gcols else (_gcols[0] if _gcols else "")
                    )
                    series_col = st.selectbox(
                        "One line per",
                        _gcols,
                        index=_gcols.index(_series_default) if _series_default in _gcols else 0,
                        help="The factor compared within each figure. The legend shows its values.",
                    )
                with pc2:
                    _facet_opts = [c for c in _gcols if c != series_col]
                    facet_cols = st.multiselect(
                        "One figure per",
                        _facet_opts,
                        default=_facet_opts,
                        help="Each combination of these gets its own figure — e.g. condition and "
                        "sex gives one chart per pulse dose per sex.",
                    )

                # Only the factors that are actually pooled INTO a figure can be filtered
                # here. The faceting columns are excluded: each of their values already
                # gets a figure of its own, so unticking one is "do not draw that figure",
                # which is what the "One figure per" picker is for.
                _filterable = [c for c in _gcols if c != series_col and c not in facet_cols]
                _incl_values = {}
                for c in _filterable:
                    vals = sorted({str(_gvals.get(g, {}).get(c, "")) for g in others} - {""})
                    if len(vals) > 1:  # a column with one value filters nothing
                        _incl_values[c] = vals

                if _incl_values:
                    with st.expander("Include in graphs", expanded=False):
                        st.caption(
                            "Untick to leave a value out of the figures, the exports and the "
                            "plotted table. The underlying comparison is not recomputed — only "
                            "what gets drawn changes. A factor with just one value is not "
                            "shown, and nor is one that already has a figure of its own."
                        )
                        _keep = {}
                        for c, vals in _incl_values.items():
                            _keep[c] = _include_boxes(_pretty(c), vals, c)

                    _before = len(others)
                    for c, picked in _keep.items():
                        others = [
                            g for g in others if str(_gvals.get(g, {}).get(c, "")) in set(picked)
                        ]
                    if not others:
                        st.warning("Nothing ticked — every group has been filtered out.")
                        st.stop()
                    if len(others) < _before:
                        st.caption(f"Showing **{len(others)} of {_before}** groups.")

                # Error bars. The plotted point is the peak of the group's MEAN trace, so
                # there is no per-fly spread to average into a SEM — the uncertainty has to
                # come from resampling the flies and re-running the whole pipeline.
                eb1, eb2, eb3 = st.columns([2, 1, 1])
                with eb1:
                    show_err = st.checkbox(
                        "Error bars (bootstrap over flies)",
                        value=False,
                        key="ps_err_on",
                        help="Resamples the flies of every group with replacement and repeats "
                        "the whole analysis, so the interval covers the control arm as well "
                        "as the pulsed one. Takes a few seconds to a minute; the result is "
                        "cached until the comparison is re-run.",
                    )
                with eb2:
                    err_kind = st.selectbox(
                        "Bars show", ["95% CI", "±1 SE"], key="ps_err_kind", disabled=not show_err
                    )
                with eb3:
                    n_boot = st.selectbox(
                        "Resamples", [50, 100, 200, 500], index=1, key="ps_err_nboot",
                        disabled=not show_err,
                    )

                boot = None
                if show_err:
                    _sig = (
                        repr(sorted(gres["params"].items())),
                        int(n_boot),
                        repr(sorted(control_map.items())),
                    )
                    _cache = st.session_state.get("_ps_boot") or {}
                    if _cache.get("key") == _sig:
                        boot = _cache["df"]
                    else:
                        _bar = st.progress(0.0, text=f"Resampling flies ({n_boot} draws)…")
                        try:
                            boot = ps_module.bootstrap_group_phase_difference(
                                ds_pulse,
                                control_map,
                                n_boot=int(n_boot),
                                seed=0,
                                progress_callback=lambda f: _bar.progress(
                                    min(1.0, f), text=f"Resampling flies ({n_boot} draws)…"
                                ),
                                **{
                                    k: v
                                    for k, v in gres["params"].items()
                                    if k
                                    in (
                                        "group_by",
                                        "day_origin",
                                        "baseline_day",
                                        "difference_sign",
                                        "filter_hours",
                                        "peak_prominence_frac",
                                        "peak_distance_hours",
                                        "search_half_width_hours",
                                    )
                                },
                            )
                            st.session_state["_ps_boot"] = {"key": _sig, "df": boot}
                        except Exception as exc:
                            st.warning(f"Could not compute error bars: {exc}")
                            boot = None
                        finally:
                            _bar.empty()
                    st.session_state["_ps_boot_df"] = boot

                _rebased = _params.get("baseline_day") is not None
                y_col = (
                    "phase_difference_from_baseline_hours" if _rebased else "phase_difference_hours"
                )
                _sign = _params.get("difference_sign", "group_minus_control")
                # Just the quantity. Spelling out the sign convention and the rebasing day here
                # made a y-title long enough to be clipped out of the exported PNG, and it is
                # not a property of the axis anyway — both are set on this page, travel with
                # the figure in the workbook export, and are restated in the caption below.
                _y_title = "Phase difference (h)"
                _axis_note = "control − pulsed" + (
                    f", relative to day {_params['baseline_day']}" if _rebased else ""
                )
                # The pulse is given on the last entrained day and shows up in the NEXT day's
                # peak, so the marker belongs between the two days, not on one of them.
                _pulse_day = -1 if _origin == "dd_onset" else (_pulse_days[0] if _pulse_days else None)
                _pulse_x = None if _pulse_day is None else _pulse_day + 0.5

                palette = [
                    "#E8722C", "#16A085", "#E86FA0", "#3B76AF", "#8E6BBF",
                    "#B5892B", "#5B8C3E", "#C0453B", "#7F8C8D", "#00838F",
                ]
                _series_vals = sorted({str(_gvals.get(g, {}).get(series_col, g)) for g in others})
                _colour = {v: palette[i % len(palette)] for i, v in enumerate(_series_vals)}

                def _facet_key(grp):
                    return tuple(str(_gvals.get(grp, {}).get(c, "")) for c in facet_cols)

                _facets = {}
                for g in others:
                    _facets.setdefault(_facet_key(g), []).append(g)

                # The range has to cover the error bars too, or the widest ones get clipped
                # at the axis edge and read as though they stopped there.
                _plot_rows = per_day[per_day["group"].isin(others)]
                _extent = [np.asarray(_plot_rows[y_col], dtype=float)]
                if boot is not None and not _plot_rows.empty:
                    _st = y_col.replace("_hours", "")
                    _b = _plot_rows.merge(boot, on=["group", "day_index"], how="left")
                    _yv = np.asarray(_b[y_col], dtype=float)
                    if err_kind == "±1 SE":
                        _sd = np.asarray(_b[f"{_st}_boot_sd"], dtype=float)
                        _extent += [_yv - _sd, _yv + _sd]
                    else:
                        _extent += [
                            np.asarray(_b[f"{_st}_lo"], dtype=float),
                            np.asarray(_b[f"{_st}_hi"], dtype=float),
                        ]
                _y_all = np.concatenate([a.ravel() for a in _extent]) if _extent else np.array([])
                _y_all = _y_all[np.isfinite(_y_all)]
                if len(_y_all):
                    _pad = max(0.35, 0.08 * (float(_y_all.max()) - float(_y_all.min())))
                    _y_range = [float(_y_all.min()) - _pad, float(_y_all.max()) + _pad]
                else:
                    _y_range = None

                _n_of = (
                    per_day.drop_duplicates("group").set_index("group")["n_flies"].to_dict()
                    if "n_flies" in per_day.columns
                    else {}
                )

                def _n(grp):
                    n = _n_of.get(grp)
                    return int(n) if n is not None and np.isfinite(n) else None

                _stem = y_col.replace("_hours", "")

                def _err_arrays(sub, grp):
                    """(below, above) distances for one group's days, or (None, None)."""
                    if boot is None or sub.empty:
                        return None, None
                    b = boot[boot["group"] == grp].set_index("day_index")
                    if b.empty:
                        return None, None
                    y = sub[y_col].to_numpy(dtype=float)
                    days_idx = sub["day_index"].to_numpy()
                    if err_kind == "±1 SE":
                        sd = b[f"{_stem}_boot_sd"].reindex(days_idx).to_numpy(dtype=float)
                        return sd, sd
                    lo = b[f"{_stem}_lo"].reindex(days_idx).to_numpy(dtype=float)
                    hi = b[f"{_stem}_hi"].reindex(days_idx).to_numpy(dtype=float)
                    # plotly wants distances from the point, never absolute bounds
                    return np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)

                for key in sorted(_facets):
                    # Facet VALUES only — the column names ("condition ZT15…") were three
                    # quarters of the title and said nothing the values do not.
                    title = " · ".join(str(v) for v in key) or "All groups"
                    _grps = sorted(_facets[key])
                    fig = go.Figure()
                    for grp in _grps:
                        sub = per_day[per_day["group"] == grp].sort_values("day_index")
                        name = str(_gvals.get(grp, {}).get(series_col, grp))
                        _elo, _ehi = _err_arrays(sub, grp)
                        fig.add_trace(
                            go.Scatter(
                                x=sub["day_index"],
                                y=sub[y_col],
                                mode="lines+markers",
                                name=f"{name} (n={_n(grp)})",
                                line=dict(color=_colour.get(name), width=2.5),
                                marker=dict(size=8),
                                error_y=(
                                    None
                                    if _elo is None
                                    else dict(
                                        type="data",
                                        symmetric=False,
                                        array=_ehi,
                                        arrayminus=_elo,
                                        thickness=1.4,
                                        width=4,
                                        color=_colour.get(name),
                                    )
                                ),
                                hovertemplate=(
                                    "day %{x}<br>%{y:.2f} h<br>"
                                    + "<br>vs "
                                    + str(control_map.get(grp))
                                    + "<extra>"
                                    + name
                                    + "</extra>"
                                ),
                            )
                        )
                    fig.add_hline(y=0, line_color="gray", line_width=1)
                    if _pulse_x is not None:
                        fig.add_vline(
                            x=_pulse_x,
                            line_dash="dash",
                            line_color="#555555",
                            annotation_text="light pulse",
                            annotation_position="top",
                        )
                    fig.update_layout(
                        title=dict(text=title, font=dict(size=21)),
                        xaxis_title=("days from first DD day" if _origin == "dd_onset" else "day"),
                        yaxis_title=_y_title,
                        # An explicit width is what the PNG export uses; without it kaleido
                        # falls back to 700 px, and since the legend needs a fixed ~300 px for
                        # "Hr38_OE_guide (n=29)" it swallowed nearly half the image and left
                        # the plot squeezed into the rest. On screen Streamlit overrides the
                        # width, so this only sizes the saved file.
                        width=1250,
                        height=560,
                        # Base size for ticks and the "light pulse" annotation. The legend is
                        # pinned SMALLER on purpose: its entries are the longest text on the
                        # figure (genotype + boxes + n), so growing it with everything else
                        # is what eats the plot area.
                        font=dict(size=15),
                        legend=dict(
                            title=dict(text=series_col, font=dict(size=13)),
                            font=dict(size=12),
                        ),
                        margin=dict(t=70, l=100, r=30, b=80),
                    )
                    # zeroline=False: plotly draws a vertical rule at x=0, which on this axis
                    # is the first DD day — a second, unlabelled marker sitting right beside
                    # the dashed light-pulse line and easily read as part of the protocol.
                    fig.update_xaxes(
                        dtick=1, zeroline=False, title_font=dict(size=17), tickfont=dict(size=15)
                    )
                    fig.update_yaxes(title_font=dict(size=17), tickfont=dict(size=15))
                    if _y_range:
                        # A shared y-range across figures keeps the doses visually comparable.
                        fig.update_yaxes(range=_y_range)
                    charts.plotly_chart(fig, filename=f"phase_shift_{title}", width="stretch")
                    # The axis names only the quantity; which way round the subtraction goes
                    # and which day was zeroed are choices made above, so they are stated
                    # here rather than crammed into the label.
                    st.caption(f"Difference direction: {_axis_note}.")

                    # Which group each line is measured against, spelled out. The
                    # legend has room for a line's own name and its n, not for its
                    # control's, and "vs its matched control" is not an answer when
                    # four genotypes are on screen.
                    _note = [
                        f"- **{_gvals.get(g, {}).get(series_col, g)}** (n={_n(g)}) "
                        f"vs **{control_map.get(g)}** (n={_n(control_map.get(g))})"
                        for g in _grps
                    ]
                    if _note:
                        st.markdown(
                            "**What is compared with what**\n\n" + "\n".join(_note)
                        )

                # what actually got drawn, for the workbook in the table tab
                st.session_state["_ps_plotted_groups"] = list(others)
                st.session_state["_ps_y_col"] = y_col
                st.session_state["_ps_y_title"] = _y_title

                if _rebased:
                    st.caption(
                        f"Every group is set to zero on day {_params['baseline_day']}, so what each "
                        "line shows is the change from that day — the phase offset the two cohorts "
                        "started with has been taken out. The raw differences are in the table tab."
                    )
                _unmatched = [
                    g for g in sorted(per_day["group"].unique()) if control_map.get(g) is None
                ]
                if _unmatched:
                    st.caption(
                        "Not plotted (no matching control): "
                        + ", ".join(f"`{g}`" for g in _unmatched)
                        + "."
                    )
                st.caption(
                    "Rows flagged `filter_edge` are each group's first and last recorded day: the "
                    "smoothing filter pads the ends of the record, so those peaks rest partly on "
                    "synthetic padding and their positions are not well determined. The first one is "
                    "a pre-pulse baseline day and carries no shift information — read the trend from "
                    "the day after it."
                )

            with tab_table:
                # Group ids exactly as the analysis keys on them. Every label used to
                # be suffixed with the flybox its flies sat in, with the raw id kept
                # in a second column — so the exported `group` column disagreed with
                # the one every other sheet, page and figure uses.
                st.dataframe(per_day, width="stretch")

                def _phase_workbook_sheets():
                    """The sheets of the per-day workbook.

                    Passed to save_excel_button as a CALLABLE, so it runs on the click rather
                    than on every rerun — it merges the bootstrap frame and pivots a matrix,
                    which is real work to do for a button nobody has pressed.
                    """
                    _plotted = st.session_state.get("_ps_plotted_groups") or []
                    _ycol = st.session_state.get("_ps_y_col", "phase_difference_hours")
                    _boot = st.session_state.get("_ps_boot_df")

                    def _with_boot(df):
                        """Attach the bootstrap interval when error bars were computed."""
                        if _boot is None or df.empty:
                            return df
                        return df.merge(_boot, on=["group", "day_index"], how="left")

                    sheets = [("per_day", _with_boot(per_day))]
                    if _plotted:
                        sub = per_day[per_day["group"].isin(_plotted)].copy()
                        sheets.append(("per_day_plotted", _with_boot(sub)))
                        # A day-by-group grid: the layout to read a shift off by eye, and the
                        # one that pastes straight into a figure or a stats package.
                        if not sub.empty:
                            _mat = sub.copy()
                            sheets.append((
                                "matrix",
                                _mat.pivot_table(
                                    index="day_index", columns="group", values=_ycol, aggfunc="mean"
                                ).reset_index(),
                            ))
                    sheets.append((
                        "controls",
                        pd.DataFrame(
                            [
                                {
                                    "group": g,
                                    "compared against": c or "— none —",
                                    "group_id": g,
                                    "control_group_id": c or "",
                                }
                                for g, c in sorted(control_map.items())
                            ]
                        ),
                    ))
                    sheets.append((
                        "parameters",
                        pd.DataFrame(
                            [{"parameter": k, "value": str(v)} for k, v in sorted(gres["params"].items())]
                        ),
                    ))
                    _sign_txt = (
                        "control minus pulsed (a delay reads negative, an advance positive)"
                    )
                    sheets.append((
                        "notes",
                        pd.DataFrame(
                            [
                                {
                                    "field": "value column",
                                    "value": st.session_state.get("_ps_y_title", _ycol),
                                },
                                {"field": "difference", "value": _sign_txt},
                                {"field": "day origin", "value": str(gres["params"].get("day_origin"))},
                                {
                                    "field": "baseline day",
                                    "value": str(gres["params"].get("baseline_day")),
                                },
                                {
                                    "field": "filter_edge rows",
                                    "value": "each group's first and last recorded day; the "
                                    "smoothing filter pads the ends, so those peak positions are "
                                    "not well determined",
                                },
                                {
                                    "field": "groups plotted",
                                    "value": ", ".join(_plotted)
                                    if _plotted
                                    else "(figures not built yet)",
                                },
                            ]
                        ),
                    ))
                    return sheets

                export_helpers.save_excel_button(
                    "Save workbook to working folder (.xlsx)",
                    _phase_workbook_sheets,
                    ds,
                    "phase_difference_per_day.xlsx",
                    key="ps_xlsx",
                    help="per_day (every group), per_day_plotted (only what the figures show), "
                    "matrix (day x group grid), controls, parameters, notes.",
                )
                st.download_button(
                    "Download per-day (CSV)",
                    per_day.to_csv(index=False).encode("utf-8"),
                    file_name="phase_difference_vs_control_per_day.csv",
                    mime="text/csv",
                )
                st.markdown("**Reference used**")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"group": g, "compared against": c or "— none —"}
                            for g, c in sorted(control_map.items())
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )

        # The two references are mutually exclusive workflows, not two halves of one.
        # This used to be a bare st.stop() at the end of the control branch, which left
        # everything below unreachable in that mode — dead by position, not by
        # structure. Same behaviour, stated explicitly.
    else:
        # ============================================================
        # Section 3: Preview one fly before committing to the cohort
        # ============================================================
        st.subheader("Preview")
        st.caption(
            "Check one fly's markers and fits before running the whole cohort — if the markers are "
            "landing in the wrong place, the parameters need adjusting, not the cohort re-run."
        )

        fly_ids = [str(i) for i in ds_pulse["id"].values]
        pulsed_ids = [f for f, m in zip(fly_ids, pulse_minutes) if np.isfinite(m)]
        preview_id = st.selectbox("Fly", pulsed_ids, index=0)

        if st.checkbox("Show preview actogram", value=True):
            with st.spinner("Computing preview..."):
                try:
                    one = ds_pulse.sel(id=[preview_id])
                    res1 = ps_module.compute_phase_shift_analysis(one, **params)
                    row1 = res1["per_fly"].iloc[0]
                    minutes = np.asarray(one["time"].values, dtype=float)
                    values = np.asarray(one["activity"].transpose("time", "id").values[:, 0], dtype=float)
                    shift_txt = (
                        f"{row1['phase_shift_hours']:+.2f} h"
                        if row1["status"] == "ok"
                        else f"no value ({row1['status']})"
                    )
                    fig1 = plotting.phase_shift_actogram(
                        minutes,
                        values,
                        day_markers=res1["day_markers"].get(preview_id),
                        pre_fit=res1["fits"].get(preview_id, {}).get("pre"),
                        post_fit=res1["fits"].get(preview_id, {}).get("post"),
                        pulse_minute=row1.get("pulse_minute"),
                        pulse_duration_minutes=(
                            float(one["pulse_duration_minutes"].values[0])
                            if "pulse_duration_minutes" in one.coords
                            else None
                        ),
                        reference_day_index=(
                            int(row1["reference_day_index"])
                            if np.isfinite(row1.get("reference_day_index", np.nan))
                            else None
                        ),
                        title=f"{preview_id} — {method} method, shift {shift_txt}",
                    )
                    charts.plotly_chart(fig1, width="stretch")
                    if row1["status"] == "ok":
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Phase shift", f"{row1['phase_shift_hours']:+.2f} h")
                        m2.metric("Period before", f"{row1['pre_period_hours']:.2f} h")
                        m3.metric("Period after", f"{row1['post_period_hours']:.2f} h")
                    else:
                        st.warning(f"No phase shift for this fly: **{row1['status']}**")
                except Exception as exc:
                    st.error(f"Preview failed: {exc}")

        # ============================================================
        # Section 4: Run
        # ============================================================
        if st.button("Run Phase Shift Analysis", type="primary"):
            with st.spinner(f"Measuring phase shifts for {n_total} flies..."):
                try:
                    results = ps_module.compute_phase_shift_analysis(ds_pulse, **params)
                    st.session_state.phase_shift_results = results

                    # Record the parameters (scalars only) so the run is reproducible and the
                    # sidebar can report the analysis as done. Results themselves stay in
                    # session state — same convention as the Sleep Deprivation page.
                    for key, value in results["params"].items():
                        if value is None:
                            continue
                        ds.attrs[f"phase_shift_{key}"] = int(value) if isinstance(value, bool) else value
                    ds.attrs["phase_shift_method"] = results["method"]
                    st.session_state.dataset = ds
                    # Phase shift records its results in attrs, not data_vars — re-detect
                    # so the home page's status grid reflects the run.
                    status.refresh(ds)

                    n_ok = int((results["per_fly"]["status"] == "ok").sum())
                    st.success(f"Done: {n_ok} of {len(results['per_fly'])} flies produced a phase shift.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Error: {exc}")
                    st.stop()

        # ============================================================
        # Section 5: Results
        # ============================================================
        if "phase_shift_results" not in st.session_state:
            st.info("Set the method above and click **Run** to analyse the whole cohort.")
            st.stop()

        results = st.session_state.phase_shift_results
        per_fly = results["per_fly"]

        st.subheader("Results")

        tab_summary, tab_actogram, tab_export = st.tabs(["Summary", "Per-Fly Actogram", "Data Export"])

        _COLORS = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        with tab_summary:
            ok = per_fly[per_fly["status"] == "ok"]
            status_counts = per_fly["status"].value_counts()

            if len(ok) == 0:
                st.warning("No fly produced a usable phase shift.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Flies with a shift", f"{len(ok)} / {len(per_fly)}")
                c2.metric("Median shift", f"{ok['phase_shift_hours'].median():+.2f} h")
                c3.metric("Median period after", f"{ok['post_period_hours'].median():.2f} h")

                groups = sorted(ok["group"].dropna().unique())
                fig = go.Figure()
                for i, grp in enumerate(groups):
                    vals = ok.loc[ok["group"] == grp, "phase_shift_hours"]
                    fig.add_trace(
                        go.Box(
                            y=vals,
                            name=str(grp),
                            boxpoints="all",
                            jitter=0.4,
                            pointpos=0,
                            marker=dict(color=_COLORS[i % len(_COLORS)]),
                        )
                    )
                fig.add_hline(y=0, line_dash="dot", line_color="gray")
                plotting.apply_category_ticks(fig, groups)
                fig.update_layout(
                    title="Phase shift by group (positive = delay, negative = advance)",
                    yaxis_title="phase shift (hours)",
                    showlegend=False,
                    height=460,
                )
                charts.plotly_chart(fig, width="stretch")

            if len(status_counts) > 1 or "ok" not in status_counts:
                st.markdown("**Why some flies have no value**")
                _explain = {
                    "ok": "phase shift measured",
                    "no_pulse": "no pulse_time in the metadata (unpulsed control)",
                    "pulse_outside_record": "the pulse time falls outside this fly's recording",
                    "insufficient_pre": "too few usable days before the pulse",
                    "insufficient_post": "too few usable days after the pulse",
                    "implausible_period": "the fitted period was not circadian — the daily markers "
                    "did not track a consistent rhythm",
                }
                for status, count in status_counts.items():
                    st.markdown(f"- `{status}` — {count} fly/flies: {_explain.get(status, '')}")

            st.dataframe(per_fly, width="stretch")

        with tab_actogram:
            st.caption(
                "The dashed line continues the pre-pulse rhythm across the pulse. The gap between it "
                "and the post-pulse line is the reported shift, so the number can be checked by eye."
            )
            view_ids = per_fly["fly_id"].tolist()
            view_id = st.selectbox("Fly", view_ids, index=0, key="actogram_fly")
            row = per_fly.set_index("fly_id").loc[view_id]

            if view_id not in results["day_markers"]:
                st.warning(f"No markers were detected for this fly (status `{row['status']}`).")
            else:
                one = ds_pulse.sel(id=[view_id])
                minutes = np.asarray(one["time"].values, dtype=float)
                values = np.asarray(one["activity"].transpose("time", "id").values[:, 0], dtype=float)
                shift_txt = (
                    f"{row['phase_shift_hours']:+.2f} h"
                    if row["status"] == "ok"
                    else f"no value ({row['status']})"
                )
                fig = plotting.phase_shift_actogram(
                    minutes,
                    values,
                    day_markers=results["day_markers"].get(view_id),
                    pre_fit=results["fits"].get(view_id, {}).get("pre"),
                    post_fit=results["fits"].get(view_id, {}).get("post"),
                    pulse_minute=row.get("pulse_minute"),
                    pulse_duration_minutes=(
                        float(one["pulse_duration_minutes"].values[0])
                        if "pulse_duration_minutes" in one.coords
                        else None
                    ),
                    reference_day_index=(
                        int(row["reference_day_index"])
                        if np.isfinite(row.get("reference_day_index", np.nan))
                        else None
                    ),
                    title=f"{view_id} ({row['group']}) — {results['method']} method, shift {shift_txt}",
                )
                charts.plotly_chart(fig, width="stretch")

        with tab_export:
            st.download_button(
                "Download per-fly phase shifts (CSV)",
                per_fly.to_csv(index=False).encode("utf-8"),
                file_name=f"phase_shift_{results['method']}_per_fly.csv",
                mime="text/csv",
            )

            marker_rows = [
                {"fly_id": fly, "day_index": day, "marker_minute": minute}
                for fly, markers in results["day_markers"].items()
                for day, minute in sorted(markers.items())
            ]
            if marker_rows:
                markers_df = pd.DataFrame(marker_rows)
                st.download_button(
                    "Download daily phase markers (CSV)",
                    markers_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"phase_shift_{results['method']}_daily_markers.csv",
                    mime="text/csv",
                    help="The per-day marker times the fits were built from — useful for checking "
                    "a suspicious result.",
                )

