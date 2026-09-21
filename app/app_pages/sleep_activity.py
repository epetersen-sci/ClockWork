"""
Activity & Sleep — the descriptive plots, one tab per measure.

**Activity**: the daily pattern, then the day/night totals behind it.
**Sleep**: the same two, plus the bout-duration curves, and the sleep detection
that produces all of them. Every chart has a matching group-level and per-fly
CSV export.

The three old tabs were cut by kind of plot — daily profiles, bouts, day/night
totals — so each held half an activity answer and half a sleep one, and reading
"what does sleep look like in this genotype" meant visiting all three and
assembling it yourself.

Sleep detection lives on the Sleep tab rather than on a page of its own, via
``ui.sleep_run``. It is the only thing on this page that writes to the dataset,
and it does so through the MASTER, never through the group-filtered view bound
below — which is precisely the hazard that used to justify a separate page. See
that module's docstring.

ONE EPOCH AT A TIME, chosen in the sidebar and defaulting to LD. This page used
to bin the whole recording onto a single ZT axis with no epoch selection at
all — harmless only while sleep could be computed for one epoch at a time,
because the other epoch was then all missing minutes and dropped out of its
own accord. Once sleep analysis could run on both, that silently became an
average of entrained and free-running days: under DD a fly runs at its own
period, so its subjective day drifts against the 24 h axis, smearing the DD
structure across the clock AND diluting the LD profile it was averaged into.
The more DD days in the record, the flatter the result, for the wrong reason.

There is deliberately no pooled option. The epoch is a viewing choice, and
"both at once" is not a meaningful average of the two.
"""


import numpy as np
import pandas as pd
import streamlit as st

import dam_utilities
import export_helpers as ex
import plotting
import sleep_analysis
from analysis_detection import detect_analyses
from dam_utilities import select_phase
from dataset_meta import (
    PHASE_DD,
    PHASE_LD,
    dataset_fingerprint,
    dataset_phase,
)
from ui import charts, sleep_run
from ui.filters import DISPLAY_GROUPS_KEY, bin_size_sidebar, group_filter_sidebar
from ui.guards import require_dataset

ds = require_dataset()
# ----------------------------------------------------------------
# Cached helpers — heavy plot/aggregation calls are wrapped so a
# Streamlit rerun (sidebar toggle, widget change) doesn't recompute
# them. The first arg of every cached function is a fingerprint tuple
# (see core/dataset_meta.dataset_fingerprint); Streamlit hashes that
# tuple while the leading-underscore ``_ds`` arg is NOT hashed (per
# Streamlit's caching convention). Returning plotly figures from a
# cached function is supported — figures pickle cleanly.
#
# The fingerprint parameter is named ``fp``, NOT ``_fp``. Streamlit's
# rule is purely syntactic — ANY leading-underscore parameter is left
# out of the cache key, not just the dataset one — so naming it ``_fp``
# excluded the very thing it exists to key on, and every one of these
# caches then ignored which flies were in ``ds``. The visible symptom
# was the sidebar group filter appearing to do nothing: narrow the
# groups and the plots kept their old traces until some other cache
# input (e.g. bin size) happened to change.
# ----------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _cached_zt_binned(fp, _ds, value_col, bin_size_minutes):
    """Wrap dam_utilities.get_zt_binned_dataframe with a fingerprint key."""
    return dam_utilities.get_zt_binned_dataframe(_ds, value_col, bin_size_minutes)

@st.cache_data(show_spinner=False)
def _cached_per_fly_summary(
    fp, _ds, variable, selected_genotypes, selected_temperatures, bin_size_minutes
):
    """Per-fly All Day / Day / Night totals — the rows every totals panel draws.

    Cached because it is now read three times per measure (one figure each) and a
    fourth time for the export, where the bars it replaced computed a group mean
    once. The group table beside it stays its own cache: it is a different
    aggregation of the same flies, and both are cheap to keep.
    """
    return plotting.per_fly_summary_table(
        _ds,
        variable,
        selected_genotypes=list(selected_genotypes) if selected_genotypes else None,
        selected_temperatures=list(selected_temperatures) if selected_temperatures else None,
        bin_size_minutes=bin_size_minutes,
    )

@st.cache_data(show_spinner=False)
def _cached_daily_pattern(
    fp,
    _ds,
    variable,
    title,
    selected_genotypes,
    selected_temperatures,
    phase_label,
    bin_size_minutes,
):
    """Cache the daily-pattern line plot. ``phase_label`` and
    ``bin_size_minutes`` are part of the cache key, so both DD/LD relabeling and
    the sidebar Bin size control correctly invalidate the cached figure."""
    return plotting.daily_pattern_line(
        _ds,
        variable,
        title,
        selected_genotypes=list(selected_genotypes) if selected_genotypes else None,
        selected_temperatures=list(selected_temperatures) if selected_temperatures else None,
        phase_label=phase_label,
        bin_size_minutes=bin_size_minutes,
    )

@st.cache_data(show_spinner=False)
def _cached_summary_table(
    fp, _ds, variable, selected_genotypes, selected_temperatures, bin_size_minutes, phase_label
):
    """Per-group summary table (mean + SEM per period) for CSV export — the
    SAME numbers plotting.summary_bars draws (one computation, no drift)."""
    return plotting.summary_table(
        _ds,
        variable,
        selected_genotypes=list(selected_genotypes) if selected_genotypes else None,
        selected_temperatures=list(selected_temperatures) if selected_temperatures else None,
        bin_size_minutes=bin_size_minutes,
        phase_label=phase_label,
    )

@st.cache_data(show_spinner=False)
def _cached_bout_duration_lines(
    fp, _ds, method, selected_genotypes, selected_temperatures, show_individual
):
    """Compute the per-fly bout-duration curves, the per-fly summary and the
    group-comparison test, then render them — the same numbers drive the plot and
    both CSV exports below, so they can't drift apart.

    The three sleep_analysis calls are HERE rather than inside
    plotting.sleep_bout_duration_lines, which used to defer-import them. That
    deferred import existed only to dodge a circular one, and this page was the
    only caller of all three functions, so the analysis was in effect being run
    by the plotting module. Compute in the caller, pass frames to the renderer.
    """
    _genos = list(selected_genotypes) if selected_genotypes else None
    _temps = list(selected_temperatures) if selected_temperatures else None
    empty_curves = pd.DataFrame(columns=["id", "group", "x", "y"])
    empty_summary = pd.DataFrame(
        columns=["id", "group", "n_bouts", "median_duration_min", "log_mean_duration_min"]
    )
    if _ds is None or "duration" not in _ds.data_vars:
        return (
            plotting.sleep_bout_duration_lines(empty_curves, method=method),
            empty_curves,
            empty_summary,
            None,
        )

    curves_df = sleep_analysis.per_fly_bout_duration_curves(
        _ds, method=method, selected_genotypes=_genos, selected_temperatures=_temps
    )
    summary_df = sleep_analysis.bout_duration_summary(
        _ds, selected_genotypes=_genos, selected_temperatures=_temps
    )
    stats_result = (
        sleep_analysis.bout_duration_group_stats(summary_df) if not curves_df.empty else None
    )
    fig = plotting.sleep_bout_duration_lines(
        curves_df,
        stats_result,
        method=method,
        show_individual=show_individual,
    )
    return fig, curves_df, summary_df, stats_result

def _summary_frame(tbl):
    """The per-group summary table with headers a reader recognises.

    A frame rather than the CSV string it used to return, because it is now a
    SHEET in a workbook beside the per-fly rows it is the mean of. Two buttons
    meant choosing twice and then remembering which file was which.
    """
    return tbl.rename(
        columns={
            "group": "Group",
            "n": "n_flies",
            "All Day": "All Day mean (min)",
            "All Day_sem": "All Day SEM (min)",
            "Day Only": "Day mean (min)",
            "Day Only_sem": "Day SEM (min)",
            "Night Only": "Night mean (min)",
            "Night Only_sem": "Night SEM (min)",
        }
    )



# The sidebar filter narrows a page-LOCAL view for plotting only. It used to
# be a hazard: sleep analysis lived above it on this same page and wrote back
# to the master dataset, so running it from below the filter would have
# dropped deselected flies for good. That block is now its own page.
analyses = detect_analyses(ds)

# ---------------------------------------------------------------------------
# Epoch selection. First in the sidebar because it governs everything below it,
# and applied before the group filter so the fingerprint the caches key on
# already reflects the epoch.
#
# A dataset stamped with a single epoch can only be shown in that epoch, and
# one with no LD/DD boundary has no epochs to choose between — neither gets a
# control it cannot honour.
# ---------------------------------------------------------------------------
_stamped = dataset_phase(ds)
_has_epochs = ("split_minute" in ds.coords) or ("first_DD_day" in ds.coords)

if _stamped in (PHASE_LD, PHASE_DD):
    phase_used = _stamped
    st.sidebar.caption(f"Phase: **{phase_used}** (this dataset holds only that epoch).")
elif not _has_epochs:
    phase_used = None  # nothing to select; the record is one undivided block
else:
    _options = [PHASE_LD, PHASE_DD]
    _phase = st.sidebar.radio(
        "Phase",
        _options,
        index=0,  # LD: sleep is conventionally read under the light cycle
        key="sleep_activity_phase",
        help=(
            "Which epoch these figures cover. Sleep is conventionally read "
            "under LD, so that is the default; DD shows the same measures on "
            "subjective time. There is no combined option — averaging "
            "entrained and free-running days onto one 24 h axis describes "
            "neither."
        ),
    )
    try:
        ds, phase_used = select_phase(ds, phase=_phase)
    except (ValueError, KeyError) as exc:
        st.error(f"Cannot show the {_phase} epoch of this dataset: {exc}")
        st.stop()

if phase_used is None:
    st.caption(
        "This dataset has no LD/DD boundary, so these figures cover the whole "
        "recording. Apply the split on **Data → Curate & split** to choose an "
        "epoch here."
    )
else:
    _other = PHASE_DD if phase_used == PHASE_LD else PHASE_LD
    st.caption(
        f"Data shown is from the **{phase_used}** dataset. To change to the "
        f"**{_other}** dataset, use the selector in the sidebar."
    )

# The unified group axis: filter on the single `group` coord defined by the
# metadata columns chosen at import, not on separate per-genotype /
# per-temperature axes. Selecting a subset narrows every plot below.
_grp_cols = dam_utilities.get_group_columns(ds)
_group_vals, _all_groups, selected_groups, ds = group_filter_sidebar(
    ds,
    key=DISPLAY_GROUPS_KEY,
    label="Groups",
    subset=True,
    help=(
        "Groups are defined by the metadata column(s) chosen at import"
        + (f": {', '.join(_grp_cols)}." if _grp_cols else ".")
    ),
)
# The unified group axis supersedes the old per-genotype / per-temperature filters;
# plots receive the already-group-filtered dataset with no further per-column slicing.
selected_genotypes = None
selected_temperatures = None

bin_size = bin_size_sidebar(key="viz_bin_size")

# One tab per MEASURE, not per kind of plot. The three old tabs — daily
# profiles, bouts, day/night totals — each held half an activity answer and half
# a sleep one, so "what does sleep look like here" meant visiting all three and
# assembling it yourself. Each tab now runs from the daily pattern down to the
# totals for one measure, which is the order the question is actually asked in.
# Keyed so the selection lives in session state and SURVIVES A RERUN. Detecting
# sleep ends in one, and without the key that rerun dropped you back on the
# Activity tab — away from the button you had just pressed and from the figures
# it had just produced. (No on_change="rerun" with it: that makes the tabs lazy,
# so only the open tab is drawn, and the page's PNG export would then silently
# cover half of what you thought was on screen.)
tab_activity, tab_sleep = st.tabs(["Activity", "Sleep"], key="sleep_activity_tab")

# Which half of the cycle the totals are named after. Hoisted above both tabs
# because both use it, and because it depends on the epoch chosen in the sidebar
# rather than on anything either tab does.
_ds_phase_label = phase_used or dataset_phase(ds)
_summary_subhead_suffix = (
    " (DD — subjective time)"
    if _ds_phase_label == PHASE_DD
    else (" (LD — Day/Night)" if _ds_phase_label == PHASE_LD else " (Day/Night)")
)


def _totals_violins(variable, y_label, key):
    """Three panels for one measure: total, day and night, groups on the x axis.

    ONE measure per panel. The first pass at this put day and night side by side in
    a single figure, which made the day-versus-night step the salient comparison
    and left group-versus-group as the hard one — backwards, since the experiment
    varies genotype, not time of day.

    Violins rather than bars: a bar was a group mean with a SEM whisker, four
    numbers standing in for thirty flies, and two groups can share both and still
    be obviously different. The per-fly values are computed either way — a group
    mean IS their mean — so the distribution costs only the drawing.
    """
    per_fly = _cached_per_fly_summary(
        dataset_fingerprint(ds),
        ds,
        variable,
        tuple(selected_genotypes) if selected_genotypes else None,
        tuple(selected_temperatures) if selected_temperatures else None,
        bin_size,
    )
    if per_fly.empty:
        st.info(f"No {variable} data to summarise for the current selection.")
        return

    # Under DD both halves are subjective, so they are named for the subjective
    # cycle rather than for a light cycle that was not running.
    _dd = _ds_phase_label == PHASE_DD
    panels = [
        ("All Day", "Total"),
        ("Day Only", "Subjective day" if _dd else "Day"),
        ("Night Only", "Subjective night" if _dd else "Night"),
    ]
    # One group order for all three, so the panels line up when read down the page.
    _order = sorted(per_fly["Group"].astype(str).unique())
    for i, (col, name) in enumerate(panels):
        fig, _ = plotting.group_violins(
            per_fly,
            col,
            title=f"{variable.capitalize()} — {name}",
            y_title=y_label,
            colour=plotting.MEASURE_COLOURS[i],
            groups=_order,
        )
        charts.plotly_chart(fig, width="stretch", theme=None)

    tbl = _cached_summary_table(
        dataset_fingerprint(ds),
        ds,
        variable,
        tuple(selected_genotypes) if selected_genotypes else None,
        tuple(selected_temperatures) if selected_temperatures else None,
        bin_size,
        _ds_phase_label,
    )
    # One workbook rather than two buttons. The group summary and the per-fly rows
    # it is the mean of are read together, and two CSVs meant choosing twice and
    # then remembering which file was which.
    ex.save_excel_button(
        f"Save {variable.capitalize()} totals (.xlsx)",
        [("summary", None if tbl.empty else _summary_frame(tbl)), ("per_fly", per_fly)],
        ds,
        f"{variable}_totals",
        key=f"dl_{key}_totals",
        help="Two sheets: the group mean ± SEM per period, and the per-fly values "
        "behind it.",
    )


with tab_activity:
    # ============================================================
    # Daily Activity Pattern
    # ============================================================
    st.subheader("Daily Activity Pattern")
    if "activity" in ds.data_vars:
        _ds_fp = dataset_fingerprint(ds)
        fig = _cached_daily_pattern(
            _ds_fp,
            ds,
            "activity",
            "Daily Activity Pattern",
            tuple(selected_genotypes) if selected_genotypes else None,
            tuple(selected_temperatures) if selected_temperatures else None,
            phase_used or dataset_phase(ds),
            bin_size,
        )
        # theme=None: let the figure's own styling (black text, transparent bg) drive both
        # the on-screen chart and the "Download plot as PNG" export (see daily_pattern_line).
        charts.plotly_chart(fig, width="stretch", theme=None)

        # CSV download of binned data (grouped: mean, SD, n per condition)
        try:
            binned_df = _cached_zt_binned(_ds_fp, ds, "activity", bin_size)
            _id_to_group = {}
            for _fid in binned_df["id"].unique():
                try:
                    _fly = ds.sel(id=_fid)
                    if "group" in ds.coords:
                        _id_to_group[_fid] = str(_fly["group"].item())
                    elif "genotype" in ds.coords and "temperature" in ds.coords:
                        _id_to_group[_fid] = f"{_fly['genotype'].item()}-{_fly['temperature'].item()}"
                    else:
                        _id_to_group[_fid] = "All"
                except Exception:
                    _id_to_group[_fid] = "All"
            binned_df["group"] = binned_df["id"].map(_id_to_group)
            # Same builder as the Export page's ZT table: Mean/SD/N per group, in
            # GraphPad's grouped-table order. This used to sort_index the columns
            # instead, which gave the alphabetical Mean/N/SD — a different header
            # row for the same quantity.
            _pivot = ex.zt_group_summary_table(binned_df, "activity", bin_size)
            # Per-fly binned time course (long) so other stats can be computed:
            # one row per fly per ZT bin (ID, Group, zt_bin_minute, zt_hours, activity).
            _act_pf = binned_df.rename(columns={"id": "ID", "group": "Group"}).copy()
            _act_pf["zt_hours"] = dam_utilities.zt_bin_to_hours(
                _act_pf["zt_bin_minute"], bin_size
            )
            _act_pf = (
                _act_pf[["ID", "Group", "zt_bin_minute", "zt_hours", "activity"]]
                .sort_values(["Group", "ID", "zt_bin_minute"])
                .reset_index(drop=True)
            )
            ex.save_excel_button(
                "Save binned activity (.xlsx)",
                [
                    ("group_summary", _pivot),
                    ("per_fly", _act_pf),
                ],
                ds,
                "activity_binned",
                key="dl_act",
                help="Two sheets: group mean ± SD ± N per ZT bin in GraphPad's "
                "grouped-table order, and the per-fly time course behind it.",
            )
        except Exception:
            pass

    st.divider()

    st.subheader(f"Activity Summary{_summary_subhead_suffix}")
    if _ds_phase_label == PHASE_DD:
        st.caption(
            "**DD note:** the bin labels are subjective time relative to the "
            "last lights-on transition (CT). Anchoring is reliable when the "
            "DD split was applied with a clean discard-first-DD-day boundary. "
            "If your recording had large data gaps at the DD start, the CT "
            "alignment may drift — verify the actogram before publishing."
        )
    if "activity" in ds.data_vars:
        _totals_violins("activity", "activity (counts)", "act")


with tab_sleep:
    # The analysis that produces everything below it, offered where the answer is
    # wanted rather than on a page of its own. ui.sleep_run computes on the MASTER,
    # never on this page's group-filtered view — see that module's docstring for
    # why that distinction is load-bearing here of all places.
    sleep_run.show_last_message("sleep_activity")
    sleep_run.ensure_sleep(key_prefix="sleep_activity")
    st.divider()

    if analyses["sleep"]:
        st.subheader("Daily Sleep Pattern")
        _ds_fp_sl = dataset_fingerprint(ds)
        fig = _cached_daily_pattern(
            _ds_fp_sl,
            ds,
            "sleep",
            "Daily Sleep Pattern",
            tuple(selected_genotypes) if selected_genotypes else None,
            tuple(selected_temperatures) if selected_temperatures else None,
            phase_used or dataset_phase(ds),
            bin_size,
        )
        # theme=None: the figure's black-text / transparent-bg styling drives screen + PNG.
        charts.plotly_chart(fig, width="stretch", theme=None)

        try:
            binned_sleep = _cached_zt_binned(_ds_fp_sl, ds, "sleep", bin_size)
            _id_to_group_sl = {}
            for _fid in binned_sleep["id"].unique():
                try:
                    _fly = ds.sel(id=_fid)
                    if "group" in ds.coords:
                        _id_to_group_sl[_fid] = str(_fly["group"].item())
                    elif "genotype" in ds.coords and "temperature" in ds.coords:
                        _id_to_group_sl[_fid] = (
                            f"{_fly['genotype'].item()}-{_fly['temperature'].item()}"
                        )
                    else:
                        _id_to_group_sl[_fid] = "All"
                except Exception:
                    _id_to_group_sl[_fid] = "All"
            binned_sleep["group"] = binned_sleep["id"].map(_id_to_group_sl)
            # Shared builder — see the activity block above.
            _pivot_sl = ex.zt_group_summary_table(binned_sleep, "sleep", bin_size)
            # Per-fly binned time course (long) so other stats can be computed:
            # one row per fly per ZT bin (ID, Group, zt_bin_minute, zt_hours, sleep).
            _sl_pf = binned_sleep.rename(columns={"id": "ID", "group": "Group"}).copy()
            _sl_pf["zt_hours"] = dam_utilities.zt_bin_to_hours(
                _sl_pf["zt_bin_minute"], bin_size
            )
            _sl_pf = (
                _sl_pf[["ID", "Group", "zt_bin_minute", "zt_hours", "sleep"]]
                .sort_values(["Group", "ID", "zt_bin_minute"])
                .reset_index(drop=True)
            )
            ex.save_excel_button(
                "Save binned sleep (.xlsx)",
                [
                    ("group_summary", _pivot_sl),
                    ("per_fly", _sl_pf),
                ],
                ds,
                "sleep_binned",
                key="dl_sleep",
                help="Two sheets: group mean ± SD ± N per ZT bin in GraphPad's "
                "grouped-table order, and the per-fly time course behind it.",
            )
        except Exception:
            pass

        st.divider()

        st.subheader(f"Sleep Summary{_summary_subhead_suffix}")
        _totals_violins("sleep", "sleep (minutes)", "sleep")

        st.divider()

        # Sleep Bout Duration
        st.subheader("Sleep Bout Duration")
        if "duration" in ds.data_vars:
            st.caption(
                "One curve per fly (not one pooled histogram) — a fly with many bouts no "
                "longer outweighs a fly with few, so genotypes overlay cleanly as lines."
            )
            _bd_col1, _bd_col2 = st.columns([2, 1])
            with _bd_col1:
                _bd_method_label = st.radio(
                    "Curve type",
                    ["KDE (log-duration)", "Survival curve (CCDF)"],
                    index=0,
                    horizontal=True,
                    key="bout_curve_method",
                )
            with _bd_col2:
                bout_show_individual = st.checkbox(
                    "Show individual flies",
                    value=False,
                    key="bout_show_individual",
                    help="One faint line per fly behind the group curves. Useful for "
                    "spotting a single fly driving a group; off by default because on "
                    "a few hundred flies it is a texture rather than a reading.",
                )
            bout_method = "kde" if _bd_method_label.startswith("KDE") else "survival"

            _ds_fp_bout = dataset_fingerprint(ds)
            bout_fig, bout_curves_df, bout_summary_df, bout_stats = _cached_bout_duration_lines(
                _ds_fp_bout,
                ds,
                bout_method,
                tuple(selected_genotypes) if selected_genotypes else None,
                tuple(selected_temperatures) if selected_temperatures else None,
                bout_show_individual,
            )
            charts.plotly_chart(bout_fig, width="stretch", theme=None)

            if bout_stats and np.isfinite(bout_stats.get("pvalue", float("nan"))):
                st.caption(
                    f"{bout_stats['test'].upper()} across groups on per-fly "
                    f"log-mean bout duration: p={bout_stats['pvalue']:.4f} "
                    f"(normality {'passed' if bout_stats['normality_passed'] else 'failed'}, "
                    f"equal variance {'passed' if bout_stats['equal_variance_passed'] else 'failed'})."
                )
                if bout_stats["pairwise"]:
                    # Folded: the p-value above is the answer most people came for,
                    # and the pairwise grid is what you open when it is interesting.
                    with st.expander("Pairwise comparisons"):
                        st.dataframe(
                            pd.DataFrame(bout_stats["pairwise"]), width="stretch"
                        )

            raw_bout_df = sleep_analysis.raw_bout_dataframe(
                ds, selected_genotypes=selected_genotypes, selected_temperatures=selected_temperatures
            )
            # Named apart from the Export page's sleep_bouts.csv on purpose. Both
            # come from raw_bout_dataframe, but this one is filtered to the group
            # selection in the sidebar while that one is every fly — under one
            # filename, whichever the user opened last silently won.
            ex.save_excel_button(
                "Save sleep bout data (.xlsx)",
                [
                    ("per_fly_summary", bout_summary_df),
                    ("bouts", raw_bout_df),
                ],
                ds,
                "sleep_bouts_filtered",
                key="dl_bouts",
                help="Two sheets: one row per fly with its bout-duration summary, and "
                "every individual bout behind it. Named apart from the Export page's "
                "sleep_bouts on purpose — this one is filtered to the sidebar group "
                "selection and that one is every fly.",
            )

    else:
        st.info(
            "Nothing to show until sleep has been detected — every figure in this "
            "tab reads the bouts it produces. The control is at the top."
        )
