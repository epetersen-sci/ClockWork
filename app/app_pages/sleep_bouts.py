"""
Sleep bouts — how a fly's sleep is distributed across bout lengths.

Two views on the bout table that Sleep analysis already wrote:
  1. Bouts per duration bin — counts (and sleep minutes) per bin, by group.
  2. Sleep under each definition — the same bouts re-totalled under each candidate
     inactivity-duration definition of sleep (>=5 ... >=60 min, plus interval
     definitions), the empirical ladder of Abhilash & Shafer (2024) Figure 6.

Nothing is re-detected here: both views read the per-bout ``duration`` table on the
dataset, so these numbers cannot drift from the sleep masks. A bout is credited to
the day it starts in. Plots and tables save into ``Graph Exports_<experiment>/
Sleep_bouts/``.
"""

import numpy as np
import streamlit as st

import bout_spectrum as bs
import export_helpers as ex
import plotting
from dataset_meta import dataset_fingerprint
from ui import charts, days, filters
from ui.guards import require_analysis, require_dataset, require_variable

EXPORT_SUBFOLDER = "Sleep_bouts"

st.caption(
    "How a fly's sleep is split across bout lengths, and how much of it each "
    "candidate definition of sleep keeps — the empirical half of Abhilash & "
    "Shafer (2024) Figure 6, without the two-process model."
)

# ============================================================
# Section 1: Prerequisites
# ============================================================
ds = require_dataset()
require_analysis(ds, "sleep", "Sleep analysis")
require_variable(
    ds,
    "duration",
    "This dataset has sleep masks but no per-bout table. Re-run **Sleep analysis** "
    "— this page reads the `duration` variable it writes, one row per bout.",
)

# The figures follow the CSVs into Sleep_bouts/ rather than landing one level up;
# the router renders the save button and would not otherwise know.
charts.set_subfolder(EXPORT_SUBFOLDER)


# ----------------------------------------------------------------
# Cached helpers — the per-fly tables are the expensive part (one pass over every
# bout of every fly), so they are keyed on the dataset fingerprint plus the options
# that change their values.
#
# `fp` takes NO leading underscore and `_ds` does. Streamlit's underscore rule is
# syntactic: it drops EVERY leading-underscore parameter from the key, so the `_fp`
# these arrived with removed the one argument that says which flies are in the
# dataset — backlog item 14, which is also why both are listed in
# tests/test_cache_keys.py.
# ----------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _cached_bout_counts(fp, _ds, bins, per_day, as_percent, days_key):
    return bs.per_fly_bout_counts(
        _ds,
        bins=list(bins),
        per_day=per_day,
        as_percent=as_percent,
        days=list(days_key) if days_key is not None else None,
    )


@st.cache_data(show_spinner=False)
def _cached_by_definition(fp, _ds, definitions, per_day, as_percent, days_key):
    return bs.per_fly_sleep_by_definition(
        _ds,
        definitions=list(definitions),
        per_day=per_day,
        as_percent=as_percent,
        days=list(days_key) if days_key is not None else None,
    )


# ============================================================
# Section 2: Group filter
# ============================================================
# The shared sidebar filter, on the same session key every other page's uses, so
# narrowing the cohort here and then switching pages does not silently widen it
# again (backlog item 11). subset=True gives back the sliced dataset.
_group_values, _all_groups, _selected_groups, ds = filters.group_filter_sidebar(
    ds,
    filters.DISPLAY_GROUPS_KEY,
    subset=True,
    help="Groups are defined by the metadata column(s) chosen on the Import page.",
)

_min_def = bs.min_definable_minutes(ds)
_phase = ds.attrs.get("sleep_phase", "unknown")

# ============================================================
# Section 3: Days
# ============================================================
st.subheader("Days")
# Day numbering needs the relative-minute axis; without it every bout is used and
# the picker is hidden rather than shown with numbers it cannot honour.
_relative = ds.attrs.get("time_is_relative_minutes", 0) == 1
sel_days, day_choice, abs_days, days_txt = None, None, None, "all days"
_sel_bouts = None  # set by the day picker; None means "no day filter applied"
if not _relative:
    st.info(
        "This dataset uses an absolute datetime axis, so days cannot be numbered — "
        "every bout in the record is included."
    )
else:
    DAYS, dd_day = days.day_table(ds, warn=st.warning)
    if DAYS.empty:
        st.info("The recording is shorter than one full day — every bout is included.")
    else:
        # Epoch first, then days within it. Mixing entrained and free-running days
        # into one bout spectrum averages two different states together, so the
        # picker asks which epoch before it offers any days.
        _epochs = [e for e in ("LD", "DD") if (DAYS.epoch == e).any()]
        # Default to the epoch the sleep analysis was actually run on — the info
        # box below names it, and bouts outside it do not exist.
        _default_epoch = _phase if _phase in _epochs else _epochs[0]
        epoch = st.radio(
            "Epoch",
            _epochs,
            index=_epochs.index(_default_epoch),
            horizontal=True,
            key="bouts_epoch",
            help="Days are numbered inside their epoch: LD from the recording "
            "start, DD from `first_DD_day`.",
        )
        st.caption(
            "A bout counts on the day it **starts**, so a bout running past midnight "
            "is counted once and never split. Per-day normalisation below divides by "
            "the valid record within the ticked days only."
        )
        pool = DAYS[DAYS.epoch == epoch]
        # Keying per epoch keeps each epoch's ticks while you switch between them.
        sel_days, day_choice = days.day_checkboxes(pool, f"bouts_days_{epoch}")
        if not day_choice:
            st.info(f"Tick at least one {epoch} day.")
            st.stop()
        abs_days = [int(d) for d in sel_days.absolute]
        days_txt = days.days_label(sel_days)
        if dd_day is None:
            st.caption("No `first_DD_day` in the metadata — every day is labelled LD.")

        # Sleep analysis is run on ONE epoch, and out-of-phase minutes carry no
        # bouts at all. Asking for the other epoch therefore yields nothing — say
        # that plainly instead of drawing empty axes and letting it read as a bug.
        _sel_bouts = bs.bout_table(ds, days=abs_days)
        if _sel_bouts.empty:
            st.warning(
                f"No sleep bouts on the selected {epoch} days. This dataset's sleep "
                f"analysis was run on the **{_phase}** epoch, so only {_phase} days "
                "carry bouts. Either pick "
                f"{'the other epoch above' if _phase in ('LD', 'DD') else 'different days'}"
                ", or re-run **Sleep analysis** for the "
                f"{epoch} phase."
            )
            st.stop()

# Slug of the same days the titles name, so a second day range writes a second file
# instead of overwriting the first (ranges spelled 'to' — see ui.days).
_days_slug = days.slug(
    days.days_label(sel_days, dash="to") if sel_days is not None else "all days"
)

st.sidebar.subheader("Sleep bouts options")
per_day = st.sidebar.checkbox(
    "Normalise per day of record",
    value=True,
    key="bout_spectrum_per_day",
    help=(
        "Divide each fly's values by its days of valid record within the selected "
        "days (minutes where sleep is measured, gaps and out-of-phase time "
        "excluded). Turn this off only if every fly contributed the same record."
    ),
)
as_percent = st.sidebar.checkbox(
    "Show as % of each fly's total",
    value=False,
    key="bout_spectrum_percent",
    help="Shape of the distribution rather than its size. Overrides per-day normalisation.",
)
chart_style = st.sidebar.radio(
    "Chart style", ["Bars", "Lines"], index=0, horizontal=True, key="bout_spectrum_chart"
)
_chart = "line" if chart_style == "Lines" else "bar"

st.info(
    f"Bouts come from the sleep analysis run on the **{_phase}** epoch with a "
    f"**{_min_def:g}-minute** threshold. Inactivity shorter than that was never "
    f"recorded as sleep and cannot be counted here, so {_min_def:g} min is the floor "
    "for every bin and definition below."
)

_fp = dataset_fingerprint(ds)
_days_key = tuple(abs_days) if abs_days is not None else None

# ============================================================
# Section 4: Bout spectrum — counts per duration bin
# ============================================================
st.subheader("Bouts per duration bin")
st.caption(
    "Every sleep bout sorted into a duration bin (`lo <= duration < hi`; the top "
    "bin is open-ended). Each point is the group mean across flies ± SEM — every "
    "fly contributes one value per bin, including a real 0 where it had no bouts."
)

edge_text = st.text_input(
    "Bin edges (minutes, ascending)",
    value=", ".join(f"{e:g}" for e in bs.DEFAULT_BIN_EDGES),
    key="bout_spectrum_edges",
    help="The last value opens an unbounded top bin (e.g. a final 240 gives a '240+ min' bin).",
)
try:
    _edges = bs.parse_edges(edge_text)
    bins = bs.make_bins(_edges)
except ValueError as exc:
    st.error(f"Bin edges: {exc}")
    st.stop()

if _edges[0] < _min_def:
    st.warning(
        f"The lowest bin edge ({_edges[0]:g} min) is below the {_min_def:g}-minute sleep "
        "threshold. Bins under the threshold will read 0 because those bouts were "
        "never recorded as sleep, not because the flies had none."
    )

metric = st.radio(
    "Value",
    ["Number of bouts", "Sleep minutes in bin"],
    index=0,
    horizontal=True,
    key="bout_spectrum_metric",
)
_value_col = "n_bouts" if metric == "Number of bouts" else "sleep_minutes"

if as_percent:
    _ylab = f"% of each fly's {'bouts' if _value_col == 'n_bouts' else 'sleep'} (mean ± SEM)"
else:
    _unit = "Bouts" if _value_col == "n_bouts" else "Sleep minutes"
    _ylab = f"{_unit} per {'day' if per_day else 'fly'} (mean ± SEM)"

counts_df = _cached_bout_counts(_fp, ds, tuple(bins), per_day, as_percent, _days_key)
_bin_order = [bs.bin_label(lo, hi) for lo, hi in bins]

# Summarised HERE, then handed to both the chart and the export — one object, so the
# plotted numbers and the saved numbers cannot be two computations that disagree
# (backlog item 7; see plotting.bout_spectrum_bars).
spec_stat = bs.summarize_by_group(counts_df, _value_col, "bin_label", _bin_order)
spec_fig = plotting.bout_spectrum_bars(
    spec_stat,
    "bin_label",
    category_order=_bin_order,
    title=f"{metric} by bout duration — {days_txt}",
    yaxis_title=_ylab,
    xaxis_title="Bout duration bin",
    chart=_chart,
)
charts.plotly_chart(
    spec_fig, filename=f"bout_spectrum_{_days_slug}", width="stretch", theme=None
)

if not spec_stat.empty:
    with st.expander("Group summary table"):
        st.dataframe(spec_stat, width="stretch", hide_index=True)
    ex.save_df_button(
        "Save Bout Spectrum (group mean±SEM) to working folder",
        spec_stat,
        ds,
        f"bout_spectrum_summary_{_days_slug}.csv",
        key="dl_bout_spectrum",
        subfolder=EXPORT_SUBFOLDER,
    )
    _spec_wide = bs.wide_by_group(counts_df, _value_col, "bin_label", _bin_order)
    ex.save_df_button(
        "Save per-fly Bout Spectrum (for stats) to working folder",
        _spec_wide,
        ds,
        f"bout_spectrum_per_fly_{_days_slug}.csv",
        key="dl_bout_spectrum_perfly",
        help="One row per fly, one column per duration bin.",
        subfolder=EXPORT_SUBFOLDER,
    )
    ex.save_df_button(
        "Save per-fly Bout Spectrum (long, counts + minutes) to working folder",
        counts_df,
        ds,
        f"bout_spectrum_per_fly_long_{_days_slug}.csv",
        key="dl_bout_spectrum_long",
        help="One row per fly per bin, with both the bout count and the sleep minutes.",
        subfolder=EXPORT_SUBFOLDER,
    )

st.divider()

# ============================================================
# Section 5: Sleep by definition — the Figure 6 ladder
# ============================================================
st.subheader("Sleep under each definition")
st.caption(
    "The same bouts re-totalled under each candidate rule for what counts as "
    "sleep. **Minimum** definitions (`>=X min`) nest inside one another, so they "
    "are not additive; **interval** definitions (`X-Y min`) are disjoint. Reading "
    "them side by side shows how much of the total each length band contributes."
)

_all_defs = bs.default_definitions()
_all_labels = [lbl for lbl, _, _ in _all_defs]
chosen = st.multiselect(
    "Definitions of sleep",
    _all_labels,
    default=_all_labels,
    key="bout_spectrum_defs",
    help="The ladder used in Abhilash & Shafer (2024) Figure 6.",
)
if not chosen:
    st.info("Select at least one definition to draw this section.")
_defs = [d for d in _all_defs if d[0] in chosen]
_def_order = [lbl for lbl, _, _ in _defs]

# Plot order keeps a blank slot between the two families. They answer different
# questions — minimums nest, intervals are disjoint — so a line running straight
# from ">=60 min" into "5-10 min" would read as a trend that isn't one. The blank
# category carries no data, which breaks the line (and leaves a gap between the
# bar groups). Export order stays _def_order, without the spacer.
_SPACER = " "
_min_defs = [lbl for lbl, _, hi in _defs if not np.isfinite(hi)]
_ival_defs = [lbl for lbl, _, hi in _defs if np.isfinite(hi)]
_plot_order = _min_defs + ([_SPACER] if _min_defs and _ival_defs else []) + _ival_defs

def_metric = st.radio(
    "Value",
    ["Sleep minutes", "Number of bouts"],
    index=0,
    horizontal=True,
    key="bout_spectrum_def_metric",
)
_def_col = "sleep_minutes" if def_metric == "Sleep minutes" else "n_bouts"

if as_percent:
    _def_ylab = (
        f"% of each fly's total {'sleep' if _def_col == 'sleep_minutes' else 'bouts'} "
        "(mean ± SEM)"
    )
else:
    _unit = "Sleep minutes" if _def_col == "sleep_minutes" else "Bouts"
    _def_ylab = f"{_unit} per {'day' if per_day else 'fly'} (mean ± SEM)"

defs_df = (
    _cached_by_definition(_fp, ds, tuple(_defs), per_day, as_percent, _days_key)
    if _defs
    else None
)
def_stat = bs.summarize_by_group(defs_df, _def_col, "definition", _plot_order)
if _defs:
    def_fig = plotting.bout_spectrum_bars(
        def_stat,
        "definition",
        category_order=_plot_order,
        title=f"{def_metric} under each definition of sleep — {days_txt}",
        yaxis_title=_def_ylab,
        xaxis_title="Definition of sleep (inactivity duration)",
        chart=_chart,
    )
    charts.plotly_chart(
        def_fig, filename=f"sleep_by_definition_{_days_slug}", width="stretch", theme=None
    )

if not def_stat.empty:
    with st.expander("Group summary table"):
        st.dataframe(def_stat, width="stretch", hide_index=True)
    ex.save_df_button(
        "Save Sleep-by-Definition (group mean±SEM) to working folder",
        def_stat,
        ds,
        f"sleep_by_definition_summary_{_days_slug}.csv",
        key="dl_sleep_by_def",
        subfolder=EXPORT_SUBFOLDER,
    )
    _def_wide = bs.wide_by_group(defs_df, _def_col, "definition", _def_order)
    ex.save_df_button(
        "Save per-fly Sleep-by-Definition (for stats) to working folder",
        _def_wide,
        ds,
        f"sleep_by_definition_per_fly_{_days_slug}.csv",
        key="dl_sleep_by_def_perfly",
        help="One row per fly, one column per definition.",
        subfolder=EXPORT_SUBFOLDER,
    )
    ex.save_df_button(
        "Save per-fly Sleep-by-Definition (long, minutes + bouts) to working folder",
        defs_df,
        ds,
        f"sleep_by_definition_per_fly_long_{_days_slug}.csv",
        key="dl_sleep_by_def_long",
        help="One row per fly per definition, with both sleep minutes and bout count.",
        subfolder=EXPORT_SUBFOLDER,
    )

st.divider()

# ============================================================
# Section 6: Raw bout table
# ============================================================
with st.expander("All bouts (one row per bout)"):
    # Built once during day selection above (it is what the "no bouts" guard reads).
    _bouts = _sel_bouts if _sel_bouts is not None else bs.bout_table(ds, days=abs_days)
    _n_flies = _bouts["ID"].nunique() if not _bouts.empty else 0
    st.caption(f"{len(_bouts):,} bouts from {_n_flies} flies (first 1000 shown).")
    st.dataframe(_bouts.head(1000), width="stretch", hide_index=True)
    ex.save_df_button(
        "Save all bouts to working folder",
        _bouts,
        ds,
        f"bout_table_{_days_slug}.csv",
        key="dl_bout_table",
        subfolder=EXPORT_SUBFOLDER,
    )

st.caption(
    f"Everything on this page — the tables here and the figures offered below — "
    f"saves into `Graph Exports/{EXPORT_SUBFOLDER}/`, with the selected days in "
    "each filename."
)
