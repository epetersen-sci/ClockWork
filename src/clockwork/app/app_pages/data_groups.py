"""
Groups & subsets — what is currently loaded, and a reversible group filter.

This sits between Import and Curate & split for two reasons. The curation
heatmap on the next page draws its y-axis from the ``group`` coord, so the
groups it renders should already be settled; and applying a subset here clears
every derived cache, so subsetting before curation avoids curating flies you are
about to drop and avoids silently invalidating curation afterwards.

Note this filter is NOT the same as the sidebar "Filter groups" on Periodograms
and Sleep & activity. Those subset a page-local view for plotting. This one
replaces the working dataset — reversibly, because the unfiltered copy is kept
in ``dataset_full``.

Groups are first DEFINED at import, inside Create Dataset, and can be REDEFINED
here — the "Redefine groups" section re-derives the ``group`` coord from the
per-fly metadata coords the dataset already carries, so a reloaded ``.nc`` can be
regrouped without going back to the raw monitor files.
"""

import numpy as np
import pandas as pd
import streamlit as st

from clockwork import pipeline
from clockwork.app.analysis_detection import detect_analyses
from clockwork.app.ui import status
from clockwork.app.ui.guards import require_dataset
from clockwork.app.ui.state import invalidate_derived_caches
from clockwork.core import dam_utilities


def _apply_group_filter(selected_groups):
    """Subset ``dataset_full`` by group and replace ``dataset`` with the
    result. Clears every derived cache so downstream pages re-derive from
    the filtered dataset.

    The groups are ticked by LABEL, but the subset is recorded by the metadata
    values behind those labels (``pipeline.keep_for_groups``), so it survives a
    reload and a regroup, and the Export settings button can write it down.
    """
    ds_full = st.session_state.dataset_full
    if ds_full is None or "group" not in ds_full.coords:
        return
    # Sanitize in case this session still holds a Dataset built before
    # ArrowStringArray coords were coerced to numpy (breaks .sel/.isel).
    ds_full = dam_utilities.ensure_numpy_backed(ds_full)
    st.session_state.dataset_full = ds_full
    filtered = pipeline.subset(ds_full, pipeline.keep_for_groups(ds_full, selected_groups))
    _replace_working_dataset(filtered)


def _apply_regroup(chosen):
    """Re-derive the ``group`` coord from ``chosen`` and replace BOTH datasets.

    ``dataset_full`` is rewritten as well as ``dataset``, because it is the
    subset filter's restore point: regrouping only the working copy would make
    *Reset to all groups* quietly put the old grouping back.

    The flies' own work (curation, the split, sleep, period results) carries
    over (``_replace_working_dataset``); what describes the old GROUPS (the HMM,
    the CWT group averages) does not, and every derived cache is cleared, since
    ``group`` feeds every group-level comparison, plot and export.
    """
    ds_full = st.session_state.get("dataset_full")
    base = ds_full if ds_full is not None else st.session_state.dataset
    base = dam_utilities.ensure_numpy_backed(base)

    regrouped_full = pipeline.apply_groups(base, pipeline.GroupsConfig(by=list(chosen)))
    invalidate_derived_caches()
    st.session_state.dataset_full = regrouped_full

    # Preserve an active subset across the regroup where the ids still exist,
    # rather than silently widening the working set back to every fly.
    current = st.session_state.get("dataset")
    if current is not None and len(current["id"]) < len(regrouped_full["id"]):
        keep = [str(i) for i in current["id"].values]
        kept = regrouped_full.sel(id=keep)
        # The ids are what is preserved; the record of WHY they are the ones kept
        # travels with them, or the regrouped dataset would claim no subset.
        if "subset_keep" in current.attrs:
            kept.attrs["subset_keep"] = current.attrs["subset_keep"]
        _replace_working_dataset(kept)
    else:
        _replace_working_dataset(regrouped_full.copy())


def _reset_group_filter():
    """Restore the unfiltered dataset and clear derived caches."""
    ds_full = st.session_state.get("dataset_full")
    if ds_full is None:
        return
    _replace_working_dataset(ds_full.copy())


#: Where a group change leaves its account of what was carried over, for the
#: rerun that follows the click (a message drawn before st.rerun() is lost).
_NOTES_KEY = "_group_change_notes"


def _replace_working_dataset(rebuilt):
    """Make ``rebuilt`` the working dataset, with the current one's work on it.

    ``rebuilt`` comes from the import copy, so it lacks curation, the split and
    every result; ``pipeline.carry_over`` re-applies or carries what it can
    (BACKLOG 22) and says what it could not.
    """
    current = st.session_state.get("dataset")
    notes = []
    if current is not None:
        with st.spinner("Carrying curation, the split and results over to the new groups..."):
            rebuilt, notes = pipeline.carry_over(rebuilt, current)
    invalidate_derived_caches()
    st.session_state.dataset = rebuilt
    st.session_state.analyses = detect_analyses(rebuilt)
    st.session_state[_NOTES_KEY] = notes


#: Display names for the analyses detect_analyses reports.
_ANALYSIS_NAMES = {
    "sleep": "sleep detection",
    "sleep_states": "sleep states",
    "lomb_scargle": "Lomb-Scargle",
    "autocorrelation": "autocorrelation",
    "cwt": "CWT",
    "mesa": "MESA",
    "hmm": "HMM",
    "sleep_deprivation": "sleep deprivation",
    "phase_shift": "phase shift",
}


def _undone_by_a_group_change(current, full):
    """What a group change here would still throw away (BACKLOG 22).

    Curation, the split, sleep and per-fly period results are carried over
    (``pipeline.carry_over``). Results that describe GROUPS are not: they were
    computed from the old groups' members, so they would describe groups that no
    longer exist. Period results also drop when a change brings in flies that
    were never analysed; which flies those are is only known once curation has
    been re-applied, so that is said after the change, in its notes.
    """
    if full is None:
        return []
    undone = []
    ran = {k for k, v in detect_analyses(current).items() if v}
    if "hmm" in ran:
        undone.append("**the HMM**: its model was fitted per group or pooled across flies")
    if "cwt_group_average_paths" in current.attrs:
        undone.append("**the CWT group-averaged scalograms**")
    for key in ("sleep_deprivation", "phase_shift"):
        if key in ran:
            undone.append(f"**{_ANALYSIS_NAMES[key]}** results")
    return undone


def _warn_if_work_would_be_undone(current, full):
    undone = _undone_by_a_group_change(current, full)
    if undone:
        st.warning(
            "**This discards some results.** Curation, the LD/DD split, sleep and each "
            "fly's period results carry over to the new groups, but these do not:\n\n"
            + "\n".join(f"- {item}" for item in undone)
            + "\n\nRe-run them after changing groups."
        )


ds = require_dataset()
ds_full = st.session_state.get("dataset_full")

# What the last group change carried over and what it could not, said once.
_notes = st.session_state.pop(_NOTES_KEY, None)
if _notes is not None:
    st.success(
        "**Groups changed.**" + ("\n\n" + "\n".join(f"- {n}" for n in _notes) if _notes else "")
    )

# ---- Active-filter status badge ----------------------------------
if ds_full is not None and len(ds["id"]) < len(ds_full["id"]) and "group" in ds_full.coords:
    full_groups = {str(g) for g in ds_full["group"].values}
    cur_groups = {str(g) for g in ds["group"].values} if "group" in ds.coords else set()
    st.warning(
        f"**Group filter active** — using **{len(ds['id'])} of "
        f"{len(ds_full['id'])} flies**, "
        f"**{len(cur_groups)} of {len(full_groups)} groups**. "
        "Use *Reset to all groups* below to restore the full set."
    )

col1, col2, col3 = st.columns(3)
col1.metric("Flies", len(ds["id"]))
col2.metric("Timepoints", len(ds["time"]))
if "group" in ds.coords:
    col3.metric("Groups", len(set(ds["group"].values)))
st.dataframe(
    pd.DataFrame(
        {
            "ID": ds["id"].values,
            "Group": ds["group"].values if "group" in ds.coords else "N/A",
        }
    ),
    width="stretch",
    height=200,
)

# Import-time data-quality counters, if this dataset carries them. This page is
# "what is currently loaded", and for a reloaded .nc it is the only place that
# can say anything about the quality of the recording behind it.
status.render_integrity_counters(ds)

# ============================================================
# Redefine groups — re-derive the `group` coord from the per-fly
# metadata coords the dataset already carries. Matters most for a
# reloaded .nc, where the alternative is re-reading the raw DAM files.
# ============================================================
_regroup_candidates = dam_utilities.group_defining_coords(ds)
if _regroup_candidates:
    with st.expander("Redefine groups"):
        _current_cols = dam_utilities.get_group_columns(ds)
        st.markdown(
            "Groups are built by joining one or more metadata columns with `-`. "
            "Changing them **re-derives the `group` coord in place**. Curation, the "
            "LD/DD split, sleep and each fly's period results carry over; results "
            "that describe whole groups (the HMM, CWT group averages) do not."
        )
        _chosen = st.multiselect(
            "Metadata columns that define a group",
            options=_regroup_candidates,
            default=[c for c in _current_cols if c in _regroup_candidates],
            key="regroup_columns",
            help="Only columns stored per fly at import are offered. Datetime, "
            "monitor, region and id columns are excluded automatically.",
        )
        _warn_if_work_would_be_undone(ds, ds_full)
        if not _chosen:
            st.caption("Select at least one column.")
        else:
            _preview = dam_utilities.regroup_dataset(ds, _chosen)
            _new_groups = sorted({str(g) for g in _preview["group"].values})
            _unchanged = list(_chosen) == list(_current_cols)
            st.caption(
                f"**{len(_new_groups)}** group(s) — "
                + ", ".join(_new_groups[:8])
                + (" …" if len(_new_groups) > 8 else "")
                + (
                    f"  (currently **{len({str(g) for g in ds['group'].values})}** "
                    f"from {', '.join(_current_cols) or 'nothing recorded'})"
                    if not _unchanged
                    else "  — unchanged from the current grouping"
                )
            )
            if st.button(
                "Apply grouping",
                key="apply_regroup",
                disabled=_unchanged,
                help="Disabled while the selection matches the current grouping."
                if _unchanged
                else None,
            ):
                _apply_regroup(_chosen)
                st.rerun()

# ============================================================
# Group selection — subset the dataset before any downstream
# page sees it. Reversible via the Reset button below; the
# unfiltered dataset is preserved in `dataset_full`.
# ============================================================
if ds_full is not None and "group" in ds_full.coords:
    with st.expander("Group selection (subset for downstream analyses)"):
        st.markdown(
            "Pick which groups to keep. Applying a selection **drops the "
            "other flies from the working dataset**. Curation, the LD/DD "
            "split, sleep and each kept fly's period results carry over. "
            "Click *Reset* to restore the full set."
        )

        # Per-group fly counts from the unfiltered dataset.
        full_groups_arr = np.asarray([str(g) for g in ds_full["group"].values])
        group_counts = (
            pd.Series(full_groups_arr)
            .value_counts()
            .sort_index()
            .rename_axis("group")
            .reset_index(name="n_flies")
        )
        cur_groups_set = {str(g) for g in ds["group"].values} if "group" in ds.coords else set()
        group_counts["currently_kept"] = group_counts["group"].isin(cur_groups_set)
        st.dataframe(group_counts, width="stretch", height=180)

        all_group_options = group_counts["group"].tolist()
        # Default the picker to whatever is currently kept; first-time
        # users see all groups selected.
        default_selection = sorted(cur_groups_set) if cur_groups_set else all_group_options
        selected = st.multiselect(
            "Groups to keep",
            options=all_group_options,
            default=default_selection,
            key="group_filter_select",
        )

        # Live preview of the selection's effect.
        preview_n = (
            int(group_counts.loc[group_counts["group"].isin(selected), "n_flies"].sum())
            if selected
            else 0
        )
        st.caption(
            f"Will keep **{preview_n} / {len(ds_full['id'])} flies** "
            f"({len(selected)} / {len(all_group_options)} groups)."
        )

        _warn_if_work_would_be_undone(ds, ds_full)
        col_apply, col_reset = st.columns(2)
        with col_apply:
            apply_disabled = len(selected) == 0
            if st.button(
                "Apply selection",
                key="apply_group_filter",
                disabled=apply_disabled,
                help=(
                    "Pick at least one group"
                    if apply_disabled
                    else "Subset the dataset and clear cached analyses."
                ),
            ):
                _apply_group_filter(selected)
                st.rerun()
        with col_reset:
            reset_disabled = len(ds["id"]) == len(ds_full["id"])
            if st.button(
                "Reset to all groups",
                key="reset_group_filter",
                disabled=reset_disabled,
                help=(
                    "No filter active"
                    if reset_disabled
                    else "Restore the full unfiltered dataset."
                ),
            ):
                _reset_group_filter()
                st.rerun()
