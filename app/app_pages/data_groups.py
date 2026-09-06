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

import dam_utilities
from analysis_detection import detect_analyses
from ui.guards import require_dataset
from ui.state import invalidate_derived_caches


def _apply_group_filter(selected_groups):
    """Subset ``dataset_full`` by group and replace ``dataset`` with the
    result. Clears every derived cache so downstream pages re-derive from
    the filtered dataset."""
    ds_full = st.session_state.dataset_full
    if ds_full is None or "group" not in ds_full.coords:
        return
    # Sanitize in case this session still holds a Dataset built before
    # ArrowStringArray coords were coerced to numpy (breaks .sel/.isel).
    ds_full = dam_utilities.ensure_numpy_backed(ds_full)
    st.session_state.dataset_full = ds_full
    selected = {str(g) for g in selected_groups}
    mask = np.array([str(g) in selected for g in ds_full["group"].values])
    filtered = ds_full.isel(id=np.flatnonzero(mask))
    invalidate_derived_caches()
    st.session_state.dataset = filtered
    st.session_state.analyses = detect_analyses(filtered)


def _apply_regroup(chosen):
    """Re-derive the ``group`` coord from ``chosen`` and replace BOTH datasets.

    ``dataset_full`` is rewritten as well as ``dataset``, because it is the
    subset filter's restore point: regrouping only the working copy would make
    *Reset to all groups* quietly put the old grouping back.

    Every derived cache goes too. ``group`` feeds every group-level comparison,
    plot and export, so a period analysis computed under the previous grouping
    describes labels that no longer exist — showing it against the new ones would
    be worse than making the user re-run it.
    """
    ds_full = st.session_state.get("dataset_full")
    base = ds_full if ds_full is not None else st.session_state.dataset
    base = dam_utilities.ensure_numpy_backed(base)

    regrouped_full = dam_utilities.regroup_dataset(base, chosen)
    invalidate_derived_caches()
    st.session_state.dataset_full = regrouped_full

    # Preserve an active subset across the regroup where the ids still exist,
    # rather than silently widening the working set back to every fly.
    current = st.session_state.get("dataset")
    if current is not None and len(current["id"]) < len(regrouped_full["id"]):
        keep = [str(i) for i in current["id"].values]
        st.session_state.dataset = regrouped_full.sel(id=keep)
    else:
        st.session_state.dataset = regrouped_full.copy()
    st.session_state.analyses = detect_analyses(st.session_state.dataset)


def _reset_group_filter():
    """Restore the unfiltered dataset and clear derived caches."""
    ds_full = st.session_state.get("dataset_full")
    if ds_full is None:
        return
    invalidate_derived_caches()
    st.session_state.dataset = ds_full.copy()
    st.session_state.analyses = detect_analyses(st.session_state.dataset)


ds = require_dataset()
ds_full = st.session_state.get("dataset_full")

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
            "Changing them **re-derives the `group` coord in place** and clears "
            "every cached analysis result — `group` feeds every group-level "
            "comparison, plot and export, so previous results describe labels "
            "that no longer exist."
        )
        _chosen = st.multiselect(
            "Metadata columns that define a group",
            options=_regroup_candidates,
            default=[c for c in _current_cols if c in _regroup_candidates],
            key="regroup_columns",
            help="Only columns stored per fly at import are offered. Datetime, "
            "monitor, region and id columns are excluded automatically.",
        )
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
                st.success(
                    f"Regrouped by {', '.join(_chosen)} — {len(_new_groups)} group(s). "
                    "Cached analysis results were cleared."
                )
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
            "other flies from the working dataset** and clears every "
            "cached analysis result, so downstream pages re-compute on "
            "the subset only. Click *Reset* to restore the full set."
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
                st.success(
                    f"Filter applied — {preview_n} flies across "
                    f"{len(selected)} groups. Cached analyses were cleared."
                )
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
                st.success("Restored full dataset. Cached analyses were cleared.")
                st.rerun()
