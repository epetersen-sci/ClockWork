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

Groups themselves are DEFINED at import, inside Create Dataset. See BACKLOG.md
item 12: ``dam_utilities.regroup_dataset`` exists but is wired to nothing, so
they cannot yet be redefined here without re-importing.
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
