"""Sidebar display filters.

Two pages grew near-identical "Filter Groups" sidebars with different fallback
behaviour: Periodograms falls back to the ``genotype`` coord and then to a single
pooled group, while Sleep & activity handles only ``group`` and actually
``.sel()``s the dataset down. Sharing the implementation stops them drifting
further apart.

**The session keys stay separate on purpose.** ``pgram_groups`` and
``viz_groups`` are distinct today, so a selection on one page has no effect on
the other. Merging them into one app-wide selection is a real improvement and a
one-line change here — but it is a behaviour change, so it is BACKLOG item 11
rather than something to slip into a reorganization.

Note these are *display* filters: they subset a page-local view for plotting. The
filter on Groups & subsets is a different thing entirely — it replaces the master
dataset and invalidates every downstream cache.
"""

import numpy as np
import streamlit as st


def resolve_group_coord(ds):
    """Return the coord to group by: ``group``, else ``genotype``, else None."""
    if "group" in ds.coords:
        return "group"
    if "genotype" in ds.coords:
        return "genotype"
    return None


def group_filter_sidebar(ds, key, *, label=None, subset=False, help=None):
    """Render the sidebar group multiselect and return
    ``(group_values, all_groups, selected_groups, ds)``.

    ``group_values`` is a per-fly array of labels aligned to ``ds['id']``.
    With ``subset=True`` the returned ``ds`` is sliced to the selected groups;
    otherwise it is returned unchanged and the caller filters however it likes.
    """
    coord = resolve_group_coord(ds)

    if coord is None:
        n = ds.sizes["id"]
        return np.array(["All flies"] * n), ["All flies"], ["All flies"], ds

    group_values = np.asarray([str(v) for v in ds[coord].values])
    all_groups = sorted(set(group_values.tolist()))

    st.sidebar.subheader("Filter groups")
    selected = st.sidebar.multiselect(
        label or f"Groups ({coord})",
        all_groups,
        default=all_groups,
        key=key,
        help=help,
    )
    # multiselect returns a list in the app; guard the None the headless test
    # stub returns (and default to all groups if the widget hasn't populated).
    if not selected:
        selected = list(all_groups)

    if subset and len(selected) < len(all_groups):
        chosen = set(selected)
        keep = [str(i) for i in ds["id"].values if str(ds[coord].sel(id=i).item()) in chosen]
        if keep:
            ds = ds.sel(id=keep)

    return group_values, all_groups, selected, ds


def bin_size_sidebar(key, *, label="Bin size (minutes)", default=30):
    """ZT bin-size slider. Its own function so it stops rendering underneath the
    "Filter groups" subheader, which is where it currently appears to belong."""
    return st.sidebar.slider(label, 5, 60, default, step=5, key=key)
