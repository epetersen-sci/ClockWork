"""Sidebar display filters.

Two pages grew near-identical "Filter Groups" sidebars with different fallback
behaviour: Periodograms falls back to the ``genotype`` coord and then to a single
pooled group, while Sleep & activity handles only ``group`` and actually
``.sel()``s the dataset down. Sharing the implementation stops them drifting
further apart.

**Both pages share one session key**, :data:`DISPLAY_GROUPS_KEY`. They used to
carry a key each (``pgram_groups`` and ``viz_groups``), so narrowing to two
genotypes on Periodograms and then switching to Sleep & activity silently put
all six back — which surprised people, since the sidebar looks like one control
that follows you between pages. It now is one.

The two call sites still differ in what they do with the selection —
Sleep & activity passes ``subset=True`` and gets a ``.sel()``ed dataset back,
Periodograms filters by hand — and that asymmetry is fine. Only the key is
shared.

Note these are *display* filters: they subset a page-local view for plotting. The
filter on Groups & subsets is a different thing entirely — it replaces the master
dataset and invalidates every downstream cache.
"""

import numpy as np
import streamlit as st

#: The one app-wide session key behind every display group filter. Passed
#: explicitly by each call site rather than defaulted, so a page that genuinely
#: wants its own private selection has to say so.
DISPLAY_GROUPS_KEY = "display_groups"


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

    # Sharing `key` between the two pages is NOT enough on its own. Streamlit
    # garbage-collects widget state for widgets that were not rendered on the
    # previous run, and a page switch is exactly that — so the selection made on
    # Periodograms is gone by the time Sleep & activity instantiates its own
    # multiselect, which then falls back to `default` (all groups). Mirroring the
    # value under a plain, non-widget key survives the switch; re-seeding
    # session_state[key] from the mirror BEFORE the widget is created is the
    # supported way to restore it (assigning after instantiation raises).
    mirror = f"_{key}_persisted"
    if key not in st.session_state:
        remembered = st.session_state.get(mirror)
        # Intersect with the groups that actually exist: the dataset may have
        # been reloaded or regrouped since the selection was made, and a stale
        # label that is not in `options` would raise.
        st.session_state[key] = (
            [g for g in remembered if g in all_groups]
            if remembered is not None
            else list(all_groups)
        )

    st.sidebar.subheader("Filter groups")
    # No `default=` on purpose. session_state[key] is always seeded just above,
    # and passing both makes Streamlit warn ("created with a default value but
    # also had its value set via the Session State API") *and* return the
    # default on the run that registers the widget — so the restored selection
    # would only reach the plots one rerun late.
    selected = st.sidebar.multiselect(
        label or f"Groups ({coord})",
        all_groups,
        key=key,
        help=help,
    )
    st.session_state[mirror] = list(selected) if selected else []
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
