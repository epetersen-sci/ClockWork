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

The key alone does not survive a page switch: Streamlit drops a keyed widget's
value once the widget stops being rendered. ``persist_state="session"`` is the
built-in fix and is what this module uses.

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

    st.sidebar.subheader("Filter groups")
    # Sharing `key` between the two pages is not enough on its own: by default a
    # keyed widget's value is dropped as soon as the widget stops being rendered,
    # and a page switch is exactly that — so a selection made on Periodograms was
    # gone by the time Sleep & activity built its multiselect. `persist_state`
    # is Streamlit's own answer to this; "session" keeps the value for the whole
    # session, across page switches, and it needs `key` (which we pass).
    selected = st.sidebar.multiselect(
        label or f"Groups ({coord})",
        all_groups,
        default=all_groups,
        key=key,
        persist_state="session",
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


def column_for_coord(ds):
    """``{coord name: the metadata column it came from}`` — the renames, reversed.

    Public because a page legitimately needs it: a figure axis should carry the
    name the user ticked on Import (``pulse_time``), not the coord it is stored
    under (``pulse_zt_hour``), and a page reaching for a private name to get
    there is backlog item 8 all over again.
    """
    import dam_utilities

    return {v: k for k, v in dam_utilities.METADATA_COORD_RENAMES.items()}


def group_by_options(ds):
    """``(options, default)`` for a "one panel/figure per …" picker, in METADATA
    COLUMN names — the same names the Import page shows.

    The picker speaks the user's language: you ticked ``pulse_time`` on Import, so
    that is the chip you see here. Two of those columns are stored as coords under
    different names (``pulse_time`` -> ``pulse_zt_hour``), because the value is
    parsed on the way in; :func:`group_by_coords` does that translation, so no page
    has to know about it.

    The default is whatever was ticked on Import, so a page opens on the grouping
    the dataset already has rather than one it guessed at.
    """
    import dam_utilities

    back = column_for_coord(ds)
    options = [back.get(c, c) for c in dam_utilities.group_defining_coords(ds)]
    default = [back.get(c, c) for c in dam_utilities.get_group_coord_names(ds)]
    default = [c for c in default if c in options]
    return options, (default or options[:1])


def group_by_coords(ds, columns):
    """The coord names for a selection made in :func:`group_by_options`' terms."""
    import dam_utilities

    return tuple(dam_utilities.METADATA_COORD_RENAMES.get(c, c) for c in columns)


def is_import_grouping(ds, columns):
    """True when ``columns`` is exactly the grouping chosen at Import.

    When it is, a page should label its panels from ``ds['group']`` rather than
    rebuilding them: the stored labels carry the metadata's own values (``ZT21``),
    while rebuilding from coords gives the parsed ones (``21.0``). Same partition,
    uglier names.
    """
    import dam_utilities

    return list(group_by_coords(ds, columns)) == list(
        dam_utilities.get_group_coord_names(ds)
    )
