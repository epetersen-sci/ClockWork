"""The experiment's name — what its export folder is called.

One control, shared by the import page and the export page, because a name that
only takes effect at one moment is worse than no name at all: it was previously
stamped onto the dataset inside the "Create Dataset" button and nowhere else, so
typing it after loading, or loading a ``.nc`` at all, left the field looking
filled while every export ignored it.

The rule here is simple enough to state in one line: **the field always reflects
the loaded dataset, and editing it renames that dataset immediately.** Before any
dataset exists, it holds the value the next import will use.

Nothing about the name touches analysis, so renaming invalidates no caches. It
decides one thing — which folder exports land in — and ``ds.attrs`` remains the
only place that decision is read from (see ``dam_utilities.experiment_suffix``).
"""

import streamlit as st

import dam_utilities

#: Shared by both pages, so the field reads the same wherever it is rendered.
KEY = "experiment_name_input"

#: Where a name waits until the field can legally take it. Streamlit forbids
#: assigning a widget's session key once that widget has been created on the
#: current run, and the Import page creates this one near the top of its first
#: tab — every tab body runs on every rerun — while the thing that KNOWS the name
#: (loading a .nc) happens much further down, in another tab. Writing the key
#: directly from there raised, and because the raise landed inside the loader's
#: own ``except``, the dataset loaded but kept the previous experiment's name and
#: sent its exports to that folder.
PENDING = "_experiment_name_pending"

HELP = (
    "Names the export folder, so two experiments sharing a working folder do not "
    "overwrite each other's figures. Stored with the dataset, so it survives a "
    "save and reload."
)


def sync_from_dataset(ds):
    """Point the field at ``ds``'s own name. Call when a dataset is loaded.

    Without this the field would keep whatever the last import put there, and the
    first rerun would write that stale name onto the newly loaded dataset — the
    export-redirection bug, arriving through the widget instead of through a
    session-state fallback.

    Safe to call from ANYWHERE in a run, including after the field has been
    rendered: the name is parked in :data:`PENDING` and :func:`name_control`
    picks it up the moment it can. Assigning ``KEY`` here is what broke loading a
    ``.nc`` — see PENDING.
    """
    attrs = getattr(ds, "attrs", None) if ds is not None else None
    st.session_state[PENDING] = str((attrs or {}).get("experiment_name") or "")


def apply_to_loaded(name):
    """Stamp ``name`` onto the loaded dataset(s). Returns True if anything changed.

    Both slots, not just ``dataset``: ``dataset_full`` is the unfiltered copy the
    Groups & subsets page restores from, so a name written to only one of them
    disappears the moment a group filter is reset.
    """
    clean = dam_utilities.sanitize_experiment_name(name)
    changed = False
    for slot in ("dataset", "dataset_full"):
        ds = st.session_state.get(slot)
        if ds is not None and ds.attrs.get("experiment_name", "") != clean:
            ds.attrs["experiment_name"] = clean
            changed = True
    return changed


def take_pending():
    """Move a parked name into the field's key. Call right before the widget.

    Its own function so the handoff can be tested as a PAIR without a Streamlit
    run: ``sync_from_dataset`` then this is the whole contract, and testing either
    half alone pins a mechanism rather than the behaviour.
    """
    if PENDING in st.session_state:
        # Popped, not read: a name is consumed once, so an edit made after the
        # load is not undone on the next rerun.
        st.session_state[KEY] = st.session_state.pop(PENDING)


def name_control(*, label="Experiment name (optional)", caption=True):
    """Render the field, apply it to any loaded dataset, and return the raw text.

    The caption names the folder the current value produces, so the answer is on
    screen rather than discovered afterwards in the filesystem.
    """
    take_pending()
    name = st.text_input(label, key=KEY, help=HELP)
    apply_to_loaded(name)

    if caption:
        suffix = dam_utilities.experiment_suffix_for_name(name)
        st.caption(
            f"Exports go to `Graph Exports{suffix}/` in the working folder."
            if suffix
            else "Exports go to `Graph Exports/` — name the experiment to keep two "
            "runs in one working folder apart."
        )
    return name
