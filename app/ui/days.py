"""The shared day picker: the epoch-numbered day table, the tick-box widget, and
the label/slug helpers that turn a selection into figure titles and filenames.

Days are numbered inside their epoch — LD day 1..n from the recording start, DD day
1..n from ``first_DD_day`` — so a selection reads the way a free-run is described.

**One implementation, two pages.** Sleep (SCAMP) and Sleep bouts both pick days this
way, and they arrived with a copy each: the second was written to match the first and
carried a note saying to change any convention in both places on purpose. This
codebase has that lesson written down twice already — nine copies of the dataset
guard had drifted before ``ui.guards`` existed (item 11 is the same story for the
group filter), and the conventions here are exactly the kind that drift silently: day
numbering, whether the transition day starts ticked, and how a range is spelled in a
filename. So there is one copy, and it lives here.

The filename spelling is the subtle one. :func:`slug` and ``ui.charts`` both flatten
runs of punctuation to a single underscore, so a contiguous ``3-5`` and a
discontiguous ``3, 5`` slug identically — which is why :func:`compress_runs` takes a
``dash`` and the callers pass ``"to"`` when the text is headed for a filename.
"""

import re

import numpy as np
import pandas as pd
import streamlit as st

import dam_utilities

MINUTES_PER_DAY = 1440


def day_table(ds, *, warn=None):
    """``(DataFrame, dd_day)`` — one row per complete day of the recording.

    Columns: ``absolute`` (0-based day index into the record), ``epoch``
    (``'LD'``/``'DD'``), ``within`` (1-based day number inside that epoch) and
    ``label`` (``'DD day 3'``). ``dd_day`` is the absolute index of the first DD day,
    or ``None`` when the dataset carries no boundary — in which case every day is
    labelled LD.

    Flies released on different days give no single boundary; rather than pick one,
    the whole record is labelled LD and ``warn`` (a callable taking a message, e.g.
    ``st.warning``) is told why.
    """
    minutes = np.asarray(ds["time"].values, dtype=float)
    n_days_total = int(np.floor((minutes[-1] + 1) / MINUTES_PER_DAY)) if minutes.size else 0

    dd_day = None
    if "split_minute" in ds.coords or "first_DD_day" in ds.coords:
        try:
            split = (
                ds["split_minute"].values
                if "split_minute" in ds.coords
                else dam_utilities.add_phase_metadata(ds)["split_minute"].values
            )
            dd_days = {int(round(float(s) / MINUTES_PER_DAY)) for s in np.asarray(split)}
            if len(dd_days) == 1:
                dd_day = dd_days.pop()
            elif warn is not None:
                warn(
                    "Flies enter DD on different days of this dataset's time axis "
                    f"({sorted(dd_days)}), so there is no single LD→DD transition to "
                    "number days from and the whole record is labelled LD."
                )
        except Exception:
            dd_day = None

    rows = []
    for d in range(n_days_total):
        if dd_day is None or d < dd_day:
            epoch, within = "LD", d + 1
        else:
            epoch, within = "DD", d - dd_day + 1
        rows.append(
            {"absolute": d, "epoch": epoch, "within": within, "label": f"{epoch} day {within}"}
        )
    return pd.DataFrame(rows, columns=["absolute", "epoch", "within", "label"]), dd_day


def compress_runs(nums, dash="-"):
    """[2,3,4,7] -> '2-4, 7' — consecutive days collapse into a range.

    ``dash="to"`` gives the filename spelling (``'2to4, 7'``); see the module
    docstring for why a plain dash cannot be used there.
    """
    nums = sorted({int(n) for n in nums})
    if not nums:
        return ""
    runs, start, prev = [], nums[0], nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        runs.append((start, prev))
        start = prev = n
    runs.append((start, prev))
    return ", ".join(str(a) if a == b else f"{a}{dash}{b}" for a, b in runs)


def day_summary(rows, epoch, *, dash="-"):
    """'DD days 2-5' / 'LD days 1, 3'. The labels already carry the epoch, so
    joining them raw gave titles like 'DD DD day 2, DD day 3'."""
    nums = sorted(int(n) for n in rows.within)
    if not nums:
        return epoch
    word = "day" if len(nums) == 1 else "days"
    return f"{epoch} {word} {compress_runs(nums, dash)}"


def days_label(rows, *, dash="-"):
    """Same as :func:`day_summary` but for a selection that may span both epochs:
    ``'LD days 2-4 + DD days 1-5'``.

    This text goes into the figure TITLE and, spelled with ``dash='to'``, into the
    exported filename — so two exports of the same view over different days land in
    different files instead of the second silently overwriting the first.
    """
    order = {"LD": 0, "DD": 1}
    parts = []
    for ep in sorted(set(rows.epoch), key=lambda e: order.get(e, 9)):
        sub = rows[rows.epoch == ep]
        if len(sub):
            parts.append(day_summary(sub, ep, dash=dash))
    return " + ".join(parts)


def slug(text):
    """Filename-safe token, capped at 80 characters.

    For a CSV name built by the page. A figure name needs no slugging here —
    ``ui.charts`` slugs whatever ``filename`` it is handed.
    """
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_")[:80] or "figure"


def day_checkboxes(rows, key, *, default=None, per_row=6):
    """Tick boxes for days, laid out across columns, plus All / None shortcuts.

    A multiselect hides what is currently chosen behind a dropdown; with eight days
    and two epochs the whole selection wants to be visible at once.

    Returns ``(selected rows, chosen labels)`` — the rows are the ``day_table``
    subset, sorted by ``absolute``.
    """
    labels = list(rows.label)
    default = labels if default is None else list(default)

    def _cb_key(lbl):
        return f"{key}_cb_{lbl}"

    # Each box's own widget key IS its state. Streamlit ignores a keyed widget's
    # ``value=`` argument on every run after the first, so All / None cannot work by
    # writing to a separate dict and passing it back in as ``value`` — the widget
    # never reads it, which is why None appeared to do nothing. They have to assign
    # the widget keys themselves.
    for lbl in labels:  # a changed epoch or dataset can bring new labels
        if _cb_key(lbl) not in st.session_state:
            st.session_state[_cb_key(lbl)] = lbl in default

    b1, b2, _ = st.columns([1, 1, 6])
    # These run BEFORE the checkboxes below are instantiated in this same script
    # run, which is the only window in which assigning a widget's key still
    # changes what that widget renders.
    if b1.button("All", key=f"{key}_all"):
        for lbl in labels:
            st.session_state[_cb_key(lbl)] = True
    if b2.button("None", key=f"{key}_none"):
        for lbl in labels:
            st.session_state[_cb_key(lbl)] = False

    chosen = []
    for start in range(0, len(labels), per_row):
        chunk = labels[start : start + per_row]
        cols = st.columns(per_row)
        for col, lbl in zip(cols, chunk):
            # No ``value=``: the key already carries the state, and passing both
            # makes Streamlit warn about a value that will be ignored.
            if col.checkbox(lbl, key=_cb_key(lbl)):
                chosen.append(lbl)
    return rows[rows.label.isin(chosen)].sort_values("absolute"), chosen


def persist_checkbox_keys(*prefixes):
    """Keep day tick-boxes alive across a view switch on the same page.

    Streamlit drops a keyed widget's value once the widget stops being rendered, and
    a page that swaps one view for another does exactly that — so the days ticked in
    one view were gone on returning to it. Re-assigning each key marks it as
    script-owned and keeps it.

    An ALLOWLIST of prefixes, never a sweep over everything: a button refuses to be
    created at all if its key already carries a value
    (``StreamlitValueAssignmentNotAllowedError``), and that is raised when
    ``st.button`` runs rather than when the value is assigned, so it cannot be caught
    here. Buttons need no persisting anyway — a button's value is only True on the
    run right after its click, which is why the All/None pair is excluded by using
    the ``_cb_`` prefix rather than ``key`` itself.
    """
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith(tuple(prefixes)):
            st.session_state[k] = st.session_state[k]
