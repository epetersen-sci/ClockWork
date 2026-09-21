"""Sleep detection, rendered wherever a page first needs it.

Sleep analysis used to be a page of its own. That page existed for one reason:
it writes back to the MASTER dataset, and the page it was carved out of applies
a sidebar group filter that rebinds ``ds`` — so running detection from below the
filter would have dropped every deselected fly from the master, permanently.
Alone on its own page there was no filter to race.

It is not a page any more, because "run the analysis" is not a thing anyone sets
out to do; they set out to look at sleep, and the analysis is what has to happen
first. So the control now appears on the page that wants the answer, and this
module is what makes that safe: **everything here computes on
``st.session_state.dataset`` and never on the caller's ``ds``.** A page may have
narrowed, sliced or re-zeroed its own view by the time it calls this; the master
is the only object that is always every fly.

The work is two steps, and only one of them is expensive:

1. **Detection** — find each fly's immobility bouts and mark the sleeping
   minutes. A pass over every minute of every fly, governed by the immobility
   threshold. This is what :func:`ensure_sleep` runs.
2. **Classification** — sort those bouts into short / intermediate / long. It
   moves no bout boundary, so it is a re-cut of a table that already exists
   (:func:`core.sleep_analysis.reclassify_sleep_states`), and the Sleep states
   page runs it on its own without touching step 1.

Changing the immobility threshold invalidates both, because different bouts get
classified. Changing the state thresholds invalidates only the second.
"""

import streamlit as st

import sleep_analysis
from analysis_detection import detect_analyses
from dataset_meta import PHASE_DD, PHASE_LD, dataset_phase

#: The standard Drosophila sleep definition (Shaw et al. 2000): five minutes of
#: immobility. Every page that offers the control offers this value first.
DEFAULT_SLEEP_THRESHOLD_SEC = 300


def sleep_definition(ds):
    """What "asleep" currently means on this dataset, in one sentence.

    Printed wherever sleep is shown rather than only where it is configured. The
    threshold is a definition, not a setting — every number on the page is "sleep
    by this rule" — and a reader who did not run the analysis has no other way to
    know which rule produced what they are looking at.
    """
    if ds is None or "sleep" not in ds.data_vars:
        return None
    sec = ds.attrs.get("sleep_threshold_seconds")
    if sec is None:
        return "Sleep is whatever the last run recorded — its threshold was not saved."
    minutes = int(sec) // 60
    out = (
        f"**Sleep = {minutes} min ({int(sec)} s) or more of continuous immobility.** "
        "Gaps of up to 4 minutes are bridged so a dropped reading does not split a bout."
    )
    phase = ds.attrs.get("sleep_phase")
    if phase:
        out += f" Computed on **{phase}**."
    short, inter = ds.attrs.get("sleep_short_max_min"), ds.attrs.get("sleep_inter_max_min")
    if short is not None and inter is not None:
        out += (
            f" Bouts are called short below {int(short)} min, intermediate below "
            f"{int(inter)} min, and long at or above it."
        )
    return out


def _phase_argument(ds):
    """Which epoch to detect on, and the label to report it as.

    Always the whole recording when there is one, so the epoch stays a VIEWING
    choice in each page's own sidebar. Detecting a single epoch writes masks that
    are missing everywhere else, and every downstream figure is then blank for
    the other epoch until somebody works out why.

    Detection has no phase-dependent parameter, so this is not a different
    method: the only difference from two separate runs is that a bout straddling
    the LD/DD boundary stays one bout instead of being cut at it, which is the
    more faithful reading of the fly's behaviour.

    A file that IS a single epoch is passed its own phase rather than "both" —
    ``select_phase`` returns a matching request unchanged, and RAISES if asked
    for the epoch the file is not, which is the failure worth having.
    """
    stamped = dataset_phase(ds)
    if stamped in (PHASE_LD, PHASE_DD):
        return stamped, stamped
    return "both", "LD+DD"


def ensure_sleep(*, key_prefix, label="Detect sleep"):
    """Render the sleep-detection control. Returns the master dataset.

    Draws nothing but a definition line when sleep is already present, with the
    controls folded into an expander — rerunning detection is the rare case, and
    it is the one that throws away every state classification downstream.
    """
    ds = st.session_state.get("dataset")
    if ds is None:
        return None

    if "moving" not in ds.data_vars:
        st.warning(
            "Sleep needs the movement data that curation computes. Run "
            "**Data → Curate & split** first."
        )
        return ds

    already = "sleep" in ds.data_vars
    if already:
        st.caption(sleep_definition(ds))
        with st.expander("Change the sleep threshold and re-detect"):
            st.caption(
                "Re-detecting replaces every sleep bout and re-classifies the states "
                "with them. Only change this to use a sleep definition other than the "
                "standard five minutes."
            )
            _run_controls(ds, key_prefix=key_prefix, label="Re-detect sleep")
        return st.session_state.get("dataset", ds)

    st.info(
        "Sleep has not been detected on this dataset yet. A fly is asleep after "
        "five minutes of continuous immobility — the standard definition (Shaw et "
        "al. 2000) — and everything below reads the bouts that rule produces."
    )
    _run_controls(ds, key_prefix=key_prefix, label=label)
    return st.session_state.get("dataset", ds)


def _run_controls(ds, *, key_prefix, label):
    """The threshold input and the button, and the run itself."""
    phase_arg, phase_label = _phase_argument(ds)
    col_a, col_b = st.columns([2, 1])
    with col_a:
        threshold = st.number_input(
            "Sleep threshold (seconds of continuous immobility)",
            min_value=60,
            max_value=1800,
            value=int(
                ds.attrs.get("sleep_threshold_seconds") or DEFAULT_SLEEP_THRESHOLD_SEC
            ),
            step=60,
            key=f"{key_prefix}_threshold",
            help="300 s (5 min) is the standard Drosophila definition. Raising it "
            "counts only longer bouts as sleep; lowering it counts brief quiescence.",
        )
    with col_b:
        st.caption(f"Detecting on **{phase_label}**.")
        go = st.button(label, type="primary", key=f"{key_prefix}_run")

    if not go:
        return

    # A real bar, not a spinner: the work is per fly, and on a few hundred flies a
    # spinner leaves you unable to tell progress from a hang.
    bar = st.progress(0.0, text="Starting sleep detection…")
    try:
        # The MASTER, deliberately — see the module docstring. A page that has
        # filtered its own view must not be able to narrow what gets written back.
        out = sleep_analysis.sleep_analysis(
            ds,
            sleep_threshold_sec=int(threshold),
            short_max_min=float(ds.attrs.get("sleep_short_max_min") or 30),
            inter_max_min=float(ds.attrs.get("sleep_inter_max_min") or 60),
            phase=phase_arg,
            progress_callback=lambda done, total: bar.progress(
                min(1.0, done / max(total, 1)),
                text=f"Detecting sleep bouts: fly {done}/{total}",
            ),
        )
    except Exception as exc:
        st.error(f"Sleep detection failed: {exc}")
        return
    finally:
        # Cleared either way: a bar stuck at 90% after a failure reads as "still going".
        bar.empty()

    st.session_state.dataset = out
    st.session_state.analyses = detect_analyses(out)
    n_bouts = 0
    if "duration" in out.data_vars:
        n_bouts = int(out["duration"].notnull().sum())
    # The message rides across in session state because the rerun below discards
    # everything already drawn, including an st.success written here.
    st.session_state[f"{key_prefix}_message"] = (
        f"Sleep detected: {n_bouts} bouts at a {int(threshold) // 60} min threshold."
    )
    # Outside the try on purpose: st.rerun raises RerunException, which the handler
    # above would swallow and report as a failure.
    st.rerun()


def show_last_message(key_prefix):
    """Report the outcome of a run that ended in a rerun. Call before the control."""
    msg = st.session_state.pop(f"{key_prefix}_message", None)
    if msg:
        st.success(msg)
