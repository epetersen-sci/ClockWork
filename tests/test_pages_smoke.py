"""Every page renders without raising, with and without a dataset.

The cheapest test in the suite and the one most likely to catch a careless edit:
a page is a top-level script, so a NameError in a branch nobody clicked still
takes the page down at import time. `at.exception` is empty on a healthy run.
"""

import pytest

PAGES = [
    "home",
    "data_import",
    "data_groups",
    "data_curate_split",
    "period_analysis",
    "periodograms",
    "rhythmicity",
    "phase_shift",
    "sleep_detection",
    "sleep_activity",
    "sleep_states",
    "sleep_deprivation",
    "hmm_model_selection",
    "hmm_analysis",
    "export_data",
    "export_scamp",
]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_with_dataset(app, master_ds, page):
    at = app(ds=master_ds, page=page)
    assert not at.exception, f"{page} raised: {at.exception}"


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_dataset(app, page):
    """No dataset loaded is a supported state: pages must warn and stop, not crash.

    require_dataset() calls st.stop(), which AppTest reports as a clean run.
    """
    at = app(ds=None, page=page)
    assert not at.exception, f"{page} raised with no dataset: {at.exception}"


def test_unsplit_dataset_renders(app, unsplit_ds):
    """A dataset with no LD/DD boundary hits the fallback branch on every page
    that offers a phase choice — the branch the LD/DD fixtures never reach."""
    for page in ("sleep_detection", "hmm_model_selection", "period_analysis"):
        at = app(ds=unsplit_ds, page=page)
        assert not at.exception, f"{page} raised on an unsplit dataset: {at.exception}"


TAB_KEY = "sleep_states_tab"


def _open_tab(at, label):
    """Select a dynamic tab, then rerun.

    A Tab object in AppTest is read-only, so the only handle on the selection
    is the widget key. It also does NOT survive a rerun triggered by another
    widget, so it has to be re-asserted immediately before any `.run()` that
    follows a click — otherwise the page falls back to the landing tab and the
    assertions silently measure the wrong panel.
    """
    at.session_state[TAB_KEY] = label
    return at.run()


def test_sleep_states_tabs_are_gated(app, states_ds):
    """Only the OPEN tab's figures may render.

    st.tabs renders every tab's content by default, so this page used to draw
    all nineteen figures — waveforms, six initiation rows, six rose rows, six
    gating rings — on 189 flies for whichever single tab the user was looking
    at. With on_change="rerun" plus `tab.open` guards, the landing tab draws
    one figure.
    """
    at = app(ds=states_ds, page="sleep_states")
    assert not at.exception
    landing = len(at.get("plotly_chart"))
    # The Waveforms tab draws one panel per epoch present (Figure 1B prints LD
    # beside DD), so at most two. Anything more means another tab's figures
    # rendered too.
    assert landing <= 2, (
        f"the Waveforms tab drew {landing} figures — at most two epochs are "
        "expected, so the tab guards are not holding"
    )

    # Selecting a tab through its key is the only route a headless AppTest has;
    # its Tab objects are read-only.
    at = _open_tab(at, "Rose & gating (Fig 3)")
    assert not at.exception
    # One rose row and one gating ring per group, so strictly more than the
    # waveform tab drew — and proof the guard lets the OPEN tab through.
    assert len(at.get("plotly_chart")) > landing, "the rose tab drew nothing"


def test_sleep_states_wavelet_button_runs(app, states_ds):
    """The wavelet run is behind a button on a tab that is closed by default, so
    nothing else on this page reaches it — and it is the one path here that can
    fail on real data while an empty render looks fine.

    It caught two things worth keeping a test for: the page passed
    ``phase="both"``, which the period-analysis phase guard rejects outright,
    and a ``select_phase`` view carries NaN rather than the -1 the CWT's
    missing-value guard looked for, which turned the entire averaged surface
    into NaN.
    """
    at = app(ds=states_ds, page="sleep_states")
    at = _open_tab(at, "Scalograms (Fig 5)")

    buttons = [b for b in at.button if "wavelet" in b.label.lower()]
    assert buttons, "the wavelet run button is missing"

    buttons[0].click()
    at = _open_tab(at, "Scalograms (Fig 5)")
    assert not at.exception, f"the wavelet run raised: {at.exception}"
    assert not at.error, f"the page reported an error: {[e.value for e in at.error]}"
    assert "sleep_states_cwt" in at.session_state, "no results were stored"
    # The scalogram and the period-vs-amplitude figure.
    assert len(at.get("plotly_chart")) >= 2

    # The Ultradian tab reads the same session-state results, which is why the
    # run is not wrapped in a fragment.
    at = _open_tab(at, "Ultradian (Fig 6)")
    assert not at.exception
    assert len(at.get("plotly_chart")) >= 2, "the ultradian tab lost the results"
