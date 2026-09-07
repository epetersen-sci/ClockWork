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


def test_sleep_states_wavelet_button_runs(app, states_ds):
    """The Sleep states page keeps its wavelet run behind a button, so the
    parametrised smoke tests above never touch it — and it is the one path on
    that page that can fail on real data while an empty render looks fine.

    It caught two things worth keeping a test for: the page passed
    ``phase="both"``, which the period-analysis phase guard rejects outright,
    and a ``select_phase`` view carries NaN rather than the -1 the CWT's
    missing-value guard looked for, which turned the entire averaged surface
    into NaN.
    """
    at = app(ds=states_ds, page="sleep_states")
    buttons = [b for b in at.button if "wavelet" in b.label.lower()]
    assert buttons, "the wavelet run button is missing"

    at = buttons[0].click().run()
    assert not at.exception, f"the wavelet run raised: {at.exception}"
    assert not at.error, f"the page reported an error: {[e.value for e in at.error]}"
    # Scalograms, period-vs-amplitude and the ultradian tab all draw once the
    # results are in session state.
    assert len(at.get("plotly_chart")) > 5
