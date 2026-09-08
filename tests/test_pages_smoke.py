"""Every page renders without raising, with and without a dataset.

The cheapest test in the suite and the one most likely to catch a careless edit:
a page is a top-level script, so a NameError in a branch nobody clicked still
takes the page down at import time. `at.exception` is empty on a healthy run.
"""

import numpy as np
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
    at. With on_change="rerun" plus `tab.open` guards, only the open tab's
    figures render.

    The count is asserted EXACTLY rather than as an upper bound. Every tab is
    faceted by genotype, so a bound like "at most two" both stopped being true
    when the waveforms were split per group and would have gone on passing if
    a second tab leaked one extra figure through. An exact count is what
    actually says no other tab rendered.
    """
    n_groups = len(set(states_ds["group"].values.tolist()))
    at = app(ds=states_ds, page="sleep_states")
    assert not at.exception
    # The Waveforms tab draws one panel per genotype per epoch present — this
    # fixture spans LD and DD, and Figure 1B prints them side by side.
    landing = len(at.get("plotly_chart"))
    assert landing == 2 * n_groups, (
        f"the Waveforms tab drew {landing} figures, expected {2 * n_groups} "
        f"({n_groups} genotypes x 2 epochs) — either the facet layout changed "
        "or another tab's figures rendered too"
    )

    # Selecting a tab through its key is the only route a headless AppTest has;
    # its Tab objects are read-only.
    at = _open_tab(at, "Rose & gating")
    assert not at.exception
    # One rose row and one gating ring per genotype, and nothing else.
    assert len(at.get("plotly_chart")) == 2 * n_groups, (
        "the rose tab did not draw one rose row and one gating ring per genotype"
    )


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
    at = _open_tab(at, "Scalograms")

    buttons = [b for b in at.button if "wavelet" in b.label.lower()]
    assert buttons, "the wavelet run button is missing"

    buttons[0].click()
    at = _open_tab(at, "Scalograms")
    assert not at.exception, f"the wavelet run raised: {at.exception}"
    assert not at.error, f"the page reported an error: {[e.value for e in at.error]}"
    # AppTest's session_state is subscript-only — it has no .get().
    assert "sleep_states_cwt_runs" in at.session_state, "no run was recorded"
    runs = at.session_state["sleep_states_cwt_runs"]
    assert len(runs) == 1
    (_fp, group, _epoch), _range = next(iter(runs.items()))
    # The run is scoped to ONE genotype — the transforms are averaged over the
    # flies passed, so a pooled run returns a genotype-blind mean surface.
    assert group in set(states_ds["group"].values.tolist())
    # The scalogram and the period-vs-amplitude figure.
    assert len(at.get("plotly_chart")) >= 2

    # The Ultradian tab reads the same run record and hits the same cache,
    # which is why the run is not wrapped in a fragment.
    at = _open_tab(at, "Ultradian")
    assert not at.exception
    assert len(at.get("plotly_chart")) >= 2, "the ultradian tab lost the results"
    # Lomb-Scargle is the default test, so its table renders without a click.
    assert at.session_state["sleep_states_rhythmicity_test"] == "Lomb-Scargle"
    # And it has a genotype pulldown of its own, listing what has been run.
    ultra = [b for b in at.selectbox if b.label == "Genotype"]
    assert ultra, "the Ultradian tab has no genotype selector"
    assert list(ultra[0].options) == [group]


def test_sleep_states_genotype_selector_switches_the_view(app, states_ds):
    """Changing the genotype must change what is on screen, on its own.

    The selector used to live inside an st.form, which only submits on its own
    button — so changing it did nothing visible and the figures went on showing
    whichever genotype had been selected the last time Run was pressed. That
    reads as a selector that is simply broken.
    """
    at = app(ds=states_ds, page="sleep_states")
    at = _open_tab(at, "Scalograms")
    groups = [str(g) for g in dict.fromkeys(states_ds["group"].values.tolist())]
    assert len(groups) > 1

    # Run the first genotype.
    [b for b in at.button if "wavelet" in b.label.lower()][0].click()
    at = _open_tab(at, "Scalograms")
    first = _titles(at)
    assert any(groups[0] in t for t in first), first

    # Switch WITHOUT pressing Run: an un-run genotype prompts rather than
    # silently starting an expensive transform, and says which one it means.
    at.session_state["sleep_states_cwt_group"] = groups[1]
    at = _open_tab(at, "Scalograms")
    assert not at.exception
    assert not _titles(at), "an un-run genotype should not draw the previous one"
    assert any(groups[1] in i.value for i in at.info), [i.value for i in at.info]

    # Run it, then switch BACK: the first genotype returns from cache with no
    # Run press at all, which is the behaviour the form prevented.
    [b for b in at.button if "wavelet" in b.label.lower()][0].click()
    at = _open_tab(at, "Scalograms")
    assert any(groups[1] in t for t in _titles(at))

    at.session_state["sleep_states_cwt_group"] = groups[0]
    at = _open_tab(at, "Scalograms")
    back = _titles(at)
    assert back and any(groups[0] in t for t in back), back
    assert not any(groups[1] in t for t in back), back


def _titles(at):
    """Plotly figure titles on the page."""
    import json

    out = []
    for el in at.get("plotly_chart"):
        title = json.loads(el.proto.spec).get("layout", {}).get("title", {})
        if isinstance(title, dict) and title.get("text"):
            out.append(title["text"])
    return out


def test_sleep_states_wavelet_is_per_genotype(app, states_ds):
    """Each genotype must get its own surfaces, not a shared pooled one.

    ``sleep_cwt_analysis`` averages the per-fly transforms across whatever
    flies it is handed, so running the whole dataset produced ONE surface per
    state with every genotype averaged into it — the only thing these two tabs
    could show, and not a description of any genotype in the experiment.
    """
    from dam_utilities import select_phase
    from periodograms import sleep_cwt_analysis

    view, used = select_phase(states_ds, phase="DD")
    ids = [str(i) for i in view["id"].values]
    labels = [str(g) for g in view["group"].values]
    groups = list(dict.fromkeys(labels))
    assert len(groups) > 1, "this fixture needs several genotypes to prove anything"

    surfaces = {}
    for group in groups[:2]:
        fly_ids = [i for i, g in zip(ids, labels) if g == group]
        out = sleep_cwt_analysis(
            view, states=("standard",), fly_ids=fly_ids, full_range=(1.0, 32.0), phase=used
        )
        surfaces[group] = out["sleep_cwt_standard_full_avg_surface"].values

    a, b = (surfaces[g] for g in groups[:2])
    assert a.shape == b.shape
    assert not np.allclose(a, b, equal_nan=True), (
        "two genotypes produced identical surfaces — fly_ids is not scoping the run"
    )
