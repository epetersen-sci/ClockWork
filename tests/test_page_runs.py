"""Pressing the Run buttons the smoke tests never press.

``test_pages_smoke`` renders every page, but each heavy analysis sits behind an
``st.button``, and a page that renders fine can still fail the moment Run is
pressed: the page builds the arguments, wires the progress callback, and stores
the result, and none of that runs until the click. These press them — Period &
rhythmicity (Run analysis for each of the four methods, and Classify), Sleep
deprivation, and Curate & split —
on four real flies from example_data, and assert each run neither raised nor
reported an error and left its result where the next page looks for it.

What the analyses compute is pinned elsewhere (test_period_estimators_snapshot,
test_curation, test_sleep_deprivation). This file is about the pages.
"""

import pytest

from clockwork.core import dam_utilities, periodograms, sleep_analysis
from clockwork.core.preprocessing import ac_default_config, ls_default_config, preprocess_activity

FLIES = ["20250115_17_1", "20250115_17_2", "20250115_18_1", "20250115_18_2"]


@pytest.fixture(scope="module")
def page_ds(example_ds):
    return example_ds.sel(id=FLIES).copy()


def _ok(at, what):
    assert not at.exception, f"{what} raised: {at.exception}"
    assert not at.error, f"{what} reported: {[e.value for e in at.error]}"


def _click(at, key, *, tab_key=None, tab=None):
    """Click a button, re-selecting the lazy tab it lives on across the rerun
    (a tab selection does not survive a rerun caused by another widget)."""
    at.button(key=key).click()
    if tab_key:
        at.session_state[tab_key] = tab
    return at.run()


PAGE = "period_rhythmicity"
PERIOD_TAB = "period_rhythm_tab"
METHODS = ("cwt", "ls", "ac", "mesa")


@pytest.mark.parametrize(
    "method, result_var",
    [("ls", "ls_period"), ("ac", "ac_period"), ("mesa", "mesa_period"), ("cwt", "cwt_period")],
)
def test_period_analysis_runs(app, page_ds, method, result_var):
    # Tick only this method, so each parametrisation presses Run for one of them.
    ticks = {f"run_method_{m}": m == method for m in METHODS}
    at = app(ds=page_ds, page=PAGE, **{PERIOD_TAB: "Analysis"}, **ticks)
    _ok(at, f"period & rhythmicity (Analysis tab, {method} ticked)")
    at = _click(at, "run_period_analysis", tab_key=PERIOD_TAB, tab="Analysis")
    _ok(at, f"the {method} run")
    assert result_var in at.session_state["dataset"], f"{method} stored no {result_var}"


@pytest.fixture(scope="module")
def period_results_ds(page_ds):
    """What the Rhythmicity page expects to find: LS and AC already run."""
    dd, _ = dam_utilities.select_phase(page_ds, "DD")
    ls = periodograms.lomb_scargle_analysis(
        preprocess_activity(dd, ls_default_config()), phase="DD", n_processes=2
    )
    ac = periodograms.autocorrelation_analysis(
        preprocess_activity(dd, ac_default_config()), phase="DD", n_processes=2
    )
    ls_vars = [v for v in ls.data_vars if v.startswith("ls_")]
    ac_vars = [v for v in ac.data_vars if v.startswith("ac_")]
    merge_kw = {"compat": "no_conflicts", "join": "outer"}
    return page_ds.merge(ls[ls_vars], **merge_kw).merge(ac[ac_vars], **merge_kw)


def test_rhythmicity_classification_runs(app, period_results_ds):
    tab = "Rhythmicity cutoff"
    at = app(ds=period_results_ds, page=PAGE, **{PERIOD_TAB: tab})
    _ok(at, "rhythmicity cutoff")
    at = _click(at, "classify_rhythmicity", tab_key=PERIOD_TAB, tab=tab)
    _ok(at, "the classification run")
    ds = at.session_state["dataset"]
    assert "ac_rhythmic" in ds.coords
    assert ds["ac_rhythmic"].sizes["id"] == len(FLIES)


@pytest.fixture(scope="module")
def sleep_ds(page_ds):
    return sleep_analysis.sleep_analysis(
        dam_utilities._compute_moving(page_ds), phase="LD", sleep_threshold_sec=300
    )


def test_sleep_deprivation_runs(app, sleep_ds):
    at = app(ds=sleep_ds, page="sleep_deprivation")
    _ok(at, "sleep deprivation")
    buttons = [b for b in at.button if b.label == "Run Sleep Deprivation Analysis"]
    assert buttons, "the Run button is missing"
    buttons[0].click()
    at = at.run()
    _ok(at, "the SD run")
    results = at.session_state["sd_results"]
    assert results["n_baseline_days"] == 1 and results["n_recovery_days"] == 1
    assert at.session_state["dataset"].attrs["sd_day_number"] == 2


def test_curation_runs_and_hands_on_the_curated_dataset(app, page_ds):
    at = app(ds=page_ds, page="data_curate_split")
    _ok(at, "curate & split")
    at = _click(at, "run_curation")
    _ok(at, "the curation run")
    curated = at.session_state["dataset"]
    assert "is_alive" in curated
    assert curated.attrs["curation_time_window_hours"] == 24
    assert "curated_dead_data" in at.session_state
