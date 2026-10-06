"""The Period & rhythmicity page records what it ran, as the pipeline's config.

test_page_runs presses the buttons and checks they produce a result. This
checks the result is DESCRIBED correctly: that what the page ran reads back off
the dataset as the PeriodConfig the page's controls mean, and that it is the
same computation the pipeline does when a config file asks for it.
"""

import numpy as np
import pytest

from clockwork import pipeline
from clockwork.pipeline import PeriodConfig
from conftest import requires_example_data

pytestmark = requires_example_data

FLIES = ["20250115_17_1", "20250115_17_2", "20250115_18_1", "20250115_18_2"]
PAGE, TAB = "period_rhythmicity", "period_rhythm_tab"
METHODS = ("cwt", "ls", "ac", "mesa")
# A non-default range and floor, set where the page remembers them, so a page
# that ignored them or recorded the defaults would be caught.
SETTINGS = {"_persist_period_min_h": 18.0, "_persist_period_max_h": 30.0, "_persist_min_days_floor_shared": 3.0}


@pytest.fixture(scope="module")
def page_ds(example_ds):
    return example_ds.sel(id=FLIES).copy()


def _run(app, ds, method):
    ticks = {f"run_method_{m}": m == method for m in METHODS}
    at = app(ds=ds, page=PAGE, **{TAB: "Analysis"}, **ticks, **SETTINGS)
    at.button(key="run_period_analysis").click()
    at.session_state[TAB] = "Analysis"
    at.run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    return at


@pytest.fixture(scope="module")
def expected():
    return PeriodConfig(
        period_range_hours=(18.0, 30.0), min_dd_days=3.0, methods={"autocorrelation": {"classify": False}}
    )


class TestAnalysisTab:
    def test_a_run_reads_back_as_the_config_its_controls_mean(self, app, page_ds, expected):
        at = _run(app, page_ds, "ac")
        ds = at.session_state["dataset"]
        assert PeriodConfig.from_attrs(ds.attrs, ds.coords) == expected
        assert ds.attrs["ac_prep_detrend"] == "linear", "preprocessing must reach the master"

    def test_and_it_is_the_pipeline_s_computation(self, app, page_ds, expected):
        at = _run(app, page_ds, "ac")
        page = at.session_state["dataset"]
        direct, _ = pipeline.run_period_method(page_ds, expected, "autocorrelation")
        for v in (v for v in direct.data_vars if v.startswith("ac_")):
            np.testing.assert_array_equal(
                page[v].sel(id=FLIES).values, direct[v].sel(id=FLIES).values, err_msg=v
            )


class TestCutoffTab:
    def test_classify_records_its_threshold(self, app, page_ds, expected):
        ran, _ = pipeline.run_period_method(page_ds, expected, "autocorrelation")
        at = app(ds=ran, page=PAGE, **{TAB: "Rhythmicity cutoff"}, **SETTINGS, expl_thr_ac=0.4)
        at.button(key="classify_rhythmicity").click()
        at.session_state[TAB] = "Rhythmicity cutoff"
        at.run()
        assert not at.exception, at.exception
        ds = at.session_state["dataset"]
        back = PeriodConfig.from_attrs(ds.attrs, ds.coords).methods.autocorrelation
        assert back.classify is True
        assert back.rhythmic_threshold == pytest.approx(0.4)


class TestGroupScalograms:
    def test_the_checkbox_is_recorded_and_the_files_are_written(self, app, page_ds, tmp_path):
        """The page's "group-averaged scalograms" choice is part of the CWT's
        config, so Export settings carries it to `clockwork run`."""
        ticks = {f"run_method_{m}": m == "cwt" for m in METHODS}
        at = app(
            ds=page_ds,
            page=PAGE,
            **{TAB: "Analysis"},
            **ticks,
            **SETTINGS,
            working_dir=str(tmp_path),
            cwt_compute_group_averages=True,
            # Every fly: no autocorrelation call exists to filter by.
            cwt_avg_filter_nonrhythmic=False,
        )
        at.button(key="run_period_analysis").click()
        at.session_state[TAB] = "Analysis"
        at.run()
        assert not at.exception, at.exception
        ds = at.session_state["dataset"]
        cwt = PeriodConfig.from_attrs(ds.attrs, ds.coords).methods.cwt
        assert cwt.group_scalograms is True
        assert cwt.scalogram_flies == "all"
        pngs = list((tmp_path / "Averaged Scalograms").glob("averaged_scalogram_*_DD_*.png"))
        assert len(pngs) == len({str(g) for g in page_ds["group"].values})
