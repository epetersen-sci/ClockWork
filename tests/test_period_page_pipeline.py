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

pytestmark = [requires_example_data, pytest.mark.usefixtures("shared_worker_pool")]

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
    @pytest.fixture(scope="class")
    def ac_run(self, page_ds):
        """One press of Run analysis, shared by both checks on what it left."""
        from conftest import make_app

        return _run(make_app, page_ds, "ac")

    def test_a_run_reads_back_as_the_config_its_controls_mean(self, ac_run, expected):
        ds = ac_run.session_state["dataset"]
        assert PeriodConfig.from_attrs(ds.attrs, ds.coords) == expected
        assert ds.attrs["ac_prep_detrend"] == "linear", "preprocessing must reach the master"

    def test_and_it_is_the_pipeline_s_computation(self, ac_run, page_ds, expected):
        page = ac_run.session_state["dataset"]
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
    @pytest.fixture(scope="class")
    def cwt_ds(self, page_ds):
        config = PeriodConfig(
            period_range_hours=(18.0, 30.0),
            min_dd_days=3.0,
            methods={"cwt": {"group_scalograms": True, "scalogram_flies": "all"}},
        )
        master, averages = pipeline.run_period_method(page_ds, config, "cwt")
        return master, averages

    def test_the_tab_recomputes_what_the_run_averaged(self, cwt_ds):
        """Recomputed afterwards from the recorded settings, the averages are the
        ones the CWT run itself produced."""
        from clockwork.pipeline import scalograms

        master, run_averages = cwt_ds
        again, counts = scalograms.compute_group_average_scalograms(master)
        by_group = {a["group"]: a for a in run_averages}
        assert {a["group"] for a in again} == set(by_group)
        for a in again:
            ref = by_group[a["group"]]
            assert a["n_flies"] == ref["n_flies"] == counts[a["group"]][0]
            np.testing.assert_allclose(a["mean_power"], ref["mean_power"], rtol=1e-5)

    def test_a_rhythmic_call_narrows_the_flies(self, cwt_ds):
        from clockwork.pipeline import scalograms

        master, _ = cwt_ds
        flag = np.zeros(master.sizes["id"], dtype=bool)
        flag[0] = True
        marked = master.assign_coords(ac_rhythmic=("id", flag))
        averages, counts = scalograms.compute_group_average_scalograms(marked, flag="ac_rhythmic")
        assert sum(n for n, _ in counts.values()) == 1
        assert len(averages) == 1

    def test_without_a_cwt_run_it_says_so(self, page_ds):
        from clockwork.pipeline import scalograms

        with pytest.raises(ValueError, match="CWT has not been run"):
            scalograms.compute_group_average_scalograms(page_ds)

    def test_the_tab_draws_and_saves_them(self, app, cwt_ds, tmp_path):
        master, _ = cwt_ds
        at = app(ds=master, page=PAGE, **{TAB: "Scalograms"}, **SETTINGS, working_dir=str(tmp_path))
        assert not at.exception, at.exception
        at.button(key="scalogram_run").click()
        at.session_state[TAB] = "Scalograms"
        at.run()
        assert not at.exception, at.exception
        assert not at.error, [e.value for e in at.error]
        pngs = list((tmp_path / "Averaged Scalograms").glob("averaged_scalogram_*.png"))
        assert len(pngs) == len({str(g) for g in master["group"].values})
        # And the one-fly view below it draws a figure.
        assert at.get("plotly_chart")
