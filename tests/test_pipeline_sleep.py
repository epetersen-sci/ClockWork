"""The sleep step: detection, re-classification, and the round trip.

Equivalence with the core call the pages made, the config reading back off the
dataset (through a .nc too), and the two pages that run it — Activity & Sleep
(detect) and Sleep states (re-cut the boundaries) — recording what they ran.
"""

import pytest
import xarray as xr
from pydantic import ValidationError

from clockwork import pipeline
from clockwork.core import sleep_analysis
from clockwork.core.load_and_save_datasets import save_dataset_to_netcdf
from clockwork.pipeline import CurationConfig, SleepConfig
from conftest import requires_example_data

pytestmark = requires_example_data

FLIES = ["20250115_17_1", "20250115_17_2", "20250115_18_1", "20250115_18_2"]


@pytest.fixture(scope="module")
def curated(example_ds):
    return pipeline.curate(example_ds.sel(id=FLIES), CurationConfig()).live


@pytest.fixture(scope="module")
def slept(curated):
    return pipeline.detect_sleep(curated, SleepConfig(threshold_seconds=420, short_max_minutes=20))


class TestEquivalence:
    def test_detection_is_the_core_call(self, curated, slept):
        direct = sleep_analysis.sleep_analysis(
            curated, sleep_threshold_sec=420, short_max_min=20.0, inter_max_min=60.0, phase="both"
        )
        xr.testing.assert_identical(slept, direct)

    def test_reclassification_is_the_core_call(self, slept):
        cfg = SleepConfig(threshold_seconds=420, short_max_minutes=15, intermediate_max_minutes=90)
        via = pipeline.reclassify_sleep(slept, cfg)
        direct = sleep_analysis.reclassify_sleep_states(slept, short_max_min=15.0, inter_max_min=90.0)
        xr.testing.assert_identical(via, direct)


class TestRoundTrip:
    def test_detection_reads_back(self, slept):
        assert SleepConfig.from_attrs(slept.attrs) == SleepConfig(threshold_seconds=420, short_max_minutes=20)

    def test_and_through_a_netcdf(self, slept, tmp_path):
        path = tmp_path / "sleep.nc"
        save_dataset_to_netcdf(slept, str(path))
        back = pipeline.load_netcdf(path)
        assert SleepConfig.from_attrs(back.attrs) == SleepConfig(threshold_seconds=420, short_max_minutes=20)

    def test_a_reclassification_keeps_the_threshold_and_changes_the_boundaries(self, slept):
        cfg = SleepConfig(threshold_seconds=420, short_max_minutes=15, intermediate_max_minutes=90)
        out = pipeline.reclassify_sleep(slept, cfg)
        assert SleepConfig.from_attrs(out.attrs) == cfg

    def test_no_detection_reads_back_as_none(self, curated):
        assert SleepConfig.from_attrs(curated.attrs) is None

    def test_overrides(self):
        assert SleepConfig().overrides() == {}
        assert SleepConfig(threshold_seconds=420).overrides() == {"threshold_seconds": 420}


class TestValidation:
    def test_detection_without_curation_says_why(self, example_ds):
        with pytest.raises(ValueError, match="curation"):
            pipeline.detect_sleep(example_ds.sel(id=FLIES), SleepConfig())

    def test_the_boundaries_must_be_in_order(self):
        with pytest.raises(ValidationError, match="must be above"):
            SleepConfig(short_max_minutes=60, intermediate_max_minutes=30)

    def test_the_threshold_has_the_page_s_bounds(self):
        with pytest.raises(ValidationError):
            SleepConfig(threshold_seconds=30)


class TestPages:
    def test_detect_on_activity_and_sleep_records_the_threshold(self, app, curated):
        # The Sleep tab is lazy (on_change="rerun"): open it, and keep it open
        # across the rerun the click causes, as test_sleep_reorganisation does.
        at = app(ds=curated, page="sleep_activity", sleep_activity_tab="Sleep")
        at.number_input(key="sleep_activity_threshold").set_value(480)
        at.button(key="sleep_activity_run").click()
        at.session_state["sleep_activity_tab"] = "Sleep"
        at.run()
        assert not at.exception, at.exception
        ds = at.session_state["dataset"]
        assert SleepConfig.from_attrs(ds.attrs) == SleepConfig(threshold_seconds=480)

    def test_reclassify_on_sleep_states_records_the_boundaries(self, app, slept):
        at = app(ds=slept, page="sleep_states", ss_short_max=10, ss_inter_max=45)
        at.button(key="ss_apply_thresholds").click().run()
        assert not at.exception, at.exception
        ds = at.session_state["dataset"]
        assert SleepConfig.from_attrs(ds.attrs) == SleepConfig(
            threshold_seconds=420, short_max_minutes=10, intermediate_max_minutes=45
        )
