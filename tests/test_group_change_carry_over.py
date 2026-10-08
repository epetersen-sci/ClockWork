"""BACKLOG 22: a group change keeps the work done on the flies.

The Groups page rebuilds every subset and regroup from the import-time copy,
which predates curation, the split and every analysis. pipeline.carry_over puts
that work back. The contract it is held to here: carrying work across a subset
gives exactly what doing the work after the subset gives. Curation, the split
and sleep are re-applied from their recorded settings; period results are kept
fly by fly; results that describe GROUPS (the HMM) are dropped and said so.
"""

import shutil

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from clockwork import pipeline
from clockwork.core.dataset_meta import is_split_applied
from clockwork.pipeline import CurationConfig, PeriodConfig, SleepConfig, SplitConfig
from conftest import EXAMPLE_DIR, EXAMPLE_MONITORS, requires_example_data

pytestmark = requires_example_data

LS_ONLY = PeriodConfig(methods={"lomb_scargle": {}})
KEEP_ONE = {"genotype": ["dsmcherry"]}


@pytest.fixture(scope="module")
def imported(tmp_path_factory):
    folder = tmp_path_factory.mktemp("carry_raw")
    for m in EXAMPLE_MONITORS:
        shutil.copy(EXAMPLE_DIR / f"Monitor{m}.txt", folder / f"Monitor{m}.txt")
    meta = pd.read_excel(EXAMPLE_DIR / "metadata.xlsx")
    meta[meta["Monitor"].isin(EXAMPLE_MONITORS)].to_csv(folder / "metadata.csv", index=False)
    inputs = pipeline.InputsConfig(metadata=folder / "metadata.csv", monitors=folder)
    return pipeline.build_dataset(pipeline.read_monitors(inputs), inputs)


def _work(ds):
    """Curate, split, detect sleep, run Lomb-Scargle: the work a group change must keep."""
    ds = pipeline.curate(ds, CurationConfig(min_alive_days=3)).live
    ds = pipeline.split(ds, SplitConfig(gap_threshold_minutes=30))
    ds = pipeline.detect_sleep(ds, SleepConfig(threshold_seconds=420))
    return pipeline.run_period(ds, LS_ONLY)


@pytest.fixture(scope="module")
def worked(imported):
    return _work(imported)


class TestSubsetAfterTheWork:
    @pytest.fixture(scope="class")
    def carried(self, imported, worked):
        return pipeline.carry_over(pipeline.subset(imported, KEEP_ONE), worked)

    @pytest.fixture(scope="class")
    def done_after(self, imported):
        return _work(pipeline.subset(imported, KEEP_ONE))

    def test_it_is_the_work_done_after_the_subset(self, carried, done_after):
        ds, _ = carried
        assert list(ds["id"].values) == list(done_after["id"].values)
        for v in ("activity", "moving", "is_alive", "sleep", "sleep_long"):
            xr.testing.assert_equal(ds[v], done_after[v])
        assert CurationConfig.from_attrs(ds.attrs) == CurationConfig(min_alive_days=3)
        assert SplitConfig.from_attrs(ds.attrs) == SplitConfig(gap_threshold_minutes=30)
        assert is_split_applied(ds)
        assert SleepConfig.from_attrs(ds.attrs) == SleepConfig(threshold_seconds=420)

    def test_period_results_are_kept_fly_by_fly(self, carried, worked, done_after):
        ds, notes = carried
        ids = ds["id"].values
        for v in ("ls_period", "ls_power", "ls_periodogram"):
            np.testing.assert_array_equal(ds[v].values, worked[v].sel(id=ids).values, err_msg=v)
        np.testing.assert_array_equal(ds["ls_rhythmic"].values, worked["ls_rhythmic"].sel(id=ids).values)
        # And they are what running it after the subset gives.
        np.testing.assert_allclose(ds["ls_period"].values, done_after["ls_period"].values)
        assert PeriodConfig.from_attrs(ds.attrs, ds.coords) == LS_ONLY
        assert any("Period" in n and "kept" in n for n in notes)

    def test_the_notes_say_what_was_re_applied(self, carried):
        _, notes = carried
        text = " ".join(notes)
        assert "Curation re-applied" in text and "split re-applied" in text and "Sleep re-detected" in text


class TestWideningBringsInFliesNeverAnalysed:
    def test_period_results_are_dropped_and_said_so(self, imported, worked):
        narrowed, _ = pipeline.carry_over(pipeline.subset(imported, KEEP_ONE), worked)
        # Back to every group: the other genotype was never analysed.
        widened, notes = pipeline.carry_over(imported, narrowed)
        assert "ls_period" not in widened.data_vars
        assert PeriodConfig.from_attrs(widened.attrs, widened.coords) is None
        assert any("never analysed" in n for n in notes)
        # The flies' own steps are still re-applied for the whole set.
        assert CurationConfig.from_attrs(widened.attrs) is not None
        assert "sleep" in widened.data_vars
        assert widened.sizes["id"] > narrowed.sizes["id"]


class TestGroupLevelResultsAreDropped:
    def test_the_hmm_does_not_carry_over(self, imported, worked):
        with_hmm = worked.assign(hmm_state=xr.zeros_like(worked["sleep"]))
        ds, notes = pipeline.carry_over(pipeline.subset(imported, KEEP_ONE), with_hmm)
        assert "hmm_state" not in ds.data_vars
        assert any("HMM" in n for n in notes)


class TestTheGroupsPage:
    def test_a_subset_keeps_curation_split_sleep_and_periods(self, app, imported, worked):
        at = app(ds=worked, page="data_groups", dataset_full=imported)
        label = sorted({str(g) for g in worked["group"].values})[0]
        at.multiselect(key="group_filter_select").set_value([label]).run()
        at.button(key="apply_group_filter").click().run()
        assert not at.exception, at.exception
        ds = at.session_state["dataset"]
        assert {str(g) for g in ds["group"].values} == {label}
        assert CurationConfig.from_attrs(ds.attrs) == CurationConfig(min_alive_days=3)
        assert is_split_applied(ds)
        assert "sleep" in ds.data_vars and "ls_period" in ds.data_vars
        # The rerun after the click says what happened.
        assert any("Groups changed" in s.value for s in at.success)

    def test_a_regroup_keeps_them_too(self, app, imported, worked):
        at = app(ds=worked, page="data_groups", dataset_full=imported)
        at.multiselect(key="regroup_columns").set_value(["Monitor"]).run()
        at.button(key="apply_regroup").click().run()
        assert not at.exception, at.exception
        ds = at.session_state["dataset"]
        assert pipeline.GroupsConfig.from_attrs(ds.attrs).by == ["Monitor"]
        assert ds.sizes["id"] == worked.sizes["id"]
        assert CurationConfig.from_attrs(ds.attrs) is not None
        np.testing.assert_array_equal(
            ds["ls_period"].sel(id=worked["id"].values).values, worked["ls_period"].values
        )

    def test_no_warning_when_nothing_would_be_lost(self, app, imported, worked):
        at = app(ds=worked, page="data_groups", dataset_full=imported)
        assert not [w for w in at.warning if "discards some results" in w.value]

    def test_the_warning_names_what_would_be_lost(self, app, imported, worked):
        with_hmm = worked.assign(hmm_state=xr.zeros_like(worked["sleep"]))
        at = app(ds=with_hmm, page="data_groups", dataset_full=imported)
        warnings = [w.value for w in at.warning if "discards some results" in w.value]
        assert len(warnings) == 2  # beside both group-change buttons
        assert "HMM" in warnings[0]
        assert "curation" not in warnings[0].split("do not:")[1]  # named as carried, not lost
