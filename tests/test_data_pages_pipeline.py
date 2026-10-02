"""The Data pages, driven through their buttons, now that they call the pipeline.

tests/test_pipeline_data.py holds the steps to account; this holds the PAGES to
account for calling them. Before these, no test pressed Load & Validate, Create
Dataset, Apply LD/DD Split or Apply selection, so a page could have stopped
recording its settings without anything failing.
"""

import shutil

import pandas as pd
import pytest

from clockwork.pipeline import CurationConfig, GroupsConfig, InputsConfig, SplitConfig
from conftest import EXAMPLE_DIR, EXAMPLE_MONITORS, requires_example_data

pytestmark = requires_example_data


@pytest.fixture(scope="module")
def raw_folder(tmp_path_factory):
    folder = tmp_path_factory.mktemp("page_raw")
    for m in EXAMPLE_MONITORS:
        shutil.copy(EXAMPLE_DIR / f"Monitor{m}.txt", folder / f"Monitor{m}.txt")
    meta = pd.read_excel(EXAMPLE_DIR / "metadata.xlsx")
    meta[meta["Monitor"].isin(EXAMPLE_MONITORS)].to_csv(folder / "metadata_exp17.csv", index=False)
    return folder


def _import_through_the_page(app, folder, gap_hours=1.0):
    at = app(page="data_import")
    at.text_input(key="data_dir_input").set_value(str(folder))
    at.text_input(key="metadata_path_input").set_value(str(folder / "metadata_exp17.csv"))
    at.number_input[0].set_value(gap_hours)
    at.button(key="load_raw").click().run()
    assert not at.exception, at.exception
    at.button(key="create_dataset").click().run()
    assert not at.exception, at.exception
    return at


class TestImportPage:
    def test_create_dataset_records_the_inputs(self, app, raw_folder):
        at = _import_through_the_page(app, raw_folder, gap_hours=2.0)
        ds = at.session_state["dataset"]
        assert ds.sizes["id"] == 64
        back = InputsConfig.from_attrs(ds.attrs)
        assert back.metadata.name == "metadata_exp17.csv"
        assert back.monitors.resolve() == raw_folder.resolve()
        assert back.gap_threshold_hours == 2.0

    def test_it_is_the_same_dataset_the_pipeline_builds(self, app, raw_folder):
        from clockwork import pipeline

        at = _import_through_the_page(app, raw_folder)
        inputs = InputsConfig(metadata=raw_folder / "metadata_exp17.csv", monitors=raw_folder)
        direct = pipeline.build_dataset(pipeline.read_monitors(inputs), inputs, experiment_name="exp17")
        page = at.session_state["dataset"]
        assert page.attrs["experiment_name"] == direct.attrs["experiment_name"] == "exp17"
        assert page.identical(direct)


class TestGroupsPage:
    def test_apply_selection_records_the_subset_by_value(self, app, built_master):
        at = app(ds=built_master, page="data_groups")
        label = sorted({str(g) for g in built_master["group"].values})[0]
        at.multiselect(key="group_filter_select").set_value([label]).run()
        at.button(key="apply_group_filter").click().run()
        assert not at.exception, at.exception
        ds = at.session_state["dataset"]
        assert {str(g) for g in ds["group"].values} == {label}
        keep = GroupsConfig.from_attrs(ds.attrs).keep
        assert keep and all("genotype" in combo for combo in keep)

    def test_a_regroup_keeps_the_subset_and_its_record(self, app, built_master):
        at = app(ds=built_master, page="data_groups")
        label = sorted({str(g) for g in built_master["group"].values})[0]
        at.multiselect(key="group_filter_select").set_value([label]).run()
        at.button(key="apply_group_filter").click().run()
        n_kept = at.session_state["dataset"].sizes["id"]
        at.multiselect(key="regroup_columns").set_value(["Monitor"]).run()
        at.button(key="apply_regroup").click().run()
        assert not at.exception, at.exception
        ds = at.session_state["dataset"]
        assert ds.sizes["id"] == n_kept
        assert GroupsConfig.from_attrs(ds.attrs).by == ["Monitor"]
        assert "subset_keep" in ds.attrs


class TestCurateSplitPage:
    def test_curation_and_split_are_recorded_on_the_master(self, app, built_master):
        at = app(ds=built_master, page="data_curate_split")
        at.button(key="run_curation").click().run()
        assert not at.exception, at.exception
        assert CurationConfig.from_attrs(at.session_state["dataset"].attrs) == CurationConfig()
        at.checkbox(key="discard_first_dd_checkbox").check().run()
        at.number_input(key="gap_threshold_input").set_value(30).run()
        at.button(key="apply_split").click().run()
        assert not at.exception, at.exception
        ds = at.session_state["dataset"]
        assert SplitConfig.from_attrs(ds.attrs) == SplitConfig(
            discard_first_dd_day=True, gap_threshold_minutes=30
        )
        assert CurationConfig.from_attrs(ds.attrs) == CurationConfig()


@pytest.fixture(scope="module")
def built_master(raw_folder):
    from clockwork import pipeline

    inputs = InputsConfig(metadata=raw_folder / "metadata_exp17.csv", monitors=raw_folder)
    return pipeline.build_dataset(pipeline.read_monitors(inputs), inputs)


class TestGroupChangesWarnBeforeUndoingWork:
    """BACKLOG 22: changing groups rebuilds from the import-time copy, so it undoes
    curation and the split. The page has to say so before the button is pressed."""

    @staticmethod
    def _warnings(at):
        return [w.value for w in at.warning if "This undoes work" in w.value]

    def test_nothing_to_undo_means_no_warning(self, app, built_master):
        at = app(ds=built_master, page="data_groups")
        assert not at.exception
        assert self._warnings(at) == []

    def test_curation_and_split_are_named(self, app, built_master):
        from clockwork import pipeline

        curated = pipeline.curate(built_master, CurationConfig()).live
        worked = pipeline.split(curated, SplitConfig())
        at = app(ds=worked, page="data_groups", dataset_full=built_master)
        assert not at.exception
        warnings = self._warnings(at)
        assert warnings, "a curated, split dataset must warn before a group change"
        # Both expanders carry it, beside their buttons.
        assert len(warnings) == 2
        assert "curation" in warnings[0] and "LD/DD split" in warnings[0]
