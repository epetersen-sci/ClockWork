"""The dataset-building steps: import, groups & subsets, curation, the split.

Two properties carry the weight (docs/cli-config.md):

- EQUIVALENCE. The pipeline builds the dataset the Import page built before it
  existed. The pages now call these steps, so this is what says moving the code
  did not move a number.
- ROUND TRIP. Every config reads back off the dataset it produced exactly as it
  went in, including through a .nc save and reload. The Export settings button
  is only as good as this.
"""

import dataclasses
import shutil

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pydantic import ValidationError

from clockwork import pipeline
from clockwork.core import dam_processor, dam_utilities
from clockwork.core.dataset_meta import PHASE_FULL, is_split_applied, stamp_phase
from clockwork.core.load_and_save_datasets import save_dataset_to_netcdf
from clockwork.pipeline import (
    CurationConfig,
    GroupsConfig,
    InputsConfig,
    SplitConfig,
)
from conftest import EXAMPLE_DIR, EXAMPLE_MONITORS, requires_example_data

pytestmark = requires_example_data

# The new attrs build_dataset records; everything else must match the old path.
NEW_IMPORT_ATTRS = {"import_metadata_file", "import_monitor_dir", "import_gap_threshold_hours"}


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    """Monitors 17 + 18 in a folder of their own, as a user would lay them out."""
    folder = tmp_path_factory.mktemp("exp_raw")
    for m in EXAMPLE_MONITORS:
        shutil.copy(EXAMPLE_DIR / f"Monitor{m}.txt", folder / f"Monitor{m}.txt")
    meta = pd.read_excel(EXAMPLE_DIR / "metadata.xlsx")
    meta[meta["Monitor"].isin(EXAMPLE_MONITORS)].to_csv(folder / "metadata_exp17.csv", index=False)
    return InputsConfig(metadata=folder / "metadata_exp17.csv", monitors=folder)


@pytest.fixture(scope="module")
def raw(inputs):
    return pipeline.read_monitors(inputs)


@pytest.fixture(scope="module")
def built(raw, inputs):
    return pipeline.build_dataset(raw, inputs)


@pytest.fixture(scope="module")
def chain(built):
    """The whole Data section, as a CLI run takes it."""
    groups = GroupsConfig(by=["genotype"], keep={"genotype": ["dsmcherry"]})
    curation = CurationConfig(min_alive_days=1.5, immobility_proportion=0.02)
    split = SplitConfig(discard_first_dd_day=True, gap_threshold_minutes=45)
    ds = pipeline.apply_groups(built, groups)
    ds = pipeline.curate(ds, curation).live
    ds = pipeline.split(ds, split)
    return ds, groups, curation, split


def _old_import_page(inputs):
    """What the Import page did before the pipeline existed, call for call."""
    processor = dam_processor.MetadataProcessor(
        str(inputs.metadata), str(inputs.monitors), gap_threshold_hours=inputs.gap_threshold_hours
    )
    metadata, data = processor.run()
    ds = dam_utilities.create_xarray_dataset(
        dam_utilities.convert_to_relative_time(data, metadata), metadata
    )
    stamp_phase(ds, PHASE_FULL, split_applied=False)
    ds.attrs["source_data_dir"] = str(inputs.metadata.parent.resolve())
    ds.attrs["experiment_name"] = dam_utilities.sanitize_experiment_name(
        dam_utilities.experiment_name_from_path(str(inputs.metadata))
    )
    ds.attrs.update(processor.integrity_scalars())
    return ds


class TestEquivalence:
    def test_the_pipeline_builds_what_the_import_page_built(self, built, inputs):
        old = _old_import_page(inputs)
        new = built.copy()
        new.attrs = {k: v for k, v in built.attrs.items() if k not in NEW_IMPORT_ATTRS}
        xr.testing.assert_identical(new, old)

    def test_it_only_adds_attrs(self, built, inputs):
        """ARCHITECTURE rule 1: the core is append-only."""
        old = _old_import_page(inputs)
        assert set(built.attrs) - set(old.attrs) == NEW_IMPORT_ATTRS

    def test_curation_is_the_core_function(self, built):
        config = CurationConfig()
        via_pipeline = pipeline.curate(built, config)
        live, dead, *_ = dam_utilities.curate_dead_animals(built)
        xr.testing.assert_identical(via_pipeline.live, live)
        xr.testing.assert_identical(via_pipeline.dead, dead)

    def test_the_split_marks_the_master_and_changes_no_data(self, built):
        out = pipeline.split(built, SplitConfig(discard_first_dd_day=True, gap_threshold_minutes=30))
        assert is_split_applied(out) and not is_split_applied(built)
        assert out.attrs["split_discard_first_dd_day"] == 1
        assert out.attrs["gap_threshold_minutes"] == 30
        xr.testing.assert_identical(out.drop_attrs(), built.drop_attrs())


class TestRoundTrip:
    def test_every_config_reads_back_as_it_went_in(self, chain, inputs):
        ds, groups, curation, split = chain
        assert CurationConfig.from_attrs(ds.attrs) == curation
        assert SplitConfig.from_attrs(ds.attrs) == split
        assert GroupsConfig.from_attrs(ds.attrs) == groups
        back = InputsConfig.from_attrs(ds.attrs)
        assert back.metadata == inputs.metadata.resolve()
        assert back.gap_threshold_hours == inputs.gap_threshold_hours

    def test_and_still_does_after_a_netcdf_round_trip(self, chain, tmp_path):
        """netCDF has no bool, no None and no nesting, and hands a one-element list
        back as a scalar: each of those is a way for a config to come back changed."""
        ds, groups, curation, split = chain
        path = tmp_path / "chain.nc"
        save_dataset_to_netcdf(ds, str(path))
        reloaded = pipeline.load_netcdf(path)
        assert CurationConfig.from_attrs(reloaded.attrs) == curation
        assert SplitConfig.from_attrs(reloaded.attrs) == split
        assert GroupsConfig.from_attrs(reloaded.attrs) == groups

    def test_a_step_that_never_ran_reads_back_as_none(self, built):
        """Not as "ran with its defaults", which would put it in an exported config."""
        assert CurationConfig.from_attrs(built.attrs) is None
        assert SplitConfig.from_attrs(built.attrs) is None

    def test_overrides_are_only_what_differs(self):
        assert CurationConfig().overrides() == {}
        assert CurationConfig(min_alive_days=3).overrides() == {"min_alive_days": 3.0}

    def test_a_regrouped_dataset_reports_column_names(self, built):
        """regroup_dataset records the coord name pulse_zt_hour where import records
        the column pulse_time. The config must speak the metadata's language."""
        regrouped = pipeline.apply_groups(built, GroupsConfig(by=["genotype", "Monitor"]))
        assert GroupsConfig.from_attrs(regrouped.attrs).by == ["genotype", "Monitor"]
        attrs = dict(regrouped.attrs, group_columns=["pulse_zt_hour"])
        assert GroupsConfig.from_attrs(attrs).by == ["pulse_time"]


class TestSubsets:
    def test_the_product_form_keeps_every_listed_value(self, built):
        out = pipeline.subset(built, {"genotype": ["dsmcherry"]})
        assert set(out["genotype"].values) == {"dsmcherry"}
        assert out.sizes["id"] == 32

    def test_values_match_by_meaning_not_type(self, built):
        """Monitor is an integer column; a config written by hand may say 17 or "17"."""
        a = pipeline.subset(built, {"Monitor": [17]})
        b = pipeline.subset(built, {"Monitor": ["17"]})
        c = pipeline.subset(built, {"Monitor": [17.0]})
        assert a.sizes["id"] == b.sizes["id"] == c.sizes["id"] == 32

    def test_the_combination_form_is_either_or(self, built):
        out = pipeline.subset(
            built,
            [{"genotype": "dsmcherry", "Monitor": 17}, {"genotype": "dsmcherry+Ldhmut"}],
        )
        assert out.sizes["id"] == 64

    def test_a_subset_that_keeps_nothing_is_an_error(self, built):
        with pytest.raises(ValueError, match="matches no flies"):
            pipeline.subset(built, {"genotype": ["no such genotype"]})

    def test_an_unknown_column_is_an_error_naming_it(self, built):
        with pytest.raises(ValueError, match="sex"):
            pipeline.subset(built, {"sex": ["F"]})

    def test_it_keeps_the_same_flies_whatever_the_grouping(self, built):
        keep = {"genotype": ["dsmcherry"]}
        before = pipeline.apply_groups(built, GroupsConfig(keep=keep))
        after = pipeline.apply_groups(built, GroupsConfig(by=["Monitor"], keep=keep))
        assert list(before["id"].values) == list(after["id"].values)

    def test_ticked_group_labels_become_a_keep_that_means_the_same_flies(self, built):
        """The Groups page's path: labels ticked by name -> keep -> the same flies."""
        label = str(built["group"].values[0])
        keep = pipeline.keep_for_groups(built, [label])
        out = pipeline.subset(built, keep)
        expected = [i for i, g in zip(built["id"].values, built["group"].values) if str(g) == label]
        assert list(out["id"].values) == expected
        assert all(set(combo) == {"genotype", "temperature"} for combo in keep)


class TestValidation:
    def test_a_misspelt_key_is_an_error(self):
        with pytest.raises(ValidationError, match="min_alive_day"):
            CurationConfig(min_alive_day=2)

    def test_out_of_range_values_are_errors(self):
        with pytest.raises(ValidationError):
            CurationConfig(immobility_proportion=1.5)
        with pytest.raises(ValidationError):
            SplitConfig(gap_threshold_minutes=-1)

    def test_inputs_need_exactly_one_source(self, tmp_path):
        with pytest.raises(ValidationError, match="not both"):
            InputsConfig(metadata=tmp_path, monitors=tmp_path, dataset=tmp_path / "x.nc")
        with pytest.raises(ValidationError, match="both metadata and monitors"):
            InputsConfig(metadata=tmp_path)
        with pytest.raises(ValidationError):
            InputsConfig()

    def test_configs_are_frozen(self):
        config = CurationConfig()
        with pytest.raises(ValidationError):
            config.min_alive_days = 5

    def test_splitting_without_a_dd_boundary_says_why(self, built):
        with pytest.raises(ValueError, match="first_DD_day"):
            pipeline.split(built.drop_vars("first_DD_day"), SplitConfig())

    def test_splitting_twice_is_refused(self, built):
        once = pipeline.split(built, SplitConfig())
        with pytest.raises(ValueError, match="already been split"):
            pipeline.split(once, SplitConfig())

    def test_an_empty_import_is_an_error_not_an_empty_dataset(self, raw, inputs):
        empty = dataclasses.replace(raw, data=raw.data.iloc[:, :0])
        with pytest.raises(pipeline.ImportFailed, match="no flies"):
            pipeline.build_dataset(empty, inputs)

    def test_a_non_string_legacy_phase_is_ambiguous(self, tmp_path):
        """The one case the loader cannot resolve alone, exactly as on the Import
        page before: a legacy split_phase it cannot read, and no canonical phase.
        (A string split_phase is migrated; no evidence at all means 'full'.)"""
        path = tmp_path / "legacy.nc"
        xr.Dataset({"activity": ("time", np.zeros(3))}, attrs={"split_phase": 1}).to_netcdf(path)
        with pytest.raises(pipeline.AmbiguousPhase) as caught:
            pipeline.load_netcdf(path)
        assert caught.value.dataset.sizes["time"] == 3

    def test_a_plain_nc_loads_as_the_full_recording(self, tmp_path):
        path = tmp_path / "plain.nc"
        xr.Dataset({"activity": ("time", np.zeros(3))}).to_netcdf(path)
        assert pipeline.load_netcdf(path).attrs["phase"] == PHASE_FULL
