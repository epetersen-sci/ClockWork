"""A page's grouping must be the dataset's grouping; a column has one name.

Reported first: a dataset grouped on Import by genotype + pulse_time +
pulse_duration_min opened on the Actograms and Phase shift pages grouped by
**genotype alone**, because the two pulse columns were stored under other coord
names (``pulse_zt_hour``, ``pulse_duration_minutes``) and nothing could find them.

The first fix recorded the coord names beside the column names. BACKLOG 21 found
the deeper problem: one column under two names meant two spellings of its values
too, so the same grouping gave ``dsmcherry-ZT21`` from Import and
``dsmcherry-21.0`` from Redefine groups. Now every metadata column is stored under
its own name, ``pulse_time`` holds the text the metadata wrote, and the hour is
parsed when an analysis needs it. These tests hold that, and the read-side
migration that keeps older ``.nc`` files working.
"""

import numpy as np
import pandas as pd
import pytest

from clockwork.core import dam_utilities
from clockwork.core import phase_shift as ps


@pytest.fixture
def pulse_metadata():
    """Metadata with the two columns that get renamed on the way in."""
    rows = []
    for monitor, (cond, zt, dur) in enumerate(
        [("LP", "ZT21", 20), ("noLP", "none", 0)], start=1
    ):
        for gene in ("Mito", "per"):
            rows.append(
                {
                    "id": f"{gene}_{cond}",
                    "Monitor": monitor,
                    "genotype": gene,
                    "condition": cond,
                    "pulse_time": zt,
                    "pulse_duration_min": dur,
                    "start_datetime": pd.Timestamp("2025-01-15 09:00"),
                    "stop_datetime": pd.Timestamp("2025-01-20 09:00"),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def pulse_ds_built(pulse_metadata):
    """A dataset built the way the Import page builds one, grouped on all three."""
    n_time = 2 * 1440
    ids = list(pulse_metadata["id"])
    data = pd.DataFrame(
        np.random.default_rng(0).integers(0, 5, (n_time, len(ids))),
        columns=ids,
        index=pd.RangeIndex(n_time),
    )
    return dam_utilities.create_xarray_dataset(
        data,
        pulse_metadata,
        group_columns=["genotype", "pulse_time", "pulse_duration_min"],
    )


class TestEachColumnHasOneName:
    def test_the_pulse_columns_are_offered_for_grouping(self, pulse_ds_built):
        """The reported symptom: they were missing from the picker entirely."""
        offered = dam_utilities.group_defining_coords(pulse_ds_built)
        assert "pulse_time" in offered
        assert "pulse_duration_min" in offered

    def test_the_grouping_is_recorded_under_the_names_ticked(self, pulse_ds_built):
        ticked = ["genotype", "pulse_time", "pulse_duration_min"]
        assert dam_utilities.get_group_columns(pulse_ds_built) == ticked
        assert dam_utilities.get_group_coord_names(pulse_ds_built) == ticked
        assert all(c in pulse_ds_built.coords for c in ticked)

    def test_pulse_time_is_kept_as_written(self, pulse_ds_built):
        values = {str(v) for v in pulse_ds_built["pulse_time"].values}
        assert values == {"ZT21", "none"}

    def test_the_hour_is_parsed_when_needed(self, pulse_ds_built):
        hours = dam_utilities.pulse_zt_hours(pulse_ds_built)
        by_id = dict(zip(map(str, pulse_ds_built["id"].values), hours))
        assert by_id["Mito_LP"] == 21.0
        assert np.isnan(by_id["Mito_noLP"])

    def test_import_and_regroup_give_the_same_labels(self, pulse_ds_built):
        """BACKLOG 21: the same grouping set two ways named the groups two ways."""
        cols = ["genotype", "pulse_time", "pulse_duration_min"]
        regrouped = dam_utilities.regroup_dataset(pulse_ds_built, cols)
        assert list(regrouped["group"].values) == list(pulse_ds_built["group"].values)
        # pulse_time as written; the duration is a number, so it reads as one.
        assert "Mito-ZT21-20.0" in {str(g) for g in pulse_ds_built["group"].values}

    def test_a_blank_reads_nan_in_every_label(self, pulse_ds_built):
        """The phase-shift page builds its own labels; an unpulsed control's
        pulse_time is "" and must read "nan" there too, not vanish ("Mito_")."""
        labels, _ = ps.group_labels(pulse_ds_built, ("genotype", "pulse_duration_min"))
        assert set(labels) >= {"Mito_20.0", "Mito_0.0"}
        blank = pulse_ds_built.assign_coords(
            pulse_time=("id", ["" if "noLP" in str(i) else "ZT21" for i in pulse_ds_built["id"].values])
        )
        labels, _ = ps.group_labels(blank, ("genotype", "pulse_time"))
        assert set(labels) == {"Mito_ZT21", "Mito_nan", "per_ZT21", "per_nan"}

    def test_spellings_are_the_owners_to_choose(self, pulse_metadata):
        """"ZT21" and "zt21" are the same hour but kept apart: a lab may write them
        differently on purpose, and it is not ClockWork's call to merge them."""
        meta = pulse_metadata.copy()
        meta.loc[meta["id"] == "per_LP", "pulse_time"] = "zt21"
        data = pd.DataFrame(
            np.zeros((1440, len(meta))), columns=list(meta["id"]), index=pd.RangeIndex(1440)
        )
        ds = dam_utilities.create_xarray_dataset(data, meta, group_columns=["pulse_time"])
        assert {str(g) for g in ds["group"].values} == {"ZT21", "zt21", "none"}

    def test_an_unreadable_pulse_time_fails_at_import(self, pulse_metadata):
        meta = pulse_metadata.copy()
        meta.loc[0, "pulse_time"] = "after lunch"
        data = pd.DataFrame(np.zeros((10, len(meta))), columns=list(meta["id"]), index=pd.RangeIndex(10))
        with pytest.raises(ValueError, match="Unparseable pulse_time"):
            dam_utilities.create_xarray_dataset(data, meta)

    def test_the_recorded_names_reproduce_the_datasets_own_partition(self, pulse_ds_built):
        """Re-deriving the grouping from the recorded names has to give the SAME
        groups as the group coord."""
        by_coords, _ = ps.group_labels(
            pulse_ds_built, tuple(dam_utilities.get_group_coord_names(pulse_ds_built))
        )
        by_group, _ = ps.group_labels(pulse_ds_built, ("group",))
        assert len(set(by_coords)) == len(set(by_group)) == 4
        # Same partition: two flies share a re-derived label iff they share a group.
        assert {frozenset(np.flatnonzero(by_coords == g)) for g in set(by_coords)} == {
            frozenset(np.flatnonzero(by_group == g)) for g in set(by_group)
        }


def _as_saved_before_backlog_21(ds):
    """``ds`` as a dataset saved before every column kept its own name."""
    hours = dam_utilities.pulse_zt_hours(ds)
    old = ds.drop_vars(["pulse_time", "pulse_duration_min"]).assign_coords(
        pulse_zt_hour=("id", hours.astype("float32")),
        pulse_duration_minutes=("id", ds["pulse_duration_min"].values),
    )
    old.attrs = dict(ds.attrs)
    old.attrs["group_coord_names"] = ["genotype", "pulse_zt_hour", "pulse_duration_minutes"]
    old.attrs["metadata_coords"] = [
        "pulse_zt_hour" if c == "pulse_time" else "pulse_duration_minutes" if c == "pulse_duration_min" else c
        for c in ds.attrs["metadata_coords"]
    ]
    return old


class TestAttrsSurviveNetCDF:
    """The grouping attrs must be plain string lists; the pulse columns have no
    attr of their own because their values could not be serialized."""

    def test_they_are_string_lists(self, pulse_ds_built):
        for key in ("metadata_coords", "group_coord_names"):
            val = pulse_ds_built.attrs[key]
            assert isinstance(val, list), f"{key} is {type(val).__name__}"
            assert all(isinstance(v, str) for v in val), f"{key} holds non-strings"

    def test_a_round_trip_keeps_the_text_and_the_names(self, pulse_ds_built, tmp_path):
        """Through the app's own saver, not a bare ``to_netcdf`` (which cannot
        write the Timestamp attrs this dataset carries)."""
        from clockwork.core.load_and_save_datasets import (
            load_dataset_from_netcdf,
            save_dataset_to_netcdf,
        )

        path = tmp_path / "rt.nc"
        save_dataset_to_netcdf(pulse_ds_built, str(path))
        back = load_dataset_from_netcdf(str(path))
        assert dam_utilities.get_group_coord_names(back) == [
            "genotype",
            "pulse_time",
            "pulse_duration_min",
        ]
        assert [str(v) for v in back["pulse_time"].values] == [
            str(v) for v in pulse_ds_built["pulse_time"].values
        ]
        assert "pulse_time" in dam_utilities.group_defining_coords(back)


class TestOlderFilesAreMigratedOnLoad:
    def test_an_older_nc_loads_under_the_new_names(self, pulse_ds_built, tmp_path):
        from clockwork.core.load_and_save_datasets import (
            load_dataset_from_netcdf,
            save_dataset_to_netcdf,
        )

        path = tmp_path / "old.nc"
        save_dataset_to_netcdf(_as_saved_before_backlog_21(pulse_ds_built), str(path))
        back = load_dataset_from_netcdf(str(path))
        assert "pulse_zt_hour" not in back.coords and "pulse_duration_minutes" not in back.coords
        # The original spelling was not kept by those versions; "ZT21" is chosen.
        assert {str(v) for v in back["pulse_time"].values} == {"ZT21", ""}
        np.testing.assert_array_equal(
            back["pulse_duration_min"].values, pulse_ds_built["pulse_duration_min"].values
        )
        assert dam_utilities.get_group_coord_names(back) == [
            "genotype",
            "pulse_time",
            "pulse_duration_min",
        ]
        assert {"pulse_time", "pulse_duration_min"} <= set(dam_utilities.group_defining_coords(back))
        np.testing.assert_array_equal(
            dam_utilities.pulse_zt_hours(back), dam_utilities.pulse_zt_hours(pulse_ds_built)
        )

    def test_the_settings_export_reads_older_attrs_as_column_names(self, pulse_ds_built):
        from clockwork.pipeline import GroupsConfig

        attrs = dict(pulse_ds_built.attrs, group_columns=["genotype", "pulse_zt_hour"])
        assert GroupsConfig.from_attrs(attrs).by == ["genotype", "pulse_time"]


class TestPagesDefaultToTheDatasetGrouping:
    def test_the_picker_speaks_metadata_column_names(self, pulse_ds_built):
        """The chips say what you ticked on Import."""
        from clockwork.app.ui import filters

        options, default = filters.group_by_options(pulse_ds_built)
        assert default == ["genotype", "pulse_time", "pulse_duration_min"]
        assert "pulse_time" in options and "pulse_zt_hour" not in options

    def test_the_selection_is_the_coord_names(self, pulse_ds_built):
        from clockwork.app.ui import filters

        assert filters.group_by_coords(
            pulse_ds_built, ["genotype", "pulse_time", "pulse_duration_min"]
        ) == ("genotype", "pulse_time", "pulse_duration_min")

    def test_the_import_grouping_is_recognised_so_labels_stay_readable(self, pulse_ds_built):
        """When the selection IS the import grouping, a page labels from ds['group'],
        so its labels match every other page's."""
        from clockwork.app.ui import filters

        assert filters.is_import_grouping(
            pulse_ds_built, ["genotype", "pulse_time", "pulse_duration_min"]
        )
        assert not filters.is_import_grouping(pulse_ds_built, ["genotype"])

    def test_a_dataset_with_no_group_coord_falls_back(self, pulse_ds_built):
        from clockwork.app.ui import filters

        bare = pulse_ds_built.drop_vars("group")
        options, default = filters.group_by_options(bare)
        assert default and all(c in options for c in default)
        # The recorded grouping is still readable even with the coord gone, so a page
        # can rebuild the same partition from columns — it just cannot shortcut to
        # ds['group'] for the labels.
        assert filters.is_import_grouping(bare, default)
        assert "group" not in bare.coords
