"""A page's grouping must be the dataset's grouping.

Reported: a dataset grouped on Import by genotype + pulse_time + pulse_duration_min
opened on the Actograms and Phase shift pages grouped by **genotype alone** — a
different partition, presented as the default, with no way to rebuild the real one
because the two pulse columns were not even offered.

The cause is a name mismatch that looks like nothing. ``attrs['group_columns']``
records the metadata COLUMNS ticked at import, which is the honest record of the
choice; but two of those columns are stored as coords under different names
(``pulse_time`` -> ``pulse_zt_hour``), because their values are transformed on the
way in. Filtering ``group_columns`` down to "columns that are also coords" therefore
dropped them silently, and ``group_defining_coords`` could not offer them either: it
identifies metadata coords by "is a per-id coord AND an attr key", and those two
columns deliberately have no attr, since their unique values mix strings, numbers and
the NaN of an unpulsed cohort — which NetCDF cannot serialize.

So the fix records the provenance directly, and these tests hold it in place.
"""

import numpy as np
import pandas as pd
import pytest

import dam_utilities
import phase_shift as ps


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


class TestRenamedColumnsStayFindable:
    def test_the_renamed_coords_are_offered_for_grouping(self, pulse_ds_built):
        """The reported symptom: they were missing from the picker entirely."""
        offered = dam_utilities.group_defining_coords(pulse_ds_built)
        assert "pulse_zt_hour" in offered
        assert "pulse_duration_minutes" in offered

    def test_group_columns_still_records_what_was_ticked(self, pulse_ds_built):
        """Unchanged, deliberately: it answers "which columns did I choose", which is
        a different question from "which coords hold them"."""
        assert dam_utilities.get_group_columns(pulse_ds_built) == [
            "genotype",
            "pulse_time",
            "pulse_duration_min",
        ]

    def test_group_coord_names_gives_names_the_dataset_actually_has(self, pulse_ds_built):
        got = dam_utilities.get_group_coord_names(pulse_ds_built)
        assert got == ["genotype", "pulse_zt_hour", "pulse_duration_minutes"]
        assert all(c in pulse_ds_built.coords for c in got), (
            "every name returned has to be readable off the dataset — that is the "
            "whole point of it existing alongside get_group_columns"
        )

    def test_an_older_dataset_without_the_attr_still_resolves(self, pulse_ds_built):
        """A .nc saved before the attr existed maps its columns through the rename
        table instead, so it is not stuck with the broken default."""
        old = pulse_ds_built.copy()
        del old.attrs["group_coord_names"]
        assert dam_utilities.get_group_coord_names(old) == [
            "genotype",
            "pulse_zt_hour",
            "pulse_duration_minutes",
        ]

    def test_the_recorded_names_reproduce_the_datasets_own_partition(self, pulse_ds_built):
        """The test that would have caught the bug: re-deriving the grouping from the
        recorded coord names has to give the SAME groups as the group coord."""
        by_coords, _ = ps.group_labels(
            pulse_ds_built, tuple(dam_utilities.get_group_coord_names(pulse_ds_built))
        )
        by_group, _ = ps.group_labels(pulse_ds_built, ("group",))
        assert len(set(by_coords)) == len(set(by_group)) == 4
        # Same partition: two flies share a re-derived label iff they share a group.
        assert {frozenset(np.flatnonzero(by_coords == g)) for g in set(by_coords)} == {
            frozenset(np.flatnonzero(by_group == g)) for g in set(by_group)
        }

    def test_filtering_group_columns_to_coords_is_the_bug(self, pulse_ds_built):
        """Pinning the wrong answer so nobody reinvents it: this is exactly what the
        pages used to do, and it collapses three grouping factors to one."""
        offered = dam_utilities.group_defining_coords(pulse_ds_built)
        naive = [c for c in dam_utilities.get_group_columns(pulse_ds_built) if c in offered]
        assert naive == ["genotype"], "the old default; kept here to show what it cost"
        labels, _ = ps.group_labels(pulse_ds_built, tuple(naive))
        assert len(set(labels)) == 2, "two genotypes, not the four real groups"


class TestAttrsSurviveNetCDF:
    """Both new attrs must be plain string lists. The pulse columns have no attr of
    their own precisely because theirs could not be serialized."""

    def test_they_are_string_lists(self, pulse_ds_built):
        for key in ("metadata_coords", "group_coord_names"):
            val = pulse_ds_built.attrs[key]
            assert isinstance(val, list), f"{key} is {type(val).__name__}"
            assert all(isinstance(v, str) for v in val), f"{key} holds non-strings"

    def test_a_round_trip_keeps_them_readable(self, pulse_ds_built, tmp_path):
        """Through the app's own saver, not a bare ``to_netcdf``.

        A bare one cannot write this dataset at all — ``attrs['start_datetime']``
        holds Timestamps, which NetCDF refuses — and ``save_dataset_to_netcdf``
        exists to convert exactly those. Testing the raw call would have been
        testing a path the app never takes.
        """
        from load_and_save_datasets import load_dataset_from_netcdf, save_dataset_to_netcdf

        path = tmp_path / "rt.nc"
        save_dataset_to_netcdf(pulse_ds_built, str(path))
        back = load_dataset_from_netcdf(str(path))
        assert dam_utilities.get_group_coord_names(back) == [
            "genotype",
            "pulse_zt_hour",
            "pulse_duration_minutes",
        ]
        assert "pulse_zt_hour" in dam_utilities.group_defining_coords(back)


class TestPagesDefaultToTheDatasetGrouping:
    def test_the_actogram_picker_leads_with_the_group_coord(self, pulse_ds_built):
        from ui import filters

        options, default = filters.group_by_options(pulse_ds_built)
        assert default == ["group"]
        assert options[0] == "group"
        assert "pulse_zt_hour" in options, "re-grouping must still be possible"

    def test_the_group_option_says_what_it_is(self, pulse_ds_built):
        from ui import filters

        label = filters.group_by_label(pulse_ds_built, "group")
        assert "import" in label.lower()
        # It names the columns behind it, so the picker is self-explaining.
        for col in ("genotype", "pulse_time", "pulse_duration_min"):
            assert col in label

    def test_a_dataset_with_no_group_coord_falls_back(self, pulse_ds_built):
        from ui import filters

        bare = pulse_ds_built.drop_vars("group")
        options, default = filters.group_by_options(bare)
        assert "group" not in options
        assert default and default[0] in options
