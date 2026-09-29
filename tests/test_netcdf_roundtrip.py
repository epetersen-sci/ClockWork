"""A dataset saved to .nc and loaded back is the dataset that was saved.

The .nc file is how an analysed experiment leaves one session and comes back in
the next ("Path B" on the Import page), so anything the round trip changes is a
change to every later analysis of that experiment. NetCDF cannot store booleans,
Timestamps, or a one-element list (it comes back as a scalar), and
``save_dataset_to_netcdf`` / ``load_dataset_from_netcdf`` exist to paper over
exactly that. Only two of their attributes had a test (test_grouping_provenance).

This uses real flies from example_data, with sleep analysis run on them, so the
file carries what a real saved experiment carries: Timestamp-list attrs from the
loader, the int8 sleep masks with their -1 "no data" sentinel, the (id, time)
masks next to (time, id) activity, and the per-bout table.
"""

import numpy as np
import pandas as pd
import pytest

import dam_utilities
import sleep_analysis
from load_and_save_datasets import load_dataset_from_netcdf, save_dataset_to_netcdf


@pytest.fixture(scope="module")
def saved_ds(example_ds):
    small = example_ds.isel(id=slice(0, 6)).copy()
    # Knock out an hour so the -1 sentinel really appears in the sleep masks.
    act = small["activity"].values.copy()
    act[3000:3060, 0] = np.nan
    small["activity"] = (small["activity"].dims, act)
    # 'moving' is added by curation on the real path; derive it the same way.
    small = dam_utilities._compute_moving(small)
    ds = sleep_analysis.sleep_analysis(small, phase="both", sleep_threshold_sec=300)
    # The attribute types the saver converts, beyond what the loader writes.
    ds.attrs["a_flag"] = True
    ds.attrs["numpy_flag"] = np.bool_(False)
    ds.attrs["one_time"] = pd.Timestamp("2025-01-18 09:00:00")
    ds.attrs["time_array"] = np.array(
        ["2025-01-15T09:00", "2025-01-18T09:00"], dtype="datetime64[ns]"
    )
    return ds


@pytest.fixture(scope="module")
def reloaded(saved_ds, tmp_path_factory):
    path = tmp_path_factory.mktemp("nc") / "roundtrip.nc"
    save_dataset_to_netcdf(saved_ds, str(path))
    back = load_dataset_from_netcdf(str(path))
    yield back
    back.close()


def test_the_saver_does_not_modify_the_dataset_it_is_given(saved_ds, reloaded):
    assert saved_ds.attrs["a_flag"] is True
    assert isinstance(saved_ds.attrs["one_time"], pd.Timestamp)


def test_every_variable_and_coord_comes_back(saved_ds, reloaded):
    assert set(reloaded.data_vars) == set(saved_ds.data_vars)
    assert set(reloaded.coords) == set(saved_ds.coords)
    assert list(reloaded["id"].values) == list(saved_ds["id"].values)


def test_dimension_order_survives(saved_ds, reloaded):
    """activity is (time, id), the sleep masks (id, time); code that extracts
    per-fly series relies on that and would index the wrong axis otherwise."""
    for var in saved_ds.data_vars:
        assert reloaded[var].dims == saved_ds[var].dims, var


def test_values_survive_including_the_missing_data_sentinel(saved_ds, reloaded):
    np.testing.assert_array_equal(reloaded["activity"].values, saved_ds["activity"].values)
    assert reloaded["sleep"].dtype == saved_ds["sleep"].dtype
    np.testing.assert_array_equal(reloaded["sleep"].values, saved_ds["sleep"].values)
    assert (reloaded["sleep"].values == -1).any(), "fixture should carry the -1 sentinel"


def test_boolean_attrs_come_back_as_booleans(reloaded):
    assert reloaded.attrs["a_flag"] is True
    assert reloaded.attrs["numpy_flag"] is False


def test_timestamp_attrs_come_back_as_timestamps(reloaded):
    assert reloaded.attrs["one_time"] == pd.Timestamp("2025-01-18 09:00:00")
    arr = reloaded.attrs["time_array"]
    assert isinstance(arr, np.ndarray) and arr.dtype == "datetime64[ns]"
    assert list(arr) == list(
        np.array(["2025-01-15T09:00", "2025-01-18T09:00"], dtype="datetime64[ns]")
    )


def test_a_one_element_timestamp_list_stays_a_list(saved_ds, reloaded):
    """The loader writes ``start_datetime`` as ``[Timestamp]``; NetCDF hands a
    one-element list back as a scalar, which the loader must re-wrap."""
    assert saved_ds.attrs["start_datetime"] == [pd.Timestamp("2025-01-15 09:00:00")]
    assert reloaded.attrs["start_datetime"] == [pd.Timestamp("2025-01-15 09:00:00")]


def test_list_attrs_are_readable_through_their_getters(saved_ds, reloaded):
    """A one-element STRING list (``temperature: ['25C']``) has no type recorded
    and does come back as a bare string. That is why the app reads list attrs
    through getters that re-wrap, never directly — which is what this holds."""
    assert saved_ds.attrs["temperature"] == ["25C"]
    assert dam_utilities.get_group_columns(reloaded) == dam_utilities.get_group_columns(saved_ds)
    assert dam_utilities.get_group_coord_names(reloaded) == dam_utilities.get_group_coord_names(
        saved_ds
    )


def test_a_reloaded_dataset_still_splits_into_the_same_phases(saved_ds, reloaded):
    for phase in ("LD", "DD"):
        before, _ = dam_utilities.select_phase(saved_ds, phase)
        after, _ = dam_utilities.select_phase(reloaded, phase)
        np.testing.assert_array_equal(after["activity"].values, before["activity"].values)
