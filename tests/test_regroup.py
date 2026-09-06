"""Redefining groups after import (backlog item 12).

``regroup_dataset`` existed and was correct but nothing called it, so the `group`
coord could only ever be set at import — and the README claimed otherwise. This
covers the wiring: which columns are offered, what regrouping produces, and the
cache invalidation, which is the part with teeth. `group` feeds every group-level
comparison, so a period analysis computed under the old labels must not survive
into the new ones.
"""

import numpy as np
import pytest

import dam_utilities
from ui.state import DERIVED_CACHE_KEYS


class TestCandidateColumns:
    def test_offers_the_metadata_columns(self, master_ds):
        assert set(dam_utilities.group_defining_coords(master_ds)) == {
            "genotype",
            "temperature",
        }

    def test_excludes_group_itself(self, master_ds):
        """`group` is the output of grouping, not an input to it."""
        assert "group" not in dam_utilities.group_defining_coords(master_ds)

    def test_excludes_timing_columns(self, master_ds):
        """start_datetime / stop_datetime / first_DD_day are per-id coords and
        attrs, so only the dtype and exclude-list filters keep them out."""
        offered = dam_utilities.group_defining_coords(master_ds)
        for col in ("start_datetime", "stop_datetime", "first_DD_day"):
            assert col in master_ds.coords, "fixture precondition"
            assert col not in offered

    def test_excludes_analysis_derived_coords(self, master_ds):
        """A coord added later by an analysis is not a metadata column, and is
        distinguishable because create_xarray_dataset never wrote an attr for it."""
        ds = master_ds.assign_coords(
            split_minute=("id", np.zeros(master_ds.sizes["id"], dtype=np.int32)),
            ac_rhythmic=("id", np.ones(master_ds.sizes["id"], dtype=bool)),
        )
        offered = dam_utilities.group_defining_coords(ds)
        assert "split_minute" not in offered
        assert "ac_rhythmic" not in offered


class TestRegroup:
    def test_narrowing_reduces_the_group_count(self, master_ds):
        before = len({str(g) for g in master_ds["group"].values})
        out = dam_utilities.regroup_dataset(master_ds, ["genotype"])
        after = len({str(g) for g in out["group"].values})
        assert after < before, f"expected fewer groups, got {before} -> {after}"

    def test_records_the_columns_it_used(self, master_ds):
        out = dam_utilities.regroup_dataset(master_ds, ["genotype"])
        assert dam_utilities.get_group_columns(out) == ["genotype"]

    def test_labels_join_with_a_dash(self, master_ds):
        out = dam_utilities.regroup_dataset(master_ds, ["genotype", "temperature"])
        for fly in out["id"].values:
            g = str(out["group"].sel(id=fly).values)
            geno = str(out["genotype"].sel(id=fly).values)
            temp = str(out["temperature"].sel(id=fly).values)
            assert g == f"{geno}-{temp}"

    def test_does_not_mutate_the_input(self, master_ds):
        before = [str(g) for g in master_ds["group"].values]
        dam_utilities.regroup_dataset(master_ds, ["genotype"])
        assert [str(g) for g in master_ds["group"].values] == before

    def test_unknown_column_is_skipped_not_fatal(self, master_ds):
        out = dam_utilities.regroup_dataset(master_ds, ["genotype", "not_a_column"])
        assert dam_utilities.get_group_columns(out) == ["genotype"]


class TestPageWiring:
    """The item's Verify step, driven through the page."""

    def test_regroup_section_is_offered(self, app, master_ds):
        at = app(ds=master_ds, page="data_groups")
        assert not at.exception
        assert at.multiselect(key="regroup_columns"), "the regroup control should render"

    def test_applying_a_regroup_updates_both_datasets(self, app, master_ds):
        at = app(ds=master_ds, page="data_groups")
        at.multiselect(key="regroup_columns").set_value(["genotype"]).run()
        at.button(key="apply_regroup").click().run()
        assert not at.exception

        for key in ("dataset", "dataset_full"):
            ds = at.session_state[key]
            assert dam_utilities.get_group_columns(ds) == ["genotype"], (
                f"{key} still carries the old grouping; dataset_full is the "
                "subset filter's restore point, so leaving it stale would put "
                "the old grouping back on Reset"
            )

    def test_regroup_clears_a_previously_computed_analysis(self, app, master_ds):
        """The part that matters: a period analysis computed under the old labels
        must be cleared rather than shown against the new ones."""
        stale = dict.fromkeys(("circ_df", "rhythmicity_df", "hmm_results"), "stale-result")
        at = app(ds=master_ds, page="data_groups", **stale)
        for key in stale:
            assert at.session_state[key] == "stale-result", "fixture precondition"

        at.multiselect(key="regroup_columns").set_value(["genotype"]).run()
        at.button(key="apply_regroup").click().run()
        assert not at.exception

        for key in stale:
            assert key not in at.session_state, (
                f"{key} survived a regroup; it was computed against group labels "
                "that no longer exist"
            )

    @pytest.mark.parametrize("key", ["circ_df", "hmm_results", "rhythmicity_summary"])
    def test_cleared_keys_are_registered_as_derived(self, key):
        """Guard the registry rather than each call site: anything cleared here is
        cleared because it is listed in DERIVED_CACHE_KEYS."""
        assert key in DERIVED_CACHE_KEYS
