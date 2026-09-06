"""Export shape contracts (backlog items 1, 2 and 5).

These are the tests that would have caught the drift the backlog recorded:
downstream GraphPad templates read columns POSITIONALLY, so a header row that
reorders or re-cases is a silent data corruption, not a cosmetic change.
"""

import numpy as np
import pytest

import export_helpers
from export_helpers import ZT_STAT_ORDER, phase_slice, zt_group_summary_table


class TestZTSummaryTable:
    """Item 2: one builder, so the two ZT exports cannot drift apart."""

    def test_stat_order_is_graphpad_not_alphabetical(self, binned_df):
        out = zt_group_summary_table(binned_df, "activity", 30)
        stats = [stat for _, stat in out.columns if stat]
        assert stats[:3] == ["Mean", "SD", "N"], (
            "GraphPad's grouped-table order is Mean, SD, N. Sorting the columns "
            "gives the alphabetical Mean, N, SD — the exact drift item 2 fixed."
        )

    def test_groups_are_alphabetical(self, binned_df):
        out = zt_group_summary_table(binned_df, "activity", 30)
        groups = [g for g, stat in out.columns if stat]
        assert groups == sorted(groups)

    def test_leading_zt_hours_column(self, binned_df):
        out = zt_group_summary_table(binned_df, "activity", 30)
        assert out.columns[0] == ("zt_hours", "")

    def test_casing_is_capitalised(self, binned_df):
        """Lowercase mean/sd/n was the other half of the drift."""
        out = zt_group_summary_table(binned_df, "activity", 30)
        assert not any(stat in ("mean", "sd", "n") for _, stat in out.columns)

    def test_stat_order_constant_matches_output(self, binned_df):
        out = zt_group_summary_table(binned_df, "activity", 30)
        first_group = out.columns[1][0]
        block = [stat for g, stat in out.columns if g == first_group]
        assert tuple(block) == ZT_STAT_ORDER


class TestPhaseSlice:
    """Item 5: the export pages re-slice on demand instead of reading caches."""

    @pytest.mark.parametrize("phase", ["LD", "DD"])
    def test_returns_a_real_slice_not_a_masked_view(self, master_ds, phase):
        """select_phase is NOT a substitute here — SCAMP needs equal-length files
        per board, so the time axis must actually shrink."""
        out = phase_slice(master_ds, phase)
        assert out.sizes["time"] < master_ds.sizes["time"]
        assert out.sizes["id"] == master_ds.sizes["id"]

    @pytest.mark.parametrize("phase", ["LD", "DD"])
    def test_sleep_masks_stay_int8(self, master_ds, phase):
        """Slicing pads with NaN and upcasts to float. Losing the restoration
        would silently change every per-phase .nc from int8 to float64."""
        out = phase_slice(master_ds, phase)
        for var in ("sleep", "sleep_short", "sleep_intermediate", "sleep_long"):
            if var in out.data_vars:
                assert out[var].dtype == np.int8, f"{var} upcast during {phase} slice"

    def test_uses_split_params_recorded_on_the_master(self, master_ds):
        """The parameters come from the master, which is what makes a reloaded
        .nc slice the same way the original session did."""
        ds = master_ds.copy()
        ds.attrs["gap_threshold_minutes"] = 45
        out = phase_slice(ds, "LD")
        assert out.attrs["gap_threshold_minutes"] == 45

    def test_stamps_the_canonical_phase(self, master_ds):
        assert phase_slice(master_ds, "DD").attrs["phase"] == "DD"


class TestBoutDataframeIsSingleSourceOfTruth:
    """Item 1: both sleep-bout exports come from raw_bout_dataframe."""

    def test_export_page_uses_the_shared_builder(self):
        from conftest import REPO_ROOT

        src = (REPO_ROOT / "app" / "app_pages" / "export_data.py").read_text(encoding="utf-8")
        assert "sleep_analysis.raw_bout_dataframe(ds)" in src
        assert "ds[bout_vars].to_dataframe()" not in src, (
            "the ad-hoc re-derivation was a strict subset of raw_bout_dataframe "
            "(no group, no sleep_state) and made the two pages disagree"
        )

    def test_the_two_files_are_named_apart(self):
        from conftest import REPO_ROOT

        export = (REPO_ROOT / "app" / "app_pages" / "export_data.py").read_text(encoding="utf-8")
        activity = (REPO_ROOT / "app" / "app_pages" / "sleep_activity.py").read_text(
            encoding="utf-8"
        )
        assert '"sleep_bouts.csv"' in export
        assert '"sleep_bouts_filtered.csv"' in activity
        assert '"sleep_bouts.csv"' not in activity, (
            "same filename, different contents: whichever the user opened last won"
        )


def test_phase_slice_is_the_only_slicer_the_export_pages_use():
    """Both export pages must go through the shared helper, or they can drift the
    way the two ZT tables did."""
    from conftest import REPO_ROOT

    for page in ("export_data.py", "export_scamp.py"):
        src = (REPO_ROOT / "app" / "app_pages" / page).read_text(encoding="utf-8")
        assert "export_helpers.phase_slice(" in src, f"{page} should slice via the helper"
        assert "session_state.dataset_LD" not in src
        assert "session_state.dataset_DD" not in src


def test_helper_exports_expected_names():
    for name in ("phase_slice", "zt_group_summary_table", "ZT_STAT_ORDER"):
        assert hasattr(export_helpers, name)
