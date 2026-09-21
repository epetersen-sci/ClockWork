"""Sleep detection moved onto the page that needs it. These hold the move safe.

Detection had a page of its own for one reason, and it was a good one: it writes
to the MASTER dataset, and the page it was carved out of narrows ``ds`` with a
sidebar group filter. Running it from below that filter would have dropped every
deselected fly from the master, permanently and silently. Alone on a page there
was no filter to race.

It is back on that page, so the race is back — unless ``ui.sleep_run`` always
reaches past the caller for the master, which is the first test here and the one
that matters most.

The second theme is that sleep analysis is TWO steps. Finding the bouts is a
per-fly pass over every minute; sorting them into short / intermediate / long is
a re-cut of the resulting table. Re-running the first to change the second is
minutes of work to answer a question that needs none, so the state thresholds
call ``reclassify_sleep_states`` — which has to agree with a full re-run exactly,
or it is a second implementation of the classification rather than a shortcut to
the same one.
"""

import numpy as np
import pytest
from conftest import _build_with_sleep_structure

import sleep_analysis


@pytest.fixture(scope="module")
def detected():
    """A dataset with sleep detected at the standard threshold."""
    return sleep_analysis.sleep_analysis(
        _build_with_sleep_structure(),
        phase="both",
        sleep_threshold_sec=300,
        short_max_min=30,
        inter_max_min=60,
    )


class TestReclassifyingIsTheSameAnswerFaster:
    NEW = dict(short_max_min=15, inter_max_min=45)

    @pytest.fixture(scope="class")
    def full_rerun(self):
        return sleep_analysis.sleep_analysis(
            _build_with_sleep_structure(), phase="both", sleep_threshold_sec=300, **self.NEW
        )

    @pytest.fixture(scope="class")
    def reclassified(self, detected):
        return sleep_analysis.reclassify_sleep_states(detected, **self.NEW)

    @pytest.mark.parametrize(
        "var", ["sleep_short", "sleep_intermediate", "sleep_long"]
    )
    def test_the_state_masks_are_identical(self, full_rerun, reclassified, var):
        """Not "close": identical. A shortcut that disagrees anywhere is a second
        implementation of the rule, and the two will drift."""
        a = np.asarray(full_rerun[var].transpose("id", "time").values)
        b = np.asarray(reclassified[var].transpose("id", "time").values)
        assert np.array_equal(a, b), f"{int((a != b).sum())} cells differ"

    def test_the_per_bout_labels_are_identical(self, full_rerun, reclassified):
        assert np.array_equal(
            full_rerun["sleep_state"].values.astype(str),
            reclassified["sleep_state"].values.astype(str),
        )

    def test_it_changes_the_answer_at_all(self, detected, reclassified):
        """Guards the trap the baseline-day correction fell into: a "shortcut" that
        silently does nothing also passes an equality test against itself."""
        before = np.asarray(detected["sleep_short"].transpose("id", "time").values)
        after = np.asarray(reclassified["sleep_short"].transpose("id", "time").values)
        assert not np.array_equal(before, after), (
            "narrowing short sleep from 30 to 15 min must move some minutes"
        )

    def test_the_sleep_mask_itself_is_untouched(self, detected, reclassified):
        """Where short ends moves no bout boundary, so which minutes are ASLEEP
        cannot change. If it does, the shortcut is re-detecting."""
        assert np.array_equal(
            np.asarray(detected["sleep"].values),
            np.asarray(reclassified["sleep"].values),
        )

    def test_the_bounds_are_recorded_and_the_threshold_is_not_disturbed(self, reclassified):
        assert reclassified.attrs["sleep_short_max_min"] == 15.0
        assert reclassified.attrs["sleep_inter_max_min"] == 45.0
        assert reclassified.attrs["sleep_threshold_seconds"] == 300

    def test_a_dataset_with_no_bouts_says_so(self):
        """Rather than writing empty masks, which read as "no short sleep"."""
        with pytest.raises(ValueError, match="no sleep bouts"):
            sleep_analysis.reclassify_sleep_states(_build_with_sleep_structure())


class TestDetectionRunsOnTheMaster:
    """The hazard the separate page existed to avoid."""

    def test_the_sleep_tab_offers_detection(self, app, master_ds):
        """Whether or not sleep is already there — folded into an expander when it
        is, offered outright when it is not, but present either way. It is the only
        place detection lives now."""
        at = app(ds=master_ds, page="sleep_activity")
        assert not at.exception, at.exception
        assert any(b.key == "sleep_activity_run" for b in at.button)

    def test_a_narrowed_group_filter_does_not_narrow_what_is_written(
        self, app, master_ds
    ):
        """The regression in one test. Narrow the sidebar to a single group, run
        detection, and every fly must still be on the master — the filter is a
        VIEW, and detection reaches past it."""
        from ui.filters import DISPLAY_GROUPS_KEY

        at = app(ds=master_ds, page="sleep_activity")
        groups = list(at.multiselect(key=DISPLAY_GROUPS_KEY).value)
        assert len(groups) > 1, "fixture needs more than one group to mean anything"
        at = at.multiselect(key=DISPLAY_GROUPS_KEY).set_value(groups[:1]).run()
        assert not at.exception

        at = at.button(key="sleep_activity_run").click().run()
        assert not at.exception, at.exception

        master = at.session_state["dataset"]
        assert "sleep" in master.data_vars, "detection did not write anything"
        assert master.sizes["id"] == master_ds.sizes["id"], (
            "flies were dropped from the MASTER by a display filter"
        )

    def test_the_threshold_defaults_to_the_standard_definition(self, app, master_ds):
        at = app(ds=master_ds, page="sleep_activity")
        assert at.number_input(key="sleep_activity_threshold").value == 300


class TestTheDefinitionIsPrinted:
    def test_it_says_what_asleep_means(self, states_ds):
        from ui import sleep_run

        line = sleep_run.sleep_definition(states_ds)
        assert "5 min" in line and "300 s" in line
        assert "immobility" in line

    def test_a_dataset_without_sleep_has_no_definition_to_print(self):
        from ui import sleep_run

        assert sleep_run.sleep_definition(_build_with_sleep_structure()) is None

    def test_the_page_prints_it_once_sleep_exists(self, app, states_ds):
        at = app(ds=states_ds, page="sleep_activity")
        assert not at.exception
        assert any("or more of continuous immobility" in c.value for c in at.caption), (
            "every figure on the page is 'sleep by this rule'; the rule has to be on it"
        )


class TestTheStateBoundsLiveWithTheStates:
    def test_sleep_states_offers_both_bounds(self, app, states_ds):
        at = app(ds=states_ds, page="sleep_states")
        assert not at.exception, at.exception
        assert at.number_input(key="ss_short_max").value == 30
        assert at.number_input(key="ss_inter_max").value == 60

    def test_the_apply_button_is_dead_until_something_changes(self, app, states_ds):
        """A live button that would recompute the same answer invites the click."""
        at = app(ds=states_ds, page="sleep_states")
        assert at.button(key="ss_apply_thresholds").disabled

    def test_changing_a_bound_reclassifies_the_master(self, app, states_ds):
        at = app(ds=states_ds, page="sleep_states")
        at = at.number_input(key="ss_short_max").set_value(15).run()
        assert not at.button(key="ss_apply_thresholds").disabled
        at = at.button(key="ss_apply_thresholds").click().run()
        assert not at.exception, at.exception
        assert at.session_state["dataset"].attrs["sleep_short_max_min"] == 15.0
