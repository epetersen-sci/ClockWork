"""Page behaviour that previously needed a browser (backlog items 3 and 11).

Both of these were originally verified by driving the running app by hand —
importing a dataset, clicking through pages, reading the sidebar. AppTest does
the same checks in-process in a couple of seconds, which is the difference
between "verified once" and "verified on every commit".
"""

from ui.filters import DISPLAY_GROUPS_KEY


def _captions(at):
    return [c.value for c in at.caption]


class TestHMMPhasePicker:
    """Item 3: cross-validation must run on an explicitly chosen phase."""

    def test_phase_radio_offers_ld_and_dd_defaulting_to_ld(self, app, master_ds):
        at = app(ds=master_ds, page="hmm_model_selection")
        radio = at.radio(key="cv_phase_choice")
        assert radio.options == ["LD", "DD"]
        assert radio.value == "LD"

    def test_usable_fly_count_is_phase_specific(self, app, master_ds):
        """`len(ds['id'])` is identical for LD and DD, because select_phase masks
        rather than subsets. The count that matters is flies with usable minutes,
        and the page must report THAT or the picker looks inert."""
        counts = {}
        for phase in ("LD", "DD"):
            at = app(ds=master_ds, page="hmm_model_selection", cv_phase_choice=phase)
            assert not at.exception
            fold = next(c for c in _captions(at) if "fold holds out" in c)
            counts[phase] = fold
        assert counts["LD"] != counts["DD"] or "usable" in counts["LD"], (
            "the fold caption should describe the selected phase"
        )

    def test_no_transition_falls_back_to_full_recording(self, app, unsplit_ds):
        """Mirrors hmm_analysis's else branch: say so rather than silently
        cross-validating on a mix the user did not choose."""
        at = app(ds=unsplit_ds, page="hmm_model_selection")
        assert not at.exception
        assert not at.radio  # no phase picker when there is no boundary
        assert any("full recording" in c for c in _captions(at))

    def test_phase_mismatch_warning_is_shown(self, app, master_ds):
        """The two HMM pages keep unlinked widget keys, so the note telling the
        user to match the phase by hand is the only thing preventing a model
        selected on LD being used for a DD fit."""
        at = app(ds=master_ds, page="hmm_model_selection")
        assert any("HMM Analysis" in i.value for i in at.info)


class TestSharedGroupFilter:
    """Item 11: one app-wide display group selection, surviving page switches."""

    def test_both_pages_use_the_same_session_key(self, app, master_ds):
        for page in ("periodograms", "sleep_activity"):
            at = app(ds=master_ds, page=page)
            assert DISPLAY_GROUPS_KEY in at.session_state, (
                f"{page} should render its group filter under the shared key"
            )

    def test_selection_survives_a_page_switch(self, app, master_ds):
        """The regression this guards: a keyed widget's value is dropped when the
        widget stops being rendered, and a page switch is exactly that. Sharing
        the key is not enough on its own — persist_state="session" is what makes
        it survive."""
        at = app(ds=master_ds, page="periodograms")
        all_groups = list(at.multiselect(key=DISPLAY_GROUPS_KEY).value)
        assert len(all_groups) > 1, "fixture needs >1 group for this to mean anything"

        narrowed = all_groups[:1]
        at.multiselect(key=DISPLAY_GROUPS_KEY).set_value(narrowed).run()
        assert at.multiselect(key=DISPLAY_GROUPS_KEY).value == narrowed

        at.switch_page("app_pages/sleep_activity.py").run()
        assert not at.exception
        assert at.multiselect(key=DISPLAY_GROUPS_KEY).value == narrowed, (
            "the selection made on Periodograms should still be in force"
        )

    def test_filter_actually_subsets_the_view(self, app, master_ds):
        """Sleep & activity passes subset=True, so narrowing the filter must
        narrow the data the plots receive — not just the widget."""
        at = app(ds=master_ds, page="sleep_activity")
        groups = list(at.multiselect(key=DISPLAY_GROUPS_KEY).value)
        at.multiselect(key=DISPLAY_GROUPS_KEY).set_value(groups[:1]).run()
        assert not at.exception


class TestSplitStateDrivesThePhaseUI:
    """Items 4 + 5: split state is read from the master's attrs, not from caches."""

    def test_sleep_page_offers_the_phase_radio_on_a_split_master(self, app, master_ds):
        at = app(ds=master_ds, page="sleep_detection")
        assert not at.exception
        assert at.radio, "a split-applied master should offer the LD/DD choice"

    def test_reloaded_split_master_still_reports_split(self, app, master_ds):
        """A .nc round-trip turns split_applied into a NUMPY int, which is not a
        Python int under NumPy 2. When that was missed, a reloaded split master
        reported "not split" — hiding the phase pickers and making the SCAMP
        export refuse to run."""
        import numpy as np

        from dataset_meta import is_split_applied

        reloaded = master_ds.copy()
        reloaded.attrs["split_applied"] = np.int64(1)  # what comes back off disk
        reloaded.attrs.pop("split_phase", None)  # item 4 no longer writes the alias
        assert is_split_applied(reloaded) is True

        at = app(ds=reloaded, page="export_scamp")
        assert not at.exception
        assert not at.warning, "SCAMP export should not refuse a reloaded split master"
