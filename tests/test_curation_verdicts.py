"""Curation's three verdicts, and why there are three.

``curate_dead_animals`` returns ``(live_data, dead_data, ...)``, and ``dead_data``
holds two different things: flies it REMOVED, and flies it only TRIMMED — a dead
tail cut off and the fly itself kept. A trimmed fly is in BOTH returned datasets and
neither carries a flag saying which happened, so membership of ``live_data`` is the
only thing that can tell them apart.

Reading ``dead_data`` as "the dropped flies" therefore counts a trimmed fly twice
and reports a fly as discarded that is still in the analysis. That is the mistake
:func:`curation_verdict_table` exists to prevent, and these tests are what keep it
prevented.
"""

import numpy as np
import pytest
import xarray as xr

import dam_utilities

MPD = 1440


def _ds(ids, *, active_until=None, genotypes=None, n_days=4):
    """Per-fly activity of 1 count/min up to ``active_until`` days, then silence."""
    n_time = n_days * MPD
    active_until = active_until or {}
    act = np.zeros((n_time, len(ids)))
    for j, fly in enumerate(ids):
        stop = int(active_until.get(fly, n_days) * MPD)
        act[:stop, j] = 1.0
    coords = {"id": list(ids), "time": np.arange(n_time, dtype=np.int64)}
    if genotypes:
        coords["genotype"] = ("id", np.array([genotypes[i] for i in ids]))
    return xr.Dataset({"activity": (("time", "id"), act)}, coords=coords)


def test_a_fly_only_in_live_is_kept():
    table = dam_utilities.curation_verdict_table(_ds(["a", "b"]), _ds([]))
    assert dict(zip(table.id, table.verdict)) == {"a": "kept", "b": "kept"}


def test_a_fly_only_in_dead_is_dropped():
    table = dam_utilities.curation_verdict_table(_ds(["a"]), _ds(["b"]))
    assert dict(zip(table.id, table.verdict)) == {"a": "kept", "b": "dropped"}


def test_a_fly_in_both_is_trimmed_and_appears_exactly_once():
    """The case the three verdicts exist for. Calling it 'dropped' would both
    double-count it and report a fly as discarded that is still being analysed."""
    table = dam_utilities.curation_verdict_table(_ds(["a", "b"]), _ds(["b"]))
    assert len(table) == 2, f"a trimmed fly appeared twice: {table.to_dict('records')}"
    assert dict(zip(table.id, table.verdict)) == {"a": "kept", "b": "trimmed"}


def test_the_counts_add_up_to_the_flies_involved():
    """kept + trimmed + dropped must equal the union, with nothing counted twice."""
    live, dead = _ds(["a", "b", "c"]), _ds(["b", "d", "e"])
    table = dam_utilities.curation_verdict_table(live, dead)
    counts = table.verdict.value_counts()
    assert int(counts.get("kept", 0)) == 2  # a, c
    assert int(counts.get("trimmed", 0)) == 1  # b
    assert int(counts.get("dropped", 0)) == 2  # d, e
    assert len(table) == 5 == table.id.nunique()


def test_before_curation_every_fly_is_simply_kept():
    """dead_data is None until curation runs; the table must still render."""
    table = dam_utilities.curation_verdict_table(_ds(["a", "b"]), None)
    assert set(table.verdict) == {"kept"}
    assert len(table) == 2


def test_no_dataset_gives_an_empty_frame_not_a_crash():
    assert dam_utilities.curation_verdict_table(None, None).empty
    assert dam_utilities.curation_verdict_table(_ds([]), _ds([])).empty


def test_last_active_day_is_the_last_minute_with_a_beam_break():
    """The statistic the dead call rests on, so it is what makes a borderline
    verdict checkable. A fly silent from day 2 must read ~2, not the record length."""
    table = dam_utilities.curation_verdict_table(
        _ds(["alive", "died"], active_until={"died": 2}, n_days=4)
    )
    by_id = dict(zip(table.id, table["last active (day)"]))
    assert by_id["died"] == pytest.approx(2.0, abs=0.01)
    assert by_id["alive"] == pytest.approx(4.0, abs=0.01)


def test_a_fly_that_never_moved_reads_zero_rather_than_raising():
    table = dam_utilities.curation_verdict_table(_ds(["silent"], active_until={"silent": 0}))
    assert table["last active (day)"].iloc[0] == 0.0
    assert table["total counts"].iloc[0] == 0.0


def test_gaps_are_not_counted_as_activity():
    """NaN is "no reading". It must not set last-active, which would make a fly
    that died early look as though it survived to its last gap."""
    ds = _ds(["a"], active_until={"a": 1}, n_days=4)
    ds["activity"][3 * MPD :, 0] = np.nan
    table = dam_utilities.curation_verdict_table(ds)
    assert table["last active (day)"].iloc[0] == pytest.approx(1.0, abs=0.01)


def test_total_counts_sums_the_activity():
    table = dam_utilities.curation_verdict_table(_ds(["a"], active_until={"a": 2}, n_days=4))
    assert table["total counts"].iloc[0] == 2 * MPD


def test_metadata_columns_appear_only_when_the_dataset_has_them():
    plain = dam_utilities.curation_verdict_table(_ds(["a"]))
    assert "genotype" not in plain.columns

    labelled = dam_utilities.curation_verdict_table(
        _ds(["a", "b"], genotypes={"a": "ctrl", "b": "mut"})
    )
    assert list(labelled["genotype"]) == ["ctrl", "mut"]
    assert all(c not in labelled.columns for c in ("sex", "flybox", "block"))


class TestTheVerdictFilterNoticesCuration:
    """The expander's whole purpose is to show the dropped flies beside the kept
    ones, and a keyed multiselect with ``default=`` cannot do that.

    Before curation the only verdict is "kept", so the widget stores ``['kept']``.
    Streamlit then ignores ``default=`` on every later run, so the selection
    survived curation and went on hiding exactly the flies there was now something
    to see. Found in the running app: the control read "Show: kept" while the
    caption promised the dropped flies were there too.
    """

    @staticmethod
    def _show(at):
        return next(m for m in at.multiselect if m.label == "Show")

    def test_curation_reveals_the_new_verdicts(self, app, master_ds):
        at = app(ds=master_ds, page="data_curate_split")
        assert self._show(at).value == ["kept"], "nothing is dropped before curation"

        # What curation leaves behind: a fly in both datasets was trimmed, one only
        # in dead_data was dropped.
        at.session_state["dataset"] = master_ds.isel(id=[0, 1, 2, 3])
        at.session_state["curated_dead_data"] = master_ds.isel(id=[3, 4, 5])
        at = at.run()
        assert not at.exception
        assert set(self._show(at).value) == {"kept", "trimmed", "dropped"}

    def test_a_hand_narrowed_selection_is_left_alone(self, app, master_ds):
        """The re-seed must be tied to the options changing, not run every time,
        or the filter could never be narrowed at all."""
        at = app(ds=master_ds, page="data_curate_split")
        at.session_state["dataset"] = master_ds.isel(id=[0, 1, 2, 3])
        at.session_state["curated_dead_data"] = master_ds.isel(id=[3, 4, 5])
        at = at.run()

        at.session_state["flyview_verdict"] = ["dropped"]
        at = at.run()
        assert self._show(at).value == ["dropped"]

    def test_only_the_verdicts_present_are_offered(self, app, master_ds):
        """A run where nothing was dropped should not offer "dropped" to look at."""
        at = app(ds=master_ds, page="data_curate_split")
        assert list(self._show(at).options) == ["kept"]


def test_the_real_curation_output_reconciles(master_ds):
    """Against curate_dead_animals itself rather than a hand-made pair, so the
    contract being read here is the one that function actually returns."""
    result = dam_utilities.curate_dead_animals(master_ds, min_alive_days=0.5)
    live, dead = result[0], result[1]
    table = dam_utilities.curation_verdict_table(live, dead)

    assert table.id.nunique() == len(table), "a fly was counted twice"
    kept_and_trimmed = set(table.loc[table.verdict != "dropped", "id"])
    assert kept_and_trimmed == {str(v) for v in live["id"].values}, (
        "every fly still in live_data must read kept or trimmed, and no other"
    )
