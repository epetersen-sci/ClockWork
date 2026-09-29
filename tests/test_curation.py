"""Dead-fly curation on real flies: who is kept, who is removed, where each is cut.

``curate_dead_animals`` decides which flies every later analysis sees, and no test
called it. It sits behind the Curate page's Run button, which nothing pressed.

Two kinds of check, both on example_data's Monitors 17 and 18:

* a pinned snapshot (see snapshot_util.py) of the page-default outcome per fly —
  kept or removed, and how many minutes of it survive. Monitor 17 has flies with
  no DD data at all, so this holds real removals as well as real keeps;
* a planted death: a fly that is clearly alive in the real record, with its
  activity zeroed from day 6 on. Its right answer is known — kept, trimmed near
  day 6 — which is what ``_compute_moving``'s docstring said
  ``tests/test_simulated_death_curation`` checked, a file that never existed here.
"""

import numpy as np
import pytest
from snapshot_util import check_snapshot

import dam_utilities

# The Curate page's defaults.
PAGE_KW = {"time_window": 24, "prop_immobile": 0.01, "min_alive_days": 2.0}
MINUTES_PER_DAY = 1440


@pytest.fixture(scope="module")
def curated(example_ds):
    return dam_utilities.curate_dead_animals(example_ds, **PAGE_KW)


def test_page_default_outcome_is_pinned(example_ds, curated, update_snapshots):
    live, dead, *_ = curated
    kept = set(live["id"].values)
    records = {}
    for fly in example_ds["id"].values:
        rec = {"kept": fly in kept}
        if rec["kept"]:
            alive = live["is_alive"].sel(id=fly).values
            rec["alive_minutes"] = int((alive == 1).sum())
        records[str(fly)] = rec
    n_removed = sum(not r["kept"] for r in records.values())
    check_snapshot(
        "curation_example",
        records,
        update_snapshots,
        rtol=0,
        summary={"lines": [f"kept {len(kept)}/{len(records)}, removed {n_removed}"]},
    )


def test_the_counts_agree_with_the_datasets(example_ds, curated):
    live, dead, errors, success, before, after, removed, unchanged, trimmed = curated
    assert before == example_ds.sizes["id"]
    assert after == live.sizes["id"] == len(success)
    assert removed == before - after
    assert unchanged + trimmed == after
    assert errors == []


def test_curation_does_not_modify_its_input(example_ds, curated):
    assert "is_alive" not in example_ds
    assert "moving" not in example_ds


def test_the_parameters_are_recorded_on_the_result(curated):
    live = curated[0]
    assert live.attrs["curation_time_window_hours"] == 24
    assert live.attrs["curation_prop_immobile_threshold"] == 0.01
    assert live.attrs["curation_min_alive_days"] == 2.0


def _most_active_fly(ds):
    return str(ds["id"].values[int(np.nanargmax(ds["activity"].sum("time").values))])


def test_a_planted_death_is_trimmed_where_the_fly_stopped(example_ds):
    fly = _most_active_fly(example_ds)
    death_minute = 6 * MINUTES_PER_DAY
    ds = example_ds.sel(id=[fly]).copy(deep=True)
    act = ds["activity"].values
    act[death_minute:, 0] = 0
    ds["activity"] = (ds["activity"].dims, act)

    live, dead, *_ = dam_utilities.curate_dead_animals(ds, **PAGE_KW)
    assert list(live["id"].values) == [fly]
    alive = live["is_alive"].sel(id=fly).values
    last_alive = int(np.flatnonzero(alive == 1)[-1])
    # The rolling window is centred, so the last "active" minute sits up to half
    # a window past the true death — never before it, and never a day beyond.
    assert death_minute - 60 <= last_alive <= death_minute + 12 * 60


def test_a_death_before_min_alive_days_removes_the_fly(example_ds):
    fly = _most_active_fly(example_ds)
    ds = example_ds.sel(id=[fly]).copy(deep=True)
    act = ds["activity"].values
    act[MINUTES_PER_DAY:, 0] = 0  # dead after one day
    ds["activity"] = (ds["activity"].dims, act)

    live, dead, *_ = dam_utilities.curate_dead_animals(ds, **PAGE_KW)
    assert live.sizes["id"] == 0
    assert list(dead["id"].values) == [fly]
