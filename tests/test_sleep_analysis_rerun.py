"""Re-running sleep analysis on a dataset that already has results (item 15).

That is the "Re-run sleep analysis with different parameters" checkbox, and it is
the normal path for any ``.nc`` loaded through Path B — such a file always
carries the previous run's ``sleep`` masks and its ``(id, sleep_bout_number)``
bout variables.

The failure mode was not a clean error. ``sleep_analysis`` derives ``analysis_ds``
from ``data`` and the per-fly loop calls ``.to_dataframe()`` on a slice of it, so
a slice still carrying the bout dimension makes pandas take the cartesian product
with ``time``: 17,281 timepoints x 134 bouts = 2.3 million rows per fly, every
timestamp repeated 134 times. The run spent a long time building those frames and
then died on the duplicated time index.
"""

import numpy as np
import pytest
import xarray as xr

import sleep_analysis as sa

BOUT_DIM = "sleep_bout_number"
SLEEP_MASKS = ("sleep", "sleep_short", "sleep_intermediate", "sleep_long")


@pytest.fixture
def analysed_ds(master_ds):
    """A dataset shaped like one reloaded from ``.nc``: sleep masks present AND
    the bout dimension present, which is what a saved analysis looks like."""
    ds = sa.sleep_analysis(master_ds, phase="both")
    assert BOUT_DIM in ds.dims, "fixture precondition: the first run creates bouts"
    assert "sleep" in ds.data_vars
    return ds


def test_rerun_does_not_raise(analysed_ds):
    out = sa.sleep_analysis(analysed_ds, phase="both")
    assert "sleep" in out.data_vars


def test_rerun_is_idempotent(analysed_ds):
    """Same parameters, same answer — a second run must not accumulate or shift
    results just because the first run's variables were present on the input."""
    out = sa.sleep_analysis(analysed_ds, phase="both")
    for var in SLEEP_MASKS:
        if var in analysed_ds.data_vars:
            np.testing.assert_array_equal(
                out[var].transpose("id", "time").values,
                analysed_ds[var].transpose("id", "time").values,
                err_msg=f"{var} changed on a re-run with identical parameters",
            )


def test_rerun_does_not_explode_the_row_count(analysed_ds, monkeypatch):
    """Guard the mechanism, not just the symptom.

    The crash was a consequence of the cartesian product; a future change could
    reintroduce the blow-up while dodging the duplicate-index error. Assert that
    no per-fly frame is ever wider than the time axis.
    """
    seen = []
    real_to_dataframe = xr.Dataset.to_dataframe

    def spy(self, *args, **kwargs):
        df = real_to_dataframe(self, *args, **kwargs)
        if "time" in self.dims:
            seen.append((len(df), self.sizes["time"]))
        return df

    monkeypatch.setattr(xr.Dataset, "to_dataframe", spy)
    sa.sleep_analysis(analysed_ds, phase="both")

    assert seen, "expected the per-fly loop to build dataframes"
    for n_rows, n_time in seen:
        assert n_rows <= n_time, (
            f"a per-fly frame had {n_rows} rows for {n_time} timepoints — the bout "
            "dimension is being broadcast against time again"
        )


def test_rerun_with_different_threshold_changes_the_result(analysed_ds):
    """The point of the checkbox: a different threshold must actually re-detect."""
    loose = sa.sleep_analysis(analysed_ds, phase="both", sleep_threshold_sec=60)
    strict = sa.sleep_analysis(analysed_ds, phase="both", sleep_threshold_sec=1500)
    n_loose = int((loose["sleep"].values == 1).sum())
    n_strict = int((strict["sleep"].values == 1).sum())
    assert n_loose > n_strict, (
        f"a 60s threshold should mark more minutes asleep than a 1500s one "
        f"({n_loose} vs {n_strict})"
    )


def test_bout_dimension_is_rebuilt_not_appended(analysed_ds):
    """The old bout dim is dropped wholesale, so its length tracks the new run
    rather than being the union of both."""
    out = sa.sleep_analysis(analysed_ds, phase="both", sleep_threshold_sec=1500)
    assert BOUT_DIM in out.dims
    assert out.sizes[BOUT_DIM] <= analysed_ds.sizes[BOUT_DIM], (
        "a stricter threshold should not produce more bout slots than the looser "
        "run it replaced — the old dimension is leaking through"
    )
